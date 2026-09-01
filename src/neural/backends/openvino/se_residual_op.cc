/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "neural/backends/openvino/se_residual_op.h"

#include <cmath>
#include <vector>

namespace lczero {
namespace openvino_backend {

SEResidualOp::SEResidualOp(const ov::Output<ov::Node>& se_input,
                          const ov::Output<ov::Node>& skip_input,
                          const ov::Output<ov::Node>& w1,
                          const ov::Output<ov::Node>& b1,
                          const ov::Output<ov::Node>& w2,
                          const ov::Output<ov::Node>& b2,
                          Activation activation)
    : Op({se_input, skip_input, w1, b1, w2, b2}), activation_(activation) {
  constructor_validate_and_infer_types();
}

void SEResidualOp::validate_and_infer_types() {
  // Output shape matches se_input (== skip_input) exactly: [N, C, 8, 8].
  const auto& in_shape = get_input_partial_shape(0);
  const auto& w1_shape = get_input_partial_shape(2);
  if (in_shape.rank().is_static() && in_shape.size() == 4 &&
      in_shape[1].is_static()) {
    channels_ = static_cast<int>(in_shape[1].get_length());
  }
  if (w1_shape.rank().is_static() && w1_shape.size() == 2 &&
      w1_shape[1].is_static()) {
    se_filters_ = static_cast<int>(w1_shape[1].get_length());
  }
  set_output_type(0, get_input_element_type(0), in_shape);
}

std::shared_ptr<ov::Node> SEResidualOp::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
  check_new_args_count(this, new_args);
  return std::make_shared<SEResidualOp>(new_args.at(0), new_args.at(1),
                                        new_args.at(2), new_args.at(3),
                                        new_args.at(4), new_args.at(5),
                                        activation_);
}

bool SEResidualOp::visit_attributes(ov::AttributeVisitor&) { return true; }

bool SEResidualOp::has_evaluate() const {
  return get_input_element_type(0) == ov::element::f32;
}

namespace {
inline float Softplus(float x) {
  return (x > 0.0f ? x : 0.0f) + std::log1p(std::exp(-std::fabs(x)));
}
inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float Mish(float x) { return x * std::tanh(Softplus(x)); }

inline float Activate(float x, SEResidualOp::Activation act) {
  switch (act) {
    case SEResidualOp::Activation::kRelu:
      return x > 0.0f ? x : 0.0f;
    case SEResidualOp::Activation::kMish:
      return Mish(x);
  }
  return x;
}

}  // namespace

bool SEResidualOp::evaluate(ov::TensorVector& outputs,
                            const ov::TensorVector& inputs) const {
  if (inputs[0].get_element_type() != ov::element::f32) return false;

  const auto& shape = inputs[0].get_shape();  // [N, C, 8, 8]
  const int N = static_cast<int>(shape[0]);
  const int C = static_cast<int>(shape[1]);
  const int HW = static_cast<int>(shape[2] * shape[3]);
  const int se_filters = static_cast<int>(inputs[2].get_shape()[1]);

  outputs[0].set_shape(shape);

  const float* se_input = inputs[0].data<const float>();
  const float* skip_input = inputs[1].data<const float>();
  const float* w1 = inputs[2].data<const float>();  // [C, se_filters]
  const float* b1 = inputs[3].data<const float>();  // [se_filters]
  const float* w2 = inputs[4].data<const float>();  // [se_filters, 2*C]
  const float* b2 = inputs[5].data<const float>();  // [2*C]
  float* output = outputs[0].data<float>();

  std::vector<float> avg(C), h1(se_filters), o(2 * C);

  for (int n = 0; n < N; ++n) {
    const float* x = se_input + static_cast<size_t>(n) * C * HW;
    const float* skip = skip_input + static_cast<size_t>(n) * C * HW;
    float* out = output + static_cast<size_t>(n) * C * HW;

    for (int c = 0; c < C; ++c) {
      float sum = 0.0f;
      const float* row = x + static_cast<size_t>(c) * HW;
      for (int i = 0; i < HW; ++i) sum += row[i];
      avg[c] = sum / static_cast<float>(HW);
    }

    for (int j = 0; j < se_filters; ++j) {
      float sum = b1[j];
      for (int c = 0; c < C; ++c) sum += avg[c] * w1[c * se_filters + j];
      h1[j] = Activate(sum, activation_);
    }

    for (int k = 0; k < 2 * C; ++k) {
      float sum = b2[k];
      for (int j = 0; j < se_filters; ++j) sum += h1[j] * w2[j * 2 * C + k];
      o[k] = sum;
    }

    for (int c = 0; c < C; ++c) {
      const float gamma = Sigmoid(o[c]);
      const float beta = o[C + c];
      const float* x_row = x + static_cast<size_t>(c) * HW;
      const float* skip_row = skip + static_cast<size_t>(c) * HW;
      float* out_row = out + static_cast<size_t>(c) * HW;
      for (int i = 0; i < HW; ++i) {
        out_row[i] =
            Activate(gamma * x_row[i] + beta + skip_row[i], activation_);
      }
    }
  }

  return true;
}

}  // namespace openvino_backend
}  // namespace lczero
