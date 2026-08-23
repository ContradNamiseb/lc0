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

#include "neural/backends/openvino/kda_scan_op.h"

#include "neural/kda_directions.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace lczero {
namespace openvino_backend {

KdaScanOp::KdaScanOp(const ov::Output<ov::Node>& q,
                     const ov::Output<ov::Node>& k,
                     const ov::Output<ov::Node>& v,
                     const ov::Output<ov::Node>& raw_decay,
                     const ov::Output<ov::Node>& beta,
                     const ov::Output<ov::Node>& dt_bias,
                     const ov::Output<ov::Node>& neg_decay_scale,
                     int direction_count,
                     std::vector<int> directions)
    : Op({q, k, v, raw_decay, beta, dt_bias, neg_decay_scale}),
      direction_count_(direction_count),
      directions_(std::move(directions)) {
  constructor_validate_and_infer_types();
}

void KdaScanOp::validate_and_infer_types() {
  // Output shape is q's [N, 64, heads] with v's value_dim appended.
  const auto& q_shape = get_input_partial_shape(0);
  const auto& v_shape = get_input_partial_shape(2);
  ov::Dimension value_dim = ov::Dimension::dynamic();
  if (v_shape.rank().is_static() && v_shape.size() == 4 &&
      v_shape[3].is_static()) {
    value_dim = v_shape[3];
    value_dim_ = static_cast<int>(v_shape[3].get_length());
  }
  ov::PartialShape out_shape{ov::Dimension::dynamic(), 64,
                             ov::Dimension::dynamic(), value_dim};
  if (q_shape.rank().is_static() && q_shape.size() == 4) {
    out_shape[0] = q_shape[0];
    out_shape[2] = q_shape[2];
    if (q_shape[2].is_static()) heads_ = static_cast<int>(q_shape[2].get_length());
    if (q_shape[3].is_static()) key_dim_ = static_cast<int>(q_shape[3].get_length());
  }
  set_output_type(0, get_input_element_type(0), out_shape);
}

std::shared_ptr<ov::Node> KdaScanOp::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
  check_new_args_count(this, new_args);
  return std::make_shared<KdaScanOp>(new_args.at(0), new_args.at(1),
                                     new_args.at(2), new_args.at(3),
                                     new_args.at(4), new_args.at(5),
                                     new_args.at(6), direction_count_,
                                     directions_);
}

bool KdaScanOp::visit_attributes(ov::AttributeVisitor&) { return true; }

bool KdaScanOp::has_evaluate() const {
  return get_input_element_type(0) == ov::element::f32;
}

namespace {
inline float Softplus(float x) {
  return (x > 0.0f ? x : 0.0f) + std::log1p(std::exp(-std::fabs(x)));
}
inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

bool KdaScanOp::evaluate(ov::TensorVector& outputs,
                         const ov::TensorVector& inputs) const {
  if (inputs[0].get_element_type() != ov::element::f32) return false;

  const auto& q_shape = inputs[0].get_shape();  // [N, 64, H, key_dim]
  const int N = static_cast<int>(q_shape[0]);
  const int heads = static_cast<int>(q_shape[2]);
  const int key_dim = static_cast<int>(q_shape[3]);
  const int value_dim = static_cast<int>(inputs[2].get_shape()[3]);

  outputs[0].set_shape({static_cast<size_t>(N), 64,
                        static_cast<size_t>(heads),
                        static_cast<size_t>(value_dim)});

  const float* q = inputs[0].data<const float>();
  const float* k = inputs[1].data<const float>();
  const float* v = inputs[2].data<const float>();
  const float* raw_decay = inputs[3].data<const float>();
  const float* beta = inputs[4].data<const float>();
  const float* dt_bias = inputs[5].data<const float>();          // [H, key_dim]
  const float* neg_decay_scale = inputs[6].data<const float>();  // [H]
  float* mixed = outputs[0].data<float>();

  const float scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
  std::vector<float> state(static_cast<size_t>(key_dim) * value_dim);

  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < heads; ++h) {
      std::fill(state.begin(), state.end(), 0.0f);
      const float neg_scale_h = neg_decay_scale[h];
      const int dir_group = h / (heads / direction_count_);
      const int dir = directions_[dir_group];

      for (int t = 0; t < 64; ++t) {
        const int sq = KdaSquareForToken(dir, t);
        const size_t qk_base =
            ((static_cast<size_t>(n) * 64 + sq) * heads + h) * key_dim;
        const size_t v_base =
            ((static_cast<size_t>(n) * 64 + sq) * heads + h) * value_dim;
        const size_t beta_idx = (static_cast<size_t>(n) * 64 + sq) * heads + h;

        const float* q_t = q + qk_base;
        const float* k_t = k + qk_base;
        const float* v_t = v + v_base;
        const float* decay_t = raw_decay + qk_base;

        float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
        for (int key = 0; key < key_dim; ++key) {
          q_norm_sq += q_t[key] * q_t[key];
          k_norm_sq += k_t[key] * k_t[key];
        }
        const float q_norm =
            1.0f / std::sqrt(q_norm_sq > 1e-12f ? q_norm_sq : 1e-12f);
        const float k_norm =
            1.0f / std::sqrt(k_norm_sq > 1e-12f ? k_norm_sq : 1e-12f);

        for (int key = 0; key < key_dim; ++key) {
          const float decay_x = decay_t[key] + dt_bias[h * key_dim + key];
          const float log_decay_raw = neg_scale_h * Softplus(decay_x);
          const float log_decay = log_decay_raw > KdaScanOp::kLogDecayFloor
                                       ? log_decay_raw
                                       : KdaScanOp::kLogDecayFloor;
          const float decay = std::exp(log_decay);
          float* row = &state[static_cast<size_t>(key) * value_dim];
          for (int val = 0; val < value_dim; ++val) row[val] *= decay;
        }

        const float update_rate = Sigmoid(beta[beta_idx]);

        std::vector<float> prediction(value_dim, 0.0f);
        for (int key = 0; key < key_dim; ++key) {
          const float k_normed = k_t[key] * k_norm;
          const float* row = &state[static_cast<size_t>(key) * value_dim];
          for (int val = 0; val < value_dim; ++val) {
            prediction[val] += k_normed * row[val];
          }
        }

        std::vector<float> delta(value_dim);
        for (int val = 0; val < value_dim; ++val) {
          delta[val] = update_rate * (v_t[val] - prediction[val]);
        }

        std::vector<float> out(value_dim, 0.0f);
        const float q_scale = q_norm * scale;
        for (int key = 0; key < key_dim; ++key) {
          const float k_normed = k_t[key] * k_norm;
          const float q_normed = q_t[key] * q_scale;
          float* row = &state[static_cast<size_t>(key) * value_dim];
          for (int val = 0; val < value_dim; ++val) {
            row[val] += k_normed * delta[val];
            out[val] += q_normed * row[val];
          }
        }

        float* out_ptr = mixed + v_base;
        for (int val = 0; val < value_dim; ++val) out_ptr[val] = out[val];
      }
    }
  }

  return true;
}

}  // namespace openvino_backend
}  // namespace lczero
