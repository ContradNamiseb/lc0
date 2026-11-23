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

#pragma once

#include <openvino/op/op.hpp>

namespace lczero {
namespace openvino_backend {

// Fused replacement for converter.cc's Squeeze-and-Excitation sub-block plus
// the residual-block tail that immediately follows it (skip-add + final
// activation, Converter::MakeConvBlock). SE is 10 generic ONNX ops per
// residual block (ReduceMean/MatMul/Add/activation/MatMul/Add/Reshape/Split/
// Sigmoid/Mul) plus 2 more for the residual add and activation -- exactly
// the "many small ops -> per-op GPU dispatch overhead" shape this backend
// already measured as the bottleneck once KdaScanOp fixed the KDA
// recurrence (see the profiling comment on ReplaceKdaScan/de5fe4e).
//
// Deliberately does NOT attempt to fuse the surrounding convolutions --
// unlike the KDA recurrence, OpenVINO's native Conv is a mature, tuned
// primitive on this hardware; only the small-op SE/residual tail is fused.
//
// Inputs (all float32):
//   0: se_input    [N, C, 8, 8]   -- conv2's raw output
//   1: skip_input  [N, C, 8, 8]   -- the residual block's own input
//   2: w1          [C, se_filters]        (constant)
//   3: b1          [se_filters]           (constant)
//   4: w2          [se_filters, 2*C]      (constant)
//   5: b2          [2*C]                  (constant)
// Output:
//   0: output      [N, C, 8, 8]
//
// Math (matches Converter::MakeSqueezeAndExcite, converter.cc:371-401, plus
// MakeConvBlock's trailing mixin-add + activation, converter.cc:432-436):
//   avg[c]        = mean_hw(se_input[c,:,:])
//   h[j]          = activation(sum_c avg[c]*w1[c,j] + b1[j])
//   o[k]          = sum_j h[j]*w2[j,k] + b2[k]                 k in [0, 2C)
//   gamma[c]      = o[c],  beta[c] = o[C+c]
//   output[c,h,w] = activation(sigmoid(gamma[c])*se_input[c,h,w] + beta[c] +
//                              skip_input[c,h,w])
// The same `activation` is used both places, matching converter.cc using
// default_activation_ for both MakeSqueezeAndExcite's FC1 and the block's
// final activation.
class SEResidualOp : public ov::op::Op {
 public:
  OPENVINO_OP("SEResidual", "lc0_extension");

  // Only the two activation shapes converter.cc's default_activation_ can
  // actually produce for this backend (opset=17 forces the Mish softplus
  // decomposition rather than the native opset-18+ Mish op -- see
  // se_residual_pass.cc). Any other activation kind found downstream of a
  // matched SE block is a converter.cc drift, not a case to silently guess
  // at.
  enum class Activation { kRelu, kMish };

  SEResidualOp() = default;
  SEResidualOp(const ov::Output<ov::Node>& se_input,
               const ov::Output<ov::Node>& skip_input,
               const ov::Output<ov::Node>& w1, const ov::Output<ov::Node>& b1,
               const ov::Output<ov::Node>& w2, const ov::Output<ov::Node>& b2,
               Activation activation);

  void validate_and_infer_types() override;
  std::shared_ptr<ov::Node> clone_with_new_inputs(
      const ov::OutputVector& new_args) const override;
  bool visit_attributes(ov::AttributeVisitor& visitor) override;

  bool evaluate(ov::TensorVector& outputs,
                const ov::TensorVector& inputs) const override;
  bool has_evaluate() const override;

  // Static dims, resolved by validate_and_infer_types() from input shapes.
  // -1 until shape inference has run. Used by network_openvino.cc to
  // JIT-compile the GPU custom-layer kernel for this model instance's dims.
  int channels() const { return channels_; }
  int se_filters() const { return se_filters_; }
  Activation activation() const { return activation_; }

 private:
  int channels_ = -1;
  int se_filters_ = -1;
  Activation activation_ = Activation::kRelu;
};

}  // namespace openvino_backend
}  // namespace lczero
