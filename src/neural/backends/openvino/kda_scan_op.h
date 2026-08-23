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

// Fused replacement for the TensorIterator that OpenVINO's ONNX frontend
// builds from converter.cc's KDA-recurrence Scan op. Profiling showed that
// TensorIterator (per-step state copy-in/copy-out across a 64-step
// sequential scan) is >98% of OpenVINO's total inference time on a
// KDA-hybrid net -- see docs/inference-backends-handoff.md.
//
// This op is direction-agnostic: converter.cc already reorders q/k/v/decay/
// beta into each direction's board-traversal order (and un-reorders the
// output afterwards) *outside* the Scan node, so all this op does is the
// generic per-step recurrence body over axis 1, matching:
//   - converter.cc's EmitKdaLayer Scan body, and
//   - sycl/common_kernels.dp.cpp's kdaRecurrenceValueParallel
// bit-for-bit. If either of those changes, this must change with it.
//
// Inputs (all float32, batch dim N dynamic):
//   0: q           [N, 64, heads, key_dim]
//   1: k           [N, 64, heads, key_dim]
//   2: v           [N, 64, heads, value_dim]
//   3: raw_decay   [N, 64, heads, key_dim]
//   4: beta        [N, 64, heads, 1]
//   5: dt_bias         [1, heads, key_dim]  (constant)
//   6: neg_decay_scale [1, heads, 1]        (constant, = -exp(a_log))
// Output:
//   0: mixed       [N, 64, heads, value_dim]  (the scan outputs; the final
//                   recurrent state is not exposed -- nothing downstream of
//                   the original Scan node ever consumed it either)
class KdaScanOp : public ov::op::Op {
 public:
  OPENVINO_OP("KdaScan", "lc0_extension");

  KdaScanOp() = default;
  KdaScanOp(const ov::Output<ov::Node>& q, const ov::Output<ov::Node>& k,
            const ov::Output<ov::Node>& v,
            const ov::Output<ov::Node>& raw_decay,
            const ov::Output<ov::Node>& beta,
            const ov::Output<ov::Node>& dt_bias,
            const ov::Output<ov::Node>& neg_decay_scale,
            int direction_count = 8,
            std::vector<int> directions = {1, 2, 3, 4, 5, 6, 7, 8});

  void validate_and_infer_types() override;
  std::shared_ptr<ov::Node> clone_with_new_inputs(
      const ov::OutputVector& new_args) const override;
  bool visit_attributes(ov::AttributeVisitor& visitor) override;

  bool evaluate(ov::TensorVector& outputs,
                const ov::TensorVector& inputs) const override;
  bool has_evaluate() const override;

  // Static dims, resolved by validate_and_infer_types() from input shapes
  // (only the batch dim is ever dynamic). -1 until shape inference has run.
  // Used by network_openvino.cc to JIT-compile the GPU custom-layer kernel
  // for the exact dims this model instance uses.
  int heads() const { return heads_; }
  int key_dim() const { return key_dim_; }
  int value_dim() const { return value_dim_; }
  int direction_count() const { return direction_count_; }
  const std::vector<int>& directions() const { return directions_; }

  // Lower clamp on log_decay. Must match converter.cc's kLogDecayFloor.
  static constexpr float kLogDecayFloor = -10.0f;

 private:
  int heads_ = -1;
  int key_dim_ = -1;
  int value_dim_ = -1;
  int direction_count_ = 8;
  std::vector<int> directions_{1, 2, 3, 4, 5, 6, 7, 8};
};

}  // namespace openvino_backend
}  // namespace lczero
