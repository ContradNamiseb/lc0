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

#include <openvino/pass/matcher_pass.hpp>

namespace lczero {
namespace openvino_backend {

// Finds each ONNX-imported KDA recurrence (an ov::op::v0::TensorIterator
// produced by OpenVINO's ONNX frontend from converter.cc's Scan op) and
// replaces it with a single fused KdaScanOp node. TensorIterator pays a
// per-iteration state copy-in/copy-out cost across 64 sequential steps;
// profiling on a real KDA-hybrid net showed this is >98% of OpenVINO's total
// inference time, so this -- and its GPU custom-kernel counterpart -- is the
// actual fix for OpenVINO's speed deficit vs SYCL, not the ONNX-vs-native-IR
// import path (see docs/inference-backends-handoff.md).
//
// Matches purely on external input naming (each TensorIterator's data
// inputs are named /encoderN/scan/q, .../k, .../v, .../decay, .../beta by
// converter.cc's exporter, which this pass owns and controls), so it does
// not need to understand the internal body subgraph OpenVINO's ONNX
// frontend generated -- except for pulling the two per-head constants
// (dt_bias, neg_decay_scale) out of the body, since the frontend folds
// those in as internal Constants rather than external invariant inputs.
class ReplaceKdaScan : public ov::pass::MatcherPass {
 public:
  OPENVINO_RTTI("ReplaceKdaScan", "0");
  ReplaceKdaScan();
};

}  // namespace openvino_backend
}  // namespace lczero
