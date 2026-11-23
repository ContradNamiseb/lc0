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

// Finds each ONNX-imported Squeeze-and-Excitation sub-block (the fixed
// ReduceMean/MatMul/Add/activation/MatMul/Add/Reshape/Split/Sigmoid/Mul/Add
// chain Converter::MakeSqueezeAndExcite emits, converter.cc:371-401) plus
// the residual-block tail that immediately follows it in MakeConvBlock
// (mixin-add + final activation, converter.cc:432-436), and replaces the
// whole thing with a single fused SEResidualOp node.
//
// Matches on op-type chain shape anchored by friendly-name suffix
// ("/se/reduce_mean", which this pass owns jointly with converter.cc's
// naming), the same two-stage approach ReplaceKdaScan uses for
// TensorIterator + "/scan". Unlike ReplaceKdaScan, no TensorIterator body is
// involved here -- w1/b1/w2/b2 are plain top-level ONNX initializers, so
// they're pulled directly off the matched MatMul/Add nodes' second inputs.
class ReplaceSqueezeExcite : public ov::pass::MatcherPass {
 public:
  OPENVINO_RTTI("ReplaceSqueezeExcite", "0");
  ReplaceSqueezeExcite();
};

}  // namespace openvino_backend
}  // namespace lczero
