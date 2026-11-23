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

#include "neural/backends/openvino/kda_scan_pass.h"

#include <openvino/core/rt_info.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/tensor_iterator.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/strided_slice.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include "neural/backends/openvino/kda_scan_op.h"
#include "utils/exception.h"

namespace lczero {
namespace openvino_backend {

namespace {

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Finds the external data input of `ti` whose source node's friendly name
// ends with `suffix`. Throws if not found -- a miss here means converter.cc's
// naming and this pass have drifted apart, which must fail loudly rather
// than silently fall back to TensorIterator.
//
// `fallback_suffix` covers one frontend variance: for a single-direction net
// the pre-scan Concat (named "/scan/q") has exactly one input, and a frontend
// is free to fold that away -- leaving the direction Gather, named
// "/dir0/q", as the TensorIterator input. converter.cc now always emits the
// Concat, so the primary suffix is what current exports produce; the
// fallback keeps older single-direction exports loadable.
ov::Output<ov::Node> FindInputBySuffix(
    const std::shared_ptr<ov::op::v0::TensorIterator>& ti,
    const std::string& suffix, const std::string& fallback_suffix = "") {
  for (size_t i = 0; i < ti->get_input_size(); ++i) {
    auto src = ti->input_value(i);
    if (EndsWith(src.get_node()->get_friendly_name(), suffix)) return src;
  }
  if (!fallback_suffix.empty()) {
    for (size_t i = 0; i < ti->get_input_size(); ++i) {
      auto src = ti->input_value(i);
      if (EndsWith(src.get_node()->get_friendly_name(), fallback_suffix)) {
        return src;
      }
    }
  }
  throw Exception(
      "OpenVINO KdaScan pass: TensorIterator '" + ti->get_friendly_name() +
      "' has no external input ending in '" + suffix +
      "' -- converter.cc's KDA export naming must have changed.");
}

// The per-head constants (dt_bias, neg_decay_scale) are referenced from
// outer scope inside the Scan body in the ONNX graph, which OpenVINO's ONNX
// frontend folds into the TensorIterator's *body* as internal Constant
// nodes rather than exposing as external invariant inputs. Pull them out
// and rebuild them as ordinary outer-graph constants.
std::shared_ptr<ov::op::v0::Constant> FindBodyConstantBySuffix(
    const std::shared_ptr<ov::op::v0::TensorIterator>& ti,
    const std::string& suffix) {
  for (const auto& node : ti->get_body()->get_ops()) {
    if (!EndsWith(node->get_friendly_name(), suffix)) continue;
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (c) return c;
  }
  throw Exception("OpenVINO KdaScan pass: TensorIterator '" +
                  ti->get_friendly_name() +
                  "' body has no Constant ending in '" + suffix + "'.");
}

// Walks back from a KdaScanOp input to the unpermuted, square-major
// tensor. The ONNX export builds each scan input as a Concat over the
// per-direction groups, each of which is a Gather (the direction's
// token->square permutation) over a Slice of the full tensor. The kernel
// now applies that permutation itself from kDirectionTable, so the graph
// copy has to be bypassed -- feeding it the Concat as well would permute
// twice.
//
// Do not compare get_type_name() against a literal: it returns const char*,
// so `==` compares pointers and is always false. That silently turned this
// whole function into a no-op.
ov::Output<ov::Node> TraceToUnpermutedInput(ov::Output<ov::Node> input) {
  auto node = input.get_node_shared_ptr();
  if (ov::is_type<ov::op::v0::Concat>(node) && node->get_input_size() > 0) {
    node = node->input_value(0).get_node_shared_ptr();
  }
  if ((ov::is_type<ov::op::v8::Gather>(node) ||
       ov::is_type<ov::op::v7::Gather>(node) ||
       ov::is_type<ov::op::v1::Gather>(node)) &&
      node->get_input_size() > 0) {
    node = node->input_value(0).get_node_shared_ptr();
  }
  if ((ov::is_type<ov::op::v8::Slice>(node) ||
       ov::is_type<ov::op::v1::StridedSlice>(node)) &&
      node->get_input_size() > 0) {
    return node->input_value(0);
  }
  throw Exception(
      "OpenVINO KdaScan pass: could not trace '" +
      input.get_node()->get_friendly_name() +
      "' back to an unpermuted tensor (stopped at a " +
      std::string(node->get_type_name()) +
      "). The kernel permutes internally, so feeding it the graph's "
      "permuted tensor would double-permute and silently produce wrong "
      "results -- failing instead.");
}

// Finds the node that reassembles the per-direction scan outputs back into
// square order. The kernel already writes square-major, so that reassembly
// is what the fused op must replace -- replacing ti->output(1) instead
// would leave the inverse permutation in place on top of a result that is
// already in square order.
//
// The reassembly is the Slice->Gather->Concat chain for multi-direction
// nets. With one direction the trailing Concat has a single input and may
// be folded away by the frontend (converter.cc emits it either way), in
// which case the Gather itself -- the inverse permutation -- is the last
// node to bypass, and returning it is correct: replacing its output puts
// the square-major result exactly where the Gather's output used to be.
std::shared_ptr<ov::Node> FindDownstreamReorderedOutput(
    const ov::Output<ov::Node>& ti_out, int direction_count) {
  for (const auto& slice_in : ti_out.get_target_inputs()) {
    auto slice_node = slice_in.get_node()->shared_from_this();
    for (const auto& gather_in : slice_node->output(0).get_target_inputs()) {
      auto gather_node = gather_in.get_node()->shared_from_this();
      for (const auto& concat_in :
           gather_node->output(0).get_target_inputs()) {
        auto concat_node = concat_in.get_node()->shared_from_this();
        if (ov::is_type<ov::op::v0::Concat>(concat_node)) return concat_node;
      }
      // No downstream Concat. Only legitimate for a single direction, where
      // the reassembly Concat is trivially foldable; for more directions the
      // Concat is load-bearing and its absence means the graph changed.
      if (direction_count == 1 &&
          ov::is_type<ov::op::v8::Gather>(gather_node)) {
        return gather_node;
      }
      if (direction_count == 1 &&
          ov::is_type<ov::op::v7::Gather>(gather_node)) {
        return gather_node;
      }
      if (direction_count == 1 &&
          ov::is_type<ov::op::v1::Gather>(gather_node)) {
        return gather_node;
      }
    }
  }
  return nullptr;
}

}  // namespace

ReplaceKdaScan::ReplaceKdaScan(int direction_count,
                               std::vector<int> directions) {
  auto pattern = ov::pass::pattern::wrap_type<ov::op::v0::TensorIterator>();

  ov::matcher_pass_callback callback =
      [direction_count,
       directions = std::move(directions)](ov::pass::pattern::Matcher& m) {
    auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(m.get_match_root());
    if (!ti) return false;

    // Only touches KDA-recurrence TensorIterators, identified the same way
    // the OpenVINO backend's profiling dump identified them: by the "/scan"
    // suffix converter.cc gives every KDA Scan node's friendly name.
    if (!EndsWith(ti->get_friendly_name(), "/scan")) return false;

    auto q = TraceToUnpermutedInput(FindInputBySuffix(
        ti, "/scan/q", direction_count == 1 ? "/dir0/q" : ""));
    auto k = TraceToUnpermutedInput(FindInputBySuffix(
        ti, "/scan/k", direction_count == 1 ? "/dir0/k" : ""));
    auto v = TraceToUnpermutedInput(FindInputBySuffix(
        ti, "/scan/v", direction_count == 1 ? "/dir0/v" : ""));
    auto decay = TraceToUnpermutedInput(FindInputBySuffix(
        ti, "/scan/decay", direction_count == 1 ? "/dir0/decay" : ""));
    auto beta = TraceToUnpermutedInput(FindInputBySuffix(
        ti, "/scan/beta", direction_count == 1 ? "/dir0/beta" : ""));
    auto dt_bias_c = FindBodyConstantBySuffix(ti, "/dt_bias");
    auto neg_decay_scale_c =
        FindBodyConstantBySuffix(ti, "/neg_decay_scale");

    auto dt_bias = ov::op::v0::Constant::create(
        ov::element::f32, dt_bias_c->get_shape(),
        dt_bias_c->get_vector<float>());
    auto neg_decay_scale = ov::op::v0::Constant::create(
        ov::element::f32, neg_decay_scale_c->get_shape(),
        neg_decay_scale_c->get_vector<float>());

    auto kda_scan = std::make_shared<KdaScanOp>(
        q, k, v, decay, beta, dt_bias, neg_decay_scale, direction_count,
        directions);
    kda_scan->set_friendly_name(ti->get_friendly_name() + "/fused");

    // Output 0 of the original Scan/TensorIterator is the final recurrent
    // state, which converter.cc never consumes (only scan_out[1] is used) --
    // confirm that still holds before dropping it, rather than silently
    // producing a wrong graph if some future exporter change starts using
    // it.
    if (!ti->output(0).get_target_inputs().empty()) {
      throw Exception(
          "OpenVINO KdaScan pass: TensorIterator '" + ti->get_friendly_name() +
          "' final-state output is consumed downstream -- KdaScanOp does "
          "not compute it and this replacement would be silently wrong.");
    }

    auto mixed4 = FindDownstreamReorderedOutput(ti->output(1), direction_count);
    if (!mixed4) {
      throw Exception(
          "OpenVINO KdaScan pass: TensorIterator '" +
          ti->get_friendly_name() +
          "' has no downstream Slice->Gather->Concat reassembling the "
          "per-direction outputs. Falling back to replacing the scan "
          "output directly would leave the inverse permutation applied to "
          "an already square-major result -- failing instead.");
    }
    mixed4->output(0).replace(kda_scan->output(0));
    ov::copy_runtime_info(ti, kda_scan);
    return true;
  };

  auto matcher = std::make_shared<ov::pass::pattern::Matcher>(
      pattern, "ReplaceKdaScan");
  register_matcher(matcher, callback);
}

}  // namespace openvino_backend
}  // namespace lczero
