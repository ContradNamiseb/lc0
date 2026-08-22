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
// ends with `suffix` (e.g. "/scan/q"). Throws if not found -- a miss here
// means converter.cc's naming and this pass have drifted apart, which must
// fail loudly rather than silently fall back to TensorIterator.
ov::Output<ov::Node> FindInputBySuffix(
    const std::shared_ptr<ov::op::v0::TensorIterator>& ti,
    const std::string& suffix) {
  for (size_t i = 0; i < ti->get_input_size(); ++i) {
    auto src = ti->input_value(i);
    if (EndsWith(src.get_node()->get_friendly_name(), suffix)) return src;
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

}  // namespace

ReplaceKdaScan::ReplaceKdaScan() {
  auto pattern = ov::pass::pattern::wrap_type<ov::op::v0::TensorIterator>();

  ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
    auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(m.get_match_root());
    if (!ti) return false;

    // Only touches KDA-recurrence TensorIterators, identified the same way
    // the OpenVINO backend's profiling dump identified them: by the "/scan"
    // suffix converter.cc gives every KDA Scan node's friendly name.
    if (!EndsWith(ti->get_friendly_name(), "/scan")) return false;

    auto q = FindInputBySuffix(ti, "/scan/q");
    auto k = FindInputBySuffix(ti, "/scan/k");
    auto v = FindInputBySuffix(ti, "/scan/v");
    auto decay = FindInputBySuffix(ti, "/scan/decay");
    auto beta = FindInputBySuffix(ti, "/scan/beta");
    auto dt_bias_c = FindBodyConstantBySuffix(ti, "/dt_bias");
    auto neg_decay_scale_c =
        FindBodyConstantBySuffix(ti, "/neg_decay_scale");

    auto dt_bias = ov::op::v0::Constant::create(
        ov::element::f32, dt_bias_c->get_shape(),
        dt_bias_c->get_vector<float>());
    auto neg_decay_scale = ov::op::v0::Constant::create(
        ov::element::f32, neg_decay_scale_c->get_shape(),
        neg_decay_scale_c->get_vector<float>());

    auto kda_scan = std::make_shared<KdaScanOp>(q, k, v, decay, beta,
                                                dt_bias, neg_decay_scale);
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

    ti->output(1).replace(kda_scan->output(0));
    ov::copy_runtime_info(ti, kda_scan);
    return true;
  };

  auto matcher = std::make_shared<ov::pass::pattern::Matcher>(
      pattern, "ReplaceKdaScan");
  register_matcher(matcher, callback);
}

}  // namespace openvino_backend
}  // namespace lczero
