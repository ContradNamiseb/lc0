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

#include "neural/backends/openvino/se_residual_pass.h"

#include <openvino/core/rt_info.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/softplus.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/tanh.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include "neural/backends/openvino/se_residual_op.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace openvino_backend {

namespace {

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Every intermediate tensor in a MakeSqueezeAndExcite/MakeConvBlock chain
// has exactly one consumer -- that's how the exporter builds it. More or
// fewer than one means either this isn't the block converter.cc emits, or
// some other pass/graph-sharing has drifted the assumption; fail loudly
// either way rather than guessing which consumer is "the" chain.
std::shared_ptr<ov::Node> SingleConsumer(const ov::Output<ov::Node>& output,
                                         const std::string& context) {
  const auto& targets = output.get_target_inputs();
  if (targets.size() != 1) {
    throw Exception("OpenVINO SEResidual pass: expected exactly one "
                    "consumer of '" + context + "', found " +
                    std::to_string(targets.size()) + ".");
  }
  return targets.begin()->get_node()->shared_from_this();
}

template <typename T>
std::shared_ptr<T> ExpectType(const std::shared_ptr<ov::Node>& node,
                              const std::string& what) {
  auto typed = ov::as_type_ptr<T>(node);
  if (!typed) {
    throw Exception("OpenVINO SEResidual pass: expected " + what +
                    " but found " + std::string(node->get_type_name()) +
                    " ('" + node->get_friendly_name() + "').");
  }
  return typed;
}

std::shared_ptr<ov::op::v0::Constant> ExpectConstant(
    const ov::Output<ov::Node>& input, const std::string& what) {
  auto c = ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr());
  if (!c) {
    throw Exception("OpenVINO SEResidual pass: expected " + what +
                    " to be a Constant but found " +
                    std::string(input.get_node()->get_type_name()) + ".");
  }
  return c;
}

// Matches either a bare Relu, or the Softplus->Tanh->Mul(tanh_out, input)
// decomposition Converter::MakeMish emits when it can't use the native
// opset-18+ Mish op -- which is always, for this backend: network_openvino.cc
// forces opset=17 (see converter_options.opset there), and MakeMish only
// takes the native-op branch at opset>=18 (converter.cc:305). Returns the
// activation kind and the chain's final output.
std::pair<SEResidualOp::Activation, ov::Output<ov::Node>> MatchActivation(
    const ov::Output<ov::Node>& input, const std::string& context) {
  // Mish's shape is a diamond, not a line: `input` feeds both the Softplus
  // that starts the Mish chain AND the Mul at the end of it (which
  // multiplies tanh(softplus(x)) by the original x). So `input` genuinely
  // has two consumers under Mish -- scan all of them for the one that
  // starts a recognized activation, rather than assuming exactly one.
  std::shared_ptr<ov::op::v0::Relu> relu;
  std::shared_ptr<ov::op::v4::SoftPlus> softplus;
  for (const auto& target : input.get_target_inputs()) {
    auto node = target.get_node()->shared_from_this();
    if (auto r = ov::as_type_ptr<ov::op::v0::Relu>(node)) relu = r;
    if (auto s = ov::as_type_ptr<ov::op::v4::SoftPlus>(node)) softplus = s;
  }
  if (relu) return {SEResidualOp::Activation::kRelu, relu->output(0)};
  if (!softplus) {
    throw Exception(
        "OpenVINO SEResidual pass: expected a Relu or Softplus (Mish) "
        "consumer of '" + context + "' but found neither among its " +
        std::to_string(input.get_target_inputs().size()) + " consumer(s).");
  }
  auto tanh = ExpectType<ov::op::v0::Tanh>(
      SingleConsumer(softplus->output(0), context + "/softplus"),
      "Tanh after " + context + "/softplus");
  auto mul = ExpectType<ov::op::v1::Multiply>(
      SingleConsumer(tanh->output(0), context + "/tanh"),
      "Mul after " + context + "/tanh");
  // Mish multiplies tanh(softplus(x)) by the original x, not by anything
  // else -- verify, don't assume, or a coincidental Softplus->Tanh->Mul
  // elsewhere in the graph could get silently fused as if it were Mish.
  if (mul->input_value(0) != input && mul->input_value(1) != input) {
    throw Exception(
        "OpenVINO SEResidual pass: Mul after '" + context +
        "/tanh' does not multiply by the original activation input -- "
        "this isn't Mish, failing instead of silently fusing wrong math.");
  }
  return {SEResidualOp::Activation::kMish, mul->output(0)};
}

// The operand of a two-input elementwise node that ISN'T `known`.
ov::Output<ov::Node> OtherOperand(const std::shared_ptr<ov::Node>& node,
                                  const ov::Output<ov::Node>& known,
                                  const std::string& context) {
  if (node->input_value(0) == known) return node->input_value(1);
  if (node->input_value(1) == known) return node->input_value(0);
  throw Exception("OpenVINO SEResidual pass: neither input of '" + context +
                  "' matches the expected operand.");
}

}  // namespace

ReplaceSqueezeExcite::ReplaceSqueezeExcite() {
  auto pattern = ov::pass::pattern::wrap_type<ov::op::v1::ReduceMean>();

  ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
    auto reduce_mean =
        ov::as_type_ptr<ov::op::v1::ReduceMean>(m.get_match_root());
    if (!reduce_mean) return false;

    // Only touches SE sub-blocks, identified the same way ReplaceKdaScan
    // identifies KDA scans: by the friendly-name suffix converter.cc gives
    // every SE unit's ReduceMean node.
    if (!EndsWith(reduce_mean->get_friendly_name(), "/se/reduce_mean")) {
      return false;
    }

    // Everything past this point is topology inspection, and a mismatch is
    // not necessarily an error: the name suffix only says converter.cc's
    // MakeSqueezeAndExcite produced this ReduceMean, not that it sits in a
    // residual tower. MakeConvBlock (converter.cc:432) also builds head conv
    // blocks, and an SE unit in a head has no residual skip-add after it, so
    // the "residual mixin Add" lookup below would legitimately fail to find
    // one. Failing the match leaves that block unfused and running on
    // OpenVINO's native ops; throwing would refuse to load a net the backend
    // is otherwise perfectly able to run.
    //
    // The helpers throw rather than return a status (they are shared, and
    // most of their callers want the loud version), so the conversion from
    // "fatal" to "skip this block" happens here, once, around the whole
    // inspection. Nothing below mutates the graph until the final
    // replace(), so an abandoned match leaves the model untouched. The
    // reason is still printed, because a mismatch inside a real residual
    // tower means converter.cc has drifted and that is worth seeing.
    try {
      const auto se_input = reduce_mean->input_value(0);

      auto matmul1 = ExpectType<ov::op::v0::MatMul>(
          SingleConsumer(reduce_mean->output(0), "se/reduce_mean"), "MatMul1");
      auto w1 = ExpectConstant(matmul1->input_value(1), "matmul1's weight");

      auto add1 = ExpectType<ov::op::v1::Add>(
          SingleConsumer(matmul1->output(0), "se/matmul1"), "Add1");
      auto b1 = ExpectConstant(add1->input_value(1), "add1's bias");

      auto [act1_kind, act1_out] = MatchActivation(add1->output(0), "se/add1");

      auto matmul2 = ExpectType<ov::op::v0::MatMul>(
          SingleConsumer(act1_out, "se/activation1"), "MatMul2");
      auto w2 = ExpectConstant(matmul2->input_value(1), "matmul2's weight");

      auto add2 = ExpectType<ov::op::v1::Add>(
          SingleConsumer(matmul2->output(0), "se/matmul2"), "Add2");
      auto b2 = ExpectConstant(add2->input_value(1), "add2's bias");

      auto reshape = ExpectType<ov::op::v1::Reshape>(
          SingleConsumer(add2->output(0), "se/add2"), "Reshape");

      auto split = ExpectType<ov::op::v1::Split>(
          SingleConsumer(reshape->output(0), "se/reshape"), "Split");
      if (split->get_output_size() != 2) {
        throw Exception("OpenVINO SEResidual pass: Split after 'se/reshape' "
                        "has " + std::to_string(split->get_output_size()) +
                        " outputs, expected 2.");
      }

      auto sigmoid = ExpectType<ov::op::v0::Sigmoid>(
          SingleConsumer(split->output(0), "se/split/gamma"), "Sigmoid");

      auto mul = ExpectType<ov::op::v1::Multiply>(
          SingleConsumer(sigmoid->output(0), "se/sigmoid"), "Mul");
      const auto mul_other = OtherOperand(mul, sigmoid->output(0), "se/mul");
      if (mul_other != se_input) {
        throw Exception(
            "OpenVINO SEResidual pass: 'se/mul' does not multiply by the "
            "original se_input -- SE unit shape has drifted from "
            "converter.cc's MakeSqueezeAndExcite.");
      }

      auto add3 = ExpectType<ov::op::v1::Add>(
          SingleConsumer(mul->output(0), "se/mul"), "Add3 (se output)");
      if (add3->input_value(0) != split->output(1) &&
          add3->input_value(1) != split->output(1)) {
        throw Exception(
            "OpenVINO SEResidual pass: 'se/add3' does not add the split's "
            "beta output -- SE unit shape has drifted from converter.cc's "
            "MakeSqueezeAndExcite.");
      }

      // End of MakeSqueezeAndExcite. What follows is MakeConvBlock's own
      // tail: the residual skip-add, then the block's final activation.
      auto mixin_add = ExpectType<ov::op::v1::Add>(
          SingleConsumer(add3->output(0), "se/add3"), "residual mixin Add");
      const auto skip_input =
          OtherOperand(mixin_add, add3->output(0), "mixin_add");

      auto [act2_kind, final_out] =
          MatchActivation(mixin_add->output(0), "mixin_add");

      if (act1_kind != act2_kind) {
        throw Exception(
            "OpenVINO SEResidual pass: SE's FC1 activation and the block's "
            "final activation are different kinds -- converter.cc always "
            "uses default_activation_ for both, so this net's export doesn't "
            "match what this pass assumes.");
      }

      auto se_residual = std::make_shared<SEResidualOp>(
          se_input, skip_input, w1, b1, w2, b2, act1_kind);
      se_residual->set_friendly_name(reduce_mean->get_friendly_name() +
                                     "/fused");

      final_out.replace(se_residual->output(0));
      ov::copy_runtime_info(reduce_mean, se_residual);
      return true;
    } catch (const Exception& e) {
      CERR << "OpenVINO: leaving '" << reduce_mean->get_friendly_name()
           << "' unfused -- " << e.what();
      return false;
    }
  };

  auto matcher = std::make_shared<ov::pass::pattern::Matcher>(
      pattern, "ReplaceSqueezeExcite");
  register_matcher(matcher, callback);
}

}  // namespace openvino_backend
}  // namespace lczero
