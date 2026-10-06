/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "neural/backends/directml/onnx_graph.h"

#include <algorithm>
#include <cstring>
#include <unordered_set>

#include "utils/exception.h"

namespace lczero {
namespace directml_backend {
namespace {

using DataType = pblczero::TensorProto::DataType;

std::string DimsToString(const std::vector<int64_t>& dims) {
  std::string out = "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(dims[i]);
  }
  return out + "]";
}

// The numpy broadcast of two shapes.
std::vector<int64_t> Broadcast(const std::vector<int64_t>& a,
                               const std::vector<int64_t>& b,
                               const std::string& where) {
  const size_t rank = std::max(a.size(), b.size());
  std::vector<int64_t> out(rank);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t dim_a = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    const int64_t dim_b = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      throw Exception(where + " cannot broadcast " + DimsToString(a) +
                      " with " + DimsToString(b) + ".");
    }
    out[i] = dim_a == 1 ? dim_b : dim_a;
  }
  return out;
}

int64_t NormalizeAxis(int64_t axis, size_t rank, const std::string& where) {
  const int64_t normalized =
      axis < 0 ? axis + static_cast<int64_t>(rank) : axis;
  if (normalized < 0 || normalized >= static_cast<int64_t>(rank)) {
    throw Exception(where + " has axis " + std::to_string(axis) +
                    " outside rank " + std::to_string(rank) + ".");
  }
  return normalized;
}

// Reads an INT32 or INT64 initializer. Returns false for any other type.
bool ReadIntegerTensor(const pblczero::TensorProto& tensor,
                       std::vector<int64_t>* out) {
  const std::string_view raw = tensor.raw_data();
  if (tensor.data_type() == pblczero::TensorProto::INT64) {
    out->resize(raw.size() / sizeof(int64_t));
    std::memcpy(out->data(), raw.data(), out->size() * sizeof(int64_t));
    return true;
  }
  if (tensor.data_type() == pblczero::TensorProto::INT32) {
    std::vector<int32_t> narrow(raw.size() / sizeof(int32_t));
    std::memcpy(narrow.data(), raw.data(), narrow.size() * sizeof(int32_t));
    out->assign(narrow.begin(), narrow.end());
    return true;
  }
  return false;
}

bool IsOneOf(const std::string& op_type,
             std::initializer_list<std::string_view> names) {
  return std::find(names.begin(), names.end(), op_type) != names.end();
}

}  // namespace

bool OnnxNode::HasAttribute(std::string_view attribute_name) const {
  for (const auto& attribute : proto->attribute()) {
    if (attribute.name() == attribute_name) return true;
  }
  return false;
}

int64_t OnnxNode::IntAttribute(std::string_view attribute_name,
                               int64_t default_value) const {
  for (const auto& attribute : proto->attribute()) {
    if (attribute.name() == attribute_name) return attribute.i();
  }
  return default_value;
}

std::vector<int64_t> OnnxNode::IntsAttribute(
    std::string_view attribute_name) const {
  for (const auto& attribute : proto->attribute()) {
    if (attribute.name() == attribute_name) return attribute.ints();
  }
  return {};
}

const std::vector<std::string>& OnnxGraph::SupportedOperators() {
  static const std::vector<std::string> kOperators = {"Add",
                                                      "Cast",
                                                      "Concat",
                                                      "Conv",
                                                      "Div",
                                                      "Elu",
                                                      "Exp",
                                                      "Expand",
                                                      "Gather",
                                                      "GlobalAveragePool",
                                                      "Greater",
                                                      "Identity",
                                                      "LayerNormalization",
                                                      "MatMul",
                                                      "Mish",
                                                      "Mul",
                                                      "Pad",
                                                      "Reciprocal",
                                                      "ReduceMean",
                                                      "Relu",
                                                      "Reshape",
                                                      "Scan",
                                                      "Selu",
                                                      "Shape",
                                                      "Sigmoid",
                                                      "Slice",
                                                      "Softmax",
                                                      "Softplus",
                                                      "Split",
                                                      "Sqrt",
                                                      "Squeeze",
                                                      "Sub",
                                                      "Tanh",
                                                      "Transpose",
                                                      "Where"};
  return kOperators;
}

OnnxGraph::OnnxGraph(const pblczero::ModelProto& model, int batch_size)
    : graph_(model.graph()) {
  LoadInitializers();
  for (const auto& input : graph_.input()) {
    if (FindValue(input.name()) >= 0) continue;
    OnnxValue value;
    value.name = std::string(input.name());
    const auto& tensor_type = input.type().tensor_type();
    value.data_type = static_cast<DataType>(tensor_type.elem_type());
    for (const auto& dim : tensor_type.shape().dim()) {
      // The batch dimension is the only one the converter leaves symbolic.
      const bool is_static = dim.has_dim_value() && dim.dim_value() > 0;
      value.dims.push_back(is_static ? dim.dim_value() : batch_size);
    }
    inputs_.push_back(AddValue(std::move(value)));
  }
  LoadNodes(nullptr);
}

OnnxGraph::OnnxGraph(std::string_view serialized_body, const OnnxGraph& parent,
                     const std::vector<OnnxValue>& input_values) {
  graph_.ParseFromString(serialized_body);
  LoadInitializers();
  if (graph_.input().size() != input_values.size()) {
    throw Exception("Scan body " + std::string(graph_.name()) + " declares " +
                    std::to_string(graph_.input().size()) + " inputs for " +
                    std::to_string(input_values.size()) + " values.");
  }
  for (size_t i = 0; i < input_values.size(); ++i) {
    OnnxValue value;
    value.name = std::string(graph_.input()[i].name());
    value.data_type = input_values[i].data_type;
    value.dims = input_values[i].dims;
    inputs_.push_back(AddValue(std::move(value)));
  }
  LoadNodes(&parent);
}

int OnnxGraph::FindValue(std::string_view name) const {
  const auto found = value_index_.find(std::string(name));
  return found == value_index_.end() ? -1 : found->second;
}

int OnnxGraph::AddValue(OnnxValue value) {
  const int index = static_cast<int>(values_.size());
  value_index_[value.name] = index;
  values_.push_back(std::move(value));
  return index;
}

void OnnxGraph::LoadInitializers() {
  for (const auto& tensor : graph_.initializer()) {
    OnnxValue value;
    value.name = std::string(tensor.name());
    value.data_type = tensor.data_type();
    value.dims = tensor.dims();
    value.initializer = &tensor;
    value.is_constant = ReadIntegerTensor(tensor, &value.constant);
    AddValue(std::move(value));
  }
}

void OnnxGraph::LoadNodes(const OnnxGraph* parent) {
  static const std::unordered_set<std::string> kSupported(
      SupportedOperators().begin(), SupportedOperators().end());
  nodes_.reserve(graph_.node().size());
  for (const auto& proto : graph_.node()) {
    OnnxNode node;
    node.name = std::string(proto.name());
    node.op_type = std::string(proto.op_type());
    node.proto = &proto;
    if (!kSupported.count(node.op_type)) {
      throw Exception("ONNX operator " + node.op_type + " (node " + node.name +
                      ") is not in the DirectML translator's vocabulary.");
    }
    for (const auto& input : proto.input()) {
      if (input.empty()) {
        node.inputs.push_back(-1);
        continue;
      }
      int index = FindValue(input);
      if (index < 0 && parent) {
        // A Scan body reaches the enclosing graph's values by name.
        const int outer = parent->FindValue(input);
        if (outer >= 0) index = AddValue(parent->values()[outer]);
      }
      if (index < 0) {
        throw Exception("ONNX node " + node.name + " reads " + input +
                        ", which nothing produces.");
      }
      node.inputs.push_back(index);
    }
    for (const auto& output : proto.output()) {
      OnnxValue value;
      value.name = output;
      node.outputs.push_back(AddValue(std::move(value)));
    }
    if (node.op_type == "Scan") {
      InferScan(&node);
    } else {
      InferNode(&node);
    }
    nodes_.push_back(std::move(node));
  }
  for (const auto& output : graph_.output()) {
    const int index = FindValue(output.name());
    if (index < 0) {
      throw Exception("ONNX graph output " + std::string(output.name()) +
                      " is not produced by any node.");
    }
    outputs_.push_back(index);
  }
}

const std::vector<int64_t>& OnnxGraph::ConstantOf(const OnnxNode& node,
                                                  size_t input) const {
  if (input >= node.inputs.size() || node.inputs[input] < 0 ||
      !values_[node.inputs[input]].is_constant) {
    throw Exception(node.op_type + " node " + node.name + " needs input " +
                    std::to_string(input) +
                    " to be an integer known at load time.");
  }
  return values_[node.inputs[input]].constant;
}

std::vector<int64_t> OnnxGraph::AxesOf(const OnnxNode& node,
                                       std::string_view attribute_name,
                                       size_t input) const {
  if (node.HasAttribute(attribute_name)) {
    return node.IntsAttribute(attribute_name);
  }
  if (input < node.inputs.size() && node.inputs[input] >= 0) {
    return ConstantOf(node, input);
  }
  return {};
}

void OnnxGraph::InferNode(OnnxNode* node) {
  const std::string& op = node->op_type;
  const std::string where = op + " node " + node->name;
  auto in = [&](size_t i) -> const OnnxValue& {
    if (i >= node->inputs.size() || node->inputs[i] < 0) {
      throw Exception(where + " lacks input " + std::to_string(i) + ".");
    }
    return values_[node->inputs[i]];
  };
  const std::vector<int64_t> x = in(0).dims;
  const size_t rank = x.size();
  DataType data_type = in(0).data_type;
  std::vector<int64_t> dims;
  bool is_constant = false;
  std::vector<int64_t> constant;

  if (IsOneOf(op, {"Cast", "Elu", "Exp", "Identity", "LayerNormalization",
                   "Mish", "Reciprocal", "Relu", "Selu", "Sigmoid", "Softmax",
                   "Softplus", "Sqrt", "Tanh"})) {
    dims = x;
    if (op == "Cast") {
      data_type = static_cast<DataType>(node->IntAttribute("to", 0));
    }
  } else if (IsOneOf(op, {"Add", "Div", "Mul", "Sub", "Greater"})) {
    dims = Broadcast(x, in(1).dims, where);
    if (op == "Greater") data_type = pblczero::TensorProto::BOOL;
  } else if (op == "Where") {
    dims = Broadcast(Broadcast(x, in(1).dims, where), in(2).dims, where);
    data_type = in(1).data_type;
  } else if (op == "MatMul") {
    const std::vector<int64_t>& y = in(1).dims;
    if (rank < 2 || y.size() < 2 || x[rank - 1] != y[y.size() - 2]) {
      throw Exception(where + " cannot multiply " + DimsToString(x) + " by " +
                      DimsToString(y) + ".");
    }
    dims = Broadcast({x.begin(), x.end() - 2}, {y.begin(), y.end() - 2}, where);
    dims.push_back(x[rank - 2]);
    dims.push_back(y.back());
  } else if (op == "Conv") {
    const std::vector<int64_t>& weights = in(1).dims;
    if (rank < 3 || weights.size() != rank) {
      throw Exception(where + " has input " + DimsToString(x) +
                      " and weights " + DimsToString(weights) + ".");
    }
    const size_t spatial = rank - 2;
    std::vector<int64_t> pads = node->IntsAttribute("pads");
    std::vector<int64_t> strides = node->IntsAttribute("strides");
    std::vector<int64_t> dilations = node->IntsAttribute("dilations");
    pads.resize(2 * spatial, 0);
    strides.resize(spatial, 1);
    dilations.resize(spatial, 1);
    dims = {x[0], weights[0]};
    for (size_t s = 0; s < spatial; ++s) {
      const int64_t extent = (weights[2 + s] - 1) * dilations[s] + 1;
      dims.push_back(
          (x[2 + s] + pads[s] + pads[s + spatial] - extent) / strides[s] + 1);
    }
  } else if (op == "GlobalAveragePool") {
    dims.assign(rank, 1);
    dims[0] = x[0];
    dims[1] = x[1];
  } else if (op == "Pad") {
    const std::vector<int64_t> pads = AxesOf(*node, "pads", 1);
    if (pads.size() != 2 * rank) {
      throw Exception(where + " has " + std::to_string(pads.size()) +
                      " pads for rank " + std::to_string(rank) + ".");
    }
    dims = x;
    for (size_t i = 0; i < rank; ++i) dims[i] += pads[i] + pads[i + rank];
  } else if (op == "Reshape") {
    const std::vector<int64_t>& shape = ConstantOf(*node, 1);
    int64_t elements = 1;
    for (const int64_t dim : x) elements *= dim;
    int64_t known = 1;
    int inferred = -1;
    dims = shape;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] == 0 && i < rank) dims[i] = x[i];
      if (dims[i] == -1) {
        inferred = static_cast<int>(i);
      } else {
        known *= dims[i];
      }
    }
    if (inferred >= 0 && known != 0) dims[inferred] = elements / known;
    int64_t reshaped = 1;
    for (const int64_t dim : dims) reshaped *= dim;
    if (reshaped != elements) {
      throw Exception(where + " cannot reshape " + DimsToString(x) + " to " +
                      DimsToString(shape) + ".");
    }
  } else if (op == "Transpose") {
    std::vector<int64_t> perm = node->IntsAttribute("perm");
    if (perm.empty()) {
      for (size_t i = rank; i-- > 0;) perm.push_back(i);
    }
    if (perm.size() != rank) {
      throw Exception(where + " has a permutation of the wrong rank.");
    }
    for (const int64_t axis : perm) {
      dims.push_back(x[NormalizeAxis(axis, rank, where)]);
    }
  } else if (op == "Concat") {
    const int64_t axis =
        NormalizeAxis(node->IntAttribute("axis", 0), rank, where);
    dims = x;
    dims[axis] = 0;
    is_constant = rank == 1;
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const OnnxValue& part = in(i);
      if (part.dims.size() != rank) {
        throw Exception(where + " joins tensors of different rank.");
      }
      dims[axis] += part.dims[axis];
      is_constant = is_constant && part.is_constant;
      if (is_constant) {
        constant.insert(constant.end(), part.constant.begin(),
                        part.constant.end());
      }
    }
  } else if (op == "Slice") {
    const std::vector<int64_t> starts = AxesOf(*node, "starts", 1);
    const std::vector<int64_t> ends = AxesOf(*node, "ends", 2);
    std::vector<int64_t> axes = AxesOf(*node, "axes", 3);
    std::vector<int64_t> steps = AxesOf(*node, "steps", 4);
    if (axes.empty()) {
      for (size_t i = 0; i < starts.size(); ++i) axes.push_back(i);
    }
    steps.resize(starts.size(), 1);
    if (ends.size() != starts.size() || axes.size() != starts.size()) {
      throw Exception(where + " has starts, ends and axes of different size.");
    }
    dims = x;
    for (size_t i = 0; i < starts.size(); ++i) {
      const int64_t axis = NormalizeAxis(axes[i], rank, where);
      if (steps[i] != 1) {
        throw Exception(where + " has a step other than 1.");
      }
      const int64_t dim = x[axis];
      const int64_t start = std::clamp<int64_t>(
          starts[i] < 0 ? starts[i] + dim : starts[i], 0, dim);
      const int64_t end =
          std::clamp<int64_t>(ends[i] < 0 ? ends[i] + dim : ends[i], 0, dim);
      dims[axis] = std::max<int64_t>(0, end - start);
      if (in(0).is_constant && rank == 1) {
        is_constant = true;
        constant.assign(in(0).constant.begin() + start,
                        in(0).constant.begin() + start + dims[axis]);
      }
    }
  } else if (op == "Gather") {
    const int64_t axis =
        NormalizeAxis(node->IntAttribute("axis", 0), rank, where);
    const OnnxValue& indices = in(1);
    dims.assign(x.begin(), x.begin() + axis);
    dims.insert(dims.end(), indices.dims.begin(), indices.dims.end());
    dims.insert(dims.end(), x.begin() + axis + 1, x.end());
    if (in(0).is_constant && rank == 1 && indices.is_constant) {
      is_constant = true;
      for (const int64_t index : indices.constant) {
        constant.push_back(
            in(0).constant[NormalizeAxis(index, in(0).constant.size(), where)]);
      }
    }
  } else if (op == "Split") {
    const int64_t axis =
        NormalizeAxis(node->IntAttribute("axis", 0), rank, where);
    std::vector<int64_t> sizes = AxesOf(*node, "split", 1);
    if (sizes.empty()) {
      sizes.assign(node->outputs.size(), x[axis] / node->outputs.size());
    }
    if (sizes.size() != node->outputs.size()) {
      throw Exception(where + " has " + std::to_string(sizes.size()) +
                      " sizes for " + std::to_string(node->outputs.size()) +
                      " outputs.");
    }
    for (size_t i = 0; i < node->outputs.size(); ++i) {
      OnnxValue& part = values_[node->outputs[i]];
      part.data_type = data_type;
      part.dims = x;
      part.dims[axis] = sizes[i];
    }
    return;
  } else if (op == "ReduceMean" || op == "Squeeze") {
    std::vector<int64_t> axes = AxesOf(*node, "axes", 1);
    const bool keep_dims =
        op == "ReduceMean" && node->IntAttribute("keepdims", 1);
    std::vector<bool> selected(rank, false);
    for (const int64_t axis : axes) {
      selected[NormalizeAxis(axis, rank, where)] = true;
    }
    for (size_t i = 0; i < rank; ++i) {
      // Without axes, ReduceMean reduces everything and Squeeze drops every
      // dimension of size 1.
      const bool hit = axes.empty() ? (op == "ReduceMean" || x[i] == 1)
                                    : static_cast<bool>(selected[i]);
      if (!hit) {
        dims.push_back(x[i]);
      } else if (keep_dims) {
        dims.push_back(1);
      }
    }
  } else if (op == "Shape") {
    dims = {static_cast<int64_t>(rank)};
    data_type = pblczero::TensorProto::INT64;
    is_constant = true;
    constant = x;
  } else if (op == "Expand") {
    dims = Broadcast(x, ConstantOf(*node, 1), where);
  } else {
    throw Exception("ONNX operator " + op + " has no shape rule.");
  }

  OnnxValue& out = values_[node->outputs[0]];
  out.data_type = data_type;
  out.dims = std::move(dims);
  out.is_constant = is_constant;
  out.constant = std::move(constant);
}

void OnnxGraph::InferScan(OnnxNode* node) {
  const std::string where = "Scan node " + node->name;
  const size_t scan_input_count = node->IntAttribute("num_scan_inputs", 0);
  if (scan_input_count == 0 || scan_input_count > node->inputs.size()) {
    throw Exception(where + " has an invalid num_scan_inputs.");
  }
  const size_t state_count = node->inputs.size() - scan_input_count;
  const std::vector<int64_t> input_axes =
      node->IntsAttribute("scan_input_axes");
  const std::vector<int64_t> output_axes =
      node->IntsAttribute("scan_output_axes");

  // A body sees each state whole and each scanned input without its scan axis.
  std::vector<OnnxValue> body_inputs;
  int64_t sequence_length = -1;
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    OnnxValue value = values_[node->inputs[i]];
    if (i >= state_count) {
      const size_t k = i - state_count;
      const int64_t axis = NormalizeAxis(
          k < input_axes.size() ? input_axes[k] : 0, value.dims.size(), where);
      if (sequence_length >= 0 && value.dims[axis] != sequence_length) {
        throw Exception(where + " scans inputs of different length.");
      }
      sequence_length = value.dims[axis];
      value.dims.erase(value.dims.begin() + axis);
    }
    body_inputs.push_back(std::move(value));
  }

  std::string_view serialized_body;
  for (const auto& attribute : node->proto->attribute()) {
    if (attribute.name() == "body") serialized_body = attribute.g();
  }
  if (serialized_body.empty()) throw Exception(where + " has no body.");
  node->body = std::make_unique<OnnxGraph>(serialized_body, *this, body_inputs);

  const OnnxGraph& body = *node->body;
  if (body.outputs().size() != node->outputs.size()) {
    throw Exception(where + " has " + std::to_string(node->outputs.size()) +
                    " outputs for " + std::to_string(body.outputs().size()) +
                    " body outputs.");
  }
  for (size_t i = 0; i < node->outputs.size(); ++i) {
    const OnnxValue& produced = body.values()[body.outputs()[i]];
    OnnxValue& out = values_[node->outputs[i]];
    out.data_type = produced.data_type;
    out.dims = produced.dims;
    if (i >= state_count) {
      // A scan output is the body's output stacked along the scan axis.
      const size_t j = i - state_count;
      const int64_t axis =
          NormalizeAxis(j < output_axes.size() ? output_axes[j] : 0,
                        out.dims.size() + 1, where);
      out.dims.insert(out.dims.begin() + axis, sequence_length);
    }
  }
}

}  // namespace directml_backend
}  // namespace lczero
