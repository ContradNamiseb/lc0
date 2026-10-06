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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "proto/onnx.pb.h"

namespace lczero {
namespace directml_backend {

// The ONNX model that ConvertWeightsToOnnx emits, in the form the DirectML
// translator works from: DirectML needs every tensor size when a graph is
// compiled, so each value carries the static shape it has at one batch size.

// One tensor of an OnnxGraph.
struct OnnxValue {
  std::string name;
  pblczero::TensorProto::DataType data_type = pblczero::TensorProto::UNDEFINED;
  std::vector<int64_t> dims;
  // The weights, for a value that is an initializer.
  const pblczero::TensorProto* initializer = nullptr;
  // The contents, for an integer value known when the graph is loaded. The
  // converter computes shapes with such values (Shape, Slice, Concat).
  bool is_constant = false;
  std::vector<int64_t> constant;
};

class OnnxGraph;

struct OnnxNode {
  std::string name;
  std::string op_type;
  // Indices into OnnxGraph::values(); -1 for an optional input that is absent.
  std::vector<int> inputs;
  std::vector<int> outputs;
  const pblczero::NodeProto* proto = nullptr;
  // The loop body, for Scan.
  std::unique_ptr<OnnxGraph> body;

  bool HasAttribute(std::string_view attribute_name) const;
  int64_t IntAttribute(std::string_view attribute_name,
                       int64_t default_value) const;
  std::vector<int64_t> IntsAttribute(std::string_view attribute_name) const;
};

class OnnxGraph {
 public:
  // Loads the model and infers every shape for @batch_size. Throws Exception
  // for an operator outside SupportedOperators() and for a shape that cannot
  // be inferred.
  OnnxGraph(const pblczero::ModelProto& model, int batch_size);
  // A Scan body, loaded by the enclosing graph. @parent resolves the names
  // the body takes from that graph, and @input_values gives the body's inputs
  // their types and shapes.
  OnnxGraph(std::string_view serialized_body, const OnnxGraph& parent,
            const std::vector<OnnxValue>& input_values);
  OnnxGraph(const OnnxGraph&) = delete;
  OnnxGraph& operator=(const OnnxGraph&) = delete;

  // Every operator the converter's OnnxBuilder can emit. A converter change
  // that adds one has to add its shape rule here and its DirectML translation.
  static const std::vector<std::string>& SupportedOperators();

  const std::vector<OnnxValue>& values() const { return values_; }
  // In execution order.
  const std::vector<OnnxNode>& nodes() const { return nodes_; }
  const std::vector<int>& inputs() const { return inputs_; }
  const std::vector<int>& outputs() const { return outputs_; }
  // Returns -1 when the graph has no value of that name.
  int FindValue(std::string_view name) const;

 private:
  void LoadInitializers();
  void LoadNodes(const OnnxGraph* parent);
  int AddValue(OnnxValue value);
  void InferNode(OnnxNode* node);
  void InferScan(OnnxNode* node);
  const std::vector<int64_t>& ConstantOf(const OnnxNode& node,
                                         size_t input) const;
  // The axes of @node, from the attribute or from the constant input at
  // @input, whichever form the opset uses.
  std::vector<int64_t> AxesOf(const OnnxNode& node,
                              std::string_view attribute_name,
                              size_t input) const;

  pblczero::GraphProto graph_;
  std::vector<OnnxValue> values_;
  std::vector<OnnxNode> nodes_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::unordered_map<std::string, int> value_index_;
};

}  // namespace directml_backend
}  // namespace lczero
