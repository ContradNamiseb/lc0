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

// The converter-driven DirectML backend: the network goes through
// ConvertWeightsToOnnx and the resulting graph is translated node by node to
// DirectMLX, so a network feature the converter can express needs no code
// here. This first slice translates the operators of the classical
// (convolutional, squeeze-and-excitation) networks.

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <span>
#include <unordered_map>
#include <version>

#include "neural/backends/directml/dml_common.h"
#include "neural/backends/directml/layers.h"
#include "neural/backends/directml/onnx_graph.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"
#include "utils/bititer.h"
#include "utils/exception.h"
#include "utils/logging.h"

// After <span> and <version>, as in layers.cc: DirectMLX.h only uses
// std::span when __cpp_lib_span is already visible.
#include <DirectMLX.h>

namespace lczero {
namespace directml_backend {
namespace {

using Sizes = std::vector<uint32_t>;

// How an initializer is laid out for the DirectML operator that reads it.
enum class WeightLayout {
  kPlain,        // The ONNX shape, as four dimensions.
  kFilter,       // A convolution filter, already [M, C, H, W].
  kChannelBias,  // [M] as [1, M, 1, 1].
  kDenseFilter,  // MatMul weights [K, M] as a 1x1 filter [M, K, 1, 1].
  kIndices,      // Gather indices as UINT32 [1, 1, 1, count].
};

// Every tensor is four-dimensional here: [N, C, H, W] as it is, [N, K] as
// [N, K, 1, 1], and a vector [K] along the channel axis.
Sizes ToFourDimensions(const std::vector<int64_t>& dims,
                       const std::string& where) {
  switch (dims.size()) {
    case 4:
      return {static_cast<uint32_t>(dims[0]), static_cast<uint32_t>(dims[1]),
              static_cast<uint32_t>(dims[2]), static_cast<uint32_t>(dims[3])};
    case 2:
      return {static_cast<uint32_t>(dims[0]), static_cast<uint32_t>(dims[1]), 1,
              1};
    case 1:
      return {1, static_cast<uint32_t>(dims[0]), 1, 1};
    default:
      throw Exception("directml-onnx: " + where + " has a tensor of rank " +
                      std::to_string(dims.size()) +
                      ", which is not translated yet.");
  }
}

Sizes SizesOf(const dml::Expression& expression) {
  const auto sizes = expression.Impl()->GetOutputDesc().sizes;
  return Sizes(sizes.begin(), sizes.end());
}

uint64_t ElementCount(const Sizes& sizes) {
  uint64_t count = 1;
  for (const uint32_t size : sizes) count *= size;
  return count;
}

// DirectML on this driver takes no implicit broadcast: the smaller operand
// becomes a view of the full size that repeats along the broadcast axes.
dml::Expression BroadcastTo(dml::Expression input, const Sizes& target,
                            const std::string& where) {
  const Sizes from = SizesOf(input);
  if (from == target) return input;
  Sizes strides(4);
  uint32_t stride = 1;
  for (size_t i = 4; i-- > 0;) {
    if (from[i] != target[i] && from[i] != 1) {
      throw Exception("directml-onnx: " + where + " cannot broadcast.");
    }
    strides[i] = from[i] == target[i] ? stride : 0;
    stride *= from[i];
  }
  return dml::Reinterpret(input, DML_TENSOR_DATA_TYPE_FLOAT32,
                          dml::TensorDimensions(target.begin(), target.end()),
                          dml::TensorStrides(strides.begin(), strides.end()));
}

void Transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = resource;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  barrier.Transition.StateBefore = before;
  barrier.Transition.StateAfter = after;
  list->ResourceBarrier(1, &barrier);
}

ComPtr<ID3D12Resource> CreateCpuBuffer(ID3D12Device* device,
                                       D3D12_HEAP_TYPE heap_type,
                                       uint64_t bytes) {
  D3D12_HEAP_PROPERTIES heap = {};
  heap.Type = heap_type;
  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = bytes;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  ComPtr<ID3D12Resource> buffer;
  ReportD3DErrors(
      device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                      heap_type == D3D12_HEAP_TYPE_UPLOAD
                                          ? D3D12_RESOURCE_STATE_GENERIC_READ
                                          : D3D12_RESOURCE_STATE_COPY_DEST,
                                      nullptr, IID_PPV_ARGS(&buffer)),
      "CreateCommittedResource (directml-onnx)");
  return buffer;
}

// Lists every distinct node pattern of the graph: the operator, the input
// shapes (w marks an initializer), the output shape and the integer
// attributes. This is what a translation has to cover, so it is the first
// thing to look at when a network stops at "not translated yet".
void PrintPatterns(const OnnxGraph& graph, const std::string& prefix) {
  std::map<std::string, int> patterns;
  auto shape = [](const OnnxValue& value) {
    std::string out = value.initializer ? "w[" : "[";
    for (size_t i = 0; i < value.dims.size(); ++i) {
      out += (i ? "," : "") + std::to_string(value.dims[i]);
    }
    return out + "]";
  };
  for (const OnnxNode& node : graph.nodes()) {
    std::string key = node.op_type;
    for (const int input : node.inputs) {
      key +=
          " " + (input < 0 ? std::string("-") : shape(graph.values()[input]));
    }
    key += " -> " + shape(graph.values()[node.outputs[0]]);
    if (node.outputs.size() > 1) {
      key += " x" + std::to_string(node.outputs.size());
    }
    for (const auto& attribute : node.proto->attribute()) {
      if (attribute.name() == "body") continue;
      key += " " + std::string(attribute.name()) + "=";
      if (attribute.ints().empty()) key += std::to_string(attribute.i());
      for (const int64_t value : attribute.ints()) {
        key += std::to_string(value) + ",";
      }
    }
    ++patterns[key];
    if (node.body) PrintPatterns(*node.body, prefix + node.op_type + " body: ");
  }
  for (const auto& [key, count] : patterns) {
    CERR << "pattern " << prefix << count << "x " << key;
  }
}

struct Outputs {
  std::vector<float> policy;
  std::vector<float> value;
  std::vector<float> moves_left;
};

class DirectMlOnnxNetwork : public Network {
 public:
  DirectMlOnnxNetwork(const WeightsFile& file, const OptionsDict& options);

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }
  std::unique_ptr<NetworkComputation> NewComputation() override;
  int GetMiniBatchSize() const override { return max_batch_; }

  bool is_wdl() const { return is_wdl_; }
  void Compute(const std::vector<InputPlanes>& inputs, Outputs* outputs);

 private:
  // One compiled graph, for one batch size.
  struct Program {
    int batch = 0;
    DmlCompiledOp op;
    // Offsets of the policy, value and moves-left outputs in the output
    // arena; a missing output has no entry.
    std::vector<uint64_t> output_offsets;
  };
  struct Weight {
    uint64_t offset = 0;
    std::vector<uint8_t> bytes;
  };

  void BuildProgram(int batch);
  uint64_t AddWeight(const OnnxValue& value, int value_index,
                     WeightLayout layout);
  void UploadWeights();
  void Execute(ID3D12CommandList* list);

  DmlDeviceContext ctx_;
  NetworkCapabilities capabilities_;
  pblczero::ModelProto model_;
  std::string policy_name_;
  std::string value_name_;
  std::string moves_left_name_;
  bool is_wdl_ = false;
  int max_batch_ = 0;

  std::vector<Program> programs_;
  std::map<std::pair<int, WeightLayout>, Weight> weights_;
  uint64_t weight_bytes_ = 0;
  uint64_t output_bytes_ = 0;
  uint64_t transient_bytes_ = 0;
  DmlArena weight_arena_;
  DmlArena input_arena_;
  DmlArena output_arena_;
  DmlArena transient_arena_;
  ComPtr<ID3D12Resource> staging_;
  ComPtr<ID3D12Resource> readback_;
  std::mutex mutex_;
};

DirectMlOnnxNetwork::DirectMlOnnxNetwork(const WeightsFile& file,
                                         const OptionsDict& options)
    : capabilities_{file.format().network_format().input(),
                    file.format().network_format().output(),
                    file.format().network_format().moves_left()} {
  ctx_.Init(options);
  const WeightsFile converted =
      ConvertWeightsToOnnx(file, WeightsToOnnxConverterOptions());
  const auto& onnx = converted.onnx_model();
  model_.ParseFromString(onnx.model());
  policy_name_ = std::string(onnx.output_policy());
  is_wdl_ = onnx.has_output_wdl();
  value_name_ = std::string(is_wdl_ ? onnx.output_wdl() : onnx.output_value());
  if (onnx.has_output_mlh()) moves_left_name_ = std::string(onnx.output_mlh());

  if (options.GetOrDefault<bool>("print_patterns", false)) {
    PrintPatterns(OnnxGraph(model_, 16), "");
  }

  // Every DirectML object has to exist before the first dispatch on this
  // driver, so one graph per batch size is compiled now, and a batch runs on
  // the smallest one that holds it.
  max_batch_ = options.GetOrDefault<int>("max_batch", 64);
  for (int batch = 1; batch < max_batch_; batch *= 2) BuildProgram(batch);
  BuildProgram(max_batch_);

  const uint64_t input_bytes =
      uint64_t{max_batch_} * kNumInputPlanes * 64 * sizeof(float);
  weight_arena_.Create(ctx_.device(), weight_bytes_, "directml-onnx weights");
  input_arena_.Create(ctx_.device(), input_bytes, "directml-onnx input");
  output_arena_.Create(ctx_.device(), output_bytes_, "directml-onnx output");
  transient_arena_.Create(ctx_.device(),
                          std::max<uint64_t>(transient_bytes_, kDmlAlignment),
                          "directml-onnx transient");
  staging_ = CreateCpuBuffer(ctx_.device(), D3D12_HEAP_TYPE_UPLOAD,
                             std::max(input_bytes, weight_bytes_));
  readback_ =
      CreateCpuBuffer(ctx_.device(), D3D12_HEAP_TYPE_READBACK, output_bytes_);
  for (auto& program : programs_) {
    for (auto& binding : program.op.bindings) {
      if (binding.kind == DmlBindingRef::Kind::kWeight) {
        binding.weight.res = weight_arena_.resource();
      }
    }
  }
  UploadWeights();

  ReportD3DErrors(ctx_.upload_allocator()->Reset(), "Reset (initialize)");
  ReportD3DErrors(ctx_.upload_list()->Reset(ctx_.upload_allocator(), nullptr),
                  "Reset list (initialize)");
  ctx_.InitializeCompiledOperators(ctx_.upload_list());
  ReportD3DErrors(ctx_.upload_list()->Close(), "Close (initialize)");
  Execute(ctx_.upload_list());
}

void DirectMlOnnxNetwork::Execute(ID3D12CommandList* list) {
  ID3D12CommandList* lists[] = {list};
  ctx_.queue()->ExecuteCommandLists(1, lists);
  const uint64_t fence_value = ctx_.NextUploadFenceValue();
  ReportD3DErrors(ctx_.queue()->Signal(ctx_.fence(), fence_value), "Signal");
  ctx_.WaitForFence(ctx_.fence(), fence_value);
}

uint64_t DirectMlOnnxNetwork::AddWeight(const OnnxValue& value, int value_index,
                                        WeightLayout layout) {
  const auto key = std::make_pair(value_index, layout);
  const auto found = weights_.find(key);
  if (found != weights_.end()) return found->second.offset;

  const std::string_view raw = value.initializer->raw_data();
  Weight weight;
  if (layout == WeightLayout::kIndices) {
    if (!value.is_constant) {
      throw Exception("directml-onnx: Gather indices " + value.name +
                      " are not integers.");
    }
    std::vector<uint32_t> indices(value.constant.begin(), value.constant.end());
    weight.bytes.resize(indices.size() * sizeof(uint32_t));
    std::memcpy(weight.bytes.data(), indices.data(), weight.bytes.size());
  } else {
    if (value.data_type != pblczero::TensorProto::FLOAT) {
      throw Exception("directml-onnx: weights " + value.name +
                      " are not float32.");
    }
    weight.bytes.assign(raw.begin(), raw.end());
    if (layout == WeightLayout::kDenseFilter) {
      // [K, M] row-major to [M, K].
      const size_t rows = value.dims[0];
      const size_t columns = value.dims[1];
      std::vector<float> from(rows * columns);
      std::memcpy(from.data(), raw.data(), from.size() * sizeof(float));
      std::vector<float> to(from.size());
      for (size_t k = 0; k < rows; ++k) {
        for (size_t m = 0; m < columns; ++m) {
          to[m * rows + k] = from[k * columns + m];
        }
      }
      std::memcpy(weight.bytes.data(), to.data(), weight.bytes.size());
    }
  }
  weight.offset = weight_bytes_;
  weight_bytes_ += AlignUp(weight.bytes.size());
  const uint64_t offset = weight.offset;
  weights_.emplace(key, std::move(weight));
  return offset;
}

void DirectMlOnnxNetwork::UploadWeights() {
  uint8_t* mapped = nullptr;
  ReportD3DErrors(staging_->Map(0, nullptr, reinterpret_cast<void**>(&mapped)),
                  "Map (weights)");
  for (const auto& [key, weight] : weights_) {
    std::memcpy(mapped + weight.offset, weight.bytes.data(),
                weight.bytes.size());
  }
  staging_->Unmap(0, nullptr);
  ID3D12GraphicsCommandList* list = ctx_.upload_list();
  ReportD3DErrors(ctx_.upload_allocator()->Reset(), "Reset (weights)");
  ReportD3DErrors(list->Reset(ctx_.upload_allocator(), nullptr),
                  "Reset list (weights)");
  Transition(list, weight_arena_.resource(),
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COPY_DEST);
  list->CopyBufferRegion(weight_arena_.resource(), 0, staging_.Get(), 0,
                         weight_bytes_);
  Transition(list, weight_arena_.resource(), D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  ReportD3DErrors(list->Close(), "Close (weights)");
  Execute(list);
  for (auto& [key, weight] : weights_) {
    weight.bytes.clear();
    weight.bytes.shrink_to_fit();
  }
}

void DirectMlOnnxNetwork::BuildProgram(int batch) {
  const OnnxGraph graph(model_, batch);
  dml::Graph dml_graph(ctx_.dml_device());
  Program program;
  program.batch = batch;
  std::unordered_map<int, dml::Expression> expressions;

  auto add_input = [&](DML_TENSOR_DATA_TYPE type, const Sizes& sizes,
                       DmlBindingRef binding) {
    dml::TensorDesc desc(type,
                         dml::TensorDimensions(sizes.begin(), sizes.end()));
    binding.bytes = desc.totalTensorSizeInBytes;
    program.op.bindings.push_back(binding);
    return dml::InputTensor(
        dml_graph, static_cast<uint32_t>(program.op.bindings.size() - 1), desc);
  };
  auto weight_input = [&](int value_index, WeightLayout layout,
                          const Sizes& sizes) {
    const OnnxValue& value = graph.values()[value_index];
    if (!value.initializer) {
      throw Exception("directml-onnx: " + value.name +
                      " has to be an initializer.");
    }
    DmlBindingRef binding;
    binding.kind = DmlBindingRef::Kind::kWeight;
    binding.weight.offset = AddWeight(value, value_index, layout);
    return add_input(layout == WeightLayout::kIndices
                         ? DML_TENSOR_DATA_TYPE_UINT32
                         : DML_TENSOR_DATA_TYPE_FLOAT32,
                     sizes, binding);
  };
  // The expression of a node input: computed earlier, or an initializer.
  auto operand = [&](const OnnxNode& node, size_t i) {
    const int value_index = node.inputs[i];
    const auto found = expressions.find(value_index);
    if (found != expressions.end()) return found->second;
    const OnnxValue& value = graph.values()[value_index];
    dml::Expression expression =
        weight_input(value_index, WeightLayout::kPlain,
                     ToFourDimensions(value.dims, node.name));
    expressions.emplace(value_index, expression);
    return expression;
  };

  {
    const int planes = graph.inputs().at(0);
    DmlBindingRef binding;
    binding.kind = DmlBindingRef::Kind::kInput;
    expressions.emplace(
        planes,
        add_input(DML_TENSOR_DATA_TYPE_FLOAT32,
                  ToFourDimensions(graph.values()[planes].dims, "the input"),
                  binding));
  }

  const uint32_t ones[] = {1, 1};
  const dml::Span<const uint32_t> unit(ones, 2);
  for (const OnnxNode& node : graph.nodes()) {
    const std::string& op = node.op_type;
    const std::string where = op + " node " + node.name;
    const Sizes out_sizes =
        ToFourDimensions(graph.values()[node.outputs[0]].dims, where);
    std::vector<dml::Expression> results;
    if (op == "Conv") {
      const OnnxValue& filter = graph.values()[node.inputs[1]];
      const auto pads = node.IntsAttribute("pads");
      const uint32_t start[] = {static_cast<uint32_t>(pads.at(0)),
                                static_cast<uint32_t>(pads.at(1))};
      const uint32_t end[] = {static_cast<uint32_t>(pads.at(2)),
                              static_cast<uint32_t>(pads.at(3))};
      dml::Optional<dml::Expression> bias;
      if (node.inputs.size() > 2 && node.inputs[2] >= 0) {
        bias = weight_input(node.inputs[2], WeightLayout::kChannelBias,
                            {1, out_sizes[1], 1, 1});
      }
      results.push_back(
          dml::Convolution(operand(node, 0),
                           weight_input(node.inputs[1], WeightLayout::kFilter,
                                        ToFourDimensions(filter.dims, where)),
                           bias, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
                           DML_CONVOLUTION_DIRECTION_FORWARD, unit, unit,
                           dml::Span<const uint32_t>(start, 2),
                           dml::Span<const uint32_t>(end, 2)));
    } else if (op == "MatMul") {
      // A dense layer on [N, K, 1, 1] is a 1x1 convolution.
      const OnnxValue& weights = graph.values()[node.inputs[1]];
      if (weights.dims.size() != 2) {
        throw Exception("directml-onnx: " + where +
                        " is not a dense layer, which is not translated yet.");
      }
      results.push_back(dml::Convolution(
          operand(node, 0),
          weight_input(node.inputs[1], WeightLayout::kDenseFilter,
                       {static_cast<uint32_t>(weights.dims[1]),
                        static_cast<uint32_t>(weights.dims[0]), 1, 1})));
    } else if (op == "Add" || op == "Mul") {
      const dml::Expression a = BroadcastTo(operand(node, 0), out_sizes, where);
      const dml::Expression b = BroadcastTo(operand(node, 1), out_sizes, where);
      results.push_back(op == "Add" ? a + b : a * b);
    } else if (op == "Relu") {
      results.push_back(dml::ActivationRelu(operand(node, 0)));
    } else if (op == "Sigmoid") {
      results.push_back(dml::ActivationSigmoid(operand(node, 0)));
    } else if (op == "Softmax") {
      const uint32_t axes[] = {
          static_cast<uint32_t>(node.IntAttribute("axis", -1))};
      results.push_back(dml::ActivationSoftmax(
          operand(node, 0), dml::Span<const uint32_t>(axes, 1)));
    } else if (op == "ReduceMean") {
      const auto axes = node.IntsAttribute("axes");
      const std::vector<uint32_t> reduced(axes.begin(), axes.end());
      results.push_back(dml::Reduce(
          operand(node, 0), DML_REDUCE_FUNCTION_AVERAGE,
          dml::Span<const uint32_t>(reduced.data(), reduced.size())));
    } else if (op == "Reshape") {
      results.push_back(dml::Reinterpret(
          operand(node, 0), DML_TENSOR_DATA_TYPE_FLOAT32,
          dml::TensorDimensions(out_sizes.begin(), out_sizes.end()),
          dml::NullOpt));
    } else if (op == "Split") {
      const uint32_t axis = static_cast<uint32_t>(node.IntAttribute("axis", 0));
      std::vector<uint32_t> parts;
      for (const int output : node.outputs) {
        parts.push_back(
            ToFourDimensions(graph.values()[output].dims, where)[axis]);
      }
      results =
          dml::Split(operand(node, 0), axis,
                     dml::Span<const uint32_t>(parts.data(), parts.size()));
    } else if (op == "Gather") {
      const OnnxValue& indices = graph.values()[node.inputs[1]];
      results.push_back(dml::Gather(
          operand(node, 0),
          weight_input(node.inputs[1], WeightLayout::kIndices,
                       {1, 1, 1, static_cast<uint32_t>(indices.dims.at(0))}),
          static_cast<uint32_t>(node.IntAttribute("axis", 0)), 1));
    } else {
      throw Exception("directml-onnx: " + where + " is not translated yet.");
    }
    for (size_t i = 0; i < results.size(); ++i) {
      const OnnxValue& value = graph.values()[node.outputs[i]];
      const Sizes expected = ToFourDimensions(value.dims, where);
      if (ElementCount(SizesOf(results[i])) != ElementCount(expected)) {
        throw Exception("directml-onnx: " + where +
                        " was translated to the wrong size.");
      }
      expressions.emplace(node.outputs[i], results[i]);
    }
  }

  std::vector<dml::Expression> outputs;
  std::vector<uint64_t> output_bytes;
  uint64_t offset = 0;
  for (const std::string* name :
       {&policy_name_, &value_name_, &moves_left_name_}) {
    if (name->empty()) continue;
    const int value_index = graph.FindValue(*name);
    if (value_index < 0 || !expressions.count(value_index)) {
      throw Exception("directml-onnx: output " + *name + " was not built.");
    }
    outputs.push_back(expressions.at(value_index));
    output_bytes.push_back(
        outputs.back().Impl()->GetOutputDesc().totalTensorSizeInBytes);
    program.output_offsets.push_back(offset);
    offset += AlignUp(output_bytes.back());
  }
  output_bytes_ = std::max(output_bytes_, offset);

  program.op.op = dml_graph.Compile(
      ctx_.meta_commands() ? DML_EXECUTION_FLAG_NONE
                           : DML_EXECUTION_FLAG_DISABLE_META_COMMANDS,
      outputs);
  program.op.output_bytes = std::move(output_bytes);
  program.op.transient_bytes =
      program.op.op->GetBindingProperties().TemporaryResourceSize;
  transient_bytes_ = std::max(transient_bytes_, program.op.transient_bytes);
  ctx_.GetOrCreateBindingTable(program.op.op.Get());
  programs_.push_back(std::move(program));
}

void DirectMlOnnxNetwork::Compute(const std::vector<InputPlanes>& inputs,
                                  Outputs* outputs) {
  const int count = static_cast<int>(inputs.size());
  if (count > max_batch_) {
    throw Exception("directml-onnx: batch of " + std::to_string(count) +
                    " exceeds max_batch=" + std::to_string(max_batch_) + ".");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  Program* program = nullptr;
  for (auto& candidate : programs_) {
    if (candidate.batch >= count) {
      program = &candidate;
      break;
    }
  }
  const uint64_t plane_floats = uint64_t{kNumInputPlanes} * 64;
  const uint64_t input_bytes = program->batch * plane_floats * sizeof(float);
  float* mapped = nullptr;
  ReportD3DErrors(staging_->Map(0, nullptr, reinterpret_cast<void**>(&mapped)),
                  "Map (input)");
  std::memset(mapped, 0, input_bytes);
  for (int n = 0; n < count; ++n) {
    float* position = mapped + n * plane_floats;
    for (size_t plane = 0; plane < inputs[n].size(); ++plane) {
      for (const auto bit : IterateBits(inputs[n][plane].mask)) {
        position[plane * 64 + bit] = inputs[n][plane].value;
      }
    }
  }
  staging_->Unmap(0, nullptr);

  ID3D12GraphicsCommandList* list = ctx_.upload_list();
  ReportD3DErrors(ctx_.upload_allocator()->Reset(), "Reset (compute)");
  ReportD3DErrors(list->Reset(ctx_.upload_allocator(), nullptr),
                  "Reset list (compute)");
  Transition(list, input_arena_.resource(),
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COPY_DEST);
  list->CopyBufferRegion(input_arena_.resource(), 0, staging_.Get(), 0,
                         input_bytes);
  Transition(list, input_arena_.resource(), D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

  std::vector<DmlPtr> bound;
  for (const auto& binding : program->op.bindings) {
    bound.push_back(binding.kind == DmlBindingRef::Kind::kWeight
                        ? binding.weight
                        : DmlPtr(input_arena_.resource(), 0));
  }
  std::vector<DmlPtr> bound_outputs;
  for (const uint64_t offset : program->output_offsets) {
    bound_outputs.emplace_back(output_arena_.resource(), offset);
  }
  ctx_.DispatchOperator(list, program->op.op.Get(), bound, program->op.bindings,
                        bound_outputs, program->op.output_bytes,
                        DmlPtr(transient_arena_.resource(), 0),
                        program->op.transient_bytes);

  Transition(list, output_arena_.resource(),
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COPY_SOURCE);
  for (size_t i = 0; i < program->output_offsets.size(); ++i) {
    list->CopyBufferRegion(readback_.Get(), program->output_offsets[i],
                           output_arena_.resource(), program->output_offsets[i],
                           program->op.output_bytes[i]);
  }
  Transition(list, output_arena_.resource(), D3D12_RESOURCE_STATE_COPY_SOURCE,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  ReportD3DErrors(list->Close(), "Close (compute)");
  Execute(list);

  const uint8_t* results = nullptr;
  ReportD3DErrors(
      readback_->Map(0, nullptr,
                     reinterpret_cast<void**>(const_cast<uint8_t**>(&results))),
      "Map (output)");
  const size_t value_floats = is_wdl_ ? 3 : 1;
  auto read = [&](size_t slot, size_t floats_per_position,
                  std::vector<float>* out) {
    out->resize(count * floats_per_position);
    std::memcpy(out->data(), results + program->output_offsets[slot],
                out->size() * sizeof(float));
  };
  read(0, kNumOutputPolicy, &outputs->policy);
  read(1, value_floats, &outputs->value);
  if (!moves_left_name_.empty()) read(2, 1, &outputs->moves_left);
  readback_->Unmap(0, nullptr);
}

class DirectMlOnnxComputation : public NetworkComputation {
 public:
  explicit DirectMlOnnxComputation(DirectMlOnnxNetwork* network)
      : network_(network) {}

  void AddInput(InputPlanes&& input) override {
    inputs_.push_back(std::move(input));
  }
  void ComputeBlocking() override { network_->Compute(inputs_, &outputs_); }
  int GetBatchSize() const override { return static_cast<int>(inputs_.size()); }
  float GetQVal(int sample) const override {
    return network_->is_wdl()
               ? outputs_.value[sample * 3] - outputs_.value[sample * 3 + 2]
               : outputs_.value[sample];
  }
  float GetDVal(int sample) const override {
    return network_->is_wdl() ? outputs_.value[sample * 3 + 1] : 0.0f;
  }
  float GetPVal(int sample, int move_id) const override {
    return outputs_.policy[sample * kNumOutputPolicy + move_id];
  }
  float GetMVal(int sample) const override {
    return outputs_.moves_left.empty() ? 0.0f : outputs_.moves_left[sample];
  }

 private:
  DirectMlOnnxNetwork* network_;
  std::vector<InputPlanes> inputs_;
  Outputs outputs_;
};

std::unique_ptr<NetworkComputation> DirectMlOnnxNetwork::NewComputation() {
  return std::make_unique<DirectMlOnnxComputation>(this);
}

std::unique_ptr<Network> MakeDirectMlOnnxNetwork(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  if (!weights) {
    throw Exception("The directml-onnx backend requires a network file.");
  }
  return std::make_unique<DirectMlOnnxNetwork>(*weights, options);
}

REGISTER_NETWORK("directml-onnx", MakeDirectMlOnnxNetwork, 20)

}  // namespace
}  // namespace directml_backend
}  // namespace lczero
