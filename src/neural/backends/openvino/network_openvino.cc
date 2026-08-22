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

// Structured to mirror the SYCL backend (network_sycl.cc.dp.cpp): this file
// only orchestrates -- construction, the InputsOutputs pool, and the
// NetworkComputation call sequence -- and inputs_outputs.h owns every
// pre-allocated buffer. There is no sycl/layers.h equivalent here: OpenVINO
// compiles and fuses the whole graph itself from the converted ONNX model,
// so there is no per-layer C++ dispatch to split out the way SYCL's
// FCLayer/SELayer/EncoderBlock classes do -- with one exception, KdaScanOp
// (kda_scan_op.h/.cc), a hand-written fused op that replaces the ONNX-Scan
// -derived TensorIterator OpenVINO would otherwise use for the KDA
// recurrence, which profiling showed was >98% of total inference time.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <openvino/core/op_extension.hpp>
#include <openvino/openvino.hpp>
#include <openvino/pass/manager.hpp>

#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"
#include "utils/bititer.h"
#include "utils/exception.h"
#include "utils/logging.h"

#include "neural/backends/openvino/inputs_outputs.h"
#include "neural/backends/openvino/kda_scan_kernel_source.h"
#include "neural/backends/openvino/kda_scan_op.h"
#include "neural/backends/openvino/kda_scan_pass.h"

namespace lczero {
namespace openvino_backend {

static constexpr int kMaxBatchSize = 256;

class OpenVinoNetwork;

class OpenVinoNetworkComputation : public NetworkComputation {
 public:
  OpenVinoNetworkComputation(OpenVinoNetwork* network, bool has_wdl,
                             bool has_mlh);
  ~OpenVinoNetworkComputation() override;

  void AddInput(InputPlanes&& input) override;
  void ComputeBlocking() override;
  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override;
  float GetDVal(int sample) const override;
  float GetPVal(int sample, int move_id) const override;
  float GetMVal(int sample) const override;

 private:
  OpenVinoNetwork* const network_;
  std::unique_ptr<InputsOutputs> io_;
  int batch_size_ = 0;
  bool has_wdl_;
  bool has_mlh_;
};

// Cached output ports, resolved once at network initialization so
// ComputeBlocking() never has to look a tensor up by name.
struct CachedOutputPorts {
  ov::Output<const ov::Node> policy_port;
  ov::Output<const ov::Node> value_port;
  ov::Output<const ov::Node> mlh_port;
};

class OpenVinoNetwork : public Network {
 public:
  OpenVinoNetwork(const WeightsFile& weights, const OptionsDict& options);
  ~OpenVinoNetwork() override = default;

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<OpenVinoNetworkComputation>(
        this, capabilities_.has_wdl(), capabilities_.has_mlh());
  }

  int GetMiniBatchSize() const override { return kMaxBatchSize; }
  bool IsCpu() const override { return is_cpu_; }

  const CachedOutputPorts& ports() const { return ports_; }

  // Mirrors sycl/network_sycl.cc.dp.cpp's GetInputsOutputs/
  // ReleaseInputsOutputs free-list: an InferRequest plus its bound tensors
  // is too expensive to build fresh on every move, so completed ones are
  // recycled instead of destroyed.
  std::unique_ptr<InputsOutputs> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (free_inputs_outputs_.empty()) {
      return std::make_unique<InputsOutputs>(
          kMaxBatchSize, compiled_model_.create_infer_request());
    }
    auto io = std::move(free_inputs_outputs_.front());
    free_inputs_outputs_.pop_front();
    return io;
  }

  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputs> io) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    free_inputs_outputs_.push_back(std::move(io));
  }

 private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  CachedOutputPorts ports_;
  NetworkCapabilities capabilities_;
  bool is_cpu_ = false;

  std::mutex io_mutex_;
  std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;
};

OpenVinoNetworkComputation::OpenVinoNetworkComputation(
    OpenVinoNetwork* network, bool has_wdl, bool has_mlh)
    : network_(network),
      io_(network->GetInputsOutputs()),
      has_wdl_(has_wdl),
      has_mlh_(has_mlh) {}

OpenVinoNetworkComputation::~OpenVinoNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(io_));
}

void OpenVinoNetworkComputation::AddInput(InputPlanes&& input) {
  if (batch_size_ >= kMaxBatchSize) {
    throw Exception("OpenVINO batch size exceeded maximum of " +
                    std::to_string(kMaxBatchSize));
  }

  float* const batch_ptr =
      io_->input_val_mem_ + batch_size_ * (kInputPlanes * 64);
  std::memset(batch_ptr, 0, kInputPlanes * 64 * sizeof(float));

  int plane_idx = 0;
  for (const auto& plane : input) {
    float* const plane_ptr = batch_ptr + plane_idx * 64;
    for (auto sq : IterateBits(plane.mask)) {
      plane_ptr[sq] = plane.value;
    }
    plane_idx++;
  }
  batch_size_++;
}

void OpenVinoNetworkComputation::ComputeBlocking() {
  if (batch_size_ == 0) return;

  // Zero-copy dynamic-batch view over the pre-allocated input tensor.
  ov::Tensor batch_view(io_->input_tensor_, {0, 0, 0, 0},
                        {static_cast<size_t>(batch_size_), kInputPlanes, 8, 8});
  io_->infer_request_.set_input_tensor(batch_view);

  io_->infer_request_.infer();

  const auto& ports = network_->ports();
  io_->op_policy_mem_ =
      io_->infer_request_.get_tensor(ports.policy_port).data<const float>();
  io_->op_value_mem_ =
      io_->infer_request_.get_tensor(ports.value_port).data<const float>();
  if (has_mlh_) {
    io_->op_moves_left_mem_ =
        io_->infer_request_.get_tensor(ports.mlh_port).data<const float>();
  }
}

float OpenVinoNetworkComputation::GetQVal(int sample) const {
  if (has_wdl_) {
    const float* const wdl = io_->op_value_mem_ + sample * 3;
    return wdl[0] - wdl[2];
  }
  return io_->op_value_mem_[sample];
}

float OpenVinoNetworkComputation::GetDVal(int sample) const {
  if (has_wdl_) {
    return io_->op_value_mem_[sample * 3 + 1];
  }
  return 0.0f;
}

float OpenVinoNetworkComputation::GetPVal(int sample, int move_id) const {
  assert(move_id >= 0 && move_id < 1858);
  return io_->op_policy_mem_[sample * 1858 + move_id];
}

float OpenVinoNetworkComputation::GetMVal(int sample) const {
  if (has_mlh_) {
    return io_->op_moves_left_mem_[sample];
  }
  return 0.0f;
}

namespace {

// Writes the GPU custom-layer kernel source and its CONFIG_FILE XML for
// this model's specific KDA dims (heads/key_dim/value_dim are static per
// net but not portable across different net shapes, so the kernel is
// JIT-compiled per model rather than shipped pre-built). Returns the XML
// path to hand to ov::Core::set_property("GPU", {{"CONFIG_FILE", ...}}).
// Must be called before compile_model().
std::filesystem::path WriteKdaScanGpuConfig(int heads, int key_dim,
                                            int value_dim) {
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "lc0_openvino_kda_scan";
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    throw Exception("OpenVINO: could not create temp dir for the KdaScan "
                    "GPU kernel config: " + ec.message());
  }

  fs::path cl_path = dir / "kda_scan.cl";
  std::ofstream cl(cl_path, std::ios::trunc);
  cl << kKdaScanKernelSource;
  cl.close();

  fs::path xml_path = dir / ("kda_scan_" + std::to_string(heads) + "x" +
                             std::to_string(key_dim) + "x" +
                             std::to_string(value_dim) + ".xml");
  std::ofstream xml(xml_path, std::ios::trunc);
  // Source filename is resolved relative to the CONFIG_FILE's own
  // directory (not the executable, despite what the docs say, and not
  // usable as an absolute path -- it gets concatenated onto that
  // directory regardless). Both files are written into the same dir
  // above, so a bare filename is enough.
  xml << "<CustomLayer name=\"KdaScan\" type=\"SimpleGPU\" version=\"1\">\n"
      << "  <Kernel entry=\"kda_scan_kernel\">\n"
      << "    <Source filename=\"" << cl_path.filename().string() << "\"/>\n"
      << "  </Kernel>\n"
      << "  <Buffers>\n"
      << "    <Tensor arg-index=\"0\" type=\"input\" port-index=\"0\" "
        "format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"1\" type=\"input\" port-index=\"1\" "
        "format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"2\" type=\"input\" port-index=\"2\" "
        "format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"3\" type=\"input\" port-index=\"3\" "
        "format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"4\" type=\"input\" port-index=\"4\" "
        "format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"5\" type=\"input\" port-index=\"5\" "
        "format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"6\" type=\"input\" port-index=\"6\" "
        "format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"7\" type=\"output\" port-index=\"0\" "
        "format=\"BFYX\"/>\n"
      << "  </Buffers>\n"
      << "  <CompilerOptions options=\"-D HEADS_=" << heads
      << " -D KEY_DIM_=" << key_dim << " -D VALUE_DIM_=" << value_dim
      << "\"/>\n"
      << "  <WorkSizes global=\"B*Y,X,1\" local=\"1,X,1\" dim=\"output\"/>\n"
      << "</CustomLayer>\n";
  xml.close();
  return xml_path;
}

}  // namespace

OpenVinoNetwork::OpenVinoNetwork(const WeightsFile& weights,
                                 const OptionsDict& options) {
  std::string device = options.GetOrDefault<std::string>("device", "GPU");
  is_cpu_ = (device == "CPU");

  const auto& format = weights.format().network_format();
  capabilities_.input_format = format.input();
  capabilities_.output_format = format.output();
  capabilities_.moves_left = format.moves_left();

  CERR << "Initializing OpenVINO backend on device [" << device << "]...";

  WeightsToOnnxConverterOptions converter_options;
  converter_options.opset = 17;
  converter_options.data_type =
      WeightsToOnnxConverterOptions::DataType::kFloat32;
  converter_options.policy_head = "vanilla";
  converter_options.value_head = "winner";

  pblczero::Net onnx_net = ConvertWeightsToOnnx(weights, converter_options);
  std::string_view model_bytes = onnx_net.onnx_model().model();

  std::shared_ptr<ov::Model> model = core_.read_model(
      std::string(model_bytes),
      ov::Tensor(ov::element::u8, {model_bytes.size()},
                 const_cast<char*>(model_bytes.data())));

  // Dynamic batch shape: -1 lets OpenVINO recompile internal shape-dependent
  // constants once here rather than reshaping (and paying that cost) inside
  // ComputeBlocking() on every call.
  model->reshape({{-1, kInputPlanes, 8, 8}});

  for (const auto& output : model->outputs()) {
    const std::string name = output.get_any_name();
    if (name.find("policy") != std::string::npos) {
      ports_.policy_port = output;
    } else if (name.find("wdl") != std::string::npos ||
               name.find("value") != std::string::npos) {
      ports_.value_port = output;
    } else if (name.find("mlh") != std::string::npos) {
      ports_.mlh_port = output;
    }
  }
  if (!ports_.policy_port.get_node() || !ports_.value_port.get_node()) {
    throw Exception(
        "OpenVINO: converted ONNX graph is missing a policy or value "
        "output -- ConvertWeightsToOnnx's naming must have changed.");
  }

  core_.add_extension(ov::OpExtension<KdaScanOp>());
  {
    ov::pass::Manager manager;
    manager.register_pass<ReplaceKdaScan>();
    manager.run_passes(model);
  }

  ov::AnyMap device_config;
  if (device == "GPU") {
    device_config[ov::hint::performance_mode.name()] =
        ov::hint::PerformanceMode::LATENCY;
    device_config[ov::hint::execution_mode.name()] =
        ov::hint::ExecutionMode::PERFORMANCE;

    // The GPU plugin defaults to running the whole graph in fp16
    // (INFERENCE_PRECISION_HINT) regardless of the model's declared f32
    // data type. The framework's own ops (FullyConnected etc.) handle that
    // transparently, but KdaScanOp's OpenCL kernel reads/writes raw
    // `float` buffers -- under silent fp16 execution it was actually being
    // handed `half` data through a `float*`, i.e. reading every value at
    // half the correct stride, which is indistinguishable from garbage.
    // Force f32 so the kernel's buffer layout assumption holds. Re-profiled
    // after KdaScanOp replaced TensorIterator: KdaScan is now only ~7% of
    // total runtime (the ~200 small FullyConnected/Gemm ops elsewhere in
    // the graph now dominate, likely per-op GPU dispatch overhead rather
    // than any single op's compute cost -- see kda_scan_pass.h's comment),
    // so losing fp16 on those ops costs more than this comment used to
    // assume; has not been re-measured against leaving fp16 on for the
    // non-KDA ops specifically.
    device_config[ov::hint::inference_precision.name()] = ov::element::f32;

    // The GPU plugin has no built-in fallback for an unrecognized op type
    // the way the CPU plugin does (which is what makes KdaScanOp::evaluate()
    // usable as a CPU reference at all) -- it needs an actual OpenCL kernel,
    // wired through the legacy-but-still-supported GPU custom-layer
    // mechanism. Only bother if this model actually has a KDA layer.
    for (const auto& node : model->get_ops()) {
      auto kda = ov::as_type_ptr<KdaScanOp>(node);
      if (!kda) continue;
      auto xml_path = WriteKdaScanGpuConfig(kda->heads(), kda->key_dim(),
                                            kda->value_dim());
      device_config["CONFIG_FILE"] = xml_path.string();
      break;
    }
  }

  compiled_model_ = core_.compile_model(model, device, device_config);
  CERR << "OpenVINO model compiled successfully on " << device << ".";
}

std::unique_ptr<Network> MakeOpenVinoNetwork(
    const std::optional<WeightsFile>& w, const OptionsDict& options) {
  if (!w) {
    throw Exception("The openvino backend requires a network file.");
  }
  return std::make_unique<OpenVinoNetwork>(*w, options);
}

std::unique_ptr<Network> MakeOpenVinoNetworkAuto(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  ov::Core core;
  auto devices = core.get_available_devices();
  OptionsDict opts = options;
  if (std::find(devices.begin(), devices.end(), "GPU") != devices.end()) {
    opts.Set("device", std::string("GPU"));
  } else {
    opts.Set("device", std::string("CPU"));
  }
  return MakeOpenVinoNetwork(weights, opts);
}

// Priority well below sycl (130-132), cudnn/dx12 (120), and cuda (102-104):
// measured GPU throughput here is currently ~5-8x slower than SYCL on this
// KDA-hybrid net (see docs/inference-backends-handoff.md), likely from
// OpenVINO serializing the KDA recurrence's ONNX Scan op into per-token
// dispatch, where SYCL runs it as one fused kernel. Still above blas/eigen
// (49-50): GPU execution should beat a CPU BLAS fallback when nothing
// faster is available. Do not raise this without re-measuring -- a wrong
// auto-selection would silently make search slower on any machine with
// both this and sycl/cuda/dx12 available.
REGISTER_NETWORK("openvino", MakeOpenVinoNetwork, 45)
REGISTER_NETWORK("openvino-auto", MakeOpenVinoNetworkAuto, 46)

}  // namespace openvino_backend
}  // namespace lczero
