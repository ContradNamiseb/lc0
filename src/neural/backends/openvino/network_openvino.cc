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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <openvino/openvino.hpp>

#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"
#include "utils/bititer.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace openvino_backend {

static constexpr int kMaxBatchSize = 256;

// Cached output ports determined once at network initialization.
struct CachedOutputPorts {
  ov::Output<const ov::Node> policy_port;
  ov::Output<const ov::Node> value_port;
  ov::Output<const ov::Node> mlh_port;
  bool has_wdl = false;
  bool has_mlh = false;
};

// Computation instance reusing retained pre-bound tensors with zero dynamic allocations.
class OpenVinoNetworkComputation : public NetworkComputation {
 public:
  OpenVinoNetworkComputation(ov::InferRequest infer_request,
                             const CachedOutputPorts& ports);
  ~OpenVinoNetworkComputation() override = default;

  void AddInput(InputPlanes&& input) override;
  void ComputeBlocking() override;
  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override;
  float GetDVal(int sample) const override;
  float GetPVal(int sample, int move_id) const override;
  float GetMVal(int sample) const override;

 private:
  ov::InferRequest infer_request_;
  CachedOutputPorts ports_;
  int batch_size_ = 0;

  // Retained input buffer for max batch size (256).
  ov::Tensor input_tensor_;
  float* input_raw_ptr_ = nullptr;

  // Direct read-only output pointers mapped after inference.
  const float* policy_data_ = nullptr;
  const float* value_data_ = nullptr;
  const float* mlh_data_ = nullptr;
};

class OpenVinoNetwork : public Network {
 public:
  OpenVinoNetwork(const WeightsFile& weights, const OptionsDict& options);
  ~OpenVinoNetwork() override = default;

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    std::lock_guard<std::mutex> lock(infer_mutex_);
    return std::make_unique<OpenVinoNetworkComputation>(
        compiled_model_.create_infer_request(), ports_);
  }

  int GetMiniBatchSize() const override { return kMaxBatchSize; }
  bool IsCpu() const override { return is_cpu_; }

  const CachedOutputPorts& GetPorts() const { return ports_; }

 private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  CachedOutputPorts ports_;
  NetworkCapabilities capabilities_;
  bool is_cpu_ = false;
  std::mutex infer_mutex_;
};

OpenVinoNetworkComputation::OpenVinoNetworkComputation(
    ov::InferRequest infer_request, const CachedOutputPorts& ports)
    : infer_request_(std::move(infer_request)),
      ports_(ports) {
  // Pre-allocate the fixed-capacity input tensor once at construction for up to 256 boards.
  input_tensor_ = ov::Tensor(
      ov::element::f32,
      {static_cast<size_t>(kMaxBatchSize), kInputPlanes, 8, 8});
  input_raw_ptr_ = input_tensor_.data<float>();
}

void OpenVinoNetworkComputation::AddInput(InputPlanes&& input) {
  if (batch_size_ >= kMaxBatchSize) {
    throw Exception("OpenVINO batch size exceeded maximum of " +
                    std::to_string(kMaxBatchSize));
  }

  float* const batch_ptr =
      input_raw_ptr_ + batch_size_ * (kInputPlanes * 64);
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

  // Zero-copy dynamic batch tensor view over pre-allocated memory
  ov::Tensor batch_view(
      input_tensor_,
      {0, 0, 0, 0},
      {static_cast<size_t>(batch_size_), kInputPlanes, 8, 8});
  infer_request_.set_input_tensor(batch_view);

  // Execute inference in-place
  infer_request_.infer();

  // Direct zero-copy output buffer mapping using pre-cached output ports
  policy_data_ = infer_request_.get_tensor(ports_.policy_port).data<const float>();
  value_data_ = infer_request_.get_tensor(ports_.value_port).data<const float>();
  if (ports_.has_mlh) {
    mlh_data_ = infer_request_.get_tensor(ports_.mlh_port).data<const float>();
  }
}

float OpenVinoNetworkComputation::GetQVal(int sample) const {
  if (ports_.has_wdl) {
    const float* const wdl = value_data_ + sample * 3;
    return wdl[0] - wdl[2];
  }
  return value_data_[sample];
}

float OpenVinoNetworkComputation::GetDVal(int sample) const {
  if (ports_.has_wdl) {
    return value_data_[sample * 3 + 1];
  }
  return 0.0f;
}

float OpenVinoNetworkComputation::GetPVal(int sample, int move_id) const {
  assert(move_id >= 0 && move_id < 1858);
  return policy_data_[sample * 1858 + move_id];
}

float OpenVinoNetworkComputation::GetMVal(int sample) const {
  if (ports_.has_mlh) {
    return mlh_data_[sample];
  }
  return 0.0f;
}

OpenVinoNetwork::OpenVinoNetwork(const WeightsFile& weights,
                                 const OptionsDict& options) {
  std::string device = options.GetOrDefault<std::string>("device", "GPU");
  is_cpu_ = (device == "CPU");

  // Determine capabilities from weight file
  const auto& format = weights.format().network_format();
  capabilities_.input_format = format.input();
  capabilities_.output_format = format.output();
  capabilities_.moves_left = format.moves_left();
  ports_.has_wdl = capabilities_.has_wdl();
  ports_.has_mlh = capabilities_.has_mlh();

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

  // Dynamic batch shape
  model->reshape({{-1, kInputPlanes, 8, 8}});

  // Cache exact output ports during initialization (cold path)
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

  ov::AnyMap device_config;
  if (device == "GPU") {
    device_config[ov::hint::performance_mode.name()] =
        ov::hint::PerformanceMode::LATENCY;
    device_config[ov::hint::execution_mode.name()] =
        ov::hint::ExecutionMode::PERFORMANCE;
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

REGISTER_NETWORK("openvino", MakeOpenVinoNetwork, 135)
REGISTER_NETWORK("openvino-auto", MakeOpenVinoNetworkAuto, 136)

}  // namespace openvino_backend
}  // namespace lczero
