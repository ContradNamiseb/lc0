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

// STATUS: this backend is NOT complete. Real and tested: D3D12 device
// bring-up and KdaRecurrenceLayer's compute shader, verified against a CPU
// reference (see test_kda_recurrence.cc). NOT built: the embedding, MHA
// block, smolgen, and policy/value/moves-left heads that would need to run
// around that recurrence to produce an actual position evaluation --
// DirectMLX's graph API has no loop/recurrence primitive, so KDA had to be
// solved first with a custom HLSL shader (see shaders/kda_recurrence.hlsl)
// before any of the rest could follow the same "compile a dml::Graph"
// pattern the earlier version of this file assumed for everything.
//
// The previous version of this file built a *complete* dml::Graph that
// only sliced the raw input tensor into policy/value/mlh-shaped pieces --
// no convolution, attention, or KDA computation ever ran. It looked
// finished (it compiled, linked, and dispatched) but its output was not a
// chess evaluation. NewComputation() below throws instead, so nothing can
// select this backend and silently get that same wrong output again.
//
// Structured to mirror the SYCL backend, per instruction: network_*.cc only
// orchestrates, inputs_outputs.h owns every D3D12 buffer (mirrors
// sycl/inputs_outputs.h), and layers.h/layers.cc hold one class per layer
// type with an Eval-equivalent entry point (mirrors sycl/layers.h) -- only
// KdaRecurrenceLayer exists there so far.

#include <cassert>
#include <memory>
#include <mutex>
#include <string>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "neural/backends/directml/inputs_outputs.h"
#include "neural/backends/directml/layers.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace directml_backend {

using Microsoft::WRL::ComPtr;

class DirectMLNetwork : public Network {
 public:
  DirectMLNetwork(const WeightsFile& weights, const OptionsDict& options);
  ~DirectMLNetwork() override = default;

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    throw Exception(
        "The directml backend only has the KDA recurrence layer built and "
        "tested so far (embedding/MHA/smolgen/heads are not implemented) -- "
        "it cannot produce a position evaluation yet. Run the "
        "kda_recurrence_test executable to exercise what is done, or use "
        "--backend=sycl / --backend=openvino for a working backend on this "
        "network. See network_directml.cc's top-of-file STATUS comment.");
  }

  int GetMiniBatchSize() const override { return 256; }
  bool IsCpu() const override { return false; }

  ID3D12Device* device() const { return device_.Get(); }
  KdaRecurrenceLayer* kda_recurrence_layer() const {
    return kda_recurrence_layer_.get();
  }

 private:
  ComPtr<IDXGIFactory4> dxgi_factory_;
  ComPtr<IDXGIAdapter1> adapter_;
  ComPtr<ID3D12Device> device_;
  std::unique_ptr<KdaRecurrenceLayer> kda_recurrence_layer_;
  NetworkCapabilities capabilities_;
  std::mutex mutex_;
};

DirectMLNetwork::DirectMLNetwork(const WeightsFile& weights,
                                 const OptionsDict& options) {
  const auto& format = weights.format().network_format();
  capabilities_.input_format = format.input();
  capabilities_.output_format = format.output();
  capabilities_.moves_left = format.moves_left();

  CERR << "Initializing directml backend on DirectX 12 GPU "
          "(KDA recurrence only -- see network_directml.cc STATUS comment)";

  HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory_));
  if (FAILED(hr)) throw Exception("Failed to create DXGI factory");

  for (UINT i = 0;
       dxgi_factory_->EnumAdapters1(i, &adapter_) != DXGI_ERROR_NOT_FOUND;
       ++i) {
    DXGI_ADAPTER_DESC1 desc;
    adapter_->GetDesc1(&desc);
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
    std::wstring gpu_name(desc.Description);
    CERR << "DirectML selected GPU: "
        << std::string(gpu_name.begin(), gpu_name.end());
    break;
  }

  hr = D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_11_0,
                         IID_PPV_ARGS(&device_));
  if (FAILED(hr)) throw Exception("Failed to create D3D12 device");

  const bool fp16 = options.GetOrDefault<bool>("fp16", false);
  kda_recurrence_layer_ =
      std::make_unique<KdaRecurrenceLayer>(device_.Get(), fp16);
}

std::unique_ptr<Network> MakeDirectMLNetwork(
    const std::optional<WeightsFile>& w, const OptionsDict& options) {
  if (!w) throw Exception("The directml backend requires a network file.");
  return std::make_unique<DirectMLNetwork>(*w, options);
}

REGISTER_NETWORK("directml", MakeDirectMLNetwork, 5)

}  // namespace directml_backend
}  // namespace lczero
