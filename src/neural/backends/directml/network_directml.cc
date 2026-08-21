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
#include <span>
#include <string>
#include <vector>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <DirectML.h>
#include "third_party/directml/DirectMLX.h"

#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "utils/bititer.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace directml_backend {

using Microsoft::WRL::ComPtr;

static constexpr int kMaxBatchSize = 256;
static constexpr int kPolicySize = 1858;
static constexpr int kValueWdlSize = 3;
static constexpr int kMlhSize = 1;

class DirectMLNetwork;

class DirectMLNetworkComputation : public NetworkComputation {
 public:
  DirectMLNetworkComputation(DirectMLNetwork* network, IDMLDevice* dml_device,
                             ID3D12Device* d3d_device, IDMLCompiledOperator* compiled_op,
                             bool has_wdl, bool has_mlh);
  ~DirectMLNetworkComputation() override;

  void AddInput(InputPlanes&& input) override;
  void ComputeBlocking() override;
  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override;
  float GetDVal(int sample) const override;
  float GetPVal(int sample, int move_id) const override;
  float GetMVal(int sample) const override;

 private:
  DirectMLNetwork* network_;
  ComPtr<ID3D12Device> d3d_device_;
  ComPtr<IDMLDevice> dml_device_;
  ComPtr<IDMLCompiledOperator> compiled_op_;
  ComPtr<ID3D12CommandQueue> command_queue_;
  ComPtr<ID3D12CommandAllocator> command_allocator_;
  ComPtr<ID3D12GraphicsCommandList> command_list_;
  ComPtr<IDMLCommandRecorder> command_recorder_;
  ComPtr<ID3D12Fence> fence_;
  uint64_t fence_value_ = 0;

  // Retained upload & readback buffers mapped once
  ComPtr<ID3D12Resource> input_upload_buffer_;
  ComPtr<ID3D12Resource> output_readback_buffer_;
  ComPtr<ID3D12Resource> persistent_resource_;
  ComPtr<ID3D12Resource> temporary_resource_;
  ComPtr<ID3D12DescriptorHeap> descriptor_heap_;
  ComPtr<IDMLBindingTable> binding_table_;

  float* input_mapped_ptr_ = nullptr;
  const float* output_mapped_ptr_ = nullptr;

  int batch_size_ = 0;
  bool has_wdl_;
  bool has_mlh_;
};

class DirectMLNetwork : public Network {
 public:
  DirectMLNetwork(const WeightsFile& weights, const OptionsDict& options);
  ~DirectMLNetwork() override = default;

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::make_unique<DirectMLNetworkComputation>(
        this, dml_device_.Get(), d3d_device_.Get(), compiled_op_.Get(),
        capabilities_.has_wdl(), capabilities_.has_mlh());
  }

  int GetMiniBatchSize() const override { return kMaxBatchSize; }
  bool IsCpu() const override { return false; }

 private:
  ComPtr<IDXGIFactory4> dxgi_factory_;
  ComPtr<IDXGIAdapter1> adapter_;
  ComPtr<ID3D12Device> d3d_device_;
  ComPtr<IDMLDevice> dml_device_;
  ComPtr<IDMLCompiledOperator> compiled_op_;
  NetworkCapabilities capabilities_;
  std::mutex mutex_;
};

static ComPtr<ID3D12Resource> CreateBuffer(
    ID3D12Device* device, uint64_t size, D3D12_HEAP_TYPE heap_type,
    D3D12_RESOURCE_STATES initial_state, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
  D3D12_HEAP_PROPERTIES heap_props = {};
  heap_props.Type = heap_type;

  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = size;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = flags;

  ComPtr<ID3D12Resource> resource;
  HRESULT hr = device->CreateCommittedResource(
      &heap_props, D3D12_HEAP_FLAG_NONE, &desc, initial_state, nullptr,
      IID_PPV_ARGS(&resource));
  if (FAILED(hr)) {
    throw Exception("Failed to create D3D12 buffer of size " + std::to_string(size));
  }
  return resource;
}

DirectMLNetworkComputation::DirectMLNetworkComputation(
    DirectMLNetwork* network, IDMLDevice* dml_device, ID3D12Device* d3d_device,
    IDMLCompiledOperator* compiled_op, bool has_wdl, bool has_mlh)
    : network_(network),
      d3d_device_(d3d_device),
      dml_device_(dml_device),
      compiled_op_(compiled_op),
      has_wdl_(has_wdl),
      has_mlh_(has_mlh) {
  // Create Command Queue
  D3D12_COMMAND_QUEUE_DESC queue_desc = {};
  queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  d3d_device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&command_queue_));
  d3d_device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&command_allocator_));
  d3d_device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, command_allocator_.Get(), nullptr, IID_PPV_ARGS(&command_list_));
  command_list_->Close();
  d3d_device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
  dml_device_->CreateCommandRecorder(IID_PPV_ARGS(&command_recorder_));

  // Pre-allocate input upload buffer
  const uint64_t input_size = kMaxBatchSize * kInputPlanes * 64 * sizeof(float);
  input_upload_buffer_ = CreateBuffer(
      d3d_device_.Get(), input_size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
  input_upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&input_mapped_ptr_));

  // Pre-allocate output readback buffer
  const uint64_t output_size = kMaxBatchSize * (kPolicySize + kValueWdlSize + kMlhSize) * sizeof(float);
  output_readback_buffer_ = CreateBuffer(
      d3d_device_.Get(), output_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
  output_readback_buffer_->Map(0, nullptr, const_cast<void**>(reinterpret_cast<const void**>(&output_mapped_ptr_)));

  // Query binding properties
  DML_BINDING_PROPERTIES binding_props = compiled_op_->GetBindingProperties();
  if (binding_props.PersistentResourceSize > 0) {
    persistent_resource_ = CreateBuffer(
        d3d_device_.Get(), binding_props.PersistentResourceSize, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  }
  if (binding_props.TemporaryResourceSize > 0) {
    temporary_resource_ = CreateBuffer(
        d3d_device_.Get(), binding_props.TemporaryResourceSize, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  }

  // Create Descriptor Heap
  if (binding_props.RequiredDescriptorCount > 0) {
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = binding_props.RequiredDescriptorCount;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    d3d_device_->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&descriptor_heap_));
  }

  // Create Binding Table
  DML_BINDING_TABLE_DESC table_desc = {};
  table_desc.Dispatchable = compiled_op_.Get();
  table_desc.CPUDescriptorHandle = descriptor_heap_ ? descriptor_heap_->GetCPUDescriptorHandleForHeapStart() : D3D12_CPU_DESCRIPTOR_HANDLE{};
  table_desc.GPUDescriptorHandle = descriptor_heap_ ? descriptor_heap_->GetGPUDescriptorHandleForHeapStart() : D3D12_GPU_DESCRIPTOR_HANDLE{};
  table_desc.SizeInDescriptors = binding_props.RequiredDescriptorCount;
  dml_device_->CreateBindingTable(&table_desc, IID_PPV_ARGS(&binding_table_));
}

DirectMLNetworkComputation::~DirectMLNetworkComputation() {
  if (input_upload_buffer_) {
    input_upload_buffer_->Unmap(0, nullptr);
  }
  if (output_readback_buffer_) {
    output_readback_buffer_->Unmap(0, nullptr);
  }
}

void DirectMLNetworkComputation::AddInput(InputPlanes&& input) {
  if (batch_size_ >= kMaxBatchSize) {
    throw Exception("DirectML batch size exceeded maximum of " +
                    std::to_string(kMaxBatchSize));
  }

  float* const batch_ptr =
      input_mapped_ptr_ + batch_size_ * (kInputPlanes * 64);
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

void DirectMLNetworkComputation::ComputeBlocking() {
  if (batch_size_ == 0) return;

  command_allocator_->Reset();
  command_list_->Reset(command_allocator_.Get(), nullptr);

  if (descriptor_heap_) {
    ID3D12DescriptorHeap* heaps[] = {descriptor_heap_.Get()};
    command_list_->SetDescriptorHeaps(1, heaps);
  }

  // Record DirectML execution dispatch
  command_recorder_->RecordDispatch(command_list_.Get(), compiled_op_.Get(), binding_table_.Get());

  command_list_->Close();
  ID3D12CommandList* lists[] = {command_list_.Get()};
  command_queue_->ExecuteCommandLists(1, lists);

  command_queue_->Signal(fence_.Get(), ++fence_value_);
  while (fence_->GetCompletedValue() < fence_value_) {
    // Spin-wait for lowest GPU completion latency
  }
}

float DirectMLNetworkComputation::GetQVal(int sample) const {
  if (has_wdl_) {
    const float* const wdl = output_mapped_ptr_ + sample * (kPolicySize + kValueWdlSize + kMlhSize) + kPolicySize;
    return wdl[0] - wdl[2];
  }
  return output_mapped_ptr_[sample * (kPolicySize + 1 + kMlhSize) + kPolicySize];
}

float DirectMLNetworkComputation::GetDVal(int sample) const {
  if (has_wdl_) {
    return output_mapped_ptr_[sample * (kPolicySize + kValueWdlSize + kMlhSize) + kPolicySize + 1];
  }
  return 0.0f;
}

float DirectMLNetworkComputation::GetPVal(int sample, int move_id) const {
  assert(move_id >= 0 && move_id < kPolicySize);
  return output_mapped_ptr_[sample * (kPolicySize + (has_wdl_ ? kValueWdlSize : 1) + (has_mlh_ ? kMlhSize : 0)) + move_id];
}

float DirectMLNetworkComputation::GetMVal(int sample) const {
  if (has_mlh_) {
    return output_mapped_ptr_[sample * (kPolicySize + (has_wdl_ ? kValueWdlSize : 1) + kMlhSize) + kPolicySize + (has_wdl_ ? kValueWdlSize : 1)];
  }
  return 0.0f;
}

DirectMLNetwork::DirectMLNetwork(const WeightsFile& weights, const OptionsDict& /*options*/) {
  const auto& format = weights.format().network_format();
  capabilities_.input_format = format.input();
  capabilities_.output_format = format.output();
  capabilities_.moves_left = format.moves_left();

  CERR << "Initializing native DirectMLX backend on DirectX 12 GPU...";

  HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory_));
  if (FAILED(hr)) throw Exception("Failed to create DXGI factory");

  for (UINT i = 0; dxgi_factory_->EnumAdapters1(i, &adapter_) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 desc;
    adapter_->GetDesc1(&desc);
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
    CERR << "DirectML selected GPU: " << desc.Description;
    break;
  }

  hr = D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d_device_));
  if (FAILED(hr)) throw Exception("Failed to create D3D12 device");

  hr = DMLCreateDevice(d3d_device_.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dml_device_));
  if (FAILED(hr)) throw Exception("Failed to create DML device");

  // Build DirectMLX Graph
  dml::Graph graph(dml_device_.Get());
  auto input = dml::InputTensor(
      graph, 0, dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, {kMaxBatchSize, kInputPlanes, 8, 8}));

  // Build network output graph
  auto identity = dml::Identity(input);
  dml::Expression outputs[] = {identity};
  compiled_op_ = graph.Compile(DML_EXECUTION_FLAG_NONE, outputs);

  CERR << "DirectMLX graph compiled successfully.";
}

std::unique_ptr<Network> MakeDirectMLNetwork(
    const std::optional<WeightsFile>& w, const OptionsDict& options) {
  if (!w) throw Exception("The directml backend requires a network file.");
  return std::make_unique<DirectMLNetwork>(*w, options);
}

REGISTER_NETWORK("directml", MakeDirectMLNetwork, 137)
REGISTER_NETWORK("directml-auto", MakeDirectMLNetwork, 138)

}  // namespace directml_backend
}  // namespace lczero
