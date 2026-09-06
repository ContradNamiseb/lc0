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
#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include "neural/network.h"
#include "utils/exception.h"

namespace lczero {
namespace directml_backend {

using Microsoft::WRL::ComPtr;

namespace detail {
// Same helper the DirectMLX scratch code used, kept here so
// InputsOutputs's constructor stays a straight list of buffer sizes rather
// than repeating CreateCommittedResource's eight arguments nine times.
inline ComPtr<ID3D12Resource> CreateBuffer(
    ID3D12Device* device, uint64_t size, D3D12_HEAP_TYPE heap_type,
    D3D12_RESOURCE_STATES initial_state,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
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
    throw Exception("Failed to create D3D12 buffer of size " +
                    std::to_string(size));
  }
  return resource;
}
}  // namespace detail

// D3D12 analog of sycl/inputs_outputs.h: one InputsOutputs owns every
// upload/default/readback buffer a single in-flight computation touches,
// sized once at construction for the backend's fixed max batch size and
// never resized or reallocated per call. Non-copyable/non-movable for the
// same reason the SYCL and OpenVINO ones are -- a command list can be
// mid-recording against these resources when a computation is destroyed,
// and there is no safe way to duplicate a mapped D3D12 upload/readback
// pointer.
struct InputsOutputs {
  InputsOutputs(const InputsOutputs&) = delete;
  InputsOutputs& operator=(const InputsOutputs&) = delete;
  InputsOutputs(InputsOutputs&&) = delete;
  InputsOutputs& operator=(InputsOutputs&&) = delete;

  InputsOutputs(ID3D12Device* device, int max_batch_size, bool wdl,
               bool moves_left)
      : has_wdl_(wdl), has_mlh_(moves_left) {
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                   IID_PPV_ARGS(&command_allocator_));
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                              command_allocator_.Get(), nullptr,
                              IID_PPV_ARGS(&command_list_));
    command_list_->Close();
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));

    const uint64_t input_bytes =
        static_cast<uint64_t>(max_batch_size) * kInputPlanes * 64 * sizeof(float);
    input_upload_ = detail::CreateBuffer(device, input_bytes,
                                         D3D12_HEAP_TYPE_UPLOAD,
                                         D3D12_RESOURCE_STATE_GENERIC_READ);
    input_upload_->Map(0, nullptr, reinterpret_cast<void**>(&input_mapped_));
    input_gpu_ = detail::CreateBuffer(
        device, input_bytes, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    constexpr int kPolicySize = 1858;
    const uint64_t policy_bytes =
        static_cast<uint64_t>(max_batch_size) * kPolicySize * sizeof(float);
    policy_gpu_ = detail::CreateBuffer(
        device, policy_bytes, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    policy_readback_ = detail::CreateBuffer(
        device, policy_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    policy_readback_->Map(
        0, nullptr, const_cast<void**>(reinterpret_cast<const void**>(&policy_mapped_)));

    const uint64_t value_bytes =
        static_cast<uint64_t>(max_batch_size) * (has_wdl_ ? 3 : 1) * sizeof(float);
    value_gpu_ = detail::CreateBuffer(
        device, value_bytes, D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    value_readback_ = detail::CreateBuffer(
        device, value_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    value_readback_->Map(
        0, nullptr, const_cast<void**>(reinterpret_cast<const void**>(&value_mapped_)));

    if (has_mlh_) {
      const uint64_t mlh_bytes = static_cast<uint64_t>(max_batch_size) * sizeof(float);
      moves_left_gpu_ = detail::CreateBuffer(
          device, mlh_bytes, D3D12_HEAP_TYPE_DEFAULT,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
      moves_left_readback_ = detail::CreateBuffer(
          device, mlh_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
      moves_left_readback_->Map(
          0, nullptr,
          const_cast<void**>(reinterpret_cast<const void**>(&moves_left_mapped_)));
    }
  }

  ~InputsOutputs() {
    if (input_upload_) input_upload_->Unmap(0, nullptr);
    if (policy_readback_) policy_readback_->Unmap(0, nullptr);
    if (value_readback_) value_readback_->Unmap(0, nullptr);
    if (moves_left_readback_) moves_left_readback_->Unmap(0, nullptr);
  }

  // Per-position command recording: one allocator/list pair per
  // InputsOutputs (not shared) so the free-list in network_directml.cc can
  // hand a fully independent recording context to each in-flight
  // computation, mirroring how sycl's InputsOutputs owns its own queue
  // reference rather than sharing one across concurrent computations.
  ComPtr<ID3D12CommandAllocator> command_allocator_;
  ComPtr<ID3D12GraphicsCommandList> command_list_;
  ComPtr<ID3D12Fence> fence_;
  uint64_t fence_value_ = 0;

  // Host-visible input staging (CPU writes here in AddInput()) and its
  // GPU-visible default-heap counterpart (copied into once per
  // ComputeBlocking(), not once per AddInput()).
  ComPtr<ID3D12Resource> input_upload_;
  ComPtr<ID3D12Resource> input_gpu_;
  float* input_mapped_ = nullptr;

  // Output default-heap buffers plus their mapped readback counterparts.
  ComPtr<ID3D12Resource> policy_gpu_;
  ComPtr<ID3D12Resource> policy_readback_;
  const float* policy_mapped_ = nullptr;

  ComPtr<ID3D12Resource> value_gpu_;
  ComPtr<ID3D12Resource> value_readback_;
  const float* value_mapped_ = nullptr;

  ComPtr<ID3D12Resource> moves_left_gpu_;
  ComPtr<ID3D12Resource> moves_left_readback_;
  const float* moves_left_mapped_ = nullptr;

  const bool has_wdl_;
  const bool has_mlh_;
};

}  // namespace directml_backend
}  // namespace lczero
