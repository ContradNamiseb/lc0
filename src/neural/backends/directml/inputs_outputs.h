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

#include <cstdint>
#include <vector>

#include "neural/backends/directml/dml_common.h"
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

  // output_elem_bytes is the byte width of ONE element of the compiled
  // graph's policy/value/moves-left output tensors: sizeof(float) for the
  // fp32 network, sizeof(DmlHalf) (2) for fp16. It sizes both the GPU-side
  // default-heap buffers those graphs write into and their readback
  // counterparts. Note 2494 / note 2523: getting this wrong doesn't just
  // mis-size an allocation, it makes the D3D12 copies below read past the
  // real (half-sized) data into whatever follows in the UAV/scratch heap.
  InputsOutputs(ID3D12Device* device, int max_batch_size, bool wdl,
               bool moves_left, size_t output_elem_bytes = sizeof(float))
      : output_elem_bytes_(output_elem_bytes),
        has_wdl_(wdl),
        has_mlh_(moves_left) {
    // agora thread 19 #620 package A1: every HRESULT here used to be
    // discarded. A failed CreateCommandList (say) left command_list_ null,
    // and the very next line called ->Close() on it -- a null-pointer AV in
    // the constructor, not a structured error. Mirrors DmlDeviceContext::
    // Init's ReportD3DErrors pattern (network_directml.cc).
    ReportD3DErrors(
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       IID_PPV_ARGS(&command_allocator_)),
        "CreateCommandAllocator (InputsOutputs)");
    ReportD3DErrors(
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                  command_allocator_.Get(), nullptr,
                                  IID_PPV_ARGS(&command_list_)),
        "CreateCommandList (InputsOutputs)");
    ReportD3DErrors(command_list_->Close(), "Close (InputsOutputs)");
    ReportD3DErrors(
        device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
        "CreateFence (InputsOutputs)");

    // agora thread 19 #696 Stream C ("ctor-throw mapped-without-Unmap, same
    // class as RR4"): RR4 (acb9711) made every Map() below throw cleanly on
    // failure instead of leaving its output pointer indeterminate -- but a
    // throw from the SECOND or later Map() here still left the FIRST one's
    // resource mapped forever: a constructor that throws never runs its own
    // destructor, so ~InputsOutputs()'s Unmap calls below never fire for a
    // partially-constructed object. Wrapping the whole buffer-creation
    // sequence and unmapping whatever succeeded before rethrowing closes
    // that gap; UnmapMapped() below is shared with the destructor so there
    // is one definition of "what got mapped," not two that can drift.
    try {
      const uint64_t input_bytes = static_cast<uint64_t>(max_batch_size) *
                                   kInputPlanes * 64 * sizeof(float);
      input_upload_ = detail::CreateBuffer(device, input_bytes,
                                           D3D12_HEAP_TYPE_UPLOAD,
                                           D3D12_RESOURCE_STATE_GENERIC_READ);
      // A2: an unchecked Map leaves input_mapped_ indeterminate on failure,
      // and every later write through it (every AddInput call) is then a
      // write through a garbage pointer instead of a clean, early failure.
      ReportD3DErrors(
          input_upload_->Map(0, nullptr,
                             reinterpret_cast<void**>(&input_mapped_)),
          "Map (input_upload_)");
      input_upload_map_ok_ = true;

      constexpr int kPolicySize = 1858;
      const uint64_t policy_bytes = static_cast<uint64_t>(max_batch_size) *
                                    kPolicySize * output_elem_bytes_;
      policy_gpu_ = detail::CreateBuffer(
          device, policy_bytes, D3D12_HEAP_TYPE_DEFAULT,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
      policy_readback_ = detail::CreateBuffer(
          device, policy_bytes, D3D12_HEAP_TYPE_READBACK,
          D3D12_RESOURCE_STATE_COPY_DEST);
      ReportD3DErrors(
          policy_readback_->Map(0, nullptr, &policy_readback_mapped_),
          "Map (policy_readback_)");
      policy_readback_map_ok_ = true;
      if (output_elem_bytes_ == sizeof(float)) {
        policy_mapped_ =
            reinterpret_cast<const float*>(policy_readback_mapped_);
      } else {
        // The compiled graph's output tensor is narrower than float (fp16
        // today); the readback buffer above holds those raw, narrower bits.
        // policy_mapped_ instead aliases this host-owned float buffer,
        // which forwardEval() batch-converts into after every readback --
        // see network_directml.cc's fp16-output conversion step.
        policy_host_.resize(static_cast<size_t>(max_batch_size) *
                            kPolicySize);
        policy_mapped_ = policy_host_.data();
      }

      // agora thread 19 #620 package C1: DWORD-aligned (4-byte) to match
      // GraphFactory::Compile's canonicalization of the compiled graph's own
      // output binding size (layers.cc) -- without this, a fp16 (2-byte)
      // output whose hand-computed byte count isn't a multiple of 4 (e.g.
      // WDL [N,3] at N=1: 6 bytes) would get a compile-time binding rounded
      // up to DirectML's own canonical size (8 bytes) while this physical
      // buffer stayed at the smaller hand-computed size -- moving the
      // under-binding bug from "declared smaller than DirectML expects" to
      // "declared larger than the buffer actually is," an out-of-bounds GPU
      // write instead of a binding-size contract violation. Aligning both
      // sides the same way keeps the physical buffer always at least as
      // large as the largest canonical binding any compiled batch size up to
      // max_batch_size can need (rounding preserves the N-monotonic
      // ordering of the hand-computed byte count).
      const uint64_t value_bytes =
          AlignUp(static_cast<uint64_t>(max_batch_size) *
                      (has_wdl_ ? 3 : 1) * output_elem_bytes_,
                  4);
      value_gpu_ = detail::CreateBuffer(
          device, value_bytes, D3D12_HEAP_TYPE_DEFAULT,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
      value_readback_ = detail::CreateBuffer(
          device, value_bytes, D3D12_HEAP_TYPE_READBACK,
          D3D12_RESOURCE_STATE_COPY_DEST);
      ReportD3DErrors(
          value_readback_->Map(0, nullptr, &value_readback_mapped_),
          "Map (value_readback_)");
      value_readback_map_ok_ = true;
      if (output_elem_bytes_ == sizeof(float)) {
        value_mapped_ = reinterpret_cast<const float*>(value_readback_mapped_);
      } else {
        value_host_.resize(static_cast<size_t>(max_batch_size) *
                           (has_wdl_ ? 3 : 1));
        value_mapped_ = value_host_.data();
      }

      if (has_mlh_) {
        // Same DWORD-alignment reasoning as value_bytes above -- a [N,1]
        // fp16 moves-left output at N=1 hand-computes to 2 bytes vs
        // DirectML's canonical 4.
        const uint64_t mlh_bytes = AlignUp(
            static_cast<uint64_t>(max_batch_size) * output_elem_bytes_, 4);
        moves_left_gpu_ = detail::CreateBuffer(
            device, mlh_bytes, D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        moves_left_readback_ = detail::CreateBuffer(
            device, mlh_bytes, D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST);
        ReportD3DErrors(
            moves_left_readback_->Map(0, nullptr,
                                      &moves_left_readback_mapped_),
            "Map (moves_left_readback_)");
        moves_left_readback_map_ok_ = true;
        if (output_elem_bytes_ == sizeof(float)) {
          moves_left_mapped_ =
              reinterpret_cast<const float*>(moves_left_readback_mapped_);
        } else {
          moves_left_host_.resize(static_cast<size_t>(max_batch_size));
          moves_left_mapped_ = moves_left_host_.data();
        }
      }
    } catch (...) {
      UnmapMapped();
      throw;
    }
  }

  ~InputsOutputs() { UnmapMapped(); }

  // agora thread 19 #696 Stream C: shared by the destructor (normal path,
  // every flag true) and the constructor's catch block (partial-
  // construction path, only the flags set before the throw are true) --
  // one definition of "what actually got mapped" instead of two that can
  // drift. Checking the _map_ok_ flag rather than just "is the ComPtr
  // non-null" matters: CreateBuffer can succeed (ComPtr set) while the
  // Map() call immediately after it still fails, and Unmap on a resource
  // whose Map never succeeded is itself a contract violation, not a safe
  // no-op.
  void UnmapMapped() {
    if (input_upload_map_ok_) input_upload_->Unmap(0, nullptr);
    if (policy_readback_map_ok_) policy_readback_->Unmap(0, nullptr);
    if (value_readback_map_ok_) value_readback_->Unmap(0, nullptr);
    if (moves_left_readback_map_ok_) moves_left_readback_->Unmap(0, nullptr);
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
  float* input_mapped_ = nullptr;
  // agora thread 19 #696 Stream C: true only once this resource's own Map()
  // call has returned success -- see UnmapMapped()'s comment for why this
  // is a separate flag from "the ComPtr is non-null" (CreateBuffer can
  // succeed and leave the ComPtr set while the Map() right after it still
  // fails).
  bool input_upload_map_ok_ = false;

  // Output default-heap buffers plus their mapped readback counterparts.
  // *_readback_mapped_ is the raw D3D12 mapping, output_elem_bytes_ wide per
  // element. *_mapped_ is always a valid float* for callers (finishEval,
  // GetQVal/GetPVal/GetMVal) to read directly: when output_elem_bytes_ ==
  // sizeof(float) it aliases *_readback_mapped_ (zero-copy, same as before
  // this struct knew about fp16); otherwise it aliases the corresponding
  // *_host_ vector, refreshed by forwardEval()'s conversion step after every
  // readback.
  ComPtr<ID3D12Resource> policy_gpu_;
  ComPtr<ID3D12Resource> policy_readback_;
  void* policy_readback_mapped_ = nullptr;
  bool policy_readback_map_ok_ = false;
  std::vector<float> policy_host_;
  const float* policy_mapped_ = nullptr;

  ComPtr<ID3D12Resource> value_gpu_;
  ComPtr<ID3D12Resource> value_readback_;
  void* value_readback_mapped_ = nullptr;
  bool value_readback_map_ok_ = false;
  std::vector<float> value_host_;
  const float* value_mapped_ = nullptr;

  ComPtr<ID3D12Resource> moves_left_gpu_;
  ComPtr<ID3D12Resource> moves_left_readback_;
  void* moves_left_readback_mapped_ = nullptr;
  bool moves_left_readback_map_ok_ = false;
  std::vector<float> moves_left_host_;
  const float* moves_left_mapped_ = nullptr;

  const size_t output_elem_bytes_;
  const bool has_wdl_;
  const bool has_mlh_;
};

}  // namespace directml_backend
}  // namespace lczero
