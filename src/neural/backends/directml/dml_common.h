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

// Shared D3D12/DirectML plumbing for the directml backend -- the analog of
// cuda/cuda_common.h. It follows the CUDA backend's conventions where the two
// APIs allow it: errors go through Report* macros that print and throw,
// device memory is carved out of big committed resources (the D3D12 answer to
// cudaMalloc arenas), and execution runs on one command queue guarded by a
// mutex, like the SYCL backend's in-order queue.

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>

#include <d3d12.h>
#include <directml.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
class OptionsDict;
namespace directml_backend {

using Microsoft::WRL::ComPtr;

constexpr int kNumInputPlanes = 112;
constexpr int kNumOutputPolicy = 1858;

inline uint32_t DivUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// All sub-allocations out of arenas are 256-byte aligned: DirectML requires
// buffer bindings to be 256-byte aligned (DML_BUFFER_BINDING offsets), and
// root SRV/UAV descriptors for the hand-written kernels want it too.
constexpr uint64_t kDmlAlignment = 256;

inline uint64_t AlignUp(uint64_t v, uint64_t a = kDmlAlignment) {
  return (v + a - 1) / a * a;
}

// Prints and throws on a failed HRESULT, cuda_common.h's ReportCUDAErrors
// equivalent for the D3D12/DirectML APIs.
inline void ReportD3DErrors(HRESULT hr, const char* what) {
  if (FAILED(hr)) {
    CERR << "D3D12 error: " << what << " failed (hr=0x" << std::hex << hr
         << std::dec << ", " << __FILE__ << ":" << __LINE__ << ")";
    throw Exception(std::string("D3D12 error in the directml backend: ") +
                    what);
  }
}

inline void ReportDmlErrors(HRESULT hr, const char* what) {
  if (FAILED(hr)) {
    CERR << "DirectML error: " << what << " failed (hr=0x" << std::hex << hr
         << std::dec << ", " << __FILE__ << ":" << __LINE__ << ")";
    throw Exception(std::string("DirectML error in the directml backend: ") +
                    what);
  }
}

// A (resource, offset) pair -- the D3D12 equivalent of the raw device pointer
// CUDA/SYCL layers exchange. Arithmetic stays inside one resource so that
// every DmlPtr can still be turned into the DML_BUFFER_BINDING or root
// descriptor the APIs actually want. nullptr resource is the backend's null
// pointer.
struct DmlPtr {
  ID3D12Resource* res = nullptr;
  uint64_t offset = 0;

  DmlPtr() = default;
  DmlPtr(ID3D12Resource* r, uint64_t o) : res(r), offset(o) {}

  explicit operator bool() const { return res != nullptr; }

  // Advance by bytes (not elements -- matches the buffer-centric API).
  DmlPtr operator+(uint64_t bytes) const { return {res, offset + bytes}; }
  DmlPtr& operator+=(uint64_t bytes) {
    offset += bytes;
    return *this;
  }
  DmlPtr ByteSub(uint64_t bytes) const { return {res, offset - bytes}; }
  uint64_t ByteOffsetFrom(const DmlPtr& base) const {
    return offset - base.offset;
  }
  D3D12_GPU_VIRTUAL_ADDRESS GpuVA() const {
    return res->GetGPUVirtualAddress() + offset;
  }
};

// One big committed resource on the default heap with ALLOW_UNORDERED_ACCESS,
// carved into 256-byte-aligned ranges. The D3D12 equivalent of the CUDA
// backend's single cudaMalloc'd scratch arena: layers receive DmlPtrs into it
// instead of owning per-layer buffers.
class DmlArena {
 public:
  DmlArena() = default;
  DmlArena(const DmlArena&) = delete;
  DmlArena& operator=(const DmlArena&) = delete;

  void Create(ID3D12Device* device, uint64_t size, const char* name,
              D3D12_RESOURCE_STATES initial_state =
                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
    size_ = AlignUp(size);
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = size_;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ReportD3DErrors(device->CreateCommittedResource(
                        &heap, D3D12_HEAP_FLAG_NONE, &desc, initial_state,
                        nullptr, IID_PPV_ARGS(&resource_)),
                    name);
    state_ = initial_state;
    name_ = name;
  }

  DmlPtr Allocate(uint64_t bytes) {
    uint64_t aligned = AlignUp(bytes);
    if (cursor_ + aligned > size_) {
      throw Exception(std::string("directml backend arena '") + name_ +
                      "' exhausted (" + std::to_string(bytes) +
                      " bytes requested)");
    }
    DmlPtr p(resource_.Get(), cursor_);
    cursor_ += aligned;
    return p;
  }

  // Reserves the rest of the arena and returns its base; used for transient
  // DirectML scratch, which is bump-allocated per recorded batch.
  DmlPtr AllocateRest() {
    DmlPtr p(resource_.Get(), cursor_);
    cursor_ = size_;
    return p;
  }

  // Rewinds the bump cursor (per-batch transient reuse).
  void ResetCursor() { cursor_ = 0; }

  uint64_t size() const { return size_; }
  uint64_t used() const { return cursor_; }
  ID3D12Resource* resource() const { return resource_.Get(); }
  D3D12_RESOURCE_STATES state() const { return state_; }
  void set_state(D3D12_RESOURCE_STATES s) { state_ = s; }

 private:
  ComPtr<ID3D12Resource> resource_;
  uint64_t size_ = 0;
  uint64_t cursor_ = 0;
  D3D12_RESOURCE_STATES state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  const char* name_ = "unnamed";
};

// Shader-visible CBV/UAV/SRV heap with a bump allocator. Every DirectML
// binding table grabs a contiguous run of descriptors from it; the whole pool
// is reset once per batch, after the batch's fence wait guarantees the GPU is
// done reading them.
class DmlDescriptorPool {
 public:
  void Create(ID3D12Device* device, uint32_t capacity) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = capacity;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ReportD3DErrors(
        device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap_)),
        "CreateDescriptorHeap");
    cpu_start_ = heap_->GetCPUDescriptorHandleForHeapStart();
    gpu_start_ = heap_->GetGPUDescriptorHandleForHeapStart();
    stride_ = device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    capacity_ = capacity;
  }

  // Reserves count consecutive slots. 64 slots per binding table is far more
  // than any layer here binds (the biggest, the KDA encoder, is ~15 buffers).
  D3D12_CPU_DESCRIPTOR_HANDLE TakeCpu(uint32_t count) {
    uint32_t first = cursor_;
    cursor_ += count;
    if (cursor_ > capacity_) {
      throw Exception(
          "directml backend descriptor pool exhausted for this batch");
    }
    D3D12_CPU_DESCRIPTOR_HANDLE h;
    h.ptr = cpu_start_.ptr + static_cast<uint64_t>(first) * stride_;
    return h;
  }

  D3D12_GPU_DESCRIPTOR_HANDLE CpuToGpu(D3D12_CPU_DESCRIPTOR_HANDLE cpu) const {
    D3D12_GPU_DESCRIPTOR_HANDLE h;
    h.ptr = gpu_start_.ptr + (cpu.ptr - cpu_start_.ptr);
    return h;
  }

  // NOTE: no Reset() -- binding tables are cached per compiled operator
  // (see DmlDeviceContext::tables_), so their descriptor slots stay
  // permanently reserved.

  ID3D12DescriptorHeap* heap() const { return heap_.Get(); }

 private:
  ComPtr<ID3D12DescriptorHeap> heap_;
  D3D12_CPU_DESCRIPTOR_HANDLE cpu_start_{};
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_start_{};
  uint32_t stride_ = 0;
  uint32_t capacity_ = 0;
  uint32_t cursor_ = 0;
};

// Owns the D3D12 device, the DirectML device/recorder, and the single command
// queue every batch is submitted to. The D3D12 analog of what the SYCL
// backend keeps in its queue + context members; there is no cudaSetDevice to
// reissue per thread, so a std::mutex replaces the CUDA backend's implicit
// stream serialization.
// One compiled dml::Graph plus the ordered list of bindings its inputs
// expect. Graph input order == the order dml::InputTensor was called during
// construction; Eval() walks the same list substituting activation pointers.
// DirectML has no transpose operator (the ONNX Runtime DML EP folds
// transposes into input strides), so layer graphs declare strided views
// over the dense token-major buffers instead of reshuffling memory.
struct DmlBindingRef {
  enum class Kind { kWeight, kInput, kInput2, kScratch, kExtra } kind;
  DmlPtr weight;  // only meaningful for kWeight
  uint64_t bytes = 0;
};

struct DmlCompiledOp {
  ComPtr<IDMLCompiledOperator> op;
  std::vector<DmlBindingRef> bindings;
  // Outputs, in graph-output order: filled with byte sizes at compile time
  // and with the actual buffers at Eval time.
  std::vector<uint64_t> output_bytes;
  uint64_t transient_bytes = 0;
};

class DmlDeviceContext {
 public:
  void Init(const OptionsDict& options);

  ID3D12Device* device() const { return device_.Get(); }

  // Cached binding table for a compiled operator. This driver degrades
  // under repeated CreateBindingTable calls (mis-floating
  // DXGI_ERROR_DEVICE_REMOVED / E_INVALIDARG after ~6-8 creations; see
  // docs/directml-handoff.md section 3), so each compiled operator gets
  // exactly one table, created on first dispatch and rebound afterwards.
  // Safe because batches are fence-serialized: a table's descriptors are
  // rewritten (CPU-side, immediate) only after the previous batch's GPU
  // work has completed.
  IDMLBindingTable* GetOrCreateBindingTable(IDMLCompiledOperator* op);
  IDMLDevice* dml_device() const { return dml_device_.Get(); }
  IDMLCommandRecorder* recorder() const { return recorder_.Get(); }
  ID3D12CommandQueue* queue() const { return queue_.Get(); }
  DmlDescriptorPool& descriptors() { return descriptors_; }

  // One-shot upload command list for the weight flush at load.
  ID3D12CommandAllocator* upload_allocator() const {
    return upload_allocator_.Get();
  }
  ID3D12GraphicsCommandList* upload_list() const { return upload_list_.Get(); }
  ID3D12Fence* fence() const { return fence_.Get(); }
  uint64_t NextUploadFenceValue() { return ++upload_fence_value_; }

  // One-place helper for the record/dispatch/binding dance every layer
  // does: takes 64 descriptor slots, creates a binding table, binds inputs
  // (with per-binding byte sizes from the compiled graph metadata),
  // outputs, and transient scratch, then records the dispatch followed by a
  // global UAV barrier (conservative, but the layers here are strictly
  // sequential). Defined in layers.cc next to the graph builders.
  void DispatchOperator(ID3D12GraphicsCommandList* list,
                        IDMLCompiledOperator* op,
                        const std::vector<DmlPtr>& inputs,
                        const std::vector<DmlBindingRef>& meta,
                        const std::vector<DmlPtr>& outputs,
                        const std::vector<uint64_t>& output_bytes,
                        DmlPtr transient, uint64_t transient_bytes);

  // Blocks until value_ on the given fence; used once per batch.
  void WaitForFence(ID3D12Fence* fence, uint64_t value);

 private:
  ComPtr<IDXGIFactory4> dxgi_factory_;
  ComPtr<IDXGIAdapter1> adapter_;
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12CommandQueue> queue_;
  ComPtr<IDMLDevice> dml_device_;
  ComPtr<IDMLCommandRecorder> recorder_;
  ComPtr<ID3D12Fence> fence_;
  HANDLE fence_event_ = nullptr;
  uint64_t upload_fence_value_ = 0;
  ComPtr<ID3D12CommandAllocator> upload_allocator_;
  ComPtr<ID3D12GraphicsCommandList> upload_list_;
  DmlDescriptorPool descriptors_;
  std::unordered_map<IDMLCompiledOperator*,
                     Microsoft::WRL::ComPtr<IDMLBindingTable>>
      tables_;
  friend class DmlUploadScope;
};

// RAII recorder for a command list plus the per-batch state hanging off it:
// the transient bump allocator and the descriptor pool reset. Mirrors the
// CUDA backend's forwardEval() running under the eval lock.
class DmlExecScope {
 public:
  DmlExecScope(DmlDeviceContext& ctx, ID3D12GraphicsCommandList* list,
               DmlArena* transient_arena)
      : ctx_(ctx), list_(list), transient_arena_(transient_arena) {}

  ID3D12GraphicsCommandList* list() const { return list_; }
  DmlDeviceContext& ctx() const { return ctx_; }

  // Bump-allocates transient (DirectML-internal scratch) memory for one
  // dispatch. Reset per batch by the network.
  DmlPtr TakeTransient(uint64_t bytes) {
    DmlPtr p = transient_arena_->Allocate(bytes);
    return p;
  }

 private:
  DmlDeviceContext& ctx_;
  ID3D12GraphicsCommandList* list_;
  DmlArena* transient_arena_;
};

}  // namespace directml_backend
}  // namespace lczero
