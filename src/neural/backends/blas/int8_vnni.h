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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#include "neural/backends/blas/int8_vnni_kernels.h"

namespace lczero {
namespace blas_backend {

// Which INT8 kernel executes W8A8 layers.
enum class Int8Kernel {
  kAuto,
  kVnniAvx512,
  kVnniAvx,
  kMkl,
};

// Runtime x86-64 ISA detection (CPUID + XGETBV). Safe on all architectures.
bool CpuSupportsAvx512Vnni();
bool CpuSupportsAvxVnni();

// True if any compiled VNNI kernel is supported on the current CPU.
bool IsVnniAvailable();

// Resolves a user option ("auto", "vnni", "avx512", "avx", "mkl") to an
// executable kernel, or throws Exception if explicitly requested and
// unavailable.
Int8Kernel SelectInt8Kernel(const std::string& requested, bool allow_mkl);

const char* Int8KernelName(Int8Kernel kernel);

// 64-byte aligned flat buffer for packed weights and SIMD scratch.
// Does not zero-initialize on growth and never shrinks once sized.
template <typename T>
class AlignedBuffer {
 public:
  AlignedBuffer() = default;
  ~AlignedBuffer() { Free(); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      Free();
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      other.data_ = nullptr;
      other.size_ = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  void Resize(size_t count) {
    if (count > capacity_) {
      Free();
      const size_t bytes = ((count * sizeof(T) + 63) / 64) * 64;
#if defined(_MSC_VER) || defined(_WIN32)
      data_ = static_cast<T*>(_aligned_malloc(bytes, 64));
#else
      data_ = static_cast<T*>(std::aligned_alloc(64, bytes));
#endif
      if (!data_) throw std::bad_alloc();
      capacity_ = bytes / sizeof(T);
    }
    size_ = count;
  }

  void ResizeZero(size_t count) {
    Resize(count);
    if (count > 0) std::memset(data_, 0, count * sizeof(T));
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

 private:
  void Free() {
    if (data_) {
#if defined(_MSC_VER) || defined(_WIN32)
      _aligned_free(data_);
#else
      std::free(data_);
#endif
      data_ = nullptr;
      size_ = 0;
      capacity_ = 0;
    }
  }

  T* data_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
};

// Lightweight lock-free spin-then-park thread pool for parallelizing whole
// INT8 layers.
// - num_threads <= 1: spawns zero threads and executes inline with zero
//   atomic overhead.
// - Concurrent callers (e.g. multiple search threads): try-acquires the pool;
//   if busy, executes inline so there is never deadlock or oversubscription.
// - `generation_` encodes `(epoch << 16) | active` so idle workers with
//   `tid >= active` never read the non-atomic job fields.
class Int8ThreadPool {
 public:
  explicit Int8ThreadPool(int num_threads);
  ~Int8ThreadPool();

  Int8ThreadPool(const Int8ThreadPool&) = delete;
  Int8ThreadPool& operator=(const Int8ThreadPool&) = delete;

  int num_threads() const { return num_threads_; }

  // Splits [0, total_items) into chunks of size >= min_chunk across up to
  // num_threads_ threads, invoking fn(begin, end, tid) where tid is in
  // [0, active_threads).
  template <typename Fn>
  void ParallelFor(size_t total_items, size_t min_chunk, Fn&& fn) {
    if (total_items == 0) return;
    if (num_threads_ <= 1 || total_items <= min_chunk) {
      fn(0, total_items, 0);
      return;
    }
    if (busy_.exchange(true, std::memory_order_acquire)) {
      fn(0, total_items, 0);
      return;
    }
    const size_t max_tasks = (total_items + min_chunk - 1) / min_chunk;
    const size_t active =
        std::min<size_t>(static_cast<size_t>(num_threads_), max_tasks);
    if (active <= 1) {
      busy_.store(false, std::memory_order_release);
      fn(0, total_items, 0);
      return;
    }

    using FnVal = std::remove_reference_t<Fn>;
    auto thunk = [](void* ctx, size_t begin, size_t end, size_t tid) {
      (*static_cast<FnVal*>(ctx))(begin, end, tid);
    };
    RunParallel(total_items, active, thunk, &fn);
    busy_.store(false, std::memory_order_release);
  }

 private:
  using ThunkFn = void (*)(void*, size_t, size_t, size_t);

  static constexpr int kActiveBits = 16;
  static constexpr uint64_t kActiveMask = (uint64_t{1} << kActiveBits) - 1;

  void RunParallel(size_t total_items, size_t active, ThunkFn thunk, void* ctx);
  void WorkerLoop(size_t tid);

  int num_threads_ = 1;
  std::atomic<bool> busy_{false};
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> generation_{0};
  std::atomic<size_t> completed_{0};
  std::atomic<size_t> sleeping_workers_{0};

  // Current task state (published before generation_ release store; only read
  // by workers with tid < active).
  ThunkFn current_thunk_ = nullptr;
  void* current_ctx_ = nullptr;
  size_t current_total_ = 0;

  std::vector<std::thread> workers_;
};

// Packs row-major int8 weights [outputs][inputs] into VNNI panel layout:
// [ceil(outputs / nr)][ceil(inputs / 4)][nr][4], zero-padded on N and K tails.
void PackGemmWeights(const int8_t* source, size_t outputs, size_t inputs,
                     size_t nr, AlignedBuffer<int8_t>& destination);

// Packs OIHW 3x3 int8 weights [outputs][input_channels][9] into direct-conv
// VNNI layout: [ceil(outputs / nr)][9 taps][ceil(input_channels / 4)][nr][4].
void PackConv3x3Weights(const int8_t* source, size_t outputs,
                        size_t input_channels, size_t nr,
                        AlignedBuffer<int8_t>& destination);

// Per-thread owned scratch buffers for direct 9-tap NHWC convolutions.
struct Conv3x3ThreadScratch {
  AlignedBuffer<uint8_t> pad1;        // 10 * 10 * padded_channels
  AlignedBuffer<uint8_t> pad2;        // 10 * 10 * padded_channels
  AlignedBuffer<float> conv2_output;  // 64 * channels
  AlignedBuffer<float> avg_pool;      // channels
  AlignedBuffer<float> se_fc1;        // se_fc_outputs
  AlignedBuffer<float> se_fc2;        // 2 * channels
};

}  // namespace blas_backend
}  // namespace lczero
