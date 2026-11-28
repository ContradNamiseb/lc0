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

#include "neural/backends/blas/int8_layers.h"

#include <algorithm>
#include <array>
#include <cstring>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#include "neural/backends/blas/blas.h"
#include "neural/backends/blas/layers.h"
#include "utils/exception.h"

#ifdef USE_ISPC
#include "int8_layers_ispc.h"
#endif

namespace lczero {
namespace blas_backend {
namespace {

constexpr int kBoardSize = 8;
constexpr int kFilterTaps = 9;

// For each square, the squares its 3x3 filter reads, in filter order; -1
// marks a tap off the board, which reads the zero point (the quantized 0.0).
constexpr std::array<std::array<int, kFilterTaps>, kSquares> MakeNeighbours() {
  std::array<std::array<int, kFilterTaps>, kSquares> neighbours{};
  for (int y = 0; y < kBoardSize; y++) {
    for (int x = 0; x < kBoardSize; x++) {
      for (int dy = 0; dy < 3; dy++) {
        for (int dx = 0; dx < 3; dx++) {
          const int row = y + dy - 1;
          const int column = x + dx - 1;
          const bool on_board = row >= 0 && row < kBoardSize && column >= 0 &&
                                column < kBoardSize;
          neighbours[y * kBoardSize + x][dy * 3 + dx] =
              on_board ? row * kBoardSize + column : -1;
        }
      }
    }
  }
  return neighbours;
}
constexpr auto kNeighbours = MakeNeighbours();

// Adding and subtracting 1.5 * 2^23 rounds a float in [0, 2^22) to the
// nearest integer, ties to even, like rint(), in a form compilers vectorize.
constexpr float kRoundingConstant = 12582912.0f;

inline void CpuPause() {
#if defined(_M_X64) || defined(__x86_64__)
  _mm_pause();
#endif
}

#if defined(_M_X64) || defined(__x86_64__)
struct CpuIdRegs {
  uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
};

CpuIdRegs QueryCpuId(uint32_t leaf, uint32_t subleaf = 0) {
  CpuIdRegs r;
#if defined(_MSC_VER)
  int regs[4] = {};
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  r.eax = static_cast<uint32_t>(regs[0]);
  r.ebx = static_cast<uint32_t>(regs[1]);
  r.ecx = static_cast<uint32_t>(regs[2]);
  r.edx = static_cast<uint32_t>(regs[3]);
#else
  __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
  return r;
}

uint64_t ReadXcr0() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  uint32_t lo = 0, hi = 0;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<uint64_t>(hi) << 32) | lo;
#endif
}
#endif

struct CpuVnniCaps {
  bool avx512_vnni = false;
  bool avx_vnni = false;
};

CpuVnniCaps DetectCpuVnniCaps() {
  CpuVnniCaps caps;
#if defined(_M_X64) || defined(__x86_64__)
  const CpuIdRegs leaf0 = QueryCpuId(0, 0);
  if (leaf0.eax < 7) return caps;
  const CpuIdRegs leaf1 = QueryCpuId(1, 0);
  // OSXSAVE (ECX bit 27) and AVX (ECX bit 28) required to read XCR0.
  const bool osxsave = (leaf1.ecx & (1u << 27)) != 0;
  const bool avx = (leaf1.ecx & (1u << 28)) != 0;
  const bool fma = (leaf1.ecx & (1u << 12)) != 0;
  if (!osxsave || !avx) return caps;

  const uint64_t xcr0 = ReadXcr0();
  const bool ymm_os = (xcr0 & 0x6u) == 0x6u;    // XMM + YMM state
  const bool zmm_os = (xcr0 & 0xE6u) == 0xE6u;  // XMM + YMM + opmask + ZMM

  const CpuIdRegs leaf7_0 = QueryCpuId(7, 0);
  const bool avx2 = (leaf7_0.ebx & (1u << 5)) != 0;
  const bool avx512f = (leaf7_0.ebx & (1u << 16)) != 0;
  const bool avx512dq = (leaf7_0.ebx & (1u << 17)) != 0;
  const bool avx512bw = (leaf7_0.ebx & (1u << 30)) != 0;
  const bool avx512vl = (leaf7_0.ebx & (1u << 31)) != 0;
  const bool avx512vnni = (leaf7_0.ecx & (1u << 11)) != 0;

  if (zmm_os && avx512f && avx512dq && avx512bw && avx512vl && avx512vnni) {
    caps.avx512_vnni = true;
  }

  if (ymm_os && avx2 && fma && leaf7_0.eax >= 1) {
    const CpuIdRegs leaf7_1 = QueryCpuId(7, 1);
    const bool avx_vnni = (leaf7_1.eax & (1u << 4)) != 0;
    if (avx_vnni) {
      caps.avx_vnni = true;
    }
  }
#endif
  return caps;
}

const CpuVnniCaps& GetCpuVnniCaps() {
  static const CpuVnniCaps caps = DetectCpuVnniCaps();
  return caps;
}

void NchwToNhwcScalar(size_t batch_size, size_t channels, const float* input,
                      float* output) {
  for (size_t b = 0; b < batch_size; ++b) {
    const float* src = input + b * channels * kSquares;
    float* dst = output + b * kSquares * channels;
    for (size_t sq = 0; sq < kSquares; ++sq) {
      for (size_t c = 0; c < channels; ++c) {
        dst[sq * channels + c] = src[c * kSquares + sq];
      }
    }
  }
}

void NhwcToNchwScalar(size_t batch_size, size_t channels, const float* input,
                      float* output) {
  for (size_t b = 0; b < batch_size; ++b) {
    const float* src = input + b * kSquares * channels;
    float* dst = output + b * channels * kSquares;
    for (size_t c = 0; c < channels; ++c) {
      for (size_t sq = 0; sq < kSquares; ++sq) {
        dst[c * kSquares + sq] = src[sq * channels + c];
      }
    }
  }
}

}  // namespace

bool CpuSupportsAvx512Vnni() {
#ifdef LC0_HAS_AVX512_VNNI
  return GetCpuVnniCaps().avx512_vnni;
#else
  return false;
#endif
}

bool CpuSupportsAvxVnni() {
#ifdef LC0_HAS_AVX_VNNI
  return GetCpuVnniCaps().avx_vnni;
#else
  return false;
#endif
}

bool IsVnniAvailable() {
  return CpuSupportsAvx512Vnni() || CpuSupportsAvxVnni();
}

bool IsInt8SupportedOnHost(bool allow_mkl) {
  return IsVnniAvailable() || (allow_mkl && kMklInt8Available);
}

Int8Kernel SelectInt8Kernel(const std::string& requested, bool allow_mkl) {
  if (requested.empty() || requested == "auto") {
    if (CpuSupportsAvx512Vnni()) return Int8Kernel::kVnniAvx512;
    if (CpuSupportsAvxVnni()) return Int8Kernel::kVnniAvx;
    if (allow_mkl && kMklInt8Available) return Int8Kernel::kMkl;
    return Int8Kernel::kMkl;
  }
  if (requested == "vnni") {
    if (CpuSupportsAvx512Vnni()) return Int8Kernel::kVnniAvx512;
    if (CpuSupportsAvxVnni()) return Int8Kernel::kVnniAvx;
    throw Exception(
        "int8_kernel=vnni requested, but this CPU/build lacks VNNI.");
  }
  if (requested == "avx512") {
    if (CpuSupportsAvx512Vnni()) return Int8Kernel::kVnniAvx512;
    throw Exception(
        "int8_kernel=avx512 requested, but this CPU/build lacks AVX-512 VNNI.");
  }
  if (requested == "avx") {
    if (CpuSupportsAvxVnni()) return Int8Kernel::kVnniAvx;
    throw Exception(
        "int8_kernel=avx requested, but this CPU/build lacks AVX-VNNI.");
  }
  if (requested == "mkl") {
    if (allow_mkl && kMklInt8Available) return Int8Kernel::kMkl;
    throw Exception("int8_kernel=mkl requested, but MKL is not available.");
  }
  throw Exception("Unknown int8_kernel '" + requested +
                  "' (expected auto, vnni, avx512, avx, or mkl).");
}

const char* Int8KernelName(Int8Kernel kernel) {
  switch (kernel) {
    case Int8Kernel::kVnniAvx512:
      return "AVX-512 VNNI";
    case Int8Kernel::kVnniAvx:
      return "AVX-VNNI";
    case Int8Kernel::kMkl:
      return "MKL cblas_gemm_s8u8s32";
    case Int8Kernel::kAuto:
      return "auto";
  }
  return "unknown";
}

// Int8ThreadPool

Int8ThreadPool::Int8ThreadPool(int num_threads)
    : num_threads_(std::clamp(num_threads, 1, static_cast<int>(kActiveMask))) {
  if (num_threads_ <= 1) return;
  workers_.reserve(static_cast<size_t>(num_threads_ - 1));
  for (int i = 1; i < num_threads_; ++i) {
    workers_.emplace_back(&Int8ThreadPool::WorkerLoop, this,
                          static_cast<size_t>(i));
  }
}

Int8ThreadPool::~Int8ThreadPool() {
  if (workers_.empty()) return;
  stop_.store(true, std::memory_order_release);
  const uint64_t next_epoch =
      (generation_.load(std::memory_order_relaxed) >> kActiveBits) + 1;
  // Publish active=0 on shutdown so waking workers never read job fields.
  generation_.store(next_epoch << kActiveBits, std::memory_order_seq_cst);
  generation_.notify_all();
  for (auto& worker : workers_) {
    if (worker.joinable()) worker.join();
  }
}

void Int8ThreadPool::RunParallel(size_t total_items, size_t active,
                                 ThunkFn thunk, void* ctx) {
  current_thunk_ = thunk;
  current_ctx_ = ctx;
  current_total_ = total_items;
  completed_.store(0, std::memory_order_relaxed);

  const uint64_t next_epoch =
      (generation_.load(std::memory_order_relaxed) >> kActiveBits) + 1;
  const uint64_t next_gen = (next_epoch << kActiveBits) |
                            (static_cast<uint64_t>(active) & kActiveMask);
  // seq_cst pairs with sleeping_workers_.fetch_add(seq_cst) and
  // generation_.wait(..., seq_cst) in WorkerLoop for zero lost wakeups.
  generation_.store(next_gen, std::memory_order_seq_cst);
  if (sleeping_workers_.load(std::memory_order_seq_cst) > 0) {
    generation_.notify_all();
  }

  // Main thread runs slice 0.
  const size_t begin0 = 0;
  const size_t end0 = total_items / active;
  thunk(ctx, begin0, end0, 0);

  // Spin until active - 1 worker threads finish.
  const size_t target = active - 1;
  while (completed_.load(std::memory_order_acquire) < target) {
    CpuPause();
  }
}

void Int8ThreadPool::WorkerLoop(size_t tid) {
  uint64_t seen_gen = 0;
  while (true) {
    uint64_t cur_gen = generation_.load(std::memory_order_acquire);
    if (cur_gen == seen_gen) {
      constexpr int kSpinIters = 4000;
      for (int spin = 0; spin < kSpinIters; ++spin) {
        CpuPause();
        cur_gen = generation_.load(std::memory_order_acquire);
        if (cur_gen != seen_gen) break;
      }
      if (cur_gen == seen_gen) {
        sleeping_workers_.fetch_add(1, std::memory_order_seq_cst);
        generation_.wait(seen_gen, std::memory_order_seq_cst);
        sleeping_workers_.fetch_sub(1, std::memory_order_relaxed);
        cur_gen = generation_.load(std::memory_order_acquire);
      }
    }
    if (stop_.load(std::memory_order_acquire)) return;
    seen_gen = cur_gen;

    // Decode `active` from atomic `cur_gen` so idle workers (tid >= active)
    // never touch the non-atomic job fields while the main thread advances.
    const size_t active = static_cast<size_t>(cur_gen & kActiveMask);
    if (tid < active) {
      const size_t total = current_total_;
      const size_t begin = (total * tid) / active;
      const size_t end = (total * (tid + 1)) / active;
      current_thunk_(current_ctx_, begin, end, tid);
      completed_.fetch_add(1, std::memory_order_release);
    }
  }
}

// Weight packing and layout helpers

void PackGemmWeights(const int8_t* source, size_t outputs, size_t inputs,
                     size_t nr, AlignedBuffer<int8_t>& destination) {
  const size_t k_groups = (inputs + 3) / 4;
  const size_t n_blocks = (outputs + nr - 1) / nr;
  const size_t total_bytes = n_blocks * k_groups * nr * 4;
  destination.ResizeZero(total_bytes);
  int8_t* out = destination.data();

  for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
    const size_t n_offset = n_block * nr;
    const size_t n_valid = std::min(nr, outputs - n_offset);
    int8_t* block_destination = out + n_block * k_groups * nr * 4;
    for (size_t k_group = 0; k_group < k_groups; ++k_group) {
      const size_t k_offset = k_group * 4;
      const size_t k_valid = std::min<size_t>(4, inputs - k_offset);
      int8_t* group_destination = block_destination + k_group * nr * 4;
      for (size_t j = 0; j < n_valid; ++j) {
        const int8_t* weight_row = source + (n_offset + j) * inputs + k_offset;
        for (size_t kb = 0; kb < k_valid; ++kb) {
          group_destination[j * 4 + kb] = weight_row[kb];
        }
      }
    }
  }
}

void PackConv3x3Weights(const int8_t* source, size_t outputs,
                        size_t input_channels, size_t nr,
                        AlignedBuffer<int8_t>& destination) {
  const size_t k_groups = (input_channels + 3) / 4;
  const size_t n_blocks = (outputs + nr - 1) / nr;
  const size_t tap_stride = k_groups * nr * 4;
  destination.ResizeZero(n_blocks * kFilterTaps * tap_stride);
  int8_t* out = destination.data();
  const size_t row_stride = input_channels * kFilterTaps;

  for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
    const size_t n_offset = n_block * nr;
    const size_t n_valid = std::min(nr, outputs - n_offset);
    for (size_t tap = 0; tap < kFilterTaps; ++tap) {
      int8_t* tap_destination =
          out + (n_block * kFilterTaps + tap) * tap_stride;
      for (size_t k_group = 0; k_group < k_groups; ++k_group) {
        const size_t c_offset = k_group * 4;
        const size_t c_valid = std::min<size_t>(4, input_channels - c_offset);
        int8_t* group_destination = tap_destination + k_group * nr * 4;
        for (size_t j = 0; j < n_valid; ++j) {
          const int8_t* weight_row = source + (n_offset + j) * row_stride;
          for (size_t cb = 0; cb < c_valid; ++cb) {
            group_destination[j * 4 + cb] =
                weight_row[(c_offset + cb) * kFilterTaps + tap];
          }
        }
      }
    }
  }
}

void NchwToNhwcFast(size_t batch_size, size_t channels, const float* input,
                    float* output, Int8ThreadPool* pool) {
  auto run = [&](size_t b0, size_t b1, size_t) {
    const size_t count = b1 - b0;
    const float* src = input + b0 * channels * kSquares;
    float* dst = output + b0 * kSquares * channels;
#ifdef LC0_HAS_AVX512_VNNI
    if (CpuSupportsAvx512Vnni()) {
      NchwToNhwcAvx512(count, channels, src, dst);
      return;
    }
#endif
    NchwToNhwcScalar(count, channels, src, dst);
  };
  if (pool && pool->num_threads() > 1 && batch_size >= 4) {
    pool->ParallelFor(batch_size, 2, run);
  } else {
    run(0, batch_size, 0);
  }
}

void NhwcToNchwFast(size_t batch_size, size_t channels, const float* input,
                    float* output, Int8ThreadPool* pool) {
  auto run = [&](size_t b0, size_t b1, size_t) {
    const size_t count = b1 - b0;
    const float* src = input + b0 * kSquares * channels;
    float* dst = output + b0 * channels * kSquares;
#ifdef LC0_HAS_AVX512_VNNI
    if (CpuSupportsAvx512Vnni()) {
      NhwcToNchwAvx512(count, channels, src, dst);
      return;
    }
#endif
    NhwcToNchwScalar(count, channels, src, dst);
  };
  if (pool && pool->num_threads() > 1 && batch_size >= 4) {
    pool->ParallelFor(batch_size, 2, run);
  } else {
    run(0, batch_size, 0);
  }
}

// Int8Weights

Int8Weights::Int8Weights(const std::vector<int8_t>& weights,
                         const std::vector<float>& weight_scales,
                         float input_scale, uint8_t input_zero_point,
                         Int8Kernel kernel, bool is_conv3x3, bool allow_nr48)
    : weights_(weights.data()),
      outputs_(weight_scales.size()),
      kernel_(kernel),
      inverse_input_scale_(1.0f / input_scale),
      input_zero_point_(input_zero_point) {
  if (outputs_ == 0 || weights.size() % outputs_ != 0) {
    throw Exception("INT8 weights don't match their scales.");
  }
  inputs_ = weights.size() / outputs_;
  output_scales_.resize(outputs_);
  zero_point_corrections_.resize(outputs_);
  for (size_t output = 0; output < outputs_; output++) {
    int32_t weight_sum = 0;
    for (size_t input = 0; input < inputs_; input++) {
      weight_sum += weights_[output * inputs_ + input];
    }
    zero_point_corrections_[output] =
        static_cast<int32_t>(input_zero_point) * weight_sum;
    output_scales_[output] = input_scale * weight_scales[output];
  }

  if (kernel_ == Int8Kernel::kVnniAvx512) {
    if (is_conv3x3) {
      const size_t in_c = inputs_ / kFilterTaps;
      conv_nr_ = (allow_nr48 && (outputs_ % 48) == 0) ? 48 : 32;
      PackConv3x3Weights(weights_, outputs_, in_c, conv_nr_, packed_weights_);
    } else {
      PackGemmWeights(weights_, outputs_, inputs_, 32, packed_weights_);
    }
  } else if (kernel_ == Int8Kernel::kVnniAvx) {
    if (is_conv3x3) {
      conv_nr_ = 16;
      PackConv3x3Weights(weights_, outputs_, inputs_ / kFilterTaps, 16,
                         packed_weights_);
    } else {
      PackGemmWeights(weights_, outputs_, inputs_, 16, packed_weights_);
    }
  }
}

void Int8Weights::QuantizeInput(size_t count, const float* input,
                                uint8_t* output, Int8ThreadPool* pool) const {
  const float inverse_scale = inverse_input_scale_;
  const float zero_point = static_cast<float>(input_zero_point_);

  auto run_slice = [&](size_t begin, size_t end, size_t) {
    const size_t n = end - begin;
    const float* in = input + begin;
    uint8_t* out = output + begin;
#ifdef LC0_HAS_AVX512_VNNI
    if (kernel_ == Int8Kernel::kVnniAvx512) {
      QuantizeToUint8Avx512(n, in, inverse_scale, zero_point, out);
      return;
    }
#endif
#ifdef LC0_HAS_AVX_VNNI
    if (kernel_ == Int8Kernel::kVnniAvx) {
      QuantizeToUint8Avx(n, in, inverse_scale, zero_point, out);
      return;
    }
#endif
#ifdef USE_ISPC
    ispc::QuantizeToUint8(static_cast<uint32_t>(n), in, inverse_scale,
                          zero_point, out);
    return;
#endif
    for (size_t i = 0; i < n; i++) {
      const float value =
          std::min(std::max(in[i] * inverse_scale + zero_point, 0.0f), 255.0f);
      out[i] = static_cast<uint8_t>(
          static_cast<int>((value + kRoundingConstant) - kRoundingConstant));
    }
  };

  if (pool && pool->num_threads() > 1 && count >= 16384) {
    // Partition in 64-byte aligned chunks to avoid false sharing.
    const size_t blocks = (count + 63) / 64;
    pool->ParallelFor(blocks, 64, [&](size_t b0, size_t b1, size_t tid) {
      const size_t i0 = b0 * 64;
      const size_t i1 = std::min(count, b1 * 64);
      run_slice(i0, i1, tid);
    });
  } else {
    run_slice(0, count, 0);
  }
}

void Int8Weights::MatrixMultiply(size_t rows, const uint8_t* quantized_input,
                                 int32_t* accumulators) const {
#ifdef USE_MKL
  // MKL's first matrix is the unsigned one, the second the signed one.
  const MKL_INT32 no_offset = 0;
  cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset,
                     static_cast<MKL_INT>(rows), static_cast<MKL_INT>(outputs_),
                     static_cast<MKL_INT>(inputs_), 1.0f, quantized_input,
                     static_cast<MKL_INT>(inputs_), 0, weights_,
                     static_cast<MKL_INT>(inputs_), 0, 0.0f, accumulators,
                     static_cast<MKL_INT>(outputs_), &no_offset);
#else
  (void)rows;
  (void)quantized_input;
  (void)accumulators;
  throw Exception("INT8 MKL kernel requested in a build without MKL.");
#endif
}

void Int8Weights::Forward(size_t rows, const uint8_t* quantized_input,
                          const float* biases, ActivationFunction activation,
                          float* output, Int8Scratch& scratch) const {
#ifdef LC0_HAS_AVX512_VNNI
  if (kernel_ == Int8Kernel::kVnniAvx512) {
    if (scratch.pool && scratch.pool->num_threads() > 1 && rows >= 16) {
      // Partition in strips of 8 rows so each thread gets full MR=8 tiles.
      const size_t strips = (rows + 7) / 8;
      scratch.pool->ParallelFor(strips, 2, [&](size_t s0, size_t s1, size_t) {
        const size_t r0 = s0 * 8;
        const size_t r1 = std::min(rows, s1 * 8);
        if (r0 < r1) {
          GemmEpilogueAvx512(r0, r1, outputs_, inputs_, quantized_input,
                             packed_weights_.data(), output_scales_.data(),
                             zero_point_corrections_.data(), biases, activation,
                             output);
        }
      });
    } else {
      GemmEpilogueAvx512(0, rows, outputs_, inputs_, quantized_input,
                         packed_weights_.data(), output_scales_.data(),
                         zero_point_corrections_.data(), biases, activation,
                         output);
    }
    return;
  }
#endif
#ifdef LC0_HAS_AVX_VNNI
  if (kernel_ == Int8Kernel::kVnniAvx) {
    if (scratch.pool && scratch.pool->num_threads() > 1 && rows >= 8) {
      const size_t strips = (rows + 3) / 4;
      scratch.pool->ParallelFor(strips, 2, [&](size_t s0, size_t s1, size_t) {
        const size_t r0 = s0 * 4;
        const size_t r1 = std::min(rows, s1 * 4);
        if (r0 < r1) {
          GemmEpilogueAvx(r0, r1, outputs_, inputs_, quantized_input,
                          packed_weights_.data(), output_scales_.data(),
                          zero_point_corrections_.data(), biases, activation,
                          output);
        }
      });
    } else {
      GemmEpilogueAvx(0, rows, outputs_, inputs_, quantized_input,
                      packed_weights_.data(), output_scales_.data(),
                      zero_point_corrections_.data(), biases, activation,
                      output);
    }
    return;
  }
#endif

  // MKL fallback path.
  GrowBuffer(scratch.accumulators, rows * outputs_);
  MatrixMultiply(rows, quantized_input, scratch.accumulators.data());
#ifdef USE_ISPC
  ispc::DequantizeRows(static_cast<uint32_t>(rows),
                       static_cast<uint32_t>(outputs_),
                       scratch.accumulators.data(), output_scales_.data(),
                       zero_point_corrections_.data(), output);
  for (size_t row = 0; row < rows; row++) {
    float* values = output + row * outputs_;
    Activate(outputs_, values, biases, values, activation);
  }
  return;
#endif
  for (size_t row = 0; row < rows; row++) {
    const int32_t* accumulators = &scratch.accumulators[row * outputs_];
    float* values = output + row * outputs_;
    for (size_t i = 0; i < outputs_; i++) {
      values[i] =
          output_scales_[i] *
          static_cast<float>(accumulators[i] - zero_point_corrections_[i]);
    }
    Activate(outputs_, values, biases, values, activation);
  }
}

void Int8Weights::ForwardQuantized(size_t rows, const uint8_t* quantized_input,
                                   const float* biases,
                                   ActivationFunction activation,
                                   float out_inverse_scale,
                                   uint8_t out_zero_point,
                                   uint8_t* quantized_output,
                                   Int8Scratch& scratch) const {
#ifdef LC0_HAS_AVX512_VNNI
  if (kernel_ == Int8Kernel::kVnniAvx512) {
    const float out_zp_f = static_cast<float>(out_zero_point);
    if (scratch.pool && scratch.pool->num_threads() > 1 && rows >= 16) {
      const size_t strips = (rows + 7) / 8;
      scratch.pool->ParallelFor(strips, 2, [&](size_t s0, size_t s1, size_t) {
        const size_t r0 = s0 * 8;
        const size_t r1 = std::min(rows, s1 * 8);
        if (r0 < r1) {
          GemmEpilogueQuantizeAvx512(
              r0, r1, outputs_, inputs_, quantized_input,
              packed_weights_.data(), output_scales_.data(),
              zero_point_corrections_.data(), biases, activation,
              out_inverse_scale, out_zp_f, quantized_output);
        }
      });
    } else {
      GemmEpilogueQuantizeAvx512(
          0, rows, outputs_, inputs_, quantized_input, packed_weights_.data(),
          output_scales_.data(), zero_point_corrections_.data(), biases,
          activation, out_inverse_scale, out_zp_f, quantized_output);
    }
    return;
  }
#endif
  (void)rows;
  (void)quantized_input;
  (void)biases;
  (void)activation;
  (void)out_inverse_scale;
  (void)out_zero_point;
  (void)quantized_output;
  (void)scratch;
  throw Exception("ForwardQuantized called without AVX-512 VNNI kernel.");
}

void Int8Weights::Forward(size_t rows, const float* input, const float* biases,
                          ActivationFunction activation, float* output,
                          Int8Scratch& scratch) const {
  scratch.quantized_input.Resize(rows * inputs_);
  QuantizeInput(rows * inputs_, input, scratch.quantized_input.data(),
                scratch.pool);
  Forward(rows, scratch.quantized_input.data(), biases, activation, output,
          scratch);
}

void Int8Weights::RunResidualBlockNhwc(
    size_t batch_size, size_t channels, float* tensor_nhwc,
    const Int8Weights& conv1, const float* conv1_biases,
    const Int8Weights& conv2, const float* conv2_biases,
    ActivationFunction activation, const ResidualSeParams& se,
    Int8Scratch& scratch) {
  const size_t max_threads =
      scratch.pool ? static_cast<size_t>(scratch.pool->num_threads()) : 1;
  if (scratch.conv3_scratch.size() < max_threads) {
    scratch.conv3_scratch.resize(max_threads);
  }
  const size_t padded_channels = ((channels + 3) / 4) * 4;
  constexpr size_t kPadSquares = 10 * 10;

  auto run_boards = [&](size_t b0, size_t b1, size_t tid) {
    Conv3x3ThreadScratch& thread_scratch = scratch.conv3_scratch[tid];
    const size_t n_boards = b1 - b0;
#ifdef LC0_HAS_AVX512_VNNI
    if (conv1.kernel_ == Int8Kernel::kVnniAvx512) {
      thread_scratch.pad1.Resize(kPadSquares * padded_channels);
      thread_scratch.pad2.Resize(kPadSquares * padded_channels);
      Conv3x3ScratchView view;
      view.pad1 = thread_scratch.pad1.data();
      view.pad2 = thread_scratch.pad2.data();
      if (se.has_se) {
        thread_scratch.conv2_output.Resize(kSquares * channels);
        thread_scratch.avg_pool.Resize(channels);
        thread_scratch.se_fc1.Resize(se.se_fc_outputs);
        thread_scratch.se_fc2.Resize(2 * channels);
        view.conv2_output = thread_scratch.conv2_output.data();
        view.avg_pool = thread_scratch.avg_pool.data();
        view.se_fc1 = thread_scratch.se_fc1.data();
        view.se_fc2 = thread_scratch.se_fc2.data();
      }
      ResidualBlockBoardsAvx512(
          n_boards, channels, tensor_nhwc + b0 * kSquares * channels,
          conv1.packed_weights_.data(), conv1.output_scales_.data(),
          conv1.zero_point_corrections_.data(), conv1_biases,
          conv1.inverse_input_scale_, conv1.input_zero_point_,
          conv2.packed_weights_.data(), conv2.output_scales_.data(),
          conv2.zero_point_corrections_.data(), conv2_biases,
          conv2.inverse_input_scale_, conv2.input_zero_point_, activation, se,
          view, conv1.conv_nr_);
      return;
    }
#endif
    thread_scratch.pad1.Resize(kPadSquares * padded_channels);
    thread_scratch.pad2.Resize(kPadSquares * padded_channels);
    Conv3x3ScratchView view;
    view.pad1 = thread_scratch.pad1.data();
    view.pad2 = thread_scratch.pad2.data();
    if (se.has_se) {
      thread_scratch.conv2_output.Resize(kSquares * channels);
      thread_scratch.avg_pool.Resize(channels);
      thread_scratch.se_fc1.Resize(se.se_fc_outputs);
      thread_scratch.se_fc2.Resize(2 * channels);
      view.conv2_output = thread_scratch.conv2_output.data();
      view.avg_pool = thread_scratch.avg_pool.data();
      view.se_fc1 = thread_scratch.se_fc1.data();
      view.se_fc2 = thread_scratch.se_fc2.data();
    }

    for (size_t batch = b0; batch < b1; ++batch) {
      float* board_io = tensor_nhwc + batch * kSquares * channels;
#ifdef LC0_HAS_AVX_VNNI
      if (conv1.kernel_ == Int8Kernel::kVnniAvx) {
        ResidualBlockBoardAvx(
            channels, board_io, conv1.packed_weights_.data(),
            conv1.output_scales_.data(), conv1.zero_point_corrections_.data(),
            conv1_biases, conv1.inverse_input_scale_, conv1.input_zero_point_,
            conv2.packed_weights_.data(), conv2.output_scales_.data(),
            conv2.zero_point_corrections_.data(), conv2_biases,
            conv2.inverse_input_scale_, conv2.input_zero_point_, activation, se,
            view);
        continue;
      }
#endif
      throw Exception("RunResidualBlockNhwc called without a VNNI kernel.");
    }
  };

  if (scratch.pool && scratch.pool->num_threads() > 1 && batch_size > 1) {
    scratch.pool->ParallelFor(batch_size, 1, run_boards);
  } else {
    run_boards(0, batch_size, 0);
  }
}

void Int8Weights::Convolve3Nhwc(size_t batch_size, const float* input_nhwc,
                                float* output_nhwc, Int8Scratch& scratch,
                                const float* biases,
                                ActivationFunction activation) const {
  const size_t input_channels = inputs_ / kFilterTaps;
  const size_t padded_channels = ((input_channels + 3) / 4) * 4;
  constexpr size_t kPadSquares = 10 * 10;
  const size_t max_threads =
      scratch.pool ? static_cast<size_t>(scratch.pool->num_threads()) : 1;
  if (scratch.conv3_scratch.size() < max_threads) {
    scratch.conv3_scratch.resize(max_threads);
  }

  auto run_boards = [&](size_t b0, size_t b1, size_t tid) {
    Conv3x3ThreadScratch& thread_scratch = scratch.conv3_scratch[tid];
    thread_scratch.pad1.Resize(kPadSquares * padded_channels);
    Conv3x3ScratchView view;
    view.pad1 = thread_scratch.pad1.data();
    for (size_t batch = b0; batch < b1; ++batch) {
      const float* in_board = input_nhwc + batch * kSquares * input_channels;
      float* out_board = output_nhwc + batch * kSquares * outputs_;
#ifdef LC0_HAS_AVX512_VNNI
      if (kernel_ == Int8Kernel::kVnniAvx512) {
        Conv3x3NhwcBoardAvx512(input_channels, outputs_, in_board,
                               packed_weights_.data(), output_scales_.data(),
                               zero_point_corrections_.data(),
                               inverse_input_scale_, input_zero_point_,
                               out_board, view, biases, activation, conv_nr_);
        continue;
      }
#endif
#ifdef LC0_HAS_AVX_VNNI
      if (kernel_ == Int8Kernel::kVnniAvx) {
        Conv3x3NhwcBoardAvx(
            input_channels, outputs_, in_board, packed_weights_.data(),
            output_scales_.data(), zero_point_corrections_.data(),
            inverse_input_scale_, input_zero_point_, out_board, view);
        continue;
      }
#endif
      throw Exception("Convolve3Nhwc called without a VNNI kernel.");
    }
  };

  if (scratch.pool && scratch.pool->num_threads() > 1 && batch_size > 1) {
    scratch.pool->ParallelFor(batch_size, 1, run_boards);
  } else {
    run_boards(0, batch_size, 0);
  }
}

void Int8Weights::Convolve3NchwToNhwc(size_t batch_size,
                                      const float* input_nchw,
                                      float plane109_scale, float* output_nhwc,
                                      Int8Scratch& scratch, const float* biases,
                                      ActivationFunction activation) const {
  const size_t input_channels = inputs_ / kFilterTaps;
  const size_t padded_channels = ((input_channels + 3) / 4) * 4;
  constexpr size_t kPadSquares = 10 * 10;
  const size_t max_threads =
      scratch.pool ? static_cast<size_t>(scratch.pool->num_threads()) : 1;
  if (scratch.conv3_scratch.size() < max_threads) {
    scratch.conv3_scratch.resize(max_threads);
  }

  auto run_boards = [&](size_t b0, size_t b1, size_t tid) {
    Conv3x3ThreadScratch& thread_scratch = scratch.conv3_scratch[tid];
    thread_scratch.pad1.Resize(kPadSquares * padded_channels);
    thread_scratch.nhwc_board.Resize(kSquares * input_channels);
    Conv3x3ScratchView view;
    view.pad1 = thread_scratch.pad1.data();
    view.nhwc_board = thread_scratch.nhwc_board.data();
    for (size_t batch = b0; batch < b1; ++batch) {
      const float* in_board = input_nchw + batch * kSquares * input_channels;
      float* out_board = output_nhwc + batch * kSquares * outputs_;
#ifdef LC0_HAS_AVX512_VNNI
      if (kernel_ == Int8Kernel::kVnniAvx512) {
        Conv3x3NhwcBoardAvx512(input_channels, outputs_, in_board,
                               packed_weights_.data(), output_scales_.data(),
                               zero_point_corrections_.data(),
                               inverse_input_scale_, input_zero_point_,
                               out_board, view, biases, activation, conv_nr_,
                               /*input_is_nchw=*/true, plane109_scale);
        continue;
      }
#endif
      throw Exception("Convolve3NchwToNhwc called without an AVX512 kernel.");
    }
  };

  if (scratch.pool && scratch.pool->num_threads() > 1 && batch_size > 1) {
    scratch.pool->ParallelFor(batch_size, 1, run_boards);
  } else {
    run_boards(0, batch_size, 0);
  }
}

void RunGemmFp32Avx512(size_t rows, size_t outputs, size_t inputs,
                       const float* input, const float* weights_transposed,
                       const float* biases, ActivationFunction activation,
                       float* output, Int8ThreadPool* pool) {
#ifdef LC0_HAS_AVX512_VNNI
  const size_t work = rows * outputs * inputs;
  if (!pool || pool->num_threads() <= 1 || work < 65536) {
    GemmFp32Avx512(0, rows, 0, outputs, outputs, inputs, input,
                   weights_transposed, biases, activation, output);
    return;
  }
  const size_t threads = static_cast<size_t>(pool->num_threads());
  if (rows >= threads * 8) {
    pool->ParallelFor(rows, 8, [&](size_t r0, size_t r1, size_t /*tid*/) {
      GemmFp32Avx512(r0, r1, 0, outputs, outputs, inputs, input,
                     weights_transposed, biases, activation, output);
    });
  } else if (outputs >= 64) {
    const size_t n_blocks = (outputs + 31) / 32;
    pool->ParallelFor(n_blocks, 1, [&](size_t nb0, size_t nb1, size_t /*tid*/) {
      const size_t c0 = nb0 * 32;
      const size_t c1 = std::min(outputs, nb1 * 32);
      GemmFp32Avx512(0, rows, c0, c1, outputs, inputs, input,
                     weights_transposed, biases, activation, output);
    });
  } else if (rows >= 2) {
    pool->ParallelFor(rows, 1, [&](size_t r0, size_t r1, size_t /*tid*/) {
      GemmFp32Avx512(r0, r1, 0, outputs, outputs, inputs, input,
                     weights_transposed, biases, activation, output);
    });
  } else {
    GemmFp32Avx512(0, rows, 0, outputs, outputs, inputs, input,
                   weights_transposed, biases, activation, output);
  }
#else
  (void)rows;
  (void)outputs;
  (void)inputs;
  (void)input;
  (void)weights_transposed;
  (void)biases;
  (void)activation;
  (void)output;
  (void)pool;
  throw Exception("RunGemmFp32Avx512 requires AVX-512.");
#endif
}

void Int8Weights::Convolve3(size_t batch_size, const float* input,
                            float* output, Int8Scratch& scratch) const {
  const size_t channels = inputs_ / kFilterTaps;
  if (uses_vnni()) {
    scratch.nhwc_input.Resize(batch_size * kSquares * channels);
    scratch.nhwc_output.Resize(batch_size * kSquares * outputs_);
    NchwToNhwcFast(batch_size, channels, input, scratch.nhwc_input.data(),
                   scratch.pool);
    Convolve3Nhwc(batch_size, scratch.nhwc_input.data(),
                  scratch.nhwc_output.data(), scratch);
    NhwcToNchwFast(batch_size, outputs_, scratch.nhwc_output.data(), output,
                   scratch.pool);
    return;
  }

  const size_t rows = batch_size * kSquares;

  // Quantize the input, then lay out each square's 3x3 neighbourhood in all
  // channels as one row (im2col), so the convolution is one matrix multiply.
  scratch.quantized_input.Resize(batch_size * channels * kSquares);
  QuantizeInput(batch_size * channels * kSquares, input,
                scratch.quantized_input.data(), scratch.pool);
  GrowBuffer(scratch.columns, rows * inputs_);
  const uint8_t* q_in = scratch.quantized_input.data();
  for (size_t batch = 0; batch < batch_size; batch++) {
    const uint8_t* planes = &q_in[batch * channels * kSquares];
    for (int square = 0; square < kSquares; square++) {
      uint8_t* row = &scratch.columns[(batch * kSquares + square) * inputs_];
      const auto& neighbours = kNeighbours[square];
      for (size_t channel = 0; channel < channels; channel++) {
        const uint8_t* plane = planes + channel * kSquares;
        uint8_t* taps = row + channel * kFilterTaps;
        for (int tap = 0; tap < kFilterTaps; tap++) {
          const int source = neighbours[tap];
          taps[tap] = source >= 0 ? plane[source] : input_zero_point_;
        }
      }
    }
  }

  GrowBuffer(scratch.accumulators, rows * outputs_);
  MatrixMultiply(rows, scratch.columns.data(), scratch.accumulators.data());

  // The products come out as [square][channel]; write them back as NCHW.
  for (size_t batch = 0; batch < batch_size; batch++) {
    const int32_t* accumulators =
        &scratch.accumulators[batch * kSquares * outputs_];
    for (size_t channel = 0; channel < outputs_; channel++) {
      float* plane = output + (batch * outputs_ + channel) * kSquares;
      const float scale = output_scales_[channel];
      const int32_t correction = zero_point_corrections_[channel];
      for (int square = 0; square < kSquares; square++) {
        plane[square] =
            scale * static_cast<float>(
                        accumulators[square * outputs_ + channel] - correction);
      }
    }
  }
}

// DynamicInt8Weights

DynamicInt8Weights::DynamicInt8Weights(const float* weights_fp32,
                                       size_t outputs, size_t inputs)
    : inputs_(inputs),
      outputs_(outputs),
      weight_scales_(outputs, 0.0f),
      weight_sum_scaled_(outputs, 0.0f) {
  std::vector<int8_t> w_q(outputs * inputs, 0);
  for (size_t o = 0; o < outputs; ++o) {
    const float* row = weights_fp32 + o * inputs;
    float max_abs = 0.0f;
    for (size_t k = 0; k < inputs; ++k) {
      max_abs = std::max(max_abs, std::abs(row[k]));
    }
    if (max_abs == 0.0f) continue;
    const float scale = max_abs / 127.0f;
    const float inv_scale = 127.0f / max_abs;
    int32_t w_sum = 0;
    int8_t* dst = w_q.data() + o * inputs;
    for (size_t k = 0; k < inputs; ++k) {
      const float qf = std::clamp(row[k] * inv_scale, -127.0f, 127.0f);
      const int32_t qi = static_cast<int32_t>(std::lround(qf));
      dst[k] = static_cast<int8_t>(qi);
      w_sum += qi;
    }
    weight_scales_[o] = scale;
    weight_sum_scaled_[o] = scale * static_cast<float>(w_sum);
  }
  PackGemmWeights(w_q.data(), outputs_, inputs_, 32, packed_weights_);
}

void DynamicInt8Weights::PrepareDynamicRows(size_t rows, const float* input,
                                            Int8Scratch& scratch) const {
#ifdef LC0_HAS_AVX512_VNNI
  // Use quantized_input2 so any pre-quantized QKV buffer in quantized_input
  // stays intact across Smolgen.
  scratch.quantized_input2.Resize(rows * inputs_);
  if (scratch.dynamic_scales.size() < rows) {
    scratch.dynamic_scales.resize(rows);
  }
  if (scratch.dynamic_mins.size() < rows) {
    scratch.dynamic_mins.resize(rows);
  }
  uint8_t* q_in = scratch.quantized_input2.data();
  float* r_scales = scratch.dynamic_scales.data();
  float* r_mins = scratch.dynamic_mins.data();

  Int8ThreadPool* pool = scratch.pool;
  if (pool && pool->num_threads() > 1 && rows * inputs_ >= 16384 && rows >= 8) {
    pool->ParallelFor(rows, 4, [&](size_t r0, size_t r1, size_t) {
      QuantizeDynamicRowsAvx512(r1 - r0, inputs_, input + r0 * inputs_,
                                q_in + r0 * inputs_, r_scales + r0,
                                r_mins + r0);
    });
  } else {
    QuantizeDynamicRowsAvx512(rows, inputs_, input, q_in, r_scales, r_mins);
  }
#else
  (void)rows;
  (void)input;
  (void)scratch;
  throw Exception("DynamicInt8Weights requires AVX-512 VNNI.");
#endif
}

void DynamicInt8Weights::ForwardPrepared(size_t rows, const float* biases,
                                         ActivationFunction activation,
                                         float* output,
                                         Int8Scratch& scratch) const {
#ifdef LC0_HAS_AVX512_VNNI
  const uint8_t* q_in = scratch.quantized_input2.data();
  const float* r_scales = scratch.dynamic_scales.data();
  const float* r_mins = scratch.dynamic_mins.data();
  Int8ThreadPool* pool = scratch.pool;

  const size_t n_blocks = (outputs_ + 31) / 32;
  if (pool && pool->num_threads() > 1) {
    const size_t num_threads = static_cast<size_t>(pool->num_threads());
    const size_t row_strips = (rows + 7) / 8;
    // Partition by rows when there are enough 8-row strips for all threads;
    // otherwise partition by 32-column output blocks (crucial for small-M
    // Smolgen dense1/dense2 where rows == batch_size == 16!).
    if (row_strips >= num_threads * 2) {
      pool->ParallelFor(row_strips, 2, [&](size_t s0, size_t s1, size_t) {
        const size_t r0 = s0 * 8;
        const size_t r1 = std::min(rows, s1 * 8);
        if (r0 < r1) {
          GemmDynamicEpilogueAvx512(
              r0, r1, 0, n_blocks, outputs_, inputs_, q_in, r_scales, r_mins,
              packed_weights_.data(), weight_scales_.data(),
              weight_sum_scaled_.data(), biases, activation, output);
        }
      });
    } else if (n_blocks >= 2) {
      pool->ParallelFor(n_blocks, 1, [&](size_t nb0, size_t nb1, size_t) {
        if (nb0 < nb1) {
          GemmDynamicEpilogueAvx512(
              0, rows, nb0, nb1, outputs_, inputs_, q_in, r_scales, r_mins,
              packed_weights_.data(), weight_scales_.data(),
              weight_sum_scaled_.data(), biases, activation, output);
        }
      });
    } else {
      GemmDynamicEpilogueAvx512(
          0, rows, 0, n_blocks, outputs_, inputs_, q_in, r_scales, r_mins,
          packed_weights_.data(), weight_scales_.data(),
          weight_sum_scaled_.data(), biases, activation, output);
    }
  } else {
    GemmDynamicEpilogueAvx512(0, rows, 0, n_blocks, outputs_, inputs_, q_in,
                              r_scales, r_mins, packed_weights_.data(),
                              weight_scales_.data(), weight_sum_scaled_.data(),
                              biases, activation, output);
  }
#else
  (void)rows;
  (void)biases;
  (void)activation;
  (void)output;
  (void)scratch;
  throw Exception("DynamicInt8Weights requires AVX-512 VNNI.");
#endif
}

void DynamicInt8Weights::ForwardPreparedRange(
    size_t row_begin, size_t row_end, const float* biases,
    ActivationFunction activation, float* output,
    const Int8Scratch& scratch) const {
#ifdef LC0_HAS_AVX512_VNNI
  if (row_begin >= row_end) return;
  const uint8_t* q_in = scratch.quantized_input2.data();
  const float* r_scales = scratch.dynamic_scales.data();
  const float* r_mins = scratch.dynamic_mins.data();
  const size_t n_blocks = (outputs_ + 31) / 32;
  GemmDynamicEpilogueAvx512(row_begin, row_end, 0, n_blocks, outputs_, inputs_,
                            q_in, r_scales, r_mins, packed_weights_.data(),
                            weight_scales_.data(), weight_sum_scaled_.data(),
                            biases, activation, output);
#else
  (void)row_begin;
  (void)row_end;
  (void)biases;
  (void)activation;
  (void)output;
  (void)scratch;
  throw Exception("DynamicInt8Weights requires AVX-512 VNNI.");
#endif
}

void DynamicInt8Weights::Forward(size_t rows, const float* input,
                                 const float* biases,
                                 ActivationFunction activation, float* output,
                                 Int8Scratch& scratch) const {
  PrepareDynamicRows(rows, input, scratch);
  ForwardPrepared(rows, biases, activation, output, scratch);
}

// DynamicConv3x3Int8

DynamicConv3x3Int8::DynamicConv3x3Int8(const float* weights_oihw,
                                       size_t output_channels,
                                       size_t input_channels)
    : input_channels_(input_channels),
      output_channels_(output_channels),
      weight_scales_(output_channels, 0.0f),
      weight_sums_(output_channels, 0) {
  const size_t row_len = input_channels * kFilterTaps;
  std::vector<int8_t> w_q(output_channels * row_len, 0);
  for (size_t o = 0; o < output_channels; ++o) {
    const float* src = weights_oihw + o * row_len;
    float max_abs = 0.0f;
    for (size_t i = 0; i < row_len; ++i) {
      max_abs = std::max(max_abs, std::abs(src[i]));
    }
    if (max_abs == 0.0f) continue;
    const float scale = max_abs / 127.0f;
    const float inv_s = 127.0f / max_abs;
    int32_t w_sum = 0;
    int8_t* dst = w_q.data() + o * row_len;
    for (size_t i = 0; i < row_len; ++i) {
      const float qf = std::clamp(src[i] * inv_s, -127.0f, 127.0f);
      const int32_t qi = static_cast<int32_t>(std::lround(qf));
      dst[i] = static_cast<int8_t>(qi);
      w_sum += qi;
    }
    weight_scales_[o] = scale;
    weight_sums_[o] = w_sum;
  }
  PackConv3x3Weights(w_q.data(), output_channels_, input_channels_, 32,
                     packed_weights_);
}

void DynamicConv3x3Int8::ForwardNhwc(size_t batch_size, const float* input_nhwc,
                                     const float* biases,
                                     ActivationFunction activation,
                                     float* output_nhwc,
                                     Int8Scratch& scratch) const {
#ifdef LC0_HAS_AVX512_VNNI
  const size_t padded_channels = ((input_channels_ + 3) / 4) * 4;
  constexpr size_t kPadSquares = 10 * 10;
  const size_t max_threads =
      scratch.pool ? static_cast<size_t>(scratch.pool->num_threads()) : 1;
  if (scratch.conv3_scratch.size() < max_threads) {
    scratch.conv3_scratch.resize(max_threads);
  }

  auto run_boards = [&](size_t b0, size_t b1, size_t tid) {
    Conv3x3ThreadScratch& thread_scratch = scratch.conv3_scratch[tid];
    thread_scratch.pad1.Resize(kPadSquares * padded_channels);
    Conv3x3ScratchView view;
    view.pad1 = thread_scratch.pad1.data();

    std::vector<float> board_scales(output_channels_);
    std::vector<int32_t> board_zp_corr(output_channels_);
    const size_t board_in_elems = kSquares * input_channels_;

    for (size_t batch = b0; batch < b1; ++batch) {
      const float* in_board = input_nhwc + batch * board_in_elems;
      float* out_board = output_nhwc + batch * kSquares * output_channels_;

      float mn = in_board[0];
      float mx = mn;
      for (size_t i = 1; i < board_in_elems; ++i) {
        const float v = in_board[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      // Include 0.0f so zero-padded board borders are represented accurately.
      if (mn > 0.0f) mn = 0.0f;
      if (mx < 0.0f) mx = 0.0f;
      const float range = mx - mn;
      const float in_scale = (range > 1e-12f) ? (range / 255.0f) : 1.0f;
      const float inv_in_scale = (range > 1e-12f) ? (255.0f / range) : 0.0f;
      const uint8_t zp =
          (mn == 0.0f)
              ? uint8_t{0}
              : static_cast<uint8_t>(std::clamp<int>(
                    static_cast<int>(std::lround(-mn * inv_in_scale)), 0, 255));

      for (size_t o = 0; o < output_channels_; ++o) {
        board_scales[o] = in_scale * weight_scales_[o];
        board_zp_corr[o] = static_cast<int32_t>(zp) * weight_sums_[o];
      }

      Conv3x3NhwcBoardAvx512(input_channels_, output_channels_, in_board,
                             packed_weights_.data(), board_scales.data(),
                             board_zp_corr.data(), inv_in_scale, zp, out_board,
                             view, biases, activation);
    }
  };

  if (scratch.pool && scratch.pool->num_threads() > 1 && batch_size > 1) {
    scratch.pool->ParallelFor(batch_size, 1, run_boards);
  } else {
    run_boards(0, batch_size, 0);
  }
#else
  (void)batch_size;
  (void)input_nhwc;
  (void)biases;
  (void)activation;
  (void)output_nhwc;
  (void)scratch;
  throw Exception("DynamicConv3x3Int8 requires AVX-512 VNNI.");
#endif
}

}  // namespace blas_backend
}  // namespace lczero
