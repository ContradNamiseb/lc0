/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors
  Copyright (C) 2023 Intel Corporation

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
   
  SPDX-License-Identifier:GNU General Public License v3.0 or later
*/

#pragma once

#include <string>
#include <sycl/sycl.hpp>

#include "utils/exception.h"

#if defined(USE_HIPBLAS)
#define SYCL_SUB_GROUP_SIZE 64
#elif defined(USE_CUBLAS)
#define SYCL_SUB_GROUP_SIZE 32
#else
#define SYCL_SUB_GROUP_SIZE 16
#endif

namespace lczero {
namespace sycldnn_backend {

static constexpr int kNumOutputPolicy = 1858;

// max supported filter count for fast path
// TODO: extend it to cover bigger networks!
// (We are limited by no of registers per thread)
static constexpr int kMaxResBlockFusingChannels = 384;  // limit on num_filters
static constexpr int kMaxResBlockFusingSeKFp16Ampere =
    512;  // (use a different kernel with reduced register pressure)
static constexpr int kMaxResBlockFusingSeK =
    128;  // limit on (num_filters / se_ratio)
static constexpr int kMaxResBlockFusingSeFp16AmpereSmem =
    72 * kMaxResBlockFusingSeKFp16Ampere *
    sizeof(sycl::half);  // shared memory used by the special
                         // kernel

#ifdef USE_CUBLAS
void CublasError(int status, const char* file, const int& line);

#define ReportCUBLASErrors(status) CublasError(status, __FILE__, __LINE__)
#endif

// The allowFusing decision for the SubGroup fused SE+output+input transform
// kernels. Shared between ResidualBlockLayer's dispatch (layers.cc.dp.cpp)
// and sycl_test.cc so the test exercises the real decision rather than a
// copy with its own constants, which had already drifted from production.
inline bool AllowSubGroupFusing(int C, int se_k, size_t shared_mem_size,
                                size_t max_wg_size, bool is_fp16) {
  bool allow_fusing =
      ((C <= kMaxResBlockFusingChannels) ||
       (is_fp16 && (shared_mem_size >= kMaxResBlockFusingSeFp16AmpereSmem) &&
        (C <= kMaxResBlockFusingSeKFp16Ampere))) &&
      (se_k <= C);

  // The SubGroup kernels launch with work-group size = C. Verify that the
  // device can support this before enabling fusing to avoid crashes or
  // silent errors.
  if (is_fp16) {
    allow_fusing = allow_fusing && (static_cast<size_t>(C) <= max_wg_size);
  }

  return allow_fusing;
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

struct SyclDeviceCache {
  int sub_group_size = 32;
  size_t max_work_group_size = 256;
  size_t l2_cache_size = 0;
  size_t local_mem_size = 0;
  size_t max_mem_alloc_size = 0;
  size_t global_mem_size = 0;
  int max_compute_units = 1;
  int max_clock_frequency = 0;
  bool supports_fp16 = false;
  bool is_gpu = false;
  std::string device_name;
};

}  // namespace sycldnn_backend
}  // namespace lczero
