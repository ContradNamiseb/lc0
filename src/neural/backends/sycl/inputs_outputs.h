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

#include <sycl/sycl.hpp>
#include "neural/network.h"
#include "sycl_common.h"

#if defined(USE_CUBLAS) || defined(USE_HIPBLAS)
#include "cuBlasContext.h"
#endif

namespace lczero {
namespace sycldnn_backend {

struct InputsOutputs {
  InputsOutputs(const InputsOutputs&) = delete;
  InputsOutputs& operator=(const InputsOutputs&) = delete;
  InputsOutputs(InputsOutputs&&) = delete;
  InputsOutputs& operator=(InputsOutputs&&) = delete;

  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left, sycl::queue& m_ct1,
                size_t tensor_mem_size = 0, size_t scratch_size = 0,
                bool cublasDisableTensorCores = false): q_ct1(m_ct1) {
  #ifdef USE_CUBLAS
    cublasHandle_t h= cuBlasContextManager::getcuBlasHandle_t();
  #endif
    // Every allocation below is checked (S5) and every one already made is
    // released if a later one throws (S4): a throwing constructor must not
    // leak the earlier buffers.
    try {
    input_masks_mem_shared_ = checkedMallocHost<uint64_t>(maxBatchSize * kInputPlanes, q_ct1);
    input_val_mem_shared_ = checkedMallocHost<float>(maxBatchSize * kInputPlanes, q_ct1);
    // Separate device memory copy for policy output.
    // It's faster to write to device memory and then copy to host memory
    // than having the kernel write directly to it (for discrete GPUs).
    // (An experiment writing straight into shared memory on Intel iGPUs via
    // a USE_INTEL define lived here; nothing ever defined USE_INTEL, so the
    // always-copy path below is what every build has actually run. Reintroduce
    // properly, keyed on a real backend option, if the iGPU trade-off is ever
    // re-measured.)
    op_policy_mem_ = checkedMallocHost<float>(maxBatchSize * kNumOutputPolicy, q_ct1);
    op_policy_mem_gpu_ = checkedMallocDevice<float>(maxBatchSize * kNumOutputPolicy, q_ct1);
    op_value_mem_shared_ = checkedMallocHost<float>(maxBatchSize * (wdl ? 3 : 1), q_ct1);

    if (moves_left) {
      op_moves_left_mem_shared_ = checkedMallocHost<float>(maxBatchSize, q_ct1);
    }

    // memory for network execution managed inside this structure
    if (tensor_mem_size) {
      multi_stream_ = true;
      scratch_mem_ = (void*)checkedMallocDevice<char>(scratch_size, q_ct1);
      for (auto& mem : tensor_mem_) {
        mem = (void*)checkedMallocDevice<char>(tensor_mem_size, q_ct1);
        q_ct1.memset(mem, 0, tensor_mem_size);
      }
    } else {
      multi_stream_ = false;
    }
    } catch (...) {
      ReleaseAll();
      throw;
    }
  }


  ~InputsOutputs() { ReleaseAll(); }

  // Frees every buffer allocated so far; null-tolerant, so it can run from
  // the constructor's failure path too.
  void ReleaseAll() {
    if (input_masks_mem_shared_) sycl::free(input_masks_mem_shared_, q_ct1);
    if (input_val_mem_shared_) sycl::free(input_val_mem_shared_, q_ct1);
    if (op_value_mem_shared_) sycl::free(op_value_mem_shared_, q_ct1);
    if (op_moves_left_mem_shared_ != nullptr)
      sycl::free(op_moves_left_mem_shared_, q_ct1);
    if (op_policy_mem_gpu_) sycl::free(op_policy_mem_gpu_, q_ct1);
    if (op_policy_mem_) sycl::free(op_policy_mem_, q_ct1);

    if (multi_stream_) {
      for (auto mem : tensor_mem_) {
        if (mem) 
          sycl::free(mem, q_ct1);
      }
      if (scratch_mem_) 
        sycl::free(scratch_mem_, q_ct1);
      if (offset_pointers_) 
        sycl::free(offset_pointers_, q_ct1);
      if (head_offset_pointers_) {
        sycl::free(head_offset_pointers_, q_ct1);
      } 
    }
    input_masks_mem_shared_ = nullptr;
    input_val_mem_shared_ = nullptr;
    op_value_mem_shared_ = nullptr;
    op_moves_left_mem_shared_ = nullptr;
    op_policy_mem_gpu_ = nullptr;
    op_policy_mem_ = nullptr;
    tensor_mem_[0] = tensor_mem_[1] = tensor_mem_[2] = nullptr;
    scratch_mem_ = nullptr;
    offset_pointers_ = nullptr;
    head_offset_pointers_ = nullptr;
  }
  uint64_t* input_masks_mem_shared_ = nullptr;
  float* input_val_mem_shared_ = nullptr;
  float* op_value_mem_shared_ = nullptr;
  float* op_moves_left_mem_shared_ = nullptr;

  // This is a seperate copy.
  float* op_policy_mem_gpu_ = nullptr;
  float* op_policy_mem_ = nullptr;

  // memory needed to run the network owned by InputsOutputs when multi_stream
  // is enabled
  bool multi_stream_ = false;
  void* tensor_mem_[3] = {nullptr, nullptr, nullptr};
  void* scratch_mem_ = nullptr;
  void** offset_pointers_ = nullptr;
  void** head_offset_pointers_ = nullptr;

  // sycl queue used to run the network
  sycl::queue& q_ct1;
};

}  // namespace sycldnn_backend
}  // namespace lczero
