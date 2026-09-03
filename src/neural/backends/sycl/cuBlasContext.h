#pragma once

#include <iostream>
#include <sycl/sycl.hpp>

#ifdef USE_CUBLAS

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

class cuBlasContextManager {
 public:
  cuBlasContextManager(const cuBlasContextManager&) = delete;
  cuBlasContextManager& operator=(const cuBlasContextManager&) = delete;

  // One handle per THREAD, not per process: every call site does
  // cublasSetStream(handle, ...) on the returned handle immediately before
  // its own gemm, so a single shared handle was a data race whenever two
  // forwardEval threads ran cublas calls concurrently (multi_stream=true).
  // The handle is created on first use in the thread and destroyed at
  // thread exit.
  static cublasHandle_t getcuBlasHandle_t() {
    thread_local cublasHandle_t handle = [] {
      cublasHandle_t h;
      cublasCreate(&h);
      return h;
    }();
    return handle;
  }

  static void destroycuBlasHandle_t() {
    // Handles are thread_local and destroyed automatically at thread exit.
  }
};

#elif defined(USE_HIPBLAS)

#include "hip/hip_runtime.h" 
#include "hipblas/hipblas.h"

class hipBlasContextManager {
 public:
  hipBlasContextManager(const hipBlasContextManager&) = delete;
  hipBlasContextManager& operator=(const hipBlasContextManager&) = delete;

  // One handle per THREAD -- same rationale as the cuBLAS manager above.
  static hipblasHandle_t gethipBlasHandle_t() {
    thread_local hipblasHandle_t handle = [] {
      hipblasHandle_t h;
      hipblasCreate(&h);
      return h;
    }();
    return handle;
  }

  static void destroyhipBlasHandle_t() {
    // Handles are thread_local and destroyed automatically at thread exit.
  }
};

#endif
