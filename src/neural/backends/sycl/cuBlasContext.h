#pragma once

#include <iostream>
#include <sycl/sycl.hpp>

#ifdef USE_CUBLAS

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

class cuBlasContextManager {
 private:
  cublasHandle_t handle;

  cuBlasContextManager() {
    cublasCreate(&handle);
  }

  ~cuBlasContextManager() {
    cublasDestroy(handle);
  }

 public:
  cuBlasContextManager(const cuBlasContextManager&) = delete;
  cuBlasContextManager& operator=(const cuBlasContextManager&) = delete;

  static cublasHandle_t getcuBlasHandle_t() {
    static cuBlasContextManager instance;
    return instance.handle;
  }

  static void destroycuBlasHandle_t() {
    // Managed automatically by function-local static destructor.
  }
};

#elif defined(USE_HIPBLAS)

#include "hip/hip_runtime.h" 
#include "hipblas/hipblas.h"

class hipBlasContextManager {
 private:
  hipblasHandle_t handle;

  hipBlasContextManager() {
    hipblasCreate(&handle);
  }

  ~hipBlasContextManager() {
    hipblasDestroy(handle);
  }

 public:
  hipBlasContextManager(const hipBlasContextManager&) = delete;
  hipBlasContextManager& operator=(const hipBlasContextManager&) = delete;

  static hipblasHandle_t gethipBlasHandle_t() {
    static hipBlasContextManager instance;
    return instance.handle;
  }

  static void destroyhipBlasHandle_t() {
    // Managed automatically by function-local static destructor.
  }
};

#endif
