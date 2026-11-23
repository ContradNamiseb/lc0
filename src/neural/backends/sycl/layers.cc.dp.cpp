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

#include <sycl/sycl.hpp>
#include "layers.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

#ifdef USE_HIPBLAS 
#include "hipblas/hipblas.h"
#include "cuBlasContext.h"
#elif defined(USE_CUBLAS)
#include <sycl/backend/cuda.hpp>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuBlasContext.h"
#else
#include "oneapi/mkl.hpp"
#include "oneapi/mkl/blas.hpp"
#endif

#ifdef USE_SYCL_DNNL
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <unordered_map>
#endif

#include "sycl_common.h"
#include "kernels.h"
#include "neural/network.h"
#include "neural/tables/attention_policy_map.h"
#include "utils/fp16_utils.h"

#include <cmath>


#ifdef USE_HIPBLAS
#if hipblasVersionMajor < 3
#define HIPBLAS_COMPUTE_16F HIPBLAS_R_16F
#define HIPBLAS_COMPUTE_32F HIPBLAS_R_32F
#endif
#define transpose_type hipblasOperation_t 
#define transpose_type_transpose HIPBLAS_OP_T  
#define transpose_type_notranspose HIPBLAS_OP_N 
#elif defined(USE_CUBLAS)
#define transpose_type cublasOperation_t 
#define transpose_type_transpose CUBLAS_OP_T  
#define transpose_type_notranspose CUBLAS_OP_N 
#else
#define transpose_type oneapi::mkl::transpose 
#define transpose_type_transpose oneapi::mkl::transpose::trans
#define transpose_type_notranspose oneapi::mkl::transpose::nontrans
#endif


namespace lczero {
namespace sycldnn_backend {

// Use Single kernel for entire SE operation.
// Right now supported only for fp16 with nhwc and it's quite a bit faster
// than using multiple passes. The flag can be set to false for debugging.
static constexpr bool kUseFusedSELayer = true;

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc,
                               sycl::queue& sycl_queue)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), sycl_queue_(sycl_queue) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, sycl::queue& sycl_queue)
    : input_(ip),
      C(c),
      H(h),
      W(w),
      nhwc_(ip ? ip->nhwc_ : false),
      sycl_queue_(sycl_queue) {}

template <typename DataType>
SELayer<DataType>::SELayer(BaseLayer<DataType>* ip, int fc1Outputs,
                           bool addPrevLayerBias, ActivationFunction activation, sycl::queue &sycl_queue)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip, sycl_queue),
      numFc1Out_(fc1Outputs),
      addPrevLayerBias_(addPrevLayerBias),
      act_(activation) {
  w1_ = (DataType*)sycl::malloc_device(C * numFc1Out_ * sizeof(DataType),
                                       sycl_queue_);
  w2_ = (DataType*)sycl::malloc_device(2 * C * numFc1Out_ * sizeof(DataType),
                                       sycl_queue_);

  if (kUseFusedSELayer && nhwc_) {
    w1_t_ = (DataType*)sycl::malloc_device(C * numFc1Out_ * sizeof(DataType),
                                           sycl_queue_);
    w2_t_ = (DataType*)sycl::malloc_device(2 * C * numFc1Out_ * sizeof(DataType),
                                           sycl_queue_);
  }

  b1_ = (DataType*)sycl::malloc_device(numFc1Out_ * sizeof(DataType),
                                       sycl_queue_);
  b2_ = (DataType*)sycl::malloc_device(2 * C * sizeof(DataType), sycl_queue_);

  bPrev_ = (DataType*)sycl::malloc_device(C * sizeof(DataType), sycl_queue_);
}

template <typename DataType>
SELayer<DataType>::~SELayer() {
  sycl::free(w1_, sycl_queue_);
  sycl::free(w2_, sycl_queue_);
  sycl::free(b1_, sycl_queue_);
  sycl::free(b2_, sycl_queue_);
  sycl::free(bPrev_, sycl_queue_);
}

template <>
void SELayer<float>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                 float* prevLayerBias, void* /*scratch*/) {
  const size_t num_weights1 = C * numFc1Out_;
  const size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t weight_size2 = 2 * weight_size1;

  // Weight for the first FC layer.
  sycl_queue_.memcpy(w1_, w1, weight_size1);

  // Weight for the second FC layer.
  sycl_queue_.memcpy(w2_, w2, weight_size2);

  // Bias for the first FC layer.
  sycl_queue_.memcpy(b1_, b1, numFc1Out_ * sizeof(float));

  // Bias for the second FC layer.
  sycl_queue_.memcpy(b2_, b2, 2 * C * sizeof(float));

  // Bias for previous layer (Convolution).
  if (prevLayerBias) {
    sycl_queue_.memcpy(bPrev_, prevLayerBias, C * sizeof(float));
  }

  sycl_queue_.wait();
}

void cpuTranspose(float* op, float* ip, int rows, int cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) op[j * rows + i] = ip[i * cols + j];
}

template <>
void SELayer<sycl::half>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                float* prevLayerBias, void* scratch) {
  const size_t num_weights1 = C * numFc1Out_;
  size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t num_weights2 = 2 * num_weights1;
  size_t weight_size2 = 2 * weight_size1;

  // Transpose the weight matrices for the fused path.
  std::vector<float> temp(weight_size2);

  // Weight for the first FC layer.
 
  sycl_queue_.memcpy(scratch, w1, weight_size1).wait();
  
  copyTypeConverted((sycl::half*)w1_, (float*)scratch, (int)num_weights1, sycl_queue_);

  if (kUseFusedSELayer && nhwc_) {
    // transposed copy for fused SE kernel
    cpuTranspose(temp.data(), w1, numFc1Out_, C);
    
    sycl_queue_.memcpy(scratch, temp.data(), weight_size1).wait();    
    
    copyTypeConverted((sycl::half*)w1_t_, (float*)scratch, (int)num_weights1, sycl_queue_);
  }

  // Weight for the second FC layer.
  sycl_queue_.memcpy(scratch, w2, weight_size2).wait();
  
  copyTypeConverted((sycl::half*)w2_, (float*)scratch, (int)num_weights2, sycl_queue_);
  if (kUseFusedSELayer && nhwc_) {
    cpuTranspose(temp.data(), w2, 2 * C, numFc1Out_);
    
    sycl_queue_.memcpy(scratch, temp.data(), weight_size2).wait();
    copyTypeConverted((sycl::half*)w2_t_, (float*)scratch, (int)num_weights2, sycl_queue_);
  }

  // Bias for the first FC layer.
    
  sycl_queue_.memcpy(scratch, b1, numFc1Out_ * sizeof(float)).wait();
  
  copyTypeConverted((sycl::half*)b1_, (float*)scratch, numFc1Out_, sycl_queue_);

  // Bias for the second FC layer.
  sycl_queue_.memcpy(scratch, b2, 2 * C * sizeof(float)).wait();

  copyTypeConverted((sycl::half*)b2_, (float*)scratch, 2 * C, sycl_queue_);

  // Bias for previous layer (Convolution).
  if (prevLayerBias) {
    
    sycl_queue_.memcpy(scratch, prevLayerBias, C * sizeof(float)).wait();
    copyTypeConverted((sycl::half*)bPrev_, (float*)scratch, C, sycl_queue_);
  }

} 

template <>
void SELayer<float>::Eval(int N, float* output, const float* input,
                          const float* /*input2*/, void* scratch,
                          size_t scratch_size, sycl::queue &sycl_queue, float***) {

  //CERR << "SELayer<float>::Eval. ";                          
  // Ping-pong between 'op1' and 'op2' (parts of scratch memory).
  float* op1 = (float*)scratch;
  float* op2 = (float*)scratch + scratch_size / sizeof(float) / 2;

  // 1. Global avg pooling (also adds previous layer bias before computing
  // averages).
  globalAvgPool(N, C, op2, input, bPrev_, false, sycl_queue);

  // 2. First fully connected layer.
  float alpha = 1.0f, beta = 0.0f;

  #ifdef USE_CUBLAS
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();

  sycl_queue.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
        
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {

        auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);  

        ReportCUBLASErrors(cublasSgemm(handle, transpose_type_transpose, transpose_type_notranspose, numFc1Out_,
                                 N, C, &alpha, w1_, C, op2, C, &beta, op1,
                                 numFc1Out_));

        });
  });
  #elif defined(USE_HIPBLAS)
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();

  sycl_queue.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
        
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);  

        hipblasSgemm(handle, transpose_type_transpose, transpose_type_notranspose, numFc1Out_,
                                 N, C, &alpha, w1_, C, op2, C, &beta, op1,
                                 numFc1Out_);
        });
  });  
  #else
  
  oneapi::mkl::blas::column_major::gemm(sycl_queue, transpose_type_transpose,
        transpose_type_notranspose, numFc1Out_, N, C, alpha, w1_, C, op2,
        C, beta, op1, numFc1Out_);

  #endif

  addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, act_, sycl_queue);

  // 3. Second fully connected layer.

  #ifdef USE_CUBLAS
  sycl_queue.submit([&](sycl::handler &cgh) {
        
        
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {

        auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);  

        ReportCUBLASErrors(cublasSgemm(handle, transpose_type_transpose, transpose_type_notranspose, 2 * C, N,
                                 numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                 numFc1Out_, &beta, op2, 2 * C));

        });
  });

  #elif defined(USE_HIPBLAS)
  sycl_queue.submit([&](sycl::handler &cgh) {
        
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);  

        hipblasSgemm(handle, transpose_type_transpose, transpose_type_notranspose, 2 * C, N,
                                 numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                 numFc1Out_, &beta, op2, 2 * C);

        
        });
  });
  #else
    oneapi::mkl::blas::column_major::gemm(sycl_queue, transpose_type_transpose,
        transpose_type_notranspose, 2 * C, N, numFc1Out_, alpha, w2_,
        numFc1Out_, op1, numFc1Out_, beta, op2, 2 * C);
  #endif

  addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, ACTIVATION_NONE, sycl_queue);

  // 4. (Optional prev layer bias add), Global scale, residual add, relu and
  // bias.
  globalScale(N, C, output, input, op2, bPrev_, false, act_, sycl_queue);

}

template <>
void SELayer<sycl::half>::Eval(int N, sycl::half* output, const sycl::half* input,
                         const sycl::half* input2, void* scratch, size_t scratch_size, sycl::queue &sycl_queue, sycl::half***) {
  //CERR << "SELayer<sycl::half>::Eval. ";

  bool se_done = false;
  if (kUseFusedSELayer && nhwc_) {
    se_done = Se_Fp16_NHWC(N, C, numFc1Out_, output, input2, input, w1_t_, b1_,
                           w2_t_, b2_, bPrev_, act_, sycl_queue);
  }
  if (!se_done) {
    assert(output == input2);
    // Ping-pong between 'op1' and 'op2' (parts of scratch memory).
    sycl::half* op1 = (sycl::half*)scratch;
    sycl::half* op2 = (sycl::half*)scratch + scratch_size / sizeof(sycl::half) / 2;

    // 1. Global avg pooling (also adds previous layer bias before computing
    // averages).
    globalAvgPool(N, C, op2, input, bPrev_, nhwc_, sycl_queue);

    // 2. First fully connected layer.
    //half_raw one_h{0x3C00};
    //half_raw zero_h{0};

    #ifdef USE_CUBLAS
    __half_raw one_h{0x3C00};
    __half_raw zero_h{0};
    half alpha = one_h;
    half beta = zero_h;

    #elif defined(USE_HIPBLAS)
    hipblasHalf alpha{1};
    hipblasHalf beta{0};

    #else
    sycl::half alpha = 1;
    sycl::half beta = 0;
    #endif

    #ifdef USE_CUBLAS
  
    cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();

    sycl_queue.submit([&](sycl::handler &cgh) {
       
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {

        auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);  
    
        ReportCUBLASErrors(cublasHgemm(handle, transpose_type_transpose, transpose_type_notranspose, numFc1Out_,
                                   N, C, &alpha, ((const half *)w1_), C, ((const half *)op2), C, &beta, ((half *)op1),
                                   numFc1Out_));
    
        });
    });

#elif defined(USE_HIPBLAS)
    hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();

    sycl_queue.submit([&](sycl::handler &cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);

        hipblasHgemm(handle, transpose_type_transpose,
                     transpose_type_notranspose,numFc1Out_, N, C, &alpha,
                     ((const hipblasHalf *)w1_), C, ((const hipblasHalf *)op2), C,
                     &beta, ((hipblasHalf *)op1), numFc1Out_);

      });
    });
#else
    oneapi::mkl::blas::column_major::gemm(
        sycl_queue, transpose_type_transpose, transpose_type_notranspose,
        numFc1Out_, N, C, alpha, ((const sycl::half *)w1_), C,
        ((const sycl::half *)op2),C, beta, ((sycl::half *)op1), numFc1Out_);
#endif

    addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, act_, sycl_queue);

    #ifdef USE_CUBLAS

    sycl_queue_.submit([&](sycl::handler &cgh) {
        
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {

        auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);   
    
        // 3. Second fully connected layer.
        ReportCUBLASErrors(cublasHgemm(handle, transpose_type_transpose, transpose_type_notranspose, 2 * C, N,
                                   numFc1Out_, &alpha, ((const half *)w2_), numFc1Out_, ((const half *)op1),
                                   numFc1Out_, &beta, ((half *)op2), 2 * C));
  
        });
    });  
    
#elif defined(USE_HIPBLAS)
    sycl_queue.submit([&](sycl::handler &cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
        hipblasSetStream(handle, hipStreamHandle);

        hipblasHgemm(
            handle, transpose_type_transpose, transpose_type_notranspose, 2 * C,
            N, numFc1Out_, &alpha,((const hipblasHalf *)w2_), numFc1Out_,
            ((const hipblasHalf *)op1), numFc1Out_, &beta, ((hipblasHalf *)op2),
            2 * C);

      });
    });
#else
    oneapi::mkl::blas::column_major::gemm(
        sycl_queue, transpose_type_transpose, transpose_type_notranspose, 2 * C,
        N, numFc1Out_, alpha, ((const sycl::half *)w2_), numFc1Out_,
        ((const sycl::half *)op1), numFc1Out_, beta, ((sycl::half *)op2),
        2 * C);
#endif
    
    addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, ACTIVATION_NONE, sycl_queue);

    // 4. (Optional prev layer bias add), Global scale, residual add, relu and
    // bias.
    globalScale(N, C, output, input, op2, bPrev_, nhwc_, act_, sycl_queue);
  }
} 

template <typename DataType>
FCLayer<DataType>::FCLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           bool bias, ActivationFunction activation, sycl::queue &sycl_queue)
    : BaseLayer<DataType>(C, H, W, ip, sycl_queue), use_bias_(bias), act_(activation)  {
  const size_t weight_size =
      sizeof(DataType) * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  const size_t bias_size = sizeof(DataType) * C * H * W;
  
  weights_ = (DataType*)sycl::malloc_device(weight_size, sycl_queue_);

  if (use_bias_) {
    biases_ = (DataType *)sycl::malloc_device(bias_size, sycl_queue_);
  } else {
    biases_ = nullptr;
  }
}

template <>
void FCLayer<sycl::half>::LoadWeights(float* cpuWeight, float* cpuBias,
                                void* scratch) {
  const size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  const size_t weight_size = sizeof(float) * num_weights;
  const size_t num_biases = C * H * W;
  const size_t bias_size = sizeof(float) * num_biases;

  // also need to convert from fp32 to fp16
  assert(scratch);
  
  sycl_queue_.memcpy(scratch, cpuWeight, weight_size).wait();

  if (nhwc_) {
    convertNCHWtoNHWC((sycl::half*)weights_, (float*)scratch, (int)num_biases,
                      input_->GetC(), (int)num_biases, input_->GetC(),
                      input_->GetH(), input_->GetW(), sycl_queue_);
  } else {
    copyTypeConverted((sycl::half*)weights_, (float*)scratch, (int)num_weights, sycl_queue_);
  }

  if (cpuBias) {
    sycl_queue_.memcpy(scratch, cpuBias, bias_size).wait();
    copyTypeConverted((sycl::half*)biases_, (float*)scratch, (int)num_biases, sycl_queue_);
  } 
} 

template <>
void FCLayer<float>::LoadWeights(float* cpuWeight, float* cpuBias,
                                 void* /*scratch*/) {
  const size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  const size_t weight_size = sizeof(float) * num_weights;
  const size_t num_biases = C * H * W;
  const size_t bias_size = sizeof(float) * num_biases;

  
  sycl_queue_.memcpy(weights_, cpuWeight, weight_size);
  
  if (use_bias_) {
    sycl_queue_.memcpy(biases_, cpuBias, bias_size);
  }

  //sycl_queue_.wait();
}

template <>
 void FCLayer<sycl::half>::Eval(int N, sycl::half* output_tensor, const sycl::half* input_tensor,
                          const sycl::half* /*input2*/, void* /*scratch*/,
                          size_t /*scratch_size*/, sycl::queue &sycl_queue, sycl::half***) {

   //CERR << "FCLayer<sycl::half>::Eval. ";

   const int num_outputs = C * H * W;
   const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

   //sycl::half alpha = float2half_rn(1.0f), 
   //beta = float2half_rn(0.0f);
   
   #ifdef USE_CUBLAS
    __half_raw one_h{0x3C00};
    __half_raw zero_h{0};
    half alpha = one_h;
    half beta = zero_h;

    #elif defined(USE_HIPBLAS)
    hipblasHalf alpha{1};
    hipblasHalf beta{0};

    #else
    sycl::half alpha = 1;
    sycl::half beta = 0;
    #endif

   #ifdef USE_CUBLAS
    cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();

    sycl_queue.submit([&](sycl::handler &cgh) {
        
         cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {

         auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
         cublasSetStream(handle, cudaStreamHandle);    
  
         ReportCUBLASErrors(cublasHgemm(handle, transpose_type_transpose, transpose_type_notranspose, num_outputs,
                                  N, num_inputs, &alpha, ((const half *)weights_), num_inputs,
                                  ((const half *)input_tensor), num_inputs, &beta, ((half *)output_tensor),
                                  num_outputs));

       });
   });  
#elif defined(USE_HIPBLAS)
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
      hipblasSetStream(handle, hipStreamHandle);

      hipblasHgemm(
          handle, transpose_type_transpose, transpose_type_notranspose,
          num_outputs, N, num_inputs, &alpha, ((const hipblasHalf *)weights_),
          num_inputs, ((const hipblasHalf *)input_tensor), num_inputs, &beta,
          ((hipblasHalf *)output_tensor), num_outputs);

      });
  });
#else
  oneapi::mkl::blas::column_major::gemm(
      sycl_queue, transpose_type_transpose, transpose_type_notranspose,
      num_outputs, N, num_inputs, alpha, ((const sycl::half *)weights_),
      num_inputs, ((const sycl::half *)input_tensor), num_inputs, beta,
      ((sycl::half *)output_tensor), num_outputs);
#endif

   if (use_bias_ || (act_ != ACTIVATION_NONE)) {
     addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
                num_outputs, num_outputs * N, act_, sycl_queue);
   }
 } 

template <>
void FCLayer<float>::Eval(int N, float* output_tensor,
                          const float* input_tensor, const float* /*input2*/,
                          void* /*scratch*/, size_t /*scratch_size*/, sycl::queue &sycl_queue, float***) {

  //CERR << "FCLayer<float>::Eval. ";

  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  float alpha = 1.0f, beta = 0.0f;
  //CERR << "FCLayer<float>::Eval - 1. " << num_inputs << " " << num_outputs;

  #ifdef USE_CUBLAS
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();

  sycl_queue.submit([&](sycl::handler &cgh) {
        
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {

        auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);    


        ReportCUBLASErrors(cublasSgemm(handle, transpose_type_transpose, transpose_type_notranspose, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs));

      });
  });  
  #elif defined(USE_HIPBLAS)
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  sycl_queue.submit([&](sycl::handler &cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);    


        hipblasSgemm(handle, transpose_type_transpose, transpose_type_notranspose, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs);

        
      });
  });
  #else
   //printf("3\n");
   oneapi::mkl::blas::column_major::gemm(sycl_queue, transpose_type_transpose,
        transpose_type_notranspose, num_outputs, N, num_inputs, alpha,
        weights_, num_inputs, input_tensor, num_inputs, beta, output_tensor,
        num_outputs);
    
    //event.wait();
  #endif


  if (use_bias_ || (act_ != ACTIVATION_NONE)) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, act_, sycl_queue);
  }
}

template <typename DataType>
FCLayer<DataType>::~FCLayer() {
  sycl::free(weights_, sycl_queue_);
  sycl::free(biases_, sycl_queue_);
}

template <typename DataType>
PolicyMapLayer<DataType>::PolicyMapLayer(BaseLayer<DataType>* ip, int C, int H,
                                         int W, int usedSize, bool attention, sycl::queue& sycl_queue)
    : BaseLayer<DataType>(C, H, W, ip, sycl_queue),
      used_size_(usedSize),
      attention_map_(attention) {

  size_t weight_size = sizeof(short) * this->input_->GetC() * 64;

  if (attention) weight_size = sizeof(short) * usedSize;
  
  weights_ = (short *)sycl::malloc_device(weight_size, sycl_queue_);
}

template <typename DataType>
void PolicyMapLayer<DataType>::LoadWeights(const short* cpuWeight,
                                           void* /*scratch*/) {

  size_t weight_size = sizeof(short) * used_size_;

  if (nhwc_ && !attention_map_) {
    // convert CHW to HWC
    int C = used_size_ / 64;
    int Cin = this->input_->GetC();

    // C is the no. of channels actually used (typically 73).
    // Cin the the no. of channels in previous layer (padded up to 80).
    // Weights of this layer is a mapping to select which output index of the
    // policy vector (1858 elements) maps to every element of input
    // tensor (assuming NCHW layout). Note that there are 73x64 valid inputs
    // (80x64 taking padding), and only 1858 outputs so the mapping isn't
    // one to one. Only few of the indices point to valid index in policy
    // vector. Invalid entries are set to -1.

    // In fp16 mode, the tensor layout is NHWC so the weights need to be
    // adjusted to make them work as intended.

    // This is how the original weights looks like (CHW layout):
    /*
               HW (64)
       ----|-------------|
           |             |
           |             |
    C (73) |             |
           |             |
           |             |
       ------------------|   Cin (80)
           |  padding    |
           |-------------|
    */
    // The padding is not part of the weights provided (used_size_ is 73 x 64).
    //
    // The weights converted to HWC looks like this
    /*
                 C (73)
            |-------------|---|
            |             | P |
            |             | a |
    HW (64) |             | d |
            |             |   |
            |             |   |
            |-----------------|
                     Cin (80)
    */
    // In HWC, because the padding is now part of each row
    // we need to increase the size of weights to account
    // for it.
    // The pad elements point to -1 (invalid output index) and the
    // same kernel works for both HWC and CHW layouts after used_size_
    // is updated to include padding (80x64).

    used_size_ = Cin * 64;
    std::vector<short> convertedWeights(used_size_);

    for (int hw = 0; hw < 64; hw++)
      for (int c = 0; c < Cin; c++) {
        if (c < C)
          convertedWeights[hw * Cin + c] = cpuWeight[c * 64 + hw];
        else
          convertedWeights[hw * Cin + c] = -1;
      }
    sycl_queue_.memcpy(weights_, convertedWeights.data(),
                       used_size_ * sizeof(short)).wait();
  } else {
    sycl_queue_.memcpy(weights_, cpuWeight, weight_size).wait();
  }
}

template <typename DataType>
void PolicyMapLayer<DataType>::Eval(
    int N, DataType* output_tensor, const DataType* input_tensor,
    const DataType* /*input2*/, void* /*scratch*/, size_t /*scratch_size*/, sycl::queue &sycl_queue, DataType***) {
  
  //CERR << "PolicyMapLayer<DataType>::Eval. ";    

  int inputSize =
      this->input_->GetC() * this->input_->GetH() * this->input_->GetW();
  if (attention_map_) inputSize = used_size_;
  int outputSize = this->C * this->H * this->W;

  PolicyMap(N, output_tensor, input_tensor, weights_, inputSize, used_size_, outputSize, sycl_queue);
}

template <typename DataType> PolicyMapLayer<DataType>::~PolicyMapLayer() {
  free(weights_, sycl_queue_);
}

template <typename DataType>
FusedWinogradConvSELayer<DataType>::FusedWinogradConvSELayer(
    BaseLayer<DataType>* ip, int C, int H, int W, int Cin,
    ActivationFunction activation, bool bias, bool skip_add, bool se, int se_k,
    sycl::queue &sycl_queue, bool op_nhcw)
    : BaseLayer<DataType>(C, H, W, ip, false, sycl_queue),
      c_input_(Cin),
      act_(activation),
      use_bias_(bias),
      skip_add_(skip_add),
      has_se_(se),
      se_k_(se_k),
      op_nhcw_(op_nhcw){

  if (act_ != ACTIVATION_RELU && act_ != ACTIVATION_MISH && act_ != ACTIVATION_NONE) {
    throw Exception("Unsupported activation for fused winograd conv SE layer.");
  }

  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * c_input_ * C * 3 * 3;

  if (use_bias_) {
    const size_t bias_size = sizeof(DataType) * C;
    biases_ = (DataType *)sycl::malloc_device(bias_size, sycl_queue_);
  }

  // 6x6 transformed filter size, for 3x3 convolution
  transformed_weights_ = (DataType *)sycl::malloc_device(weight_size * 4, sycl_queue_);

  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = sizeof(DataType) * num_weights1;
    const size_t weight_size2 = sizeof(DataType) * num_weights2;
    const size_t biases_size1 = sizeof(DataType) * num_biases1;
    const size_t biases_size2 = sizeof(DataType) * num_biases2;

    w1_ = (DataType *)sycl::malloc_device(weight_size1 * 4, sycl_queue_);
    w2_ = (DataType *)sycl::malloc_device(weight_size2 * 4, sycl_queue_);
    b1_ = (DataType *)sycl::malloc_device(biases_size1 * 4, sycl_queue_);
    b2_ = (DataType *)sycl::malloc_device(biases_size2 * 4, sycl_queue_);
  }
}

template <typename DataType> void FusedWinogradConvSELayer<DataType>::LoadWeights(float* pfilter,
                                                     float* pBias,
                                                     void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size + bias_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  //sycl_queue_.memcpy(scratch, pfilter, weight_size).wait_and_throw();
  sycl_queue_.memcpy(scratch, pfilter, weight_size).wait();
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3, sycl_queue_);

  if (pBias) {
    
    
    //sycl_queue_.memcpy(scratch, pBias, bias_size).wait();
    sycl_queue_.memcpy(scratch, pBias, bias_size);  

    float total = 0;
    for(int i = 0; i < C; i++)
      total = pBias[i] + total;

    copyTypeConverted((DataType*)biases_, (float*)scratch, C, sycl_queue_);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, c_input_, transformed_weights_, weights, sycl_queue_);
}

// TODO: Do this on the GPU to improve network load time!
static inline void CpuTranspose(float* op, float* ip, size_t rows,
                                size_t cols) {
  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++) op[j * rows + i] = ip[i * cols + j];
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::LoadSEWeights(float* w1, float* b1,
                                                       float* w2, float* b2,
                                                       void* scratch) {
  const size_t num_weights1 = C * se_k_;
  const size_t num_weights2 = num_weights1 * 2;
  const size_t num_biases1 = se_k_;
  const size_t num_biases2 = 2 * C;

  // The shader uses transposed weight matrices.
  std::vector<float> temp_transposed(num_weights2);

  CpuTranspose(temp_transposed.data(), w1, se_k_, C);
  //sycl_queue_.memcpy(scratch, temp_transposed.data(), num_weights1 * sizeof(float)).wait();
  sycl_queue_.memcpy(scratch, temp_transposed.data(), num_weights1 * sizeof(float)).wait();
  
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1, sycl_queue_);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);

  //sycl_queue_.memcpy(scratch, temp_transposed.data(), num_weights2 * sizeof(float)).wait();
  sycl_queue_.memcpy(scratch, temp_transposed.data(), num_weights2 * sizeof(float)).wait();
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2, sycl_queue_);

  //sycl_queue_.memcpy(scratch, b1, num_biases1 * sizeof(float)).wait();
  sycl_queue_.memcpy(scratch, b1, num_biases1 * sizeof(float)).wait();
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1, sycl_queue_);

  //sycl_queue_.memcpy(scratch, b2, num_biases2 * sizeof(float)).wait();
  sycl_queue_.memcpy(scratch, b2, num_biases2 * sizeof(float)).wait();
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2, sycl_queue_);

}

template <>
 void BaseLayer<sycl::half>::cublasRowMajorMatrixMul(const sycl::half* A, const sycl::half* B,
                                               sycl::half* Out, int M, int N, int K,
                                               int batchSize, sycl::queue &sycl_queue) {
   // Need to initialize 1.0 and 0.0 as hexadecimal for fp16 because typecasting
   // float to sycl::half type doesn't work before CUDA 10.0
   #ifdef USE_CUBLAS
    __half_raw one_h{0x3C00};
    __half_raw zero_h{0};
    half alpha = one_h;
    half beta = zero_h;

    #else
    sycl::half alpha = 1;
    sycl::half beta = 0;
    #endif

   // dimensions of matrix A = M x K
   // dimensions of matrix B = K x N
   // dimensions of output   = M x N

   // cublas supports only col major output
   // to multiply row major matrices, use the trick below
  
  #ifdef USE_CUBLAS
   cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  
   sycl_queue.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
         cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
  
          auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
          cublasSetStream(handle, cudaStreamHandle);

          ReportCUBLASErrors(cublasGemmStridedBatchedEx(
             handle, transpose_type_notranspose, transpose_type_notranspose, N, M, K, &one_h, B, CUDA_R_16F, N,
             N * K, A, CUDA_R_16F, K, K * M, &zero_h, Out, CUDA_R_16F, N, N * M,
             batchSize, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));

          
         });   
   });
  
#elif defined(USE_HIPBLAS)
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();

  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
      hipblasSetStream(handle, hipStreamHandle);

      hipblasGemmStridedBatchedEx(
          handle, transpose_type_notranspose, transpose_type_notranspose, N, M,
          K, &alpha, B, HIPBLAS_R_16F, N, N * K, A, HIPBLAS_R_16F, K, K * M,
          &beta, Out, HIPBLAS_R_16F, N, N * M, batchSize, HIPBLAS_COMPUTE_16F,
          HIPBLAS_GEMM_DEFAULT);

    });
  });
#else
  int64_t M_ = M;
  int64_t N_ = N;
  int64_t K_ = K;
  oneapi::mkl::blas::column_major::gemm_batch(
      sycl_queue, transpose_type_notranspose, transpose_type_notranspose, N_,
      M_, K_, alpha, B, N_, N_ * K_, A, K_, K_ * M_, beta, Out, N_, N_ * M_,
      batchSize);
#endif
 }

template <> void BaseLayer<float>::cublasRowMajorMatrixMul(const float* A, const float* B,
                                               float* Out, int M, int N, int K,
                                               int batchSize, sycl::queue &sycl_queue) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;

  int64_t M_ = M;
  int64_t N_ = N;
  int64_t K_ = K;

  #ifdef USE_CUBLAS
  //static cublasHandle_t handle;
  //ReportCUBLASErrors(cublasCreate(&handle)); 
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  #endif

  #ifdef USE_HIPBLAS
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  #endif

  {
    #ifdef USE_CUBLAS
    sycl_queue.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
            auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
            cublasSetStream(handle, cudaStreamHandle);   

          ReportCUBLASErrors(cublasGemmStridedBatchedEx(
            handle, transpose_type_notranspose, transpose_type_notranspose, N, M, K, &floatOne, B, CUDA_R_32F, N,
            N * K, A, CUDA_R_32F, K, K * M, &floatZero, Out, CUDA_R_32F, N, N * M,
          batchSize, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

          
        });
    });
    #elif defined(USE_HIPBLAS)
    sycl_queue.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
            auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
            hipblasSetStream(handle, hipStreamHandle);   

          hipblasGemmStridedBatchedEx(
            handle, transpose_type_notranspose, transpose_type_notranspose, N, M, K, &floatOne, B, HIPBLAS_R_32F, N,
            N * K, A, HIPBLAS_R_32F, K, K * M, &floatZero, Out, HIPBLAS_R_32F, N, N * M,
          batchSize, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);

          
        });
    });  
    #else
      oneapi::mkl::blas::column_major::gemm_batch(sycl_queue, transpose_type_notranspose,
            transpose_type_notranspose, N_, M_, K_, floatOne, B, N_, N_ * K_, A, K_, K_ * M_, floatZero, Out, N_, N_ * M_, batchSize);
    #endif
  }
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* input2,
    void* scratch, size_t scratch_size, sycl::queue &sycl_queue, DataType***) {
  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.

  //CERR << "FusedWinogradConvSELayer<DataType>::Eval. ";

  DataType* transformed_input = (DataType*)scratch;
  DataType* transformed_output =
      transformed_input + scratch_size / (2 * sizeof(DataType));

  InputTransform<DataType, false>(N, c_input_, transformed_input, input, sycl_queue);
  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights_, transformed_output, N * 4, C, c_input_, 36, sycl_queue);

  if (act_ == ACTIVATION_NONE) {
    if (!has_se_ && use_bias_ && !skip_add_)
      OutputTransform<DataType, false, ACTIVATION_NONE, true, false, false, false>(
          N, C, 0, output, transformed_output, nullptr, biases_, nullptr, nullptr, nullptr, nullptr, sycl_queue);
    else
      throw Exception("unsupported network type!");
  } else if (act_ == ACTIVATION_RELU) {
    if (has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, true, ACTIVATION_RELU, true, true, false, false>(
          N, C, se_k_, output, transformed_output, input2, biases_, w1_, b1_,
          w2_, b2_, sycl_queue);
    else if (!has_se_ && use_bias_ && !skip_add_) {
      if (op_nhcw_)
        OutputTransform<DataType, false, ACTIVATION_RELU, true, false, false, true>(
            N, C, 0, output, transformed_output, nullptr, biases_, nullptr,
            nullptr, nullptr, nullptr, sycl_queue);
      else
        OutputTransform<DataType, false, ACTIVATION_RELU, true, false, false, false>(
            N, C, 0, output, transformed_output, nullptr, biases_, nullptr,
            nullptr, nullptr, nullptr, sycl_queue);
    } else if (!has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, false, ACTIVATION_RELU, true, true, false, false>(
          N, C, 0, output, transformed_output, input2, biases_, nullptr,
          nullptr, nullptr, nullptr, sycl_queue);
    else
      throw Exception("unsupported network type!");
  } else if (act_ == ACTIVATION_MISH) {
    if (has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, true, ACTIVATION_MISH, true, true, false, false>(
          N, C, se_k_, output, transformed_output, input2, biases_, w1_, b1_,
          w2_, b2_, sycl_queue);
    else if (!has_se_ && use_bias_ && !skip_add_) {
      if (op_nhcw_)
        OutputTransform<DataType, false, ACTIVATION_MISH, true, false, false, true>(
            N, C, 0, output, transformed_output, nullptr, biases_, nullptr,
            nullptr, nullptr, nullptr, sycl_queue);
      else
        OutputTransform<DataType, false, ACTIVATION_MISH, true, false, false, false>(
            N, C, 0, output, transformed_output, nullptr, biases_, nullptr,
            nullptr, nullptr, nullptr, sycl_queue);
    } else if (!has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, false, ACTIVATION_MISH, true, true, false, false>(
          N, C, 0, output, transformed_output, input2, biases_, nullptr,
          nullptr, nullptr, nullptr, sycl_queue);
    else
      throw Exception("unsupported network type!");
  } else
    throw Exception("unsupported network type!");
}

template <typename DataType>
FusedWinogradConvSELayer<DataType>::~FusedWinogradConvSELayer() {
  sycl::free(transformed_weights_, sycl_queue_);
  if (use_bias_) sycl::free(biases_, sycl_queue_);
  if (has_se_) {
    sycl::free(w1_, sycl_queue_);
    sycl::free(w2_, sycl_queue_);
    sycl::free(b1_, sycl_queue_);
    sycl::free(b2_, sycl_queue_);
  }
}

template <typename DataType>
Conv1Layer<DataType>::Conv1Layer(BaseLayer<DataType>* ip, int C, int H, int W,
                                 int Cin, ActivationFunction activation,
                                 bool bias, sycl::queue& sycl_queue)
    : BaseLayer<DataType>(C, H, W, ip, false, sycl_queue),
      c_input_(Cin),
      act_(activation),
      use_bias_(bias) {

  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * c_input_ * C * 1 * 1;
  weights_ = (DataType *)sycl::malloc_device(weight_size, sycl_queue_);

  if (use_bias_) {
    const size_t bias_size = sizeof(DataType) * C;
    //CERR << "Conv1Layer using bias " << bias_size; 
    biases_ = (DataType *)sycl::malloc_device(bias_size, sycl_queue_);
  }
}

template <typename DataType> void Conv1Layer<DataType>::LoadWeights(float* pfilter, float* pBias, void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 1 * 1;
  const size_t bias_size = sizeof(float) * C;

  assert(scratch);

  sycl_queue_.memcpy(scratch, pfilter, weight_size).wait();
  copyTypeConverted((DataType*)weights_, (float*)scratch, C * c_input_ * 1 * 1, sycl_queue_);

  if (pBias) {
    sycl_queue_.memcpy(scratch, pBias, bias_size).wait();
    copyTypeConverted((DataType*)biases_, (float*)scratch, C, sycl_queue_);
  }
}


template <>
  void Conv1Layer<sycl::half>::cublasSpecialMatrixMul(const sycl::half* A, const sycl::half* B,
                                               sycl::half* Out, int M, int N, int K,
                                               int batchSize, sycl::queue &sycl_queue) {

   // Need to initialize 1.0 and 0.0 as hexadecimal for fp16 because typecasting
  // float to sycl::half type doesn't work before CUDA 10.0

  #ifdef USE_CUBLAS
   cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  #endif

  #ifdef USE_HIPBLAS
   hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  #endif


   #ifdef USE_CUBLAS
    __half_raw one_h{0x3C00};
    __half_raw zero_h{0};
    half alpha = one_h;
    half beta = zero_h;

    #else
    sycl::half alpha = 1;
    sycl::half beta = 0;
    #endif

#ifdef USE_CUBLAS
    sycl_queue.submit([&](sycl::handler &cgh) {
         
         cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
  
          auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
          cublasSetStream(handle, cudaStreamHandle);


         ReportCUBLASErrors(cublasGemmStridedBatchedEx(
         handle, transpose_type_notranspose, transpose_type_notranspose, N, M, K, &one_h, B, CUDA_R_16F, N,
         N * K, A, CUDA_R_16F, K, 0, &zero_h, Out, CUDA_R_16F, N, N * M,
         batchSize, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));

         });   
   });
#elif defined(USE_HIPBLAS)
    sycl_queue.submit([&](sycl::handler &cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
         auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
         hipblasSetStream(handle, hipStreamHandle);
         hipblasGemmStridedBatchedEx(
              handle, transpose_type_notranspose, transpose_type_notranspose,
              N, M, K, &alpha, B, HIPBLAS_R_16F, N, N * K, A, HIPBLAS_R_16F, K,
              0, &beta, Out, HIPBLAS_R_16F, N, N * M, batchSize, HIPBLAS_COMPUTE_16F,
              HIPBLAS_GEMM_DEFAULT);
      });
    });
#else
    int64_t M_ = M;
    int64_t N_ = N;
    int64_t K_ = K;
    oneapi::mkl::blas::column_major::gemm_batch(
        sycl_queue, transpose_type_notranspose, transpose_type_notranspose, N_,
        M_, K_, alpha, B, N_, N_ * K_, A, K_, 0, beta, Out, N_, N_ * M_,
        batchSize);
#endif
}

template <>
void Conv1Layer<float>::cublasSpecialMatrixMul(const float* A, const float* B,
                                               float* Out, int M, int N, int K,
                                               int batchSize, sycl::queue &sycl_queue) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;


  int64_t M_ = M;
  int64_t N_ = N;
  int64_t K_ = K;

  #ifdef USE_CUBLAS
   cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  #endif

  #ifdef USE_HIPBLAS
   hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  #endif

  // NOTE strideB set to 0 below!
  {
    #ifdef USE_CUBLAS
    sycl_queue.submit([&](sycl::handler &cgh) {
        
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
  
         auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
         cublasSetStream(handle, cudaStreamHandle);


        ReportCUBLASErrors(cublasGemmStridedBatchedEx(
          handle, transpose_type_notranspose, transpose_type_notranspose, N, M, K, &floatOne, B, CUDA_R_32F, N,
          N * K, A, CUDA_R_32F, K, 0, &floatZero, Out, CUDA_R_32F, N, N * M,
          batchSize, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

        });   
    });
    #elif defined(USE_HIPBLAS)
    sycl_queue.submit([&](sycl::handler &cgh) {
        
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
         auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
  
         hipblasSetStream(handle, hipStreamHandle);


        hipblasGemmStridedBatchedEx(
          handle, transpose_type_notranspose, transpose_type_notranspose, N, M, K, &floatOne, B, HIPBLAS_R_32F, N,
          N * K, A, HIPBLAS_R_32F, K, 0, &floatZero, Out, HIPBLAS_R_32F, N, N * M,
          batchSize, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);

        });   
    });
    #else
      oneapi::mkl::blas::column_major::gemm_batch(
        sycl_queue, transpose_type_notranspose,
        transpose_type_notranspose, N_, M_, K_, floatOne, B, N_, N_ * K_, A, K_,
        0, floatZero, Out, N_, N_ * M_, batchSize); 
    #endif
  }
}

template <typename DataType>
void Conv1Layer<DataType>::Eval(int N, DataType* output, const DataType* input,
                                const DataType* /*input2*/, void* /*scratch*/,
                                size_t /*scratch_size*/,
                                sycl::queue &sycl_queue, DataType***) {

  //CERR << "Conv1Layer<DataType>::Eval. ";

  cublasSpecialMatrixMul(weights_, input, output, C, H * W, c_input_, N, sycl_queue);
 // CERR << "cublasSpecialMatrixMul. ";

  if (use_bias_){
  // CERR << "addBias. " << N << " " << C << " " << H << " " << W;
    addBias_NCHW(output, output, biases_, N, C, H, W, act_, sycl_queue);
  } else if (act_ != ACTIVATION_NONE) {
    addVectors(output, output, (DataType*)nullptr, N * C * H * W, N * C * H * W, 0, act_, sycl_queue);
  //  CERR << "addVectors. ";
  }
}

template <typename DataType>
Conv1Layer<DataType>::~Conv1Layer() {
 
  free(weights_, sycl_queue_);
  if (use_bias_) 
    free(biases_, sycl_queue_);
}

template <typename DataType>
ResidualBlock<DataType>::ResidualBlock(BaseLayer<DataType>* ip, int C, bool se,
                                       int se_k, bool first,
                                       bool last, ActivationFunction activation,
                                       int shared_mem_size, sycl::queue& sycl_queue)
    : BaseLayer<DataType>(C, 8, 8, ip, ip->isNHWC(), sycl_queue),
      has_se_(se),
      se_k_(se_k),
      c_input_(C),
      first_block_(first),
      last_block_(last),
      shared_mem_size_(shared_mem_size),
      act_(activation) {

  if (act_ != ACTIVATION_RELU && act_ != ACTIVATION_MISH) {
    throw Exception("Unsupported activation for residual block.");
  }

  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * C * C * 3 * 3;

  const size_t bias_size = sizeof(DataType) * C;
  biases0_ = (DataType *)sycl::malloc_device(bias_size, sycl_queue_);
  biases1_ = (DataType *)sycl::malloc_device(bias_size, sycl_queue_);

  // 6x6 transformed filter size, for 3x3 convolution
  transformed_weights0_ = (DataType *)sycl::malloc_device(weight_size * 4, sycl_queue_);
  transformed_weights1_ = (DataType *)sycl::malloc_device(weight_size * 4, sycl_queue_);  


  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = sizeof(DataType) * num_weights1;
    const size_t weight_size2 = sizeof(DataType) * num_weights2;
    const size_t biases_size1 = sizeof(DataType) * num_biases1;
    const size_t biases_size2 = sizeof(DataType) * num_biases2;


    w1_ = (DataType *)sycl::malloc_device(weight_size1, sycl_queue_);
    w2_ = (DataType *)sycl::malloc_device(weight_size2, sycl_queue_);
    b1_ = (DataType *)sycl::malloc_device(biases_size1, sycl_queue_);
    b2_ = (DataType *)sycl::malloc_device(biases_size2, sycl_queue_);

  }
}

template <typename DataType>
void ResidualBlock<DataType>::LoadWeights0(float* pfilter, float* pBias,
                                           void* scratch) {

  const size_t weight_size = sizeof(float) * c_input_ * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  sycl_queue_.memcpy(scratch, pfilter, weight_size).wait();

  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3, sycl_queue_);

  if (pBias) {
    sycl_queue_.memcpy(scratch, pBias, bias_size).wait();
    copyTypeConverted((DataType*)biases0_, (float*)scratch, C, sycl_queue_);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, c_input_, transformed_weights0_, weights, sycl_queue_);
}

template <typename DataType> void ResidualBlock<DataType>::LoadWeights1(float* pfilter, float* pBias, void* scratch) {
  
  const size_t weight_size = sizeof(float) * C * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  sycl_queue_.memcpy(scratch, pfilter, weight_size).wait();

  copyTypeConverted((DataType*)weights, (float*)scratch, C * C * 3 * 3, sycl_queue_);

  if (pBias) {
    sycl_queue_.memcpy(scratch, pBias, bias_size);
    copyTypeConverted((DataType*)biases1_, (float*)scratch, C, sycl_queue_);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, C, transformed_weights1_, weights, sycl_queue_);
}

template <typename DataType> void ResidualBlock<DataType>::LoadSEWeights(float* w1, float* b1, float* w2, float* b2, void* scratch) {

  const size_t num_weights1 = C * se_k_;
  const size_t num_weights2 = num_weights1 * 2;
  const size_t num_biases1 = se_k_;
  const size_t num_biases2 = 2 * C;

  // The shader uses transposed weight matrices.
  std::vector<float> temp_transposed(num_weights2);

  CpuTranspose(temp_transposed.data(), w1, se_k_, C);
  
  sycl_queue_.memcpy(scratch, temp_transposed.data(), num_weights1 * sizeof(float)).wait();
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1, sycl_queue_);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);
  
  sycl_queue_.memcpy(scratch, temp_transposed.data(), num_weights2 * sizeof(float)).wait(); 
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2, sycl_queue_);

  
  sycl_queue_.memcpy(scratch, b1, num_biases1 * sizeof(float)).wait();
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1, sycl_queue_);

  
  sycl_queue_.memcpy(scratch, b2, num_biases2 * sizeof(float)).wait();
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2, sycl_queue_);
}

template <typename DataType>
void ResidualBlock<DataType>::Eval(int N, DataType* output,
                                   const DataType* input,
                                   const DataType* /*input2*/, void* scratch,
                                   size_t scratch_size, sycl::queue &sycl_queue, DataType***) {

  //CERR << "ResidualBlock<DataType>::Eval. ";
  // normally:
  // - "output" initially contains the transformed input,
  //    and after this layer, it contains the transformed input for next layer
  // - "input" contains the original/untransformed input
  // special cases:
  //   - for first_block_, input is real input (untransformed)
  //   - for last_block_, output is the final output of this block
  //   (untransformed)

  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.
  DataType* transformed_input;
  DataType* transformed_output;
  if (!scratch) {
    // Caller wants us to sub-allocate all memory we need from "output" tensor.
    transformed_input = output;  // This is true in normal cases too!
    transformed_output = transformed_input + (N * C * 8 * 8 * 36 / 16);
  } else {
    transformed_input = (DataType*)scratch;
    transformed_output =
        transformed_input + scratch_size / (2 * sizeof(DataType));
  }

  if (first_block_) {
    InputTransform<DataType, true>(N, c_input_, transformed_input, input, sycl_queue_);
    BaseLayer<DataType>::cublasRowMajorMatrixMul(
        transformed_input, transformed_weights0_, transformed_output, N * 4, C,
        c_input_, 36, sycl_queue);
  } else {
    BaseLayer<DataType>::cublasRowMajorMatrixMul(output, transformed_weights0_,
                                                 transformed_output, N * 4, C,
                                                 c_input_, 36, sycl_queue);
  }

  if (act_ == ACTIVATION_RELU) {
    OutputInputTransform<DataType, false, ACTIVATION_RELU, true, false>(
        N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
        nullptr, nullptr, nullptr, nullptr, sycl_queue);
  } else if (act_ == ACTIVATION_MISH) {
    OutputInputTransform<DataType, false, ACTIVATION_MISH, true, false>(
        N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
        nullptr, nullptr, nullptr, nullptr, sycl_queue);
  }
  // "transformed_input" tensor now contains transformed input for the next
  // convolution

  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights1_, transformed_output, N * 4, C, C,
      36, sycl_queue);

  const bool fp16 = std::is_same<sycl::half, DataType>::value;
  bool allowFusing =
      (C <= kMaxResBlockFusingChannels) ||
      (fp16 && (shared_mem_size_ >= kMaxResBlockFusingSeFp16AmpereSmem) &&
       (C <= kMaxResBlockFusingSeKFp16Ampere));

  if (act_ == ACTIVATION_RELU) {
    if (last_block_) {
      if (has_se_)
        OutputTransform<DataType, true, ACTIVATION_RELU, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, sycl_queue);
      else
        OutputTransform<DataType, false, ACTIVATION_RELU, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, sycl_queue);
    } else {
      if (has_se_) {
        if (allowFusing) {
          OutputInputTransform<DataType, true, ACTIVATION_RELU, true, true>(
              N, C, se_k_, output, transformed_output, input, biases1_, w1_,
              b1_, w2_, b2_, sycl_queue);
        } else {
          OutputTransform<DataType, true, ACTIVATION_RELU, true, true, true,
                          true>(N, C, se_k_, (DataType*)input,
                                transformed_output, input, biases1_, w1_, b1_,
                                w2_, b2_, sycl_queue);
          InputTransform<DataType, true>(N, C, output, (DataType*)input,
                                         sycl_queue);
        }
      } else
        OutputInputTransform<DataType, false, ACTIVATION_RELU, true, true>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, sycl_queue);
    }
  } else if (act_ == ACTIVATION_MISH) {
    if (last_block_) {
      if (has_se_)
        OutputTransform<DataType, true, ACTIVATION_MISH, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, sycl_queue);
      else
        OutputTransform<DataType, false, ACTIVATION_MISH, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, sycl_queue);
    } else {
      if (has_se_) {
        if (allowFusing) {
          OutputInputTransform<DataType, true, ACTIVATION_MISH, true, true>(
              N, C, se_k_, output, transformed_output, input, biases1_, w1_,
              b1_, w2_, b2_, sycl_queue);
        } else {
          OutputTransform<DataType, true, ACTIVATION_MISH, true, true, true,
                          true>(N, C, se_k_, (DataType*)input,
                                transformed_output, input, biases1_, w1_, b1_,
                                w2_, b2_, sycl_queue);
          InputTransform<DataType, true>(N, C, output, (DataType*)input,
                                         sycl_queue);
        }
      } else
        OutputInputTransform<DataType, false, ACTIVATION_MISH, true, true>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, sycl_queue);
    }
  }
  // "output" tensor now contains transformed input for the next
  // convolution
}

template <typename DataType>
ResidualBlock<DataType>::~ResidualBlock() {

  free(transformed_weights0_, sycl_queue_);
  free(biases0_, sycl_queue_);
  free(transformed_weights1_, sycl_queue_);
  free(biases1_, sycl_queue_);
  if (has_se_) {
    free(w1_, sycl_queue_);
    free(w2_, sycl_queue_);
    free(b1_, sycl_queue_);
    free(b2_, sycl_queue_);
  }
}

template <typename DataType>
void allocAndUpload(DataType** gpu_dest, std::vector<float> cpu_src,
                    void* scratch, sycl::queue &sycl_queue) {
  size_t size = cpu_src.size() * sizeof(DataType);
  if (size == 0) {
    *gpu_dest = nullptr;
    return;
  }


  *gpu_dest = (DataType*)sycl::malloc_device(size, sycl_queue);

   sycl_queue.memcpy(scratch, &cpu_src[0], cpu_src.size() * sizeof(float)).wait();

   copyTypeConverted((DataType*)(*gpu_dest), (float*)scratch, (int)cpu_src.size(), sycl_queue);
}

template <typename DataType>
AttentionPolicyHead<DataType>::AttentionPolicyHead(
    BaseLayer<DataType>* ip, const MultiHeadWeights::PolicyHead& weights,
    void* scratch, bool attention_body, ActivationFunction act,
    int max_batch_size, sycl::queue &sycl_queue)
    : BaseLayer<DataType>(64 * 64 + 24 * 8, 1, 1, ip, sycl_queue),
      attention_body_(attention_body),
      // Old networks without attention body (e.g. T79) use hardcoded SELU
      // activations.
      act_(attention_body ? act : ACTIVATION_SELU) {
  embedding_op_size_ = weights.ip_pol_b.size();
  wq_op_size_ = weights.ip2_pol_b.size();
  wk_op_size_ = weights.ip3_pol_b.size();

  encoder_heads_ = weights.pol_encoder_head_count;
  policy_d_model_ = wq_op_size_;

  allocAndUpload<DataType>(&ip_pol_w_, weights.ip_pol_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ip_pol_b_, weights.ip_pol_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ip2_pol_w_, weights.ip2_pol_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ip2_pol_b_, weights.ip2_pol_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ip3_pol_w_, weights.ip3_pol_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ip3_pol_b_, weights.ip3_pol_b, scratch, sycl_queue_);

  // big allocation to hold wq and wk weights one after the other
  {
    size_t elements = weights.ip2_pol_w.size();
    assert(elements == weights.ip3_pol_w.size());

    size_t size = elements * sizeof(DataType) * 2;
    wqk_w_ = (DataType*)sycl::malloc_device(size, sycl_queue_);
    sycl_queue_.memcpy(wqk_w_, ip2_pol_w_, size / 2);
    
    sycl_queue_.memcpy(wqk_w_ + elements, ip3_pol_w_, size / 2);

    elements = weights.ip2_pol_b.size();
    size = elements * sizeof(DataType) * 2;
    wqk_b_ = (DataType*)sycl::malloc_device(size, sycl_queue_);
    sycl_queue_.memcpy(wqk_b_, ip2_pol_b_, size / 2);
    sycl_queue_.memcpy(wqk_b_ + elements, ip3_pol_b_, size / 2);
  }

  allocAndUpload<DataType>(&ip4_pol_w_, weights.ip4_pol_w, scratch, sycl_queue_);

  for (const auto& enc : weights.pol_encoder) {
    EncoderBlock<DataType>* pW = new EncoderBlock<DataType>(
        enc, scratch, encoder_heads_, embedding_op_size_,
        1.0f,  // using alpha = 1 for now (TODO: may change?)
        nullptr, 0,  // smolgen weights not implemented in
                     // policy encoder heads yet.
        max_batch_size, ACTIVATION_SWISH, act_,
        1e-6, sycl_queue_);  // attentionbody nets don't have policy encoders, so using old
                // epsilon for backward compatibility with T78.
    encoder_weights_.emplace_back(pW);
  }
}

template <typename DataType>
EncoderBlock<DataType>::EncoderBlock(
    const MultiHeadWeights::EncoderLayer& cpu_weights, void* scratch, int heads,
    int size, float alpha, DataType* smolgen_global_scratch,
    int smolgen_global_size, int max_batch_size, ActivationFunction smolgen_act,
    ActivationFunction ffn_act, float default_eps, sycl::queue &sycl_queue,
    bool use_int8)
    : embedding_op_size_(size),
      encoder_heads_(heads),
      alpha_(alpha),
      has_smolgen_(cpu_weights.mha.has_smolgen),
      smolgen_activation_(smolgen_act),
      ffn_activation_(ffn_act),
      max_batch_size_(max_batch_size),
      default_eps_(default_eps),
      sycl_queue_(sycl_queue) {
  mha_q_size_ = cpu_weights.mha.q_b.size();
  mha_k_size_ = cpu_weights.mha.k_b.size();
  mha_v_size_ = cpu_weights.mha.v_b.size();
  mha_dense_size_ = cpu_weights.mha.dense_b.size();
  ffn_dense1_size_ = cpu_weights.ffn.dense1_b.size();
  ffn_dense2_size_ = cpu_weights.ffn.dense2_b.size();

  allocAndUpload<DataType>(&mha_q_w, cpu_weights.mha.q_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&mha_q_b, cpu_weights.mha.q_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&mha_k_w, cpu_weights.mha.k_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&mha_k_b, cpu_weights.mha.k_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&mha_v_w, cpu_weights.mha.v_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&mha_v_b, cpu_weights.mha.v_b, scratch, sycl_queue_);

  // big allocation to hold qkv weights one after the other
  {
    size_t elements = cpu_weights.mha.q_w.size();
    size_t size = elements * sizeof(DataType) * 3;
    
    mha_qkv_w = (DataType*)sycl::malloc_device(size, sycl_queue_);
    sycl_queue_.memcpy(mha_qkv_w, mha_q_w, size / 3);
    sycl_queue_.memcpy(mha_qkv_w + elements, mha_k_w, size / 3);
    sycl_queue_.memcpy(mha_qkv_w + elements * 2, mha_v_w, size / 3);

    elements = cpu_weights.mha.q_b.size();
    size = elements * sizeof(DataType) * 3;
    
    mha_qkv_b = (DataType*)sycl::malloc_device(size, sycl_queue_);
    sycl_queue_.memcpy(mha_qkv_b, mha_q_b, size / 3);
    sycl_queue_.memcpy(mha_qkv_b + elements, mha_k_b, size / 3);
    sycl_queue_.memcpy(mha_qkv_b + elements * 2, mha_v_b, size / 3);
  }

  allocAndUpload<DataType>(&mha_dense_w, cpu_weights.mha.dense_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&mha_dense_b, cpu_weights.mha.dense_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ln1_gammas, cpu_weights.ln1_gammas, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ln1_betas, cpu_weights.ln1_betas, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ffn_dense1_w, cpu_weights.ffn.dense1_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ffn_dense1_b, cpu_weights.ffn.dense1_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ffn_dense2_w, cpu_weights.ffn.dense2_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ffn_dense2_b, cpu_weights.ffn.dense2_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ln2_gammas, cpu_weights.ln2_gammas, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ln2_betas, cpu_weights.ln2_betas, scratch, sycl_queue_);

  // Smolgen weights.
  if (has_smolgen_) {
    smol_compress_size_ = cpu_weights.mha.smolgen.compress.size() / mha_q_size_;
    smol_dense_1_size_ = cpu_weights.mha.smolgen.dense1_b.size();
    smol_dense_2_size_ = cpu_weights.mha.smolgen.dense2_b.size();
    smol_global_size_ = smolgen_global_size;

    allocAndUpload<DataType>(&smol_compress, cpu_weights.mha.smolgen.compress,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&smol_dense1_w, cpu_weights.mha.smolgen.dense1_w,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&smol_dense1_b, cpu_weights.mha.smolgen.dense1_b,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&smol_dense2_w, cpu_weights.mha.smolgen.dense2_w,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&smol_dense2_b, cpu_weights.mha.smolgen.dense2_b,
                             scratch, sycl_queue_);

    allocAndUpload<DataType>(&smol_ln1_gammas,
                             cpu_weights.mha.smolgen.ln1_gammas, scratch, sycl_queue_);
    allocAndUpload<DataType>(&smol_ln1_betas, cpu_weights.mha.smolgen.ln1_betas,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&smol_ln2_gammas,
                             cpu_weights.mha.smolgen.ln2_gammas, scratch, sycl_queue_);
    allocAndUpload<DataType>(&smol_ln2_betas, cpu_weights.mha.smolgen.ln2_betas,
                             scratch, sycl_queue_);

    // GPU memory already allocated in AttentionBody.
    smol_global = smolgen_global_scratch;
  }

#if !defined(USE_CUBLAS) && !defined(USE_HIPBLAS)
  if (use_int8 && cpu_weights.mha.has_int8 && cpu_weights.ffn.has_int8) {
    use_int8_ = true;
    auto upload_s8 = [&](const std::vector<int8_t>& src) -> int8_t* {
      int8_t* ptr = sycl::malloc_device<int8_t>(src.size(), sycl_queue_);
      sycl_queue_.memcpy(ptr, src.data(), src.size()).wait();
      return ptr;
    };
    auto upload_comb_scales = [&](const std::vector<float>& w_sc,
                                  float act_sc) -> float* {
      std::vector<float> comb(w_sc.size());
      for (size_t i = 0; i < w_sc.size(); ++i) comb[i] = w_sc[i] * act_sc;
      float* ptr = sycl::malloc_device<float>(comb.size(), sycl_queue_);
      sycl_queue_.memcpy(ptr, comb.data(), comb.size() * sizeof(float)).wait();
      return ptr;
    };

    // Fused QKV INT8 weights [3 * d_model, embedding_op_size]
    const size_t q_elems = cpu_weights.mha.q_w_i8.size();
    std::vector<int8_t> qkv_i8(q_elems * 3);
    std::memcpy(qkv_i8.data(), cpu_weights.mha.q_w_i8.data(), q_elems);
    std::memcpy(qkv_i8.data() + q_elems, cpu_weights.mha.k_w_i8.data(), q_elems);
    std::memcpy(qkv_i8.data() + q_elems * 2, cpu_weights.mha.v_w_i8.data(), q_elems);
    mha_qkv_w_s8_ = upload_s8(qkv_i8);

    std::vector<float> qkv_sc;
    qkv_sc.reserve(cpu_weights.mha.q_w_scales.size() * 3);
    qkv_sc.insert(qkv_sc.end(), cpu_weights.mha.q_w_scales.begin(),
                  cpu_weights.mha.q_w_scales.end());
    qkv_sc.insert(qkv_sc.end(), cpu_weights.mha.k_w_scales.begin(),
                  cpu_weights.mha.k_w_scales.end());
    qkv_sc.insert(qkv_sc.end(), cpu_weights.mha.v_w_scales.begin(),
                  cpu_weights.mha.v_w_scales.end());
    mha_qkv_comb_scales_ =
        upload_comb_scales(qkv_sc, cpu_weights.mha.qkv_act_scale);
    qkv_inv_act_scale_ = 1.0f / cpu_weights.mha.qkv_act_scale;
    qkv_act_zp_ = cpu_weights.mha.qkv_act_zp;

    mha_dense_w_s8_ = upload_s8(cpu_weights.mha.dense_w_i8);
    mha_dense_comb_scales_ = upload_comb_scales(cpu_weights.mha.dense_w_scales,
                                                cpu_weights.mha.dense_act_scale);
    dense_inv_act_scale_ = 1.0f / cpu_weights.mha.dense_act_scale;
    dense_act_zp_ = cpu_weights.mha.dense_act_zp;

    ffn_dense1_w_s8_ = upload_s8(cpu_weights.ffn.dense1_w_i8);
    ffn_dense1_comb_scales_ = upload_comb_scales(
        cpu_weights.ffn.dense1_w_scales, cpu_weights.ffn.dense1_act_scale);
    ffn_dense1_inv_act_scale_ = 1.0f / cpu_weights.ffn.dense1_act_scale;
    ffn_dense1_act_zp_ = cpu_weights.ffn.dense1_act_zp;

    ffn_dense2_w_s8_ = upload_s8(cpu_weights.ffn.dense2_w_i8);
    ffn_dense2_comb_scales_ = upload_comb_scales(
        cpu_weights.ffn.dense2_w_scales, cpu_weights.ffn.dense2_act_scale);
    ffn_dense2_inv_act_scale_ = 1.0f / cpu_weights.ffn.dense2_act_scale;
    ffn_dense2_act_zp_ = cpu_weights.ffn.dense2_act_zp;
  }
#endif
}

template <typename DataType>
static void cublasXgemm(transpose_type transa,
                        transpose_type transb, int m, int n, int k,
                        float alpha, const DataType* A, int lda,
                        const DataType* B, int ldb, float beta, DataType* C,
                        int ldc, sycl::queue &sycl_queue) {



  const bool fp16 = std::is_same<sycl::half, DataType>::value;
  
  #ifdef USE_CUBLAS
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    sycl_queue.submit([&](sycl::handler &cgh) {
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);    
        ReportCUBLASErrors(cublasHgemm(
          handle, transa, transb, m, n, k, (const half*)&alpha_h, ((const half *)A),
          lda, ((const half *)B), ldb, (const half*)&beta_h, ((half *)C), ldc));
      });
    });
  } else { 
    sycl_queue.submit([&](sycl::handler &cgh) {
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {  
        auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);  
        ReportCUBLASErrors(cublasSgemm(handle, transa, transb, m, n, k, &alpha,
                                   (const float*)A, lda, (const float*)B, ldb,
                                   &beta, (float*)C, ldc));

        });
      });
  }
  #elif defined(USE_HIPBLAS)
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    sycl_queue.submit([&](sycl::handler &cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
        hipblasSetStream(handle, hipStreamHandle);
        hipblasHgemm(handle, transa, transb, m, n, k, &alpha_h, (const hipblasHalf*)A,
          lda, (const hipblasHalf*)B, ldb, &beta_h, (hipblasHalf*)C, ldc);
        });
      });
  } else {
    sycl_queue.submit([&](sycl::handler &cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {  
        auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
        hipblasSetStream(handle, hipStreamHandle);  
        hipblasSgemm(handle, transa, transb, m, n, k, &alpha, (const float*)A, lda, (const float*)B, ldb, &beta, (float*)C, ldc);
        });
      });
  }
  #else
    oneapi::mkl::blas::column_major::gemm(sycl_queue, transa, transb, m, n, k, alpha, (const DataType *)A, lda,
        (const DataType *)B, ldb, beta, (DataType *)C, ldc);
  #endif

}

template <typename DataType>
static void cublasXGemmStridedBatched(transpose_type transa, transpose_type transb,
    int m, int n, int k, float alpha, const void* A, int lda,
    long long int strideA, const void* B, int ldb, long long int strideB,
    float beta, void* C, int ldc, long long int strideC, int batchCount, sycl::queue &sycl_queue) {

  const bool fp16 = std::is_same<sycl::half, DataType>::value;
  
  #ifdef USE_CUBLAS
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    
    sycl_queue.submit([&](sycl::handler &cgh) {
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);    

        ReportCUBLASErrors(cublasGemmStridedBatchedEx(
          handle, transa, transb, m, n, k, &alpha_h, A, CUDA_R_16F, lda, strideA,
          B, CUDA_R_16F, ldb, strideB, &beta_h, C, CUDA_R_16F, ldc, strideC,
          batchCount, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
        

      });

    });
  
  } else { 
    
    sycl_queue.submit([&](sycl::handler &cgh) {
        
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
    
        auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);    
    
        ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda, strideA, B,
        CUDA_R_32F, ldb, strideB, &beta, C, CUDA_R_32F, ldc, strideC,
        batchCount, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  
  
      });
    });
  }
  #elif defined(USE_HIPBLAS)
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);

    sycl_queue.submit([&](sycl::handler &cgh) {

      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);

        hipblasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, &alpha_h, A, HIPBLAS_R_16F, lda, strideA, B,
        HIPBLAS_R_16F, ldb, strideB, &beta_h, C, HIPBLAS_R_16F, ldc, strideC,
        batchCount, HIPBLAS_COMPUTE_16F, HIPBLAS_GEMM_DEFAULT);


      });
    });
  } else {
    sycl_queue.submit([&](sycl::handler &cgh) {

      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
    
        hipblasSetStream(handle, hipStreamHandle);    
    
        hipblasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, &alpha, A, HIPBLAS_R_32F, lda, strideA, B,
        HIPBLAS_R_32F, ldb, strideB, &beta, C, HIPBLAS_R_32F, ldc, strideC,
        batchCount, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  
  
      });
    });
  }
  #else
  oneapi::mkl::blas::column_major::gemm_batch(sycl_queue, transa, transb, m, n, k,  alpha, (const DataType *)A, lda, strideA, (const DataType *)B, ldb, strideB, beta, (DataType *)C, ldc, strideC, batchCount);
  #endif
}

template <typename DataType>
static void cublasXGemmBatched(transpose_type transa,
                               transpose_type transb, int m, int n,
                               int k, float alpha, DataType** A, int lda,
                               DataType** B, int ldb, float beta, DataType** C,
                               int ldc, int batchCount, sycl::queue &sycl_queue) {

  const bool fp16 = std::is_same<sycl::half, DataType>::value;

  #ifdef USE_CUBLAS
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);


    sycl_queue.submit([&](sycl::handler &cgh) {
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);    

        ReportCUBLASErrors(cublasHgemmBatched(
        handle, transa, transb, m, n, k, (const half*)&alpha_h, (half**)A, lda,
        (half**)B, ldb, (const half*)&beta_h, (half**)C, ldc, batchCount));
        
      });

    });

  } else {
    
    sycl_queue.submit([&](sycl::handler &cgh) {
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);    

        ReportCUBLASErrors(cublasSgemmBatched(
        handle, transa, transb, m, n, k, &alpha, (float**)A, lda, (float**)B,
        ldb, &beta, (float**)C, ldc, batchCount));
        
      });

    });
  }

  #elif defined(USE_HIPBLAS)

   hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);


    sycl_queue.submit([&](sycl::handler &cgh) {
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);       

        hipblasHgemmBatched(
        handle, transa, transb, m, n, k, (const hipblasHalf*)&alpha_h, (hipblasHalf**)A, lda,
        (hipblasHalf**)B, ldb, (const hipblasHalf*)&beta_h, (hipblasHalf**)C, ldc, batchCount);
        
      });

    });

  } else {
    
    sycl_queue.submit([&](sycl::handler &cgh) {
        cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);        

        hipblasSgemmBatched(
        handle, transa, transb, m, n, k, &alpha, (float**)A, lda, (float**)B,
        ldb, &beta, (float**)C, ldc, batchCount);
        

      });

    });
  }

  #else
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);

    oneapi::mkl::blas::column_major::gemm_batch(
        sycl_queue, &transa, &transb, &m, &n, &k,  (const sycl::half*)&alpha_h,
        (const sycl::half **)A, &lda, (const sycl::half **)B, &ldb,
        (const sycl::half*)&beta_h, (sycl::half **)C, &ldc, 1, &batchCount);
  } else {
    oneapi::mkl::blas::column_major::gemm_batch(
        sycl_queue, &transa, &transb, &m, &n, &k,  &alpha, (const float **)A,
        &lda, (const float **)B, &ldb, &beta, (float **)C, &ldc, 1,
        &batchCount);
  }

  #endif
}

// input/output tensor is in_out_tensor, others are used as scratch.
template <typename DataType>
void EncoderBlock<DataType>::Eval(int N, DataType* in_out_tensor,
                                  DataType* scratch, DataType* buffer1,
                                  DataType* buffer2, sycl::queue &sycl_queue,
                                  DataType*** offset_pointers,
                                  uint8_t* int8_act_buf,
                                  int32_t* int32_gemm_buf,
                                  const int32_t* int32_zero_co) {

  //CERR << "EncoderBlock<DataType>::Eval. ";

  const int d_model = mha_q_size_;
  const int depth = d_model / encoder_heads_;
#if !defined(USE_CUBLAS) && !defined(USE_HIPBLAS)
  const bool run_int8 = use_int8_ && int8_act_buf != nullptr &&
                        int32_gemm_buf != nullptr && int32_zero_co != nullptr;
#endif

  // Calculate smolgen weights. Do this first so we can make use of
  // scratch, buffer1 and buffer2.
  if (has_smolgen_) {
    {
      // Compress.
      // input shape: N, 64, d_model
      // output shape: N, 64, hidden_channels
      const int num_inputs = d_model;
      const int num_outputs = smol_compress_size_;
      const int batch = N * 64;
      cublasXgemm<DataType>(transpose_type_transpose,
          transpose_type_notranspose, num_outputs, batch, num_inputs,
          1.0f, (const DataType*)smol_compress, num_inputs, in_out_tensor,
          num_inputs, 0.0f, scratch, num_outputs, sycl_queue);
    }

    {
      // Hidden 1 dense.
      // input shape: N, 64 * hidden_channels
      // output shape: N, hidden_sz
      const int num_inputs = 64 * smol_compress_size_;
      const int num_outputs = smol_dense_1_size_;
      const int batch = N;
      cublasXgemm<DataType>(transpose_type_transpose,
                            transpose_type_notranspose, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_dense1_w, num_inputs, scratch,
                            num_inputs, 0.0f, buffer1, num_outputs, sycl_queue);

      LayerNorm<DataType>(batch, num_outputs, scratch, buffer1, smol_dense1_b,
                          (DataType*)nullptr, smol_ln1_gammas, smol_ln1_betas,
                          1e-3, 1.0, smolgen_activation_, sycl_queue);
    }

    {
      // Hidden 2 dense (gen_from)
      // input shape: N, hidden_sz
      // output shape: N, heads * gen_sz
      const int num_inputs = smol_dense_1_size_;
      const int num_outputs = smol_dense_2_size_;
      const int batch = N;
      cublasXgemm<DataType>(transpose_type_transpose,
                            transpose_type_notranspose, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_dense2_w, num_inputs, scratch,
                            num_inputs, 0.0f, buffer1, num_outputs, sycl_queue);

      LayerNorm<DataType>(batch, num_outputs, scratch, buffer1, smol_dense2_b,
                          (DataType*)nullptr, smol_ln2_gammas, smol_ln2_betas,
                          1e-3, 1.0, smolgen_activation_, sycl_queue);
    }

    {
      // Final smolgen weights generation.
      /*
        gen_from = tf.reshape(gen_from, [-1, heads, gen_sz])
        out = self.smol_weight_gen_dense(gen_from)
      */
      const int num_inputs =
          smol_dense_2_size_ / encoder_heads_; /* num_inputs == gen_sz == 256 */
      const int num_outputs = smol_global_size_; /* hwhw: 64 * 64 */
      const int batch = N * encoder_heads_;

      cublasXgemm<DataType>(transpose_type_transpose,
                            transpose_type_notranspose, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_global, num_inputs, scratch,
                            num_inputs, 0.0f, buffer2, num_outputs, sycl_queue);
    }
  }

  DataType* mha_q;
  DataType* mha_k;
  DataType* mha_v;

  {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = d_model;
    const int batch = N * 64;
    const int max_batch = max_batch_size_ * 64;

    mha_q = scratch;
    mha_k = mha_q + num_outputs * max_batch;
    mha_v = mha_k + num_outputs * max_batch;

#if !defined(USE_CUBLAS) && !defined(USE_HIPBLAS)
    if (run_int8) {
      quantizeToU8<DataType>(int8_act_buf, in_out_tensor, batch * num_inputs,
                             qkv_inv_act_scale_, qkv_act_zp_, sycl_queue);
      oneapi::mkl::blas::column_major::gemm_bias(
          sycl_queue, transpose_type_transpose, transpose_type_notranspose,
          oneapi::mkl::offset::fix, 3 * num_outputs, batch, num_inputs, 1.0f,
          mha_qkv_w_s8_, num_inputs, int8_t(0),
          int8_act_buf, num_inputs, qkv_act_zp_,
          0.0f, int32_gemm_buf, 3 * num_outputs, int32_zero_co);
      dequantBiasUnpackQKV<DataType>(mha_q, mha_k, mha_v, int32_gemm_buf,
                                     mha_qkv_comb_scales_, mha_qkv_b,
                                     batch, num_outputs, sycl_queue);
    } else
#endif
    {
      cublasXGemmStridedBatched<DataType>(transpose_type_transpose, transpose_type_notranspose,
          num_outputs, batch, num_inputs, 1.0f, mha_qkv_w, num_inputs,
          num_inputs * num_outputs, in_out_tensor, num_inputs, 0, 0.0f, mha_q,
          num_outputs, num_outputs * max_batch, 3, sycl_queue);
      addBiasBatched<DataType>(mha_q, mha_q, mha_qkv_b, 3, batch, num_outputs,
                               max_batch, ACTIVATION_NONE, sycl_queue);
    }
  }

  // Apply split_heads() to q, k and v
  // which basically transposes (batch_size, 64, num_heads, depth)
  // to (batch_size, num_heads, 64, depth)
  // Do we really need to transpose here?
  // (Maybe not, we can play with strides of the gemm and do independent gemms
  // for each encoder head)

  // Apply scaled dot product attention:
  /*
      matmul_qk = tf.matmul(q, k, transpose_b=True)
      dk = tf.cast(tf.shape(k)[-1], self.model_dtype)
      scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
      attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
      output = tf.matmul(attention_weights, v)
  */

  // shape(k)[-1] = depth
  float factor = 1.0f / sqrt((float)depth);

  // matmul_qk = tf.matmul(q, k, transpose_b=True)
  {
    if (*offset_pointers == nullptr) {
      
      *offset_pointers = sycl::malloc_device<DataType*>(
                               encoder_heads_ * max_batch_size_ * 5,
                               sycl_queue_);
      genOffsetPointers(*offset_pointers, encoder_heads_, max_batch_size_,
                        depth, d_model, mha_k, mha_q, buffer1,
                        mha_v, buffer2, sycl_queue_);
    }

    cublasXGemmBatched<DataType>(transpose_type_transpose, transpose_type_notranspose,
        64 /*M*/, 64 /*N*/, depth /*K*/,  // A/B, and M/N are swapped for
                                          // row-major to col-major transform
        factor,            // to handle "/ tf.math.sqrt(dk)"
        *offset_pointers,  // mha_k + offset /*A*/,
        d_model /*LDA*/,   // (d_model = depth * encoder_heads_) to skip over
                           // other "depth" slices / heads
        // 64 * d_model,     /*strideA*/
        *offset_pointers +
            encoder_heads_ * max_batch_size_,  // mha_q + offset /*B*/,
        d_model /*LDB*/,  // to skip over other other "depth" slices / heads
        // 64 * d_model,     /*strideB*/
        0.0f,
        *offset_pointers + encoder_heads_ * max_batch_size_ *
                               2,  // buffer1 + outOffset /*C*/,  // output
                                   // (matmul_qk) goes to buffer1
        64 /*LDC*/,
        // 64 * 64 /*strideC*/,
        N * encoder_heads_, sycl_queue_);
  }

  // attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)
  // attention_weights -> buffer1
  if (has_smolgen_) {
    // Add smolgen weights to the scaled matmul_qk attention logits before
    // softmax.
    Softmax(encoder_heads_ * N * 64, 64, buffer1, buffer1, buffer2, sycl_queue_);
  } else {
    Softmax(encoder_heads_ * N * 64, 64, buffer1, buffer1,
            (const DataType*)nullptr, sycl_queue_);
  }

  {
    cublasXGemmBatched<DataType>(transpose_type_notranspose,
        transpose_type_notranspose, depth /*M*/, 64 /*N*/, 64 /*K*/, 1.0f,
        *offset_pointers + encoder_heads_ * max_batch_size_ *
                               3,  // mha_v + offset /*A*/,  // "v" matrix
        d_model /*LDA*/,           // to skip over other "depth" slices / heads
        // 64 * d_model,          /*strideA*/
        *offset_pointers + encoder_heads_ * max_batch_size_ *
                               2,  // buffer1 + weightsOffset /*B*/,
        64 /*LDB*/,                // 64 * 64, /*strideB*/
        0.0f,
        *offset_pointers +
            encoder_heads_ * max_batch_size_ *
                4,  // buffer2 + offset /*C*/,  // output goes to buffer2
        d_model /*LDC*/,
        // 64 * d_model /*strideC*/,
        N * encoder_heads_, sycl_queue_);
  }

  // #final dense layer (mha_dense), buffer2 -> buffer1
  {
    const int num_inputs = d_model;
    const int num_outputs = embedding_op_size_;
    const int batch = N * 64;
#if !defined(USE_CUBLAS) && !defined(USE_HIPBLAS)
    if (run_int8) {
      quantizeToU8<DataType>(int8_act_buf, buffer2, batch * num_inputs,
                             dense_inv_act_scale_, dense_act_zp_, sycl_queue);
      oneapi::mkl::blas::column_major::gemm_bias(
          sycl_queue, transpose_type_transpose, transpose_type_notranspose,
          oneapi::mkl::offset::fix, num_outputs, batch, num_inputs, 1.0f,
          mha_dense_w_s8_, num_inputs, int8_t(0),
          int8_act_buf, num_inputs, dense_act_zp_,
          0.0f, int32_gemm_buf, num_outputs, int32_zero_co);
      dequantS32ToDataType<DataType>(buffer1, int32_gemm_buf,
                                     mha_dense_comb_scales_, batch,
                                     num_outputs, sycl_queue);
    } else
#endif
    {
      cublasXgemm(transpose_type_transpose,
                  transpose_type_notranspose, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)mha_dense_w, num_inputs,
                  buffer2, num_inputs, 0.0f, buffer1, num_outputs, sycl_queue_);
    }
  }

  // LN1: skip connection and layer normalization (also bias add of prev gemm)
  // buffer1/in_out_tensor -> scratch
  LayerNorm<DataType>(N * 64, embedding_op_size_, scratch, buffer1, mha_dense_b,
                      in_out_tensor, ln1_gammas, ln1_betas, default_eps_,
                      alpha_, ACTIVATION_NONE, sycl_queue_);

#if !defined(USE_CUBLAS) && !defined(USE_HIPBLAS)
  if (run_int8) {
    // Super-fused W8A8 FFN: scratch -> u8 -> FFN1 (s32) -> [dequant+bias1+act+quant u8] -> FFN2 (s32) -> buffer1
    const int batch = N * 64;
    quantizeToU8<DataType>(int8_act_buf, scratch, batch * embedding_op_size_,
                           ffn_dense1_inv_act_scale_, ffn_dense1_act_zp_,
                           sycl_queue);
    oneapi::mkl::blas::column_major::gemm_bias(
        sycl_queue, transpose_type_transpose, transpose_type_notranspose,
        oneapi::mkl::offset::fix, ffn_dense1_size_, batch, embedding_op_size_,
        1.0f, ffn_dense1_w_s8_, embedding_op_size_, int8_t(0),
        int8_act_buf, embedding_op_size_, ffn_dense1_act_zp_,
        0.0f, int32_gemm_buf, ffn_dense1_size_, int32_zero_co);
    dequantBiasActQuantizeU8<DataType>(
        int8_act_buf, int32_gemm_buf, ffn_dense1_comb_scales_, ffn_dense1_b,
        batch, ffn_dense1_size_, ffn_activation_,
        ffn_dense2_inv_act_scale_, ffn_dense2_act_zp_, sycl_queue);
    oneapi::mkl::blas::column_major::gemm_bias(
        sycl_queue, transpose_type_transpose, transpose_type_notranspose,
        oneapi::mkl::offset::fix, embedding_op_size_, batch, ffn_dense1_size_,
        1.0f, ffn_dense2_w_s8_, ffn_dense1_size_, int8_t(0),
        int8_act_buf, ffn_dense1_size_, ffn_dense2_act_zp_,
        0.0f, int32_gemm_buf, embedding_op_size_, int32_zero_co);
    dequantS32ToDataType<DataType>(buffer1, int32_gemm_buf,
                                   ffn_dense2_comb_scales_, batch,
                                   embedding_op_size_, sycl_queue);
  } else
#endif
  {
    // #FFN dense 1, scratch -> in_out_tensor
    {
      const int num_inputs = embedding_op_size_;
      const int num_outputs = ffn_dense1_size_;  // encoder_dff
      const int batch = N * 64;
      cublasXgemm(transpose_type_transpose,
                  transpose_type_notranspose, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)ffn_dense1_w, num_inputs,
                  scratch, num_inputs, 0.0f, in_out_tensor, num_outputs, sycl_queue_);
      addBiasBatched(in_out_tensor, in_out_tensor, ffn_dense1_b, 1, batch,
                     num_outputs, ffn_activation_, sycl_queue_);
    }

    // #FFN dense 2, in_out_tensor -> buffer1
    {
      const int num_inputs = ffn_dense1_size_;  // encoder_dff
      const int num_outputs = embedding_op_size_;
      const int batch = N * 64;
      cublasXgemm(transpose_type_transpose,
                  transpose_type_notranspose, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)ffn_dense2_w, num_inputs,
                  in_out_tensor, num_inputs, 0.0f, buffer1, num_outputs, sycl_queue_);
    }
  }

  // LN2: skip connection and layer normilization (also bias add of prev gemm)
  // buffer1/scratch -> in_out_tensor
  LayerNorm<DataType>(N * 64, embedding_op_size_, in_out_tensor, buffer1,
                      ffn_dense2_b, scratch, ln2_gammas, ln2_betas,
                      default_eps_, alpha_, ACTIVATION_NONE, sycl_queue_);
}

template <typename DataType>
void AttentionPolicyHead<DataType>::Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            sycl::queue &sycl_queue, DataType*** offset_pointers) {

  //CERR << "AttentionPolicyHead<DataType>::Eval. ";

  DataType* input2_tensor = (DataType*)input2;
  DataType* buffer1 = output + scratch_size / (2 * sizeof(DataType));
  DataType* buffer2 = input2_tensor + scratch_size / (2 * sizeof(DataType));

  int inputC = this->input_->GetC();
  if (!attention_body_)
    convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, inputC, 8, 8, sycl_queue);

  // 1. Policy embedding (fully connected layer)
  // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
  DataType* pol_embedding = input2_tensor;
  {
    const int num_outputs = embedding_op_size_;
    const int num_inputs = inputC;
    const int batch = N * 64;
    cublasXgemm<DataType>(
        transpose_type_transpose, transpose_type_notranspose,
        num_outputs, batch, num_inputs, 1.0f, (const DataType*)ip_pol_w_,
        num_inputs, attention_body_ ? input : (DataType*)scratch, num_inputs,
        0.0f, pol_embedding, num_outputs, sycl_queue);

    addBiasBatched(pol_embedding, pol_embedding, ip_pol_b_, 1, batch,
                   num_outputs, act_, sycl_queue);
  }

  // 2. Encoder layers
  for (const auto pEnc : encoder_weights_) {
    pEnc->Eval(N, input2_tensor, (DataType*)scratch, buffer1, buffer2, sycl_queue, offset_pointers);
  }  // End of encoder blocks

  DataType* wq;
  DataType* wk;
  {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = policy_d_model_;
    const int batch = N * 64;
    wq = (DataType*)scratch;
    wk = wq + num_outputs * batch;

    cublasXGemmStridedBatched<DataType>(
        transpose_type_transpose, transpose_type_notranspose,
        num_outputs, batch, num_inputs, 1.0f, wqk_w_, num_inputs,
        num_inputs * num_outputs, input2_tensor, num_inputs, 0, 0.0f, wq,
        num_outputs, num_outputs * batch, 2, sycl_queue);

    addBiasBatched<DataType>(wq, wq, wqk_b_, 2, batch, num_outputs,
                             ACTIVATION_NONE, sycl_queue);
  }

  // dk = tf.math.sqrt(tf.cast(tf.shape(keys)[-1], self.model_dtype))
  // policy matmul_qk = tf.matmul(queries, keys, transpose_b=True)
  // policy_attn_logits = matmul_qk / dk
  {
    // shape(keys)[-1] = policy_d_model_
    float factor = 1.0f / sqrt((float)policy_d_model_);

    // A/B, and M/N are swapped for row-major to col-major transform
    // leave 8*24 after each batch to interleave promotion_logits (computed
    // later below)
    cublasXGemmStridedBatched<DataType>(
        transpose_type_transpose, transpose_type_notranspose,
        64 /*M*/, 64 /*N*/, policy_d_model_ /*K*/,
        factor,  // to handle "/ tf.math.sqrt(dk)"
        wk /*A*/, policy_d_model_ /*LDA*/, 64 * policy_d_model_, /*strideA*/
        wq /*B*/, policy_d_model_ /*LDB*/, 64 * policy_d_model_, /*strideB*/
        0.0f, output /*C*/,  // output (policy_attn_logits)
        64 /*LDC*/, 64 * 64 + 8 * 24 /*strideC*/, N, sycl_queue);
  }

  // Compute promotion_logits in a single kernel (and put the result just after
  // policy_attn_logits interleaved to get concat for free)
  DataType* promotion_logits = output + 64 * 64;

  ComputePromotionLogits<DataType>(N, policy_d_model_, promotion_logits, wk,
                                   ip4_pol_w_, output, sycl_queue);
}

template <typename DataType>
AttentionPolicyHead<DataType>::~AttentionPolicyHead() {
      sycl::free(ip_pol_w_, sycl_queue_);
      sycl::free(ip_pol_b_, sycl_queue_);
      sycl::free(ip2_pol_w_, sycl_queue_);
      sycl::free(ip2_pol_b_, sycl_queue_);
      sycl::free(ip3_pol_w_, sycl_queue_);
      sycl::free(ip3_pol_b_, sycl_queue_);
      sycl::free(ip4_pol_w_, sycl_queue_);
      sycl::free(wqk_w_, sycl_queue_);
      sycl::free(wqk_b_, sycl_queue_);
  for (const auto pEnc : encoder_weights_) delete pEnc;
}

template <typename DataType>
EncoderBlock<DataType>::~EncoderBlock() {
      sycl::free(mha_q_w, sycl_queue_);
      sycl::free(mha_q_b, sycl_queue_);
      sycl::free(mha_k_w, sycl_queue_);
      sycl::free(mha_k_b, sycl_queue_);
      sycl::free(mha_v_w, sycl_queue_);
      sycl::free(mha_v_b, sycl_queue_);
      sycl::free(mha_qkv_w, sycl_queue_);
      sycl::free(mha_qkv_b, sycl_queue_);
      sycl::free(mha_dense_w, sycl_queue_);
      sycl::free(mha_dense_b, sycl_queue_);
      sycl::free(ln1_gammas, sycl_queue_);
      sycl::free(ln1_betas, sycl_queue_);
      sycl::free(ffn_dense1_w, sycl_queue_);
      sycl::free(ffn_dense1_b, sycl_queue_);
      sycl::free(ffn_dense2_w, sycl_queue_);
      sycl::free(ffn_dense2_b, sycl_queue_);
      sycl::free(ln2_gammas, sycl_queue_);
      sycl::free(ln2_betas, sycl_queue_);
  if (has_smolgen_) {
      sycl::free(smol_compress, sycl_queue_);
      sycl::free(smol_dense1_w, sycl_queue_);
      sycl::free(smol_dense1_b, sycl_queue_);
      sycl::free(smol_dense2_w, sycl_queue_);
      sycl::free(smol_dense2_b, sycl_queue_);
      sycl::free(smol_ln1_gammas, sycl_queue_);
      sycl::free(smol_ln1_betas, sycl_queue_);
      sycl::free(smol_ln2_gammas, sycl_queue_);
      sycl::free(smol_ln2_betas, sycl_queue_);
  }
  if (use_int8_) {
      sycl::free(mha_qkv_w_s8_, sycl_queue_);
      sycl::free(mha_qkv_comb_scales_, sycl_queue_);
      sycl::free(mha_dense_w_s8_, sycl_queue_);
      sycl::free(mha_dense_comb_scales_, sycl_queue_);
      sycl::free(ffn_dense1_w_s8_, sycl_queue_);
      sycl::free(ffn_dense1_comb_scales_, sycl_queue_);
      sycl::free(ffn_dense2_w_s8_, sycl_queue_);
      sycl::free(ffn_dense2_comb_scales_, sycl_queue_);
  }
}

template <typename DataType>
EmbeddingLayer<DataType>::EmbeddingLayer(BaseLayer<DataType>* ip,
                                         const std::vector<float>& weights,
                                         const std::vector<float>& biases,
                                         void* scratch, ActivationFunction act,
                                         sycl::queue &sycl_queue)
    : BaseLayer<DataType>(biases.size(), 8, 8, ip, sycl_queue), act_(act) {
  allocAndUpload<DataType>(&weights_, weights, scratch, sycl_queue_);
  allocAndUpload<DataType>(&biases_, biases, scratch, sycl_queue_);
}

template <typename DataType>
EmbeddingLayer<DataType>::~EmbeddingLayer() {
    sycl::free(weights_, sycl_queue_);
    sycl::free(biases_, sycl_queue_);
}

template <typename DataType>
void EmbeddingLayer<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* /*input2*/,
    void* /*scratch*/, size_t /*scratch_size*/, sycl::queue &sycl_queue, DataType***) {

  
  //CERR << "EmbeddingLayer<DataType>::Eval. ";

  const int num_outputs = this->GetC();
  const int num_inputs = this->input_->GetC();
  const int batch = N * 64;
  cublasXgemm<DataType>(transpose_type_transpose, transpose_type_notranspose, num_outputs, batch,
                        num_inputs, 1.0f, weights_, num_inputs, input,
                        num_inputs, 0.0f, output, num_outputs, sycl_queue);
  addBiasBatched(output, output, biases_, 1, batch, num_outputs, act_, sycl_queue);
}

template <typename DataType>
AttentionBody<DataType>::AttentionBody(const MultiHeadWeights& weights,
                                       void* scratch, Activations activations,
                                       int num_res_blocks, int input_c,
                                       int max_batch_size,
                                       bool is_pe_dense_embedding,
                                       sycl::queue &sycl_queue,
                                       bool use_int8)
    : BaseLayer<DataType>(weights.ip_emb_b.size(), 8, 8, nullptr, sycl_queue),
      embedding_op_size_(weights.ip_emb_b.size()),
      encoder_head_count_(weights.encoder_head_count),
      activations_(activations),
      num_resi_blocks_(num_res_blocks),
      input_c_(input_c),
      has_gating_(weights.ip_mult_gate.size() > 0 &&
                  weights.ip_add_gate.size() > 0),
      has_smolgen_(weights.has_smolgen),
      is_pe_dense_embedding_(is_pe_dense_embedding) {
  allocAndUpload<DataType>(&ip_emb_w_, weights.ip_emb_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ip_emb_b_, weights.ip_emb_b, scratch, sycl_queue_);

  if (is_pe_dense_embedding_) {
    allocAndUpload<DataType>(&ip_emb_pre_w_, weights.ip_emb_preproc_w, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&ip_emb_pre_b_, weights.ip_emb_preproc_b, scratch,
                             sycl_queue_);

    allocAndUpload<DataType>(&ip_emb_ln_g_, weights.ip_emb_ln_gammas, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&ip_emb_ln_b_, weights.ip_emb_ln_betas, scratch,
                             sycl_queue_);

    allocAndUpload<DataType>(&ip_emb_ffn_d1_w_, weights.ip_emb_ffn.dense1_w,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&ip_emb_ffn_d1_b_, weights.ip_emb_ffn.dense1_b,
                             scratch, sycl_queue_);

    allocAndUpload<DataType>(&ip_emb_ffn_d2_w_, weights.ip_emb_ffn.dense2_w,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&ip_emb_ffn_d2_b_, weights.ip_emb_ffn.dense2_b,
                             scratch, sycl_queue_);

    allocAndUpload<DataType>(&ip_emb_ffn_ln_g_, weights.ip_emb_ffn_ln_gammas,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&ip_emb_ffn_ln_b_, weights.ip_emb_ffn_ln_betas,
                             scratch, sycl_queue_);

    // 12 is the number of input channels used for the input encoding.
    embedding_dense_size_ = weights.ip_emb_preproc_b.size() / 64;
    embedding_ffn_size_ = weights.ip_emb_ffn.dense2_b.size();
    embedding_ffn_dff_ = weights.ip_emb_ffn.dense1_b.size();
  } else {
    size_t size = 64 * kNumPosEncodingChannels * sizeof(float);
    pos_encoding_ = (DataType *)sycl::malloc_device(size, sycl_queue_);
    sycl_queue_.memcpy(scratch, kPosEncoding, size);
    copyTypeConverted(pos_encoding_, (float*)scratch, size, sycl_queue_);
  }

  if (has_gating_) {
    allocAndUpload<DataType>(&ip_mult_gate_, weights.ip_mult_gate, scratch, sycl_queue_);
    allocAndUpload<DataType>(&ip_add_gate_, weights.ip_add_gate, scratch, sycl_queue_);
  }

  if (has_smolgen_) {
    allocAndUpload<DataType>(&smolgen_global_, weights.smolgen_w, scratch, sycl_queue_);
    smolgen_global_size_ = 64 * 64;
  }

  int num_encoders = weights.encoder.size();
  float alpha = (float)pow(2.0 * num_encoders, -0.25);
  for (const auto& enc : weights.encoder) {
    EncoderBlock<DataType>* pW = new EncoderBlock<DataType>(
        enc, scratch, encoder_head_count_, embedding_op_size_, alpha,
        smolgen_global_, smolgen_global_size_, max_batch_size,
        activations_.smolgen_activation, activations_.ffn_activation,
        is_pe_dense_embedding_ ? 1e-3 : 1e-6, sycl_queue_, use_int8);

    if (pW->uses_int8()) {
      has_int8_encoders_ = true;
      const size_t act_cols =
          static_cast<size_t>(std::max({embedding_op_size_, pW->d_model(), pW->d_ff()}));
      const size_t s32_cols =
          static_cast<size_t>(std::max({3 * pW->d_model(), pW->d_ff(), embedding_op_size_}));
      int8_act_per_pos_ = std::max(int8_act_per_pos_, size_t{64} * act_cols);
      int32_gemm_per_pos_ = std::max(int32_gemm_per_pos_, size_t{64} * s32_cols);
    }
    encoder_weights_.emplace_back(pW);
  }
  if (has_int8_encoders_) {
    int32_zero_co_ = sycl::malloc_device<int32_t>(1, sycl_queue_);
    sycl_queue_.memset(int32_zero_co_, 0, sizeof(int32_t)).wait();
    ensureInt8Scratch(std::min(max_batch_size, 64), sycl_queue_);
  }
}

template <typename DataType>
void AttentionBody<DataType>::ensureInt8Scratch(int N, sycl::queue& sycl_queue) {
  if (!has_int8_encoders_ || N <= int8_scratch_cap_batch_) return;
  if (int8_act_buf_ != nullptr || int32_gemm_buf_ != nullptr) {
    sycl_queue.wait();
    if (int8_act_buf_) sycl::free(int8_act_buf_, sycl_queue_);
    if (int32_gemm_buf_) sycl::free(int32_gemm_buf_, sycl_queue_);
  }
  int8_act_buf_ = sycl::malloc_device<uint8_t>(
      static_cast<size_t>(N) * int8_act_per_pos_, sycl_queue_);
  int32_gemm_buf_ = sycl::malloc_device<int32_t>(
      static_cast<size_t>(N) * int32_gemm_per_pos_, sycl_queue_);
  int8_scratch_cap_batch_ = N;
}

template <typename DataType>
AttentionBody<DataType>::~AttentionBody() {
  if (int8_act_buf_) sycl::free(int8_act_buf_, sycl_queue_);
  if (int32_gemm_buf_) sycl::free(int32_gemm_buf_, sycl_queue_);
  if (int32_zero_co_) sycl::free(int32_zero_co_, sycl_queue_);
  sycl::free(ip_emb_w_, sycl_queue_);
  sycl::free(ip_emb_b_, sycl_queue_);
  if (is_pe_dense_embedding_) {
    sycl::free(ip_emb_pre_w_, sycl_queue_);
    sycl::free(ip_emb_pre_b_, sycl_queue_);
    sycl::free(ip_emb_ln_g_, sycl_queue_);
    sycl::free(ip_emb_ln_b_, sycl_queue_);
    sycl::free(ip_emb_ffn_d1_w_, sycl_queue_);
    sycl::free(ip_emb_ffn_d1_b_, sycl_queue_);
    sycl::free(ip_emb_ffn_d2_w_, sycl_queue_);
    sycl::free(ip_emb_ffn_d2_b_, sycl_queue_);
    sycl::free(ip_emb_ffn_ln_g_, sycl_queue_);
    sycl::free(ip_emb_ffn_ln_b_, sycl_queue_);
  } else {
    sycl::free(pos_encoding_, sycl_queue_);
  }

  if (has_gating_) {
    sycl::free(ip_mult_gate_, sycl_queue_);
    sycl::free(ip_add_gate_, sycl_queue_);
  }
  if (has_smolgen_) {
    sycl::free(smolgen_global_, sycl_queue_);
  }
  for (const auto pEnc : encoder_weights_) delete pEnc;
}

template <typename DataType>
void AttentionBody<DataType>::Eval(int N, DataType* output,
                                   const DataType* input,
                                   const DataType* input2, void* scratch,
                                   size_t scratch_size, 
                                   sycl::queue &sycl_queue,
                                   DataType*** offset_pointers) {

  //CERR << "AttentionBody<DataType>::Eval. ";

  DataType* output_tensor = (DataType*)output;
  DataType* buffer1 = (DataType*)input2;
  DataType* buffer2 = buffer1 + scratch_size / (2 * sizeof(DataType));

  int inputC = input_c_;
  if (num_resi_blocks_ == 0) {
    assert(inputC == kInputPlanes);
    /*
      # if there are no residual blocks (pure transformer), do some input
      processing
    */
    if (is_pe_dense_embedding_) {
      // New encoding is made of dense layer fed with input from a 12-channel
      // slice of the input tensor.
      // pos_info = flow[..., :12]
      // pos_info_flat = tf.reshape(pos_info, [-1, 64 * 12])
      // pos_info_processed = tf.keras.layers.Dense(64*self.embedding_dense_sz,
      //                                            name=name+"embedding/preprocess")(pos_info_flat)
      const int num_outputs = 64 * embedding_dense_size_;
      const int num_inputs = 64 * 12;
      const int batch = N;

      convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, 12, 8, 8, sycl_queue);
      cublasXgemm<DataType>(
          transpose_type_transpose, transpose_type_notranspose, num_outputs, batch, num_inputs,
          1.0f, (const DataType*)ip_emb_pre_w_, num_inputs,
          (const DataType*)scratch, num_inputs, 0.0f, buffer1, num_outputs, sycl_queue);

      // addBiasBatched(buffer1, buffer1, ip_emb_pre_b_, batch, N, num_outputs,
      //               ACTIVATION_NONE, sycl_queue);
      const int size = num_outputs * N;
      // @todo addBiasBatched has a 4096 channel limit, needs refactoring.
      addVectors(buffer1, buffer1, ip_emb_pre_b_, size, size, num_outputs,
                 ACTIVATION_NONE, sycl_queue);
      inputPreprocessForAttentionBody((DataType*)scratch, input, buffer1, N,
                                      kInputPlanes, embedding_dense_size_, true,
                                      sycl_queue);
      inputC += embedding_dense_size_;
    } else {
      /*
      flow = tf.transpose(inputs, perm=[0, 2, 3, 1])
      flow = tf.reshape(flow, [-1, 64, tf.shape(inputs)[1]])
      # add positional encoding for each square to the input
      positional_encoding = tf.broadcast_to(tf.convert_to_tensor(self.POS_ENC,
      dtype=self.model_dtype), [tf.shape(flow)[0], 64,
      tf.shape(self.POS_ENC)[2]]) flow = tf.concat([flow, positional_encoding],
      axis=2)
      */
      inputPreprocessForAttentionBody((DataType*)scratch, input, pos_encoding_,
                                      N, kInputPlanes, kNumPosEncodingChannels,
                                      false, sycl_queue);
      inputC += kNumPosEncodingChannels;
    }
  } else {
    // #redirect flow through encoder blocks
    // flow = tf.transpose(flow, perm = [ 0, 2, 3, 1 ])
    // flow = tf.reshape(flow, [ -1, 64, self.RESIDUAL_FILTERS ])
    convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, inputC, 8, 8, sycl_queue);
  }

  if (is_pe_dense_embedding_) {
    // 1. square embedding (fully connected layer)
    // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
    DataType* embedding = output_tensor;
    DataType* temp = (DataType*)scratch;
    {
      const int num_outputs = embedding_op_size_;
      const int num_inputs = inputC;
      const int batch = N * 64;
      cublasXgemm<DataType>(transpose_type_transpose, transpose_type_notranspose, num_outputs,
                            batch, num_inputs, 1.0f, (const DataType*)ip_emb_w_,
                            num_inputs, temp, num_inputs, 0.0f, embedding,
                            num_outputs, sycl_queue);
      // embedding layer norm with fused in bias add of previous gemm.
      LayerNorm<DataType>(N * 64, embedding_op_size_, temp, embedding,
                          ip_emb_b_, (DataType*)nullptr, ip_emb_ln_g_,
                          ip_emb_ln_b_, 1e-3, 1.0,
                          activations_.default_activation, sycl_queue);
    }

    // Input gating
    if (has_gating_) {
      applyInputGating<DataType>(temp, temp, ip_mult_gate_, ip_add_gate_, N, 64,
                                 embedding_op_size_, sycl_queue);
    }

    // embedding FFN dense 1
    {
      const int num_inputs = embedding_ffn_size_;
      const int num_outputs = embedding_ffn_dff_;  // encoder_dff
      const int batch = N * 64;
      cublasXgemm(transpose_type_transpose, transpose_type_notranspose, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)ip_emb_ffn_d1_w_,
                  num_inputs, temp, num_inputs, 0.0f, buffer1, num_outputs, sycl_queue);
      addBiasBatched(buffer1, buffer1, ip_emb_ffn_d1_b_, 1, batch, num_outputs,
                     activations_.ffn_activation, sycl_queue);
    }

    // embedding FFN dense 2
    {
      const int num_inputs = embedding_ffn_dff_;  // encoder_dff
      const int num_outputs = embedding_ffn_size_;
      const int batch = N * 64;
      cublasXgemm(transpose_type_transpose, transpose_type_notranspose, num_outputs, batch,
                  num_inputs, 1.0f, (const DataType*)ip_emb_ffn_d2_w_,
                  num_inputs, buffer1, num_inputs, 0.0f, buffer2, num_outputs, sycl_queue);
      // Embedding LN: skip connection and layer normilization (also bias add of
      // prev gemm) buffer2 -> embedding
      float alpha = (float)pow(2. * encoder_weights_.size(), -0.25);
      LayerNorm<DataType>(N * 64, embedding_ffn_size_, embedding, buffer2,
                          ip_emb_ffn_d2_b_, temp, ip_emb_ffn_ln_g_,
                          ip_emb_ffn_ln_b_, 1e-3, alpha, ACTIVATION_NONE,
                          sycl_queue);
    }

  } else {
    // 1. square embedding (fully connected layer)
    // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
    DataType* embedding = output_tensor;
    {
      const int num_outputs = embedding_op_size_;
      const int num_inputs = inputC;
      const int batch = N * 64;
      cublasXgemm<DataType>(transpose_type_transpose, transpose_type_notranspose, num_outputs,
                            batch, num_inputs, 1.0f, (const DataType*)ip_emb_w_,
                            num_inputs, (DataType*)scratch, num_inputs, 0.0f,
                            embedding, num_outputs, sycl_queue);
      addBiasBatched(embedding, embedding, ip_emb_b_, 1, batch, num_outputs,
                     activations_.default_activation, sycl_queue);
    }
    // Input gating
    if (has_gating_) {
      applyInputGating<DataType>(embedding, embedding, ip_mult_gate_,
                                 ip_add_gate_, N, 64, embedding_op_size_,
                                 sycl_queue);
    }
  }

  // 2. Encoder blocks
  if (has_int8_encoders_) {
    ensureInt8Scratch(N, sycl_queue);
  }
  for (const auto pEnc : encoder_weights_) {
    pEnc->Eval(N, output_tensor, (DataType*)scratch, buffer1, buffer2,
               sycl_queue, offset_pointers, int8_act_buf_, int32_gemm_buf_,
               int32_zero_co_);
  }  // End of encoder blocks
}

template <typename DataType>
ValueHead<DataType>::ValueHead(BaseLayer<DataType>* ip,
                               const MultiHeadWeights::ValueHead& weights,
                               void* scratch, bool attention_body, bool wdl,
                               ActivationFunction act, int max_batch_size,
                               sycl::queue &sycl_queue)
    : BaseLayer<DataType>(weights.ip_val_b.size(), 8, 8, ip, sycl_queue),
      attention_body_(attention_body),
      embedding_size_(attention_body ? weights.ip_val_b.size()
                                     : weights.value.biases.size()),
      value_hidden_size_(weights.ip1_val_b.size()),
      act_(act),
      wdl_(wdl) {
  if (attention_body_) {
    allocAndUpload<DataType>(&ip_val_w_, weights.ip_val_w, scratch, sycl_queue);
    allocAndUpload<DataType>(&ip_val_b_, weights.ip_val_b, scratch, sycl_queue);
  } else {
    conv_ = std::make_unique<Conv1Layer<DataType>>(
        ip, weights.value.biases.size(), 8, 8, ip->GetC(), act, true,
        sycl_queue);
    conv_->LoadWeights((float*)&weights.value.weights[0],
                       (float*)&weights.value.biases[0], scratch);
  }

  allocAndUpload<DataType>(&ip1_val_w_, weights.ip1_val_w, scratch, sycl_queue);
  allocAndUpload<DataType>(&ip1_val_b_, weights.ip1_val_b, scratch, sycl_queue);

  allocAndUpload<DataType>(&ip2_val_w_, weights.ip2_val_w, scratch, sycl_queue);
  allocAndUpload<DataType>(&ip2_val_b_, weights.ip2_val_b, scratch, sycl_queue);
}

template <typename DataType>
ValueHead<DataType>::~ValueHead() {
  if (attention_body_) {
    sycl::free(ip_val_w_, sycl_queue_);
    sycl::free(ip_val_b_, sycl_queue_);
  }
  sycl::free(ip1_val_w_, sycl_queue_);
  sycl::free(ip1_val_b_, sycl_queue_);
  sycl::free(ip2_val_w_, sycl_queue_);
  sycl::free(ip2_val_b_, sycl_queue_);
}

template <typename DataType>
void ValueHead<DataType>::Eval(int N, DataType* output, const DataType* input,
                               const DataType* input2, void* scratch,
                               size_t scratch_size, sycl::queue &sycl_queue,
                               DataType***) {
  DataType* buffer = (DataType*)input2;
  {
    const int num_inputs = this->input_->GetC();
    const int num_outputs = embedding_size_;
    const int batch = N * 64;
    if (attention_body_) {
      cublasXgemm<DataType>(transpose_type_transpose, transpose_type_notranspose, num_outputs,
                            batch, num_inputs, 1.0f, (const DataType*)ip_val_w_,
                            num_inputs, input, num_inputs, 0.0f, buffer,
                            num_outputs, sycl_queue);
      addBiasBatched<DataType>(buffer, buffer, ip_val_b_, 1, batch, num_outputs,
                               act_, sycl_queue);

    } else {
      conv_->Eval(N, buffer, input, nullptr, scratch, scratch_size, sycl_queue);
    }
  }

  {
    // Value dense 1
    const int num_inputs = embedding_size_ * 64;
    const int num_outputs = value_hidden_size_;
    const int batch = N;
    DataType* layer_out = (DataType*)scratch;
    cublasXgemm<DataType>(transpose_type_transpose, transpose_type_notranspose, num_outputs, batch,
                          num_inputs, 1.0f, (const DataType*)ip1_val_w_,
                          num_inputs, buffer, num_inputs, 0.0f, layer_out,
                          num_outputs, sycl_queue);
    addBiasBatched<DataType>(layer_out, layer_out, ip1_val_b_, 1, batch,
                             num_outputs, act_, sycl_queue);
  }

  {
    // Value dense 2
    const int num_inputs = value_hidden_size_;
    const int num_outputs = wdl_ ? 3 : 1;
    const int batch = N;
    DataType* layer_out = (DataType*)output;
    cublasXgemm<DataType>(transpose_type_transpose, transpose_type_notranspose, num_outputs, batch,
                          num_inputs, 1.0f, (const DataType*)ip2_val_w_,
                          num_inputs, (DataType*)scratch, num_inputs, 0.0f,
                          layer_out, num_outputs, sycl_queue);
    addVectors(layer_out, layer_out, ip2_val_b_, num_outputs * batch,
               num_outputs * batch, num_outputs,
               wdl_ ? ACTIVATION_NONE : ACTIVATION_TANH, sycl_queue);
  }
}

// Template instantiation.
template class FCLayer<sycl::half>;
template class FCLayer<float>;

template class SELayer<sycl::half>;
template class SELayer<float>;

template class PolicyMapLayer<sycl::half>;
template class PolicyMapLayer<float>;

template class FusedWinogradConvSELayer<sycl::half>;
template class FusedWinogradConvSELayer<float>;

template class Conv1Layer<sycl::half>;
template class Conv1Layer<float>;

template class ResidualBlock<sycl::half>;
template class ResidualBlock<float>;

template class AttentionPolicyHead<sycl::half>;
template class AttentionPolicyHead<float>;

template class EncoderBlock<sycl::half>;
template class EncoderBlock<float>;

template class AttentionBody<sycl::half>;
template class AttentionBody<float>;

template class EmbeddingLayer<sycl::half>;
template class EmbeddingLayer<float>;

template class ValueHead<sycl::half>;
template class ValueHead<float>;

#ifdef USE_SYCL_DNNL
namespace {
constexpr size_t Align256(size_t n) { return (n + 255) & ~size_t(255); }
}  // namespace

template <typename DataType>
struct Int8ResidualTower<DataType>::Impl {
  static constexpr dnnl::memory::data_type kIoType =
      std::is_same<DataType, sycl::half>::value ? dnnl::memory::data_type::f16
                                                : dnnl::memory::data_type::f32;

  struct Block {
    dnnl::memory w1_mem;
    dnnl::memory w2_mem;
    float* b1 = nullptr;
    float* b2 = nullptr;
    float* ws1 = nullptr;
    float* ws2 = nullptr;
    float* a1_sc = nullptr;
    float* a2_sc = nullptr;
    int32_t* a1_zp = nullptr;
    int32_t* a2_zp = nullptr;
    dnnl::memory m_b1;
    dnnl::memory m_b2;
    dnnl::memory m_ws1;
    dnnl::memory m_ws2;
    dnnl::memory m_a1_sc;
    dnnl::memory m_a2_sc;
    dnnl::memory m_a1_zp;
    dnnl::memory m_a2_zp;

    bool has_se = false;
    int se_k = 0;
    float* se_w1_t = nullptr;
    float* se_b1 = nullptr;
    float* se_w2_t = nullptr;
    float* se_b2 = nullptr;
  };

  struct BatchPlan {
    dnnl::memory::desc nhwc_io_md;
    dnnl::memory::desc u1_md;
    dnnl::memory::desc u2_md;
    size_t u1_bytes = 0;
    size_t u2_bytes = 0;
    dnnl::reorder reord_in;
    dnnl::convolution_forward conv1;
    dnnl::convolution_forward conv2;
    dnnl::memory m_buf_a;
    dnnl::memory m_buf_b;
    dnnl::memory m_u1;
    dnnl::memory m_u2;
    std::unordered_map<int, dnnl::memory> args_r;
    std::unordered_map<int, dnnl::memory> args_c1;
    std::unordered_map<int, dnnl::memory> args_c2;
  };

  sycl::queue cached_q_;
  dnnl::engine eng_;
  dnnl::stream strm_;
  int C_;
  ActivationFunction act_;
  dnnl::algorithm act_alg_;
  bool use_zp_ = false;
  dnnl::memory::desc w1_md_;
  dnnl::memory::desc w2_md_;
  dnnl::memory::desc bia_md_;
  dnnl::primitive_attr r_attr_;
  dnnl::primitive_attr a1_attr_;
  dnnl::primitive_attr a2_attr_;
  std::vector<Block> blocks_;
  std::unordered_map<int, BatchPlan> plans_;
  void* extra_scratch_ = nullptr;
  size_t extra_scratch_size_ = 0;

  Impl(const std::vector<MultiHeadWeights::Residual>& residuals,
       int num_filters, ActivationFunction act, sycl::queue& q)
      : cached_q_(q),
        eng_(dnnl::sycl_interop::make_engine(q.get_device(), q.get_context())),
        strm_(dnnl::sycl_interop::make_stream(eng_, q)),
        C_(num_filters),
        act_(act) {
    switch (act_) {
      case ACTIVATION_RELU:
        act_alg_ = dnnl::algorithm::eltwise_relu;
        break;
      case ACTIVATION_MISH:
        act_alg_ = dnnl::algorithm::eltwise_mish;
        break;
      default:
        throw Exception("Unsupported activation for SYCL INT8 residual tower.");
    }

    // Auto-detect whether any block needs asymmetric non-zero zero-point (e.g. Mish).
    use_zp_ = false;
    for (const auto& res : residuals) {
      if (!res.conv1.has_int8 || !res.conv2.has_int8) {
        throw Exception("Int8ResidualTower requires INT8 weights on all residual blocks.");
      }
      if (res.conv1.act_zp != 0 || res.conv2.act_zp != 0) {
        use_zp_ = true;
      }
    }

    // Configure shared primitive attributes.
    r_attr_.set_scales_mask(DNNL_ARG_DST, 0);
    if (use_zp_) r_attr_.set_zero_points_mask(DNNL_ARG_DST, 0);

    a1_attr_.set_scales_mask(DNNL_ARG_SRC, 0);
    a1_attr_.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 0);
    a1_attr_.set_scales_mask(DNNL_ARG_DST, 0);
    if (use_zp_) {
      a1_attr_.set_zero_points_mask(DNNL_ARG_SRC, 0);
      a1_attr_.set_zero_points_mask(DNNL_ARG_DST, 0);
    }
    dnnl::post_ops ops1;
    ops1.append_eltwise(act_alg_, 0.0f, 0.0f);
    a1_attr_.set_post_ops(ops1);

    a2_attr_.set_scales_mask(DNNL_ARG_SRC, 0);
    a2_attr_.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 0);
    if (use_zp_) a2_attr_.set_zero_points_mask(DNNL_ARG_SRC, 0);

    bia_md_ = dnnl::memory::desc({C_}, dnnl::memory::data_type::f32,
                                 dnnl::memory::format_tag::a);

    // Query optimal blocked weight layouts once (proven identical across B=1..64 on Xe).
    const int kRefB = 32;
    auto wei_any = dnnl::memory::desc({C_, C_, 3, 3}, dnnl::memory::data_type::s8,
                                      dnnl::memory::format_tag::any);
    auto pd1_ref = dnnl::convolution_forward::primitive_desc(
        eng_, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_direct,
        dnnl::memory::desc({kRefB, C_, 8, 8}, dnnl::memory::data_type::u8,
                           dnnl::memory::format_tag::any),
        wei_any, bia_md_,
        dnnl::memory::desc({kRefB, C_, 8, 8}, dnnl::memory::data_type::u8,
                           dnnl::memory::format_tag::any),
        {1, 1}, {1, 1}, {1, 1}, a1_attr_);
    auto pd2_ref = dnnl::convolution_forward::primitive_desc(
        eng_, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_direct,
        pd1_ref.dst_desc(), wei_any, bia_md_,
        dnnl::memory::desc({kRefB, C_, 8, 8}, kIoType,
                           dnnl::memory::format_tag::nhwc),
        {1, 1}, {1, 1}, {1, 1}, a2_attr_);
    w1_md_ = pd1_ref.weights_desc();
    w2_md_ = pd2_ref.weights_desc();

    // Helper lambdas for device upload and weight packing.
    auto wei_oihw_md = dnnl::memory::desc({C_, C_, 3, 3},
                                          dnnl::memory::data_type::s8,
                                          dnnl::memory::format_tag::oihw);
    auto* tmp_w_dev = sycl::malloc_device<int8_t>(static_cast<size_t>(C_) * C_ * 9, q);
    auto pack_weights = [&](const std::vector<int8_t>& w_i8,
                            const dnnl::memory::desc& dst_md) -> dnnl::memory {
      if (w_i8.size() != static_cast<size_t>(C_) * C_ * 9) {
        throw Exception("Unexpected INT8 conv weight size in Int8ResidualTower.");
      }
      q.memcpy(tmp_w_dev, w_i8.data(), w_i8.size()).wait();
      auto m_src = dnnl::sycl_interop::make_memory(
          wei_oihw_md, eng_, dnnl::sycl_interop::memory_kind::usm, tmp_w_dev);
      dnnl::memory m_dst(dst_md, eng_);
      dnnl::reorder(m_src, m_dst).execute(strm_, m_src, m_dst);
      strm_.wait();
      return m_dst;
    };

    auto alloc_f32 = [&](const float* data, size_t n,
                         dnnl::memory* out_mem) -> float* {
      float* p = sycl::malloc_device<float>(n, q);
      q.memcpy(p, data, n * sizeof(float)).wait();
      if (out_mem) {
        *out_mem = dnnl::sycl_interop::make_memory(
            dnnl::memory::desc({static_cast<dnnl::memory::dim>(n)},
                               dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::a),
            eng_, dnnl::sycl_interop::memory_kind::usm, p);
      }
      return p;
    };
    auto alloc_i32_scalar = [&](int32_t val, dnnl::memory* out_mem) -> int32_t* {
      int32_t* p = sycl::malloc_device<int32_t>(1, q);
      q.memcpy(p, &val, sizeof(int32_t)).wait();
      *out_mem = dnnl::sycl_interop::make_memory(
          dnnl::memory::desc({1}, dnnl::memory::data_type::s32,
                             dnnl::memory::format_tag::a),
          eng_, dnnl::sycl_interop::memory_kind::usm, p);
      return p;
    };

    blocks_.reserve(residuals.size());
    for (const auto& res : residuals) {
      Block blk;
      blk.w1_mem = pack_weights(res.conv1.weights_i8, w1_md_);
      blk.w2_mem = pack_weights(res.conv2.weights_i8, w2_md_);
      blk.b1 = alloc_f32(res.conv1.biases.data(), C_, &blk.m_b1);
      blk.b2 = alloc_f32(res.conv2.biases.data(), C_, &blk.m_b2);
      blk.ws1 = alloc_f32(res.conv1.w_scales.data(), C_, &blk.m_ws1);
      blk.ws2 = alloc_f32(res.conv2.w_scales.data(), C_, &blk.m_ws2);
      float s1 = res.conv1.act_scale;
      float s2 = res.conv2.act_scale;
      blk.a1_sc = alloc_f32(&s1, 1, &blk.m_a1_sc);
      blk.a2_sc = alloc_f32(&s2, 1, &blk.m_a2_sc);
      blk.a1_zp = alloc_i32_scalar(static_cast<int32_t>(res.conv1.act_zp), &blk.m_a1_zp);
      blk.a2_zp = alloc_i32_scalar(static_cast<int32_t>(res.conv2.act_zp), &blk.m_a2_zp);

      blk.has_se = res.has_se;
      if (blk.has_se) {
        const int se_k = static_cast<int>(res.se.b1.size());
        blk.se_k = se_k;
        std::vector<float> w1_t(static_cast<size_t>(C_) * se_k);
        for (int k = 0; k < se_k; ++k) {
          for (int c = 0; c < C_; ++c) {
            w1_t[c * se_k + k] = res.se.w1[k * C_ + c];
          }
        }
        std::vector<float> w2_t(static_cast<size_t>(se_k) * 2 * C_);
        for (int o = 0; o < 2 * C_; ++o) {
          for (int k = 0; k < se_k; ++k) {
            w2_t[k * (2 * C_) + o] = res.se.w2[o * se_k + k];
          }
        }
        blk.se_w1_t = alloc_f32(w1_t.data(), w1_t.size(), nullptr);
        blk.se_b1 = alloc_f32(res.se.b1.data(), se_k, nullptr);
        blk.se_w2_t = alloc_f32(w2_t.data(), w2_t.size(), nullptr);
        blk.se_b2 = alloc_f32(res.se.b2.data(), 2 * C_, nullptr);
      }
      blocks_.push_back(std::move(blk));
    }
    sycl::free(tmp_w_dev, q);

    // Pre-build plans through the default SYCL minibatch (2*80 = 160) to prevent MCTS JIT stalls.
    for (int b : {1, 4, 8, 16, 32, 64, 96, 128, 160}) {
      getOrCreatePlan(b);
    }
  }

  ~Impl() {
    cached_q_.wait();
    for (auto& blk : blocks_) {
      auto freep = [&](auto* p) {
        if (p) sycl::free(p, cached_q_);
      };
      freep(blk.b1);
      freep(blk.b2);
      freep(blk.ws1);
      freep(blk.ws2);
      freep(blk.a1_sc);
      freep(blk.a2_sc);
      freep(blk.a1_zp);
      freep(blk.a2_zp);
      freep(blk.se_w1_t);
      freep(blk.se_b1);
      freep(blk.se_w2_t);
      freep(blk.se_b2);
    }
    if (extra_scratch_) sycl::free(extra_scratch_, cached_q_);
  }

  BatchPlan& getOrCreatePlan(int B) {
    auto it = plans_.find(B);
    if (it != plans_.end()) return it->second;

    BatchPlan plan;
    plan.nhwc_io_md = dnnl::memory::desc({B, C_, 8, 8}, kIoType,
                                         dnnl::memory::format_tag::nhwc);
    auto pd1 = dnnl::convolution_forward::primitive_desc(
        eng_, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_direct,
        dnnl::memory::desc({B, C_, 8, 8}, dnnl::memory::data_type::u8,
                           dnnl::memory::format_tag::any),
        w1_md_, bia_md_,
        dnnl::memory::desc({B, C_, 8, 8}, dnnl::memory::data_type::u8,
                           dnnl::memory::format_tag::any),
        {1, 1}, {1, 1}, {1, 1}, a1_attr_);
    auto pd2 = dnnl::convolution_forward::primitive_desc(
        eng_, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_direct,
        pd1.dst_desc(), w2_md_, bia_md_, plan.nhwc_io_md,
        {1, 1}, {1, 1}, {1, 1}, a2_attr_);
    auto rpd = dnnl::reorder::primitive_desc(eng_, plan.nhwc_io_md, eng_,
                                             pd1.src_desc(), r_attr_);

    plan.u1_md = pd1.src_desc();
    plan.u2_md = pd1.dst_desc();
    plan.u1_bytes = plan.u1_md.get_size();
    plan.u2_bytes = plan.u2_md.get_size();
    plan.reord_in = dnnl::reorder(rpd);
    plan.conv1 = dnnl::convolution_forward(pd1);
    plan.conv2 = dnnl::convolution_forward(pd2);

    plan.m_buf_a = dnnl::sycl_interop::make_memory(
        plan.nhwc_io_md, eng_, dnnl::sycl_interop::memory_kind::usm,
        DNNL_MEMORY_NONE);
    plan.m_buf_b = dnnl::sycl_interop::make_memory(
        plan.nhwc_io_md, eng_, dnnl::sycl_interop::memory_kind::usm,
        DNNL_MEMORY_NONE);
    plan.m_u1 = dnnl::sycl_interop::make_memory(
        plan.u1_md, eng_, dnnl::sycl_interop::memory_kind::usm,
        DNNL_MEMORY_NONE);
    plan.m_u2 = dnnl::sycl_interop::make_memory(
        plan.u2_md, eng_, dnnl::sycl_interop::memory_kind::usm,
        DNNL_MEMORY_NONE);

    // Pre-populate constant keys in retained argument maps (zero per-eval map allocations).
    plan.args_r.reserve(4);
    plan.args_r[DNNL_ARG_FROM] = plan.m_buf_a;
    plan.args_r[DNNL_ARG_TO] = plan.m_u1;

    plan.args_c1.reserve(10);
    plan.args_c1[DNNL_ARG_SRC] = plan.m_u1;
    plan.args_c1[DNNL_ARG_DST] = plan.m_u2;

    plan.args_c2.reserve(8);
    plan.args_c2[DNNL_ARG_SRC] = plan.m_u2;
    plan.args_c2[DNNL_ARG_DST] = plan.m_buf_b;

    auto [ins_it, _] = plans_.emplace(B, std::move(plan));
    return ins_it->second;
  }

  static int bucketBatch(int N) {
    if (N <= 1) return 1;
    if (N <= 4) return 4;
    if (N <= 8) return 8;
    if (N <= 16) return 16;
    // Align >=17 to multiples of 32 (matches oneDNN ABcd32a32b tile and hits exact B=160).
    return ((N + 31) / 32) * 32;
  }

  void eval(int N, DataType* output, const DataType* input,
            void* scratch, size_t scratch_size, sycl::queue& sycl_queue) {
    if (sycl_queue != cached_q_) {
      strm_ = dnnl::sycl_interop::make_stream(eng_, sycl_queue);
      cached_q_ = sycl_queue;
    }

    const int B_pad = bucketBatch(N);
    BatchPlan& plan = getOrCreatePlan(B_pad);
    const size_t io_bytes = Align256(static_cast<size_t>(B_pad) * C_ * 64 * sizeof(DataType));
    const size_t u1_bytes = Align256(plan.u1_bytes);
    const size_t u2_bytes = Align256(plan.u2_bytes);
    const size_t total_needed = 2 * io_bytes + u1_bytes + u2_bytes;

    uint8_t* base_ptr = static_cast<uint8_t*>(scratch);
    if (total_needed > scratch_size) {
      if (total_needed > extra_scratch_size_) {
        if (extra_scratch_) sycl::free(extra_scratch_, sycl_queue);
        extra_scratch_ = sycl::malloc_device(total_needed, sycl_queue);
        extra_scratch_size_ = total_needed;
      }
      base_ptr = static_cast<uint8_t*>(extra_scratch_);
    }

    auto* buf_a = reinterpret_cast<DataType*>(base_ptr);
    auto* buf_b = reinterpret_cast<DataType*>(base_ptr + io_bytes);
    uint8_t* u1_ptr = base_ptr + 2 * io_bytes;
    uint8_t* u2_ptr = u1_ptr + u1_bytes;

    plan.m_buf_a.set_data_handle(buf_a);
    plan.m_buf_b.set_data_handle(buf_b);
    plan.m_u1.set_data_handle(u1_ptr);
    plan.m_u2.set_data_handle(u2_ptr);

    // 1. Convert InputConv output NCHW -> NHWC once at tower entry (zero-pad N..B_pad).
    transposeNCHWtoNHWC_Int8Tower(buf_a, input, N, B_pad, C_, sycl_queue);

    // 2. Execute all INT8 residual blocks in NHWC.
    for (const auto& blk : blocks_) {
      plan.args_r[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = blk.m_a1_sc;
      if (use_zp_) {
        plan.args_r[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST] = blk.m_a1_zp;
      }
      plan.reord_in.execute(strm_, plan.args_r);

      plan.args_c1[DNNL_ARG_WEIGHTS] = blk.w1_mem;
      plan.args_c1[DNNL_ARG_BIAS] = blk.m_b1;
      plan.args_c1[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = blk.m_a1_sc;
      plan.args_c1[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = blk.m_ws1;
      plan.args_c1[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = blk.m_a2_sc;
      if (use_zp_) {
        plan.args_c1[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = blk.m_a1_zp;
        plan.args_c1[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST] = blk.m_a2_zp;
      }
      plan.conv1.execute(strm_, plan.args_c1);

      plan.args_c2[DNNL_ARG_WEIGHTS] = blk.w2_mem;
      plan.args_c2[DNNL_ARG_BIAS] = blk.m_b2;
      plan.args_c2[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = blk.m_a2_sc;
      plan.args_c2[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = blk.m_ws2;
      if (use_zp_) {
        plan.args_c2[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = blk.m_a2_zp;
      }
      plan.conv2.execute(strm_, plan.args_c2);

      if (blk.has_se) {
        fusedSE_SkipAct_NHWC(buf_a, buf_a, buf_b, blk.se_w1_t, blk.se_b1,
                             blk.se_w2_t, blk.se_b2, N, C_, blk.se_k, act_,
                             sycl_queue);
      } else {
        addSkipAct_NHWC(buf_a, buf_a, buf_b, N * 64 * C_, act_, sycl_queue);
      }
    }

    // 3. Convert tower output NHWC -> NCHW once for Policy/Value/MLH heads.
    transposeNHWCtoNCHW_Int8Tower(output, buf_a, N, C_, sycl_queue);
  }
};

template <typename DataType>
Int8ResidualTower<DataType>::Int8ResidualTower(
    BaseLayer<DataType>* ip,
    const std::vector<MultiHeadWeights::Residual>& residuals, int num_filters,
    ActivationFunction act, sycl::queue& sycl_queue)
    : BaseLayer<DataType>(num_filters, 8, 8, ip, false, sycl_queue),
      impl_(std::make_unique<Impl>(residuals, num_filters, act, sycl_queue)) {}

template <typename DataType>
Int8ResidualTower<DataType>::~Int8ResidualTower() = default;

template <typename DataType>
void Int8ResidualTower<DataType>::Eval(int N, DataType* output,
                                       const DataType* input,
                                       const DataType* /*input2*/,
                                       void* scratch, size_t scratch_size,
                                       sycl::queue& sycl_queue,
                                       DataType***) {
  impl_->eval(N, output, input, scratch, scratch_size, sycl_queue);
}

template class Int8ResidualTower<sycl::half>;
template class Int8ResidualTower<float>;
#endif

#ifdef USE_CUBLAS
// Misc error handling stuff.
const char* CublasGetErrorString(int status) {
  switch (status) {
    case 0:
      return "CUBLAS_STATUS_SUCCESS";
    case 1:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case 3:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case 7:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case 8:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case 11:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case 13:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case 14:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case 15:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case 16:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown cublas error";
}

void CublasError(int status, const char* file, const int& line) {
  if (status != 0) {
    char message[128];
    sprintf(message, "CUBLAS error: %s (%s:%d) ", CublasGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}
#endif

}  // namespace sycldnn_backend
}  // namespace lczero
