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
#include "sycl_common.h"
#include "neural/backends/shared/activation.h"

namespace lczero {
namespace sycldnn_backend {

// Adds two vectors (possibly of different sizes), also do optional
// activation (relu, tanh or sigmoid).
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, sycl::queue &sycl_queue);

// Adds two vectors of equal size overwriting the first with the sum.
// This specialisation performs a transposition of the first 2 indexes
// of the second while performing the addition.
template <typename T>
void addVectorsHNC_NHC(T* a, T* b, int N, int H, int C, sycl::queue &sycl_queue);

// Optimized kernel to add bias to innermost dimension
// and perform optional activation (to be used with GEMMs/fully connected)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, ActivationFunction activation, sycl::queue &sycl_queue);

// Optimized kernel to add bias to innermost dimension
// and perform optional activation (to be used with GEMMs/fully connected)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, int Nstride, ActivationFunction activation, sycl::queue &sycl_queue);

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W,
                  ActivationFunction activation, sycl::queue &sycl_queue);

// Conversion from NCHW to NHWC, can also change datatype depending on template
// params, also pad/un-pad elements from Batch or Channel dimensions
template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W, sycl::queue &sycl_queue);

// Plain data-type conversion (no layout conversion).
//
// Defined inline (header-only) rather than declared here and explicitly
// instantiated in common_kernels.dp.cpp: on this toolchain a SYCL kernel
// that is only reachable via a cross-translation-unit explicit template
// instantiation can go missing from the linked device image ("No kernel
// named ... was found" at runtime, e.g. for copyTypeConverted<float,
// float> during fp32 weight loading) even though the host-side stub
// links fine. Defining the kernel in the header makes every calling TU
// (network_sycl.cc.dp.cpp, layers.cc.dp.cpp) compile and register its
// own copy, which is the portable pattern for SYCL free-function kernels.
template <typename DstType, typename SrcType>
void copyTypeConverted_kernel(DstType* op, SrcType* ip, int N,
                              const sycl::nd_item<3> &item_ct1) {
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  if (tid >= N) return;
  op[tid] = (DstType)ip[tid];
}

template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, sycl::queue &sycl_queue) {
  const int kBlockSize = 256;
  int blocks = DivUp(N, kBlockSize);
  sycl_queue.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                             sycl::range<3>(1, 1, kBlockSize),
                                         sycl::range<3>(1, 1, kBlockSize)),
                       [=](sycl::nd_item<3> item_ct1) {
                         copyTypeConverted_kernel(op, ip, N, item_ct1);
                       });
}

// Perform batch normilization.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers,
               ActivationFunction activation, sycl::queue &sycl_queue);

// Unpack planes (input to network).
void expandPlanes_Fp32_NCHW(float* output, const uint64_t* masks,
                            const float* values, int n, sycl::queue &sycl_queue);

void expandPlanes_Fp16_NHWC(sycl::half* output, const uint64_t* masks,
                            const float* values, int n, sycl::queue &sycl_queue);

void expandPlanes_Fp16_NCHW(sycl::half* output, const uint64_t* masks,
                            const float* values, int n, sycl::queue &sycl_queue);

// Perform global avg pool.
template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc, sycl::queue &sycl_queue);

// Perform global scale.
template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc,
                 ActivationFunction activation, sycl::queue &sycl_queue);

// Perform Squeeze-and-Excitation (SE) in a single fused kernel.
// Returns false if the fused kernel can't handle the sizes.
bool Se_Fp16_NHWC(int N, int C, int numFc1Out, sycl::half* output,
                  const sycl::half* skip, const sycl::half* input,
                  const sycl::half* w1, const sycl::half* b1,
                  const sycl::half* w2, const sycl::half* b2,
                  const sycl::half* bPrev, ActivationFunction activation, sycl::queue &sycl_queue);

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize, sycl::queue &sycl_queue);

// Custom winograd helper functions
template <typename T>
void FilterTransform(int N, int C, T* transformedFilter, const T* filter, sycl::queue &sycl_queue);

template <typename T, bool nhcw>
void InputTransform(int N, int C, T* transformedInput, const T* input, sycl::queue &sycl_queue);

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip, bool skipInput_nhcw, bool output_nhcw>
void OutputTransform(int N, int C, int se_K, T* output, const T* input,
                     const T* skip, const T* bias, const T* w1, const T* b1,
                     const T* w2, const T* b2, sycl::queue &sycl_queue);

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2, sycl::queue &sycl_queue);

template <typename T>
void Softmax(int N, int C, T* output, const T* input, const T* input2, sycl::queue &sycl_queue);

template <typename T>
void LayerNorm(int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act, sycl::queue &sycl_queue);

template <typename T>
void ComputePromotionLogits(int N, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits, sycl::queue &sycl_queue);

template <typename T>
void inputPreprocessForAttentionBody(T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size,
                                     bool is_pe_dense_embedding,
                                     sycl::queue &sycl_queue);

template <typename T>
void applyInputGating(T* output, const T* input, const T* mult, const T* add,
                      int N, int HW, int C, sycl::queue &sycl_queue);

template <typename T>
void genOffsetPointers(T** offsets, int heads, int max_batch, int depth,
                       int d_model, T* k, T* q, T* b1, T* v, T* b2, sycl::queue &sycl_queue);
}  // namespace sycldnn_backend
}  // namespace lczero
