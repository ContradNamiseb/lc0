

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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <sycl/sycl.hpp>

#include "neural/backends/shared/activation.h"
#include "neural/tables/attention_policy_map.h"
#include "sycl_common.h"
#include "winograd_helper.h"

namespace lczero {
namespace sycldnn_backend {
namespace {
constexpr int kInputPlanes = 112;

// Per-token square-visiting order for each of the 8 KDA board-scan
// directions, indexed [direction - 1][token]. Built once here so the hot
// recurrence loop in kdaRecurrenceValueParallel can index a row directly
// instead of re-branching on `direction` every one of its 64 iterations --
// `direction` is fixed per work-item for the whole kernel invocation, so the
// row lookup only needs to happen once, before that loop starts. Rows 0-3
// are the four orthogonal scans (forward/reverse rank-major,
// forward/reverse file-major a.k.a. transpose); rows 4-7 are the four
// diagonal scans (formerly kKdaDiagForward/Reverse/AntiDiagForward/Reverse).
constexpr int kKdaDirectionOrder[8][64] = {
    // direction 1: forward, rank-major (identity)
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
     32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
     48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
    // direction 2: reverse, rank-major
    {63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
     47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
     31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
     15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0},
    // direction 3: forward, file-major (transpose)
    {0,  8,  16, 24, 32, 40, 48, 56, 1,  9,  17, 25, 33, 41, 49, 57,
     2,  10, 18, 26, 34, 42, 50, 58, 3,  11, 19, 27, 35, 43, 51, 59,
     4,  12, 20, 28, 36, 44, 52, 60, 5,  13, 21, 29, 37, 45, 53, 61,
     6,  14, 22, 30, 38, 46, 54, 62, 7,  15, 23, 31, 39, 47, 55, 63},
    // direction 4: reverse, file-major (transpose)
    {63, 55, 47, 39, 31, 23, 15, 7,  62, 54, 46, 38, 30, 22, 14, 6,
     61, 53, 45, 37, 29, 21, 13, 5,  60, 52, 44, 36, 28, 20, 12, 4,
     59, 51, 43, 35, 27, 19, 11, 3,  58, 50, 42, 34, 26, 18, 10, 2,
     57, 49, 41, 33, 25, 17, 9,  1,  56, 48, 40, 32, 24, 16, 8,  0},
    // direction 5: diagonal, forward (formerly kKdaDiagForward)
    {7,  6,  15, 5,  14, 23, 4,  13, 22, 31, 3,  12, 21, 30, 39, 2,
     11, 20, 29, 38, 47, 1,  10, 19, 28, 37, 46, 55, 0,  9,  18, 27,
     36, 45, 54, 63, 8,  17, 26, 35, 44, 53, 62, 16, 25, 34, 43, 52,
     61, 24, 33, 42, 51, 60, 32, 41, 50, 59, 40, 49, 58, 48, 57, 56},
    // direction 6: diagonal, reverse (formerly kKdaDiagReverse)
    {56, 57, 48, 58, 49, 40, 59, 50, 41, 32, 60, 51, 42, 33, 24, 61,
     52, 43, 34, 25, 16, 62, 53, 44, 35, 26, 17, 8,  63, 54, 45, 36,
     27, 18, 9,  0,  55, 46, 37, 28, 19, 10, 1,  47, 38, 29, 20, 11,
     2,  39, 30, 21, 12, 3,  31, 22, 13, 4,  23, 14, 5,  15, 6,  7},
    // direction 7: anti-diagonal, forward (formerly kKdaAntiDiagForward)
    {0,  1,  8,  2,  9,  16, 3,  10, 17, 24, 4,  11, 18, 25, 32, 5,
     12, 19, 26, 33, 40, 6,  13, 20, 27, 34, 41, 48, 7,  14, 21, 28,
     35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51,
     58, 31, 38, 45, 52, 59, 39, 46, 53, 60, 47, 54, 61, 55, 62, 63},
    // direction 8: anti-diagonal, reverse (formerly kKdaAntiDiagReverse)
    {63, 62, 55, 61, 54, 47, 60, 53, 46, 39, 59, 52, 45, 38, 31, 58,
     51, 44, 37, 30, 23, 57, 50, 43, 36, 29, 22, 15, 56, 49, 42, 35,
     28, 21, 14, 7,  48, 41, 34, 27, 20, 13, 6,  40, 33, 26, 19, 12,
     5,  32, 25, 18, 11, 4,  24, 17, 10, 3,  16, 9,  2,  8,  1,  0},
};
}  // namespace

/////////////////////////////////////////////////////////////////////////////
//          Simple CUDA kernels used by certain layers                     //
/////////////////////////////////////////////////////////////////////////////

template <typename T>
void addVectors_kernel(T* c, T* a, T* b, int size, int asize, int bsize,
                       ActivationFunction activation,
                       const sycl::nd_item<3>& item_ct1) {
  int i = item_ct1.get_local_id(2) +
          item_ct1.get_local_range(2) * item_ct1.get_group(2);
  if (i < size) {
    float aVal = 0;
    float bVal = 0;
    if (a) aVal = (float)(a[i % asize]);
    if (b) bVal = (float)(b[i % bsize]);

    float cVal = aVal + bVal;

    cVal = activate(cVal, activation);

    c[i] = (T)cVal;
  }
}

// Adds two vectors (possibly of different sizes), also do optional relu
// activation.
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, sycl::queue& sycl_queue) {
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        addVectors_kernel(c, a, b, size, asize, bsize, activation, item_ct1);
      });
}

template <typename T>
void addVectorsHNC_NHC_kernel(T* a, T* b, int N, int H, int C,
                              const sycl::nd_item<3>& item_ct1) {
  int i = item_ct1.get_local_id(2) +
          item_ct1.get_local_range(2) * item_ct1.get_group(2);
  if (i < N * H * C) {
    int orig_i = i;
    int c = i % C;
    i /= C;
    int n = i % N;
    i /= N;
    int h = i;
    float aVal = (float)a[orig_i];
    float bVal = (float)b[n * H * C + h * C + c];

    float cVal = aVal + bVal;

    a[orig_i] = (T)cVal;
  }
}

template <typename T>
void addVectorsHNC_NHC(T* a, T* b, int N, int H, int C,
                       sycl::queue& sycl_queue) {
  const int kBlockSize = 256;
  int blocks = DivUp(N * H * C, kBlockSize);
  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        addVectorsHNC_NHC_kernel(a, b, N, H, C, item_ct1);
      });
}

template <typename T, ActivationFunction act>
void addBiasBatched_kernel(T* output, const T* input, const T* bias, int N,
                           int C, const sycl::nd_item<3>& item_ct1) {
  int batch = item_ct1.get_group(1);
  int n = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1);
  if (n >= N) return;
  int c = item_ct1.get_local_id(2) * 4;

  int biasIndex = batch * C + c;
  int tensorIndex = batch * N * C + n * C + c;

  float val[4];
  float b[4];

  // Load from memory
  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (fp16) {
    sycl::half inp[4];
    copyAs<sycl::uint2>(&inp[0], &input[tensorIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) val[i] = (float)inp[i];

    copyAs<sycl::uint2>(&inp[0], &bias[biasIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) b[i] = (float)inp[i];
  } else {
    copyAs<sycl::uint4>(&val[0], &input[tensorIndex]);
    copyAs<sycl::uint4>(&b[0], &bias[biasIndex]);
  }

  // Perform bias add and activation
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float x = val[i] + b[i];
    x = activate(x, act);
    val[i] = x;
  }

  // write to memory
  if (fp16) {
    sycl::half op[4];
#pragma unroll
    for (int i = 0; i < 4; i++) op[i] = (sycl::half)val[i];
    copyAs<sycl::uint2>(&output[tensorIndex], &op[0]);
  } else {
    copyAs<sycl::uint4>(&output[tensorIndex], &val[0]);
  }
}

// Input/output tensors are Batch * N * C
// bias tensor is N * C (i.e, different bias for each Batch dimension)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, ActivationFunction activation,
                    sycl::queue& sycl_queue) {
  // process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 4 != 0) throw Exception("unsupported filter size");
  if (C > 2048) throw Exception("unsupported filter size");

  sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
  blockDim[2] = C / 4;
  unsigned int tmp = (512 / blockDim[2]);
  blockDim[1] = sycl::min(sycl::max(tmp, 1u), (unsigned int)N);
  blockDim[0] = 1;
  gridDim[2] = DivUp(N, blockDim[1]);
  gridDim[1] = Batch;
  gridDim[0] = 1;

  switch (activation) {
    case ACTIVATION_NONE:
      // addBiasBatched_kernel<T, ACTIVATION_NONE>
      //   <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                              [=](sycl::nd_item<3> item_ct1) {
                                addBiasBatched_kernel<T, ACTIVATION_NONE>(
                                    output, input, bias, N, C, item_ct1);
                              });
      break;
    case ACTIVATION_SELU:
      // addBiasBatched_kernel<T, ACTIVATION_SELU>
      //   <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);

      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                              [=](sycl::nd_item<3> item_ct1) {
                                addBiasBatched_kernel<T, ACTIVATION_SELU>(
                                    output, input, bias, N, C, item_ct1);
                              });

      break;
    case ACTIVATION_MISH:
      // addBiasBatched_kernel<T, ACTIVATION_MISH>
      //   <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);

      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                              [=](sycl::nd_item<3> item_ct1) {
                                addBiasBatched_kernel<T, ACTIVATION_MISH>(
                                    output, input, bias, N, C, item_ct1);
                              });
      break;
    case ACTIVATION_RELU:
      // addBiasBatched_kernel<T, ACTIVATION_RELU>
      //   <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                              [=](sycl::nd_item<3> item_ct1) {
                                addBiasBatched_kernel<T, ACTIVATION_RELU>(
                                    output, input, bias, N, C, item_ct1);
                              });
      break;
    case ACTIVATION_SWISH:
      // addBiasBatched_kernel<T, ACTIVATION_SWISH>
      //   <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);

      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                              [=](sycl::nd_item<3> item_ct1) {
                                addBiasBatched_kernel<T, ACTIVATION_SWISH>(
                                    output, input, bias, N, C, item_ct1);
                              });
      break;
    case ACTIVATION_RELU_2:  // square relu
      // addBiasBatched_kernel<T, ACTIVATION_RELU_2>
      //   <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);

      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                              [=](sycl::nd_item<3> item_ct1) {
                                addBiasBatched_kernel<T, ACTIVATION_RELU_2>(
                                    output, input, bias, N, C, item_ct1);
                              });

      break;
    default:
      throw Exception(
          "unsupported activation in addBiasBatched. Add in switch-case here");
  }
}

template <typename T, ActivationFunction act>
void addBiasBatched_kernel(T* output, const T* input, const T* bias, int N,
                           int C, int Nstride,
                           const sycl::nd_item<3>& item_ct1) {
  int batch = item_ct1.get_group(1);
  int n = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1);
  if (n >= N) return;
  int c = item_ct1.get_local_id(2) * 4;

  int biasIndex = batch * C + c;
  int tensorIndex = batch * Nstride * C + n * C + c;

  float val[4];
  float b[4];

  // Load from memory
  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (fp16) {
    sycl::half inp[4];
    copyAs<sycl::uint2>(&inp[0], &input[tensorIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) val[i] = (float)inp[i];

    copyAs<sycl::uint2>(&inp[0], &bias[biasIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) b[i] = (float)inp[i];
  } else {
    copyAs<sycl::uint4>(&val[0], &input[tensorIndex]);
    copyAs<sycl::uint4>(&b[0], &bias[biasIndex]);
  }

  // Perform bias add and activation
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float x = val[i] + b[i];
    x = activate(x, act);
    val[i] = x;
  }

  // write to memory
  if (fp16) {
    sycl::half op[4];
#pragma unroll
    for (int i = 0; i < 4; i++) op[i] = (sycl::half)val[i];
    copyAs<sycl::uint2>(&output[tensorIndex], &op[0]);
  } else {
    copyAs<sycl::uint4>(&output[tensorIndex], &val[0]);
  }
}

// Input/output tensors are Batch * N * C
// bias tensor is N * C (i.e, different bias for each Batch dimension)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, int Nstride, ActivationFunction activation,
                    sycl::queue& sycl_queue) {
  // process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 4 != 0) throw Exception("unsupported filter size");
  if (C > 4096) throw Exception("unsupported filter size");

  sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
  blockDim[2] = C / 4;
  unsigned int tmp = (512 / blockDim[2]);
  blockDim[1] = sycl::min(sycl::max(tmp, 1u), (unsigned int)N);
  blockDim[0] = 1;
  gridDim[2] = DivUp(N, blockDim[1]);
  gridDim[1] = Batch;
  gridDim[0] = 1;

  switch (activation) {
    case ACTIVATION_NONE:
      // addBiasBatched_kernel<T, ACTIVATION_NONE>
      //     <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
      //                                       Nstride);
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                              [=](sycl::nd_item<3> item_ct1) {
                                addBiasBatched_kernel<T, ACTIVATION_NONE>(
                                    output, input, bias, N, C, Nstride,
                                    item_ct1);
                              });
      break;
    case ACTIVATION_SELU:
      // addBiasBatched_kernel<T, ACTIVATION_SELU>
      //   <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
      //                                    Nstride);
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                              [=](sycl::nd_item<3> item_ct1) {
                                addBiasBatched_kernel<T, ACTIVATION_SELU>(
                                    output, input, bias, N, C, Nstride,
                                    item_ct1);
                              });
      break;
    case ACTIVATION_MISH:
      // addBiasBatched_kernel<T, ACTIVATION_MISH>
      //     <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
      //                                       Nstride);
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                              [=](sycl::nd_item<3> item_ct1) {
                                addBiasBatched_kernel<T, ACTIVATION_MISH>(
                                    output, input, bias, N, C, Nstride,
                                    item_ct1);
                              });

      break;
    case ACTIVATION_RELU:
      // addBiasBatched_kernel<T, ACTIVATION_RELU>
      //   <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
      //                                    Nstride);

      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                              [=](sycl::nd_item<3> item_ct1) {
                                addBiasBatched_kernel<T, ACTIVATION_RELU>(
                                    output, input, bias, N, C, Nstride,
                                    item_ct1);
                              });
      break;
    case ACTIVATION_SWISH:
      // addBiasBatched_kernel<T, ACTIVATION_SWISH>
      //     <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
      //                                        Nstride);

      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                              [=](sycl::nd_item<3> item_ct1) {
                                addBiasBatched_kernel<T, ACTIVATION_SWISH>(
                                    output, input, bias, N, C, Nstride,
                                    item_ct1);
                              });
      break;
    case ACTIVATION_RELU_2:  // square relu
                             // addBiasBatched_kernel<T, ACTIVATION_RELU_2>
                             //     <<<gridDim, blockDim, 0, stream>>>(output,
                             //     input, bias, N, C,
      //                                       Nstride);
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                              [=](sycl::nd_item<3> item_ct1) {
                                addBiasBatched_kernel<T, ACTIVATION_RELU_2>(
                                    output, input, bias, N, C, Nstride,
                                    item_ct1);
                              });

      break;
    default:
      throw Exception(
          "unsupported activation in addBiasBatched. Add in switch-case here");
  }
}

template <typename T>
void addBias_NCHW_kernel(T* c, T* a, T* b, int N, int C, int H, int W,
                         ActivationFunction activation,
                         const sycl::nd_item<3>& item_ct1) {
  int i = item_ct1.get_local_id(2) +
          item_ct1.get_local_range(2) * item_ct1.get_group(2);
  int size = N * C * H * W;

  if (i < size) {
    float aVal = (float)a[i];

    // All this math can be optimized, but the kernel is memory bound anyway.
    int biasIndex = (i / (H * W)) % C;
    float bVal = (float)b[biasIndex];

    float cVal = aVal + bVal;

    cVal = activate(cVal, activation);

    c[i] = (T)cVal;
  }
}

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W,
                  ActivationFunction activation, sycl::queue& sycl_queue) {
  int size = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        addBias_NCHW_kernel(c, a, b, N, C, H, W, activation, item_ct1);
      });
}

template <typename dT, typename sT>
dT readNCHW(const sT* input_tensor, int n, int c, int h, int w, int Nin,
            int Cin, int H, int W) {
  if (n >= Nin || c >= Cin) return 0;

  int index;
  index = n;
  index *= Cin;
  index += c;
  index *= H;
  index += h;
  index *= W;
  index += w;

  return (dT)(input_tensor[index]);
}

template <typename dT, typename sT>
void NCHWtoNHWC_kernel(dT* output_tensor, const sT* input_tensor, int Nin,
                       int Cin, int Nout, int Cout, int H, int W,
                       const sycl::nd_item<3>& item_ct1) {
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  if (tid >= Nout * Cout * H * W) return;

  int index = tid;

  int c = (index % Cout);
  index /= Cout;
  int w = index % W;
  index /= W;
  int h = index % H;
  index /= H;
  int n = index;

  output_tensor[tid] =
      readNCHW<dT, sT>(input_tensor, n, c, h, w, Nin, Cin, H, W);
}

template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W,
                       sycl::queue& sycl_queue) {
  size_t numElements = Nout * Cout * H * W;
  const int blockSize = 256;
  int blocks = DivUp(numElements, blockSize);
  sycl_queue.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                                sycl::range<3>(1, 1, blockSize),
                                            sycl::range<3>(1, 1, blockSize)),
                          [=](sycl::nd_item<3> item_ct1) {
                            NCHWtoNHWC_kernel(output_tensor, input_tensor, Nin,
                                              Cin, Nout, Cout, H, W, item_ct1);
                          });
}

template <typename DstType, typename SrcType>
void copyTypeConverted_kernel(DstType* op, SrcType* ip, int N,
                              const sycl::nd_item<3>& item_ct1) {
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  if (tid >= N) return;

  DstType el = (DstType)ip[tid];
  op[tid] = el;
}

template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N,
                       sycl::queue& sycl_queue) {
  const int kBlockSize = 256;
  int blocks = DivUp(N, kBlockSize);
  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        copyTypeConverted_kernel(op, ip, N, item_ct1);
      });
}

template <typename T>
void batchNorm_kernel(T* output, const T* input, const T* skipInput, int N,
                      int C, int H, int W, const float* means,
                      const float* varMultipliers,
                      ActivationFunction activation,
                      const sycl::nd_item<3>& item_ct1) {
  int index = item_ct1.get_local_id(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2);

  int wIndex = 0;
  if (sizeof(T) == sizeof(float))
    wIndex = (index / (H * W)) % C;  // NCHW for fp32.
  else
    wIndex = index % C;  // NHWC for fp16.

  float el = input[index];
  float mean = means[wIndex];
  float varMulti = varMultipliers[wIndex];

  el -= mean;
  el *= varMulti;

  if (skipInput) el += (float)skipInput[index];

  el = activate(el, activation);

  output[index] = (T)el;
}

// Every thread processes single element.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers,
               ActivationFunction activation, sycl::queue& sycl_queue) {
  const int total_elements = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(total_elements, kBlockSize);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        batchNorm_kernel(output, input, skipInput, N, C, H, W, means,
                         var_multipliers, activation, item_ct1);
      });
}

void expandPlanes_kernel_Fp32_NCHW(float* output, const uint64_t* masks,
                                   const float* values, int n,
                                   const sycl::nd_item<3>& item_ct1,
                                   uint64_t* shMasks, float* shVals) {
  // Block size of 256, same mask/val for 64 consecutive threads.
  constexpr int kNumShmemElements = 256 / 64;

  int index = item_ct1.get_local_id(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2);

  int planeIndex = index >> 6;

  if (planeIndex >= n) return;

  // Load inputs to shared memory.
  if (item_ct1.get_local_id(2) < kNumShmemElements) {
    shMasks[item_ct1.get_local_id(2)] =
        masks[planeIndex + item_ct1.get_local_id(2)];
    shVals[item_ct1.get_local_id(2)] =
        values[planeIndex + item_ct1.get_local_id(2)];
  }
  /*
  DPCT1113:53: Consider replacing
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "expandPlanes_kernel_Fp32_NCHW" is called
  in a multidimensional kernel.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  uint64_t mask = shMasks[item_ct1.get_local_id(2) >> 6];

  int sqIndex = index & 0x3F;
  float op = 0;

  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    op = shVals[item_ct1.get_local_id(2) >> 6];
  }
  output[index] = op;
}

void expandPlanes_Fp32_NCHW(float* output, const uint64_t* masks,
                            const float* values, int n,
                            sycl::queue& sycl_queue) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int blockSize = 256;
  int blocks = DivUp(threads, blockSize);

  sycl_queue.submit([&](sycl::handler& cgh) {
    /*
    DPCT1101:115: 'kNumShmemElements' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<uint64_t, 1> shMasks_acc_ct1(
        sycl::range<1>(4 /*kNumShmemElements*/), cgh);
    /*
    DPCT1101:116: 'kNumShmemElements' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<float, 1> shVals_acc_ct1(
        sycl::range<1>(4 /*kNumShmemElements*/), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, blockSize),
            sycl::range<3>(1, 1, blockSize)),
        [=](sycl::nd_item<3> item_ct1) {
          expandPlanes_kernel_Fp32_NCHW(output, masks, values, n, item_ct1,
                                        shMasks_acc_ct1.get_pointer(),
                                        shVals_acc_ct1.get_pointer());
        });
  });
}

// TODO: Can optimize using shared memory if this becomes a bottleneck.
void expandPlanes_kernel_Fp16_NHWC(sycl::half* output, const uint64_t* masks,
                                   const float* values, int n,
                                   const sycl::nd_item<3>& item_ct1) {
  const int index = item_ct1.get_local_id(2) +
                    item_ct1.get_local_range(2) * item_ct1.get_group(2);
  if (index >= n * 8 * 8) return;

  const int planeIndex = index % kInputPlanes;
  const int boardIndex = index / (kInputPlanes * 8 * 8);
  const int sqIndex = (index / kInputPlanes) & 0x3F;

  uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];

  sycl::half op = 0;
  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    float val = values[boardIndex * kInputPlanes + planeIndex];
    op = (sycl::half)val;
  }
  output[index] = op;
}

void expandPlanes_Fp16_NHWC(sycl::half* output, const uint64_t* masks,
                            const float* values, int n,
                            sycl::queue& sycl_queue) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);
  {
    sycl_queue.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
            sycl::range<3>(1, 1, kBlockSize)),
        [=](sycl::nd_item<3> item_ct1) {
          expandPlanes_kernel_Fp16_NHWC(output, masks, values, n, item_ct1);
        });
  }
}

void expandPlanes_kernel_Fp16_NCHW(sycl::half* output, const uint64_t* masks,
                                   const float* values, int n,
                                   const sycl::nd_item<3>& item_ct1,
                                   uint64_t* shMasks, sycl::half* shVals) {
  // block size of 256, same mask/val for 64 consecutive threads
  constexpr int kNumShmemElements = 256 / 64;

  int index = item_ct1.get_local_id(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2);

  int planeIndex = index >> 6;

  if (planeIndex >= n) return;

  // load inputs to shared memory
  if (item_ct1.get_local_id(2) < kNumShmemElements) {
    shMasks[item_ct1.get_local_id(2)] =
        masks[planeIndex + item_ct1.get_local_id(2)];
    shVals[item_ct1.get_local_id(2)] =
        values[planeIndex + item_ct1.get_local_id(2)];
  }
  /*
  DPCT1113:56: Consider replacing
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "expandPlanes_kernel_Fp16_NCHW" is called
  in a multidimensional kernel.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  uint64_t mask = shMasks[item_ct1.get_local_id(2) >> 6];

  int sqIndex = index & 0x3F;
  sycl::half op = 0;

  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    op = (sycl::half)shVals[item_ct1.get_local_id(2) >> 6];
  }
  output[index] = op;
}

void expandPlanes_Fp16_NCHW(sycl::half* output, const uint64_t* masks,
                            const float* values, int n,
                            sycl::queue& sycl_queue) {
  int threads = n * 8 * 8;  // each thread writes a single element
  const int blockSize = 256;
  int blocks = DivUp(threads, blockSize);
  {
    sycl_queue.submit([&](sycl::handler& cgh) {
      /*
      DPCT1101:117: 'kNumShmemElements' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<uint64_t, 1> shMasks_acc_ct1(
          sycl::range<1>(4 /*kNumShmemElements*/), cgh);
      /*
      DPCT1101:118: 'kNumShmemElements' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<sycl::half, 1> shVals_acc_ct1(
          sycl::range<1>(4 /*kNumShmemElements*/), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, blockSize),
              sycl::range<3>(1, 1, blockSize)),
          [=](sycl::nd_item<3> item_ct1) {
            expandPlanes_kernel_Fp16_NCHW(output, masks, values, n, item_ct1,
                                          shMasks_acc_ct1.get_pointer(),
                                          shVals_acc_ct1.get_pointer());
          });
    });
  }
}

template <typename T>
void globalScale_kernel(T* output, const T* input, const T* scaleBias,
                        const T* prevLayerBias, int inputSize, int C,
                        ActivationFunction activation,
                        const sycl::nd_item<3>& item_ct1) {
  const int kPlaneSize = 64;

  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  if (tid >= inputSize) return;

  int nc = tid / kPlaneSize;
  int n = nc / C;
  int c = nc % C;

  float val1 = input[tid];   // Output of residual block to be scaled.
  float val2 = output[tid];  // Skip connection to be added directly.

  if (prevLayerBias) {
    val1 += (float)(prevLayerBias[c]);
  }

  int startIdx = n * 2 * C;  // Scale and bias interleaved.

  float s = scaleBias[startIdx + c];
  s = 1.0f / (1.0f + sycl::exp(-s));  // Sigmoid on scale.

  float b = scaleBias[startIdx + c + C];

  float op = val1 * s + val2 + b;
  op = activate(op, activation);
  output[tid] = (T)op;
}

void globalScale_kernel_fp16_nhwc(sycl::half* output, const sycl::half* input,
                                  const sycl::half* scaleBias,
                                  const sycl::half* prevLayerBias,
                                  int inputSize, int C, int HWC,
                                  ActivationFunction activation,
                                  const sycl::nd_item<3>& item_ct1) {
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  if (tid >= inputSize) return;

  int c = tid % C;
  int n = tid / (HWC);

  float val1 = (float)input[tid];   // Output of residual block to be scaled.
  float val2 = (float)output[tid];  // Skip connection to be added directly.
  if (prevLayerBias) {
    val1 += (float)prevLayerBias[c];
  }

  int startIdx = n * 2 * C;  // Scale and bias interleaved.

  float s = scaleBias[startIdx + c];
  s = 1.0f / (1.0f + sycl::exp(-s));  // Sigmoid on scale.

  float b = scaleBias[startIdx + c + C];

  float op = val1 * s + val2 + b;
  op = activate(op, activation);

  output[tid] = (sycl::half)op;
}

// N blocks.
// C threads per block.
// 'HWC' input data processed by thread block.
// Each thread writes a single output.
void globalAvgPool_kernel_NHWC_fp16(sycl::half* output, const sycl::half* input,
                                    const sycl::half* prevLayerBias,
                                    int outputSize, int C,
                                    const sycl::nd_item<1>& item_ct1) {
  int global_id = item_ct1.get_global_id(0);
  if (global_id >= outputSize) return;

  int batch_idx = global_id / C;
  int c_idx = global_id % C;

  const int elementsPerThread = 64;  // 8x8 board.

  int blockStart = batch_idx * C;

  float S = 0;

#pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * C + c_idx;
    int inputIndex = blockStart * elementsPerThread + localIndex;
    S += (float)(input[inputIndex]);
  }

  float avg = S / elementsPerThread;

  // Add bias from previous layer.
  if (prevLayerBias) avg += (float)(prevLayerBias[c_idx]);

  output[global_id] = (sycl::half)avg;
}

// Sub-group size agnostic globalAvgPool kernel for NCHW layout.
template <typename T>
void globalAvgPool_kernel(T* output, const T* input, const T* prevLayerBias,
                          int inputSize, int outputSize, int C,
                          const sycl::nd_item<3>& item_ct1) {
  const int elementsPerPlane = 64;
  const int sg_size = item_ct1.get_sub_group().get_max_local_range()[0];

  int localId = item_ct1.get_local_id(2);
  int laneId = localId % sg_size;
  int subGroupId = localId / sg_size;
  int subGroupsPerBlock = item_ct1.get_local_range(2) / sg_size;
  int globalSubGroupId =
      (item_ct1.get_group(2) * subGroupsPerBlock) + subGroupId;

  int planeStartIndex = globalSubGroupId * elementsPerPlane;

  // Compute per-thread partial sum for the plane.
  float S = 0;
  for (int index = planeStartIndex + laneId;
       index < planeStartIndex + elementsPerPlane; index += sg_size) {
    if (index < inputSize) S += (float)(input[index]);
  }

  // Compute sub-group wide sum across all threads in sub-group.
  for (int offset = 1; offset < sg_size; offset *= 2) {
    S += sycl::shift_group_left(item_ct1.get_sub_group(), S, offset);
  }

  float avg = S / elementsPerPlane;

  // First thread in sub-group has the sum, write it to output.
  if (laneId == 0) {
    if (globalSubGroupId < outputSize) {
      if (prevLayerBias) avg += (float)prevLayerBias[globalSubGroupId % C];
      output[globalSubGroupId] = (T)avg;
    }
  }
}

template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc, sycl::queue& sycl_queue) {
  const int kPlaneSize = 64;

  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (nhwc) {
    assert(fp16);
    // For NHWC fp16, launch a 1D grid with N * C threads to process each
    // channel independently.
    int total_threads = N * C;
    int local_size = 256;
    int blocks = DivUp(total_threads, local_size);
    sycl_queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks * local_size),
                          sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item_ct1) {
          globalAvgPool_kernel_NHWC_fp16(
              (sycl::half*)output, (sycl::half*)input,
              (sycl::half*)prevLayerBias, total_threads, C, item_ct1);
        });
  } else {
    // For NCHW layout (used with fp32),
    // each sub-group processes a full plane (64 elements), and writes a single
    // average.
    const int kTotalSubGroups = N * C;
    const int kSubGroupsPerBlock = 8;
    const int kBlockSize = kSubGroupsPerBlock * 32;

    int blocks = DivUp(kTotalSubGroups, kSubGroupsPerBlock);
    sycl_queue.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
            sycl::range<3>(1, 1, kBlockSize)),
        [=](sycl::nd_item<3> item_ct1) {
          globalAvgPool_kernel(output, input, prevLayerBias, N * C * kPlaneSize,
                               N * C, C, item_ct1);
        });
  }
}

template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc,
                 ActivationFunction activation, sycl::queue& sycl_queue) {
  const bool fp16 = std::is_same<sycl::half, T>::value;

  // Each thread writes one output.
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * 8 * 8 * C, kBlockSize);

  if (nhwc) {
    assert(fp16);
    sycl_queue.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, kBlocks) * sycl::range<3>(1, 1, kBlockSize),
            sycl::range<3>(1, 1, kBlockSize)),
        [=](sycl::nd_item<3> item_ct1) {
          globalScale_kernel_fp16_nhwc(
              (sycl::half*)output, (sycl::half*)input, (sycl::half*)scaleBias,
              (sycl::half*)prevLayerBias, N * C * 8 * 8, C, 8 * 8 * C,
              activation, item_ct1);
        });
  } else {
    sycl_queue.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, kBlocks) * sycl::range<3>(1, 1, kBlockSize),
            sycl::range<3>(1, 1, kBlockSize)),
        [=](sycl::nd_item<3> item_ct1) {
          globalScale_kernel(output, input, scaleBias, prevLayerBias,
                             N * C * 8 * 8, C, activation, item_ct1);
        });
  }
}

template <typename T>
void policyMap_kernel(T* output, const T* input, const short* indices, int N,
                      int inputSize, int usedSize, int outputSize,
                      const sycl::nd_item<3>& item_ct1) {
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  int n = tid / usedSize;
  int i = tid % usedSize;

  if (n >= N) return;

  int j = indices[i];

  if (j >= 0) {
    output[n * outputSize + j] = input[n * inputSize + i];
  }
}

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize,
               sycl::queue& sycl_queue) {
  // Each thread processes one input element
  // Only some of the threads (with valid mapping) write output
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * usedSize, kBlockSize);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, kBlocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        policyMap_kernel<T>((T*)output, (T*)input, (short*)indices, N,
                            inputSize, usedSize, outputSize, item_ct1);
      });
}

template <typename T = float, bool use_se, ActivationFunction activation,
          bool use_bias, bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2,
                          sycl::queue& sycl_queue) {
  // Each thread processes entire chess board
  if (use_se == false) {
    sycl::range<3> grid_dim(1, N, DivUp(C, kOpInpTransformBlockSize));
    {
      sycl_queue.parallel_for(
          sycl::nd_range<3>(
              grid_dim * sycl::range<3>(1, 1, kOpInpTransformBlockSize),
              sycl::range<3>(1, 1, kOpInpTransformBlockSize)),
          [=](sycl::nd_item<3> item_ct1) {
            OutputTransform_relu_InputTransform_kernel<float, activation,
                                                       use_bias, use_skip>(
                N, C, output, input, (float*)skip, bias, item_ct1);
          });
    }
  } else if (C > kMaxResBlockFusingChannels) {
    throw Exception(
        "res block fusing opt not supported for the given data type and no "
        "of filters\n");
  } else {
    /*
    DPCT1049:12: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */

    sycl_queue.submit([&](sycl::handler& cgh) {
      /*
      DPCT1101:119: 'kMaxResBlockFusingChannels' expression was replaced
      with a value. Modify the code to use the original expression, provided
      in comments, if it is correct.
      */
      sycl::local_accessor<float, 1> shared_data_acc_ct1(
          sycl::range<1>(384 /*kMaxResBlockFusingChannels*/), cgh);
      /*
      DPCT1101:120: 'kMaxResBlockFusingChannels / 32' expression was
      replaced with a value. Modify the code to use the original expression,
      provided in comments, if it is correct.
      */
      /*
      DPCT1101:121: 'kMaxResBlockFusingSeK' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<float, 2> shared_sums_acc_ct1(
          sycl::range<2>(kMaxResBlockFusingChannels / SYCL_SUB_GROUP_SIZE,
                         se_K /*kMaxResBlockFusingSeK*/),
          cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                            sycl::range<3>(1, 1, C)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
              SYCL_SUB_GROUP_SIZE)]] {
            OutputTransform_SE_relu_InputTransform_kernel<float, activation,
                                                          use_bias, use_skip>(
                N, C, se_K, output, input, (float*)skip, bias, w1, b1, w2, b2,
                item_ct1, shared_data_acc_ct1.get_pointer(),
                shared_sums_acc_ct1);
          });
    });
  }
}

// softmax along C dimension which is assumed to be 64
// each thread processes two elements. Each warp computes a sum (over 64
// elements)
template <typename T>
void softmax_opt_64_kernel(T* output, const T* input, const T* input2, int N,
                           const sycl::nd_item<3>& item_ct1, int sg_size) {
  int index = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
  if (index >= N) return;

  float x[4];
  float ex[2];

  // Load from memory
  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (fp16) {
    sycl::half inp[2];
    copyAs<int>(&inp[0], &input[index * 2]);
    x[0] = (float)inp[0];
    x[1] = (float)inp[1];
    if (input2 != nullptr) {
      copyAs<int>(&inp[0], &input2[index * 2]);
      x[2] = (float)inp[0];
      x[3] = (float)inp[1];
    }
  } else {
    copyAs<sycl::uint2>(&x[0], &input[index * 2]);
    if (input2 != nullptr) {
      copyAs<sycl::uint2>(&x[2], &input2[index * 2]);
    }
  }

  if (input2 != nullptr) {
    x[0] += x[2];
    x[1] += x[3];
  }
  float threadMax = sycl::max(x[0], x[1]);
  float maxval = warpMax(threadMax, item_ct1);
  /*
  DPCT1023:13: The SYCL sub-group does not support mask options for
  dpct::select_from_sub_group. You can specify
  "--use-experimental-features=masked-sub-group-operation" to use the
  experimental helper function to migrate __shfl_sync.
  */
  maxval = sycl::select_from_group(item_ct1.get_sub_group(), maxval, 0);

  ex[0] = sycl::exp(x[0] - maxval);
  ex[1] = sycl::exp(x[1] - maxval);

  float threadSum = ex[0] + ex[1];
  float Sum = warpReduce(threadSum, item_ct1);
  /*
  DPCT1023:14: The SYCL sub-group does not support mask options for
  dpct::select_from_sub_group. You can specify
  "--use-experimental-features=masked-sub-group-operation" to use the
  experimental helper function to migrate __shfl_sync.
  */
  Sum = sycl::select_from_group(item_ct1.get_sub_group(), Sum, 0);

  ex[0] = ex[0] / Sum;
  ex[1] = ex[1] / Sum;

  // Store to memory
  if (fp16) {
    sycl::half op[2];
    op[0] = (sycl::half)ex[0];
    op[1] = (sycl::half)ex[1];
    copyAs<int>(&output[index * 2], &op[0]);
  } else {
    copyAs<sycl::uint2>(&output[index * 2], &ex[0]);
  }
}

// N * C Tensors
// performs softmax along the C dimension
// Each thread processes one element
// Sums are computed in shared memory
// C threads per block, N blocks
template <typename T>
void softmax_kernel(T* output, const T* input, const T* input2,
                    const sycl::nd_item<3>& item_ct1, float& localsum,
                    float& localmax, int sg_size) {
  int n = item_ct1.get_group(2);
  int c = item_ct1.get_local_id(2);
  int C = item_ct1.get_local_range(2);
  int index = n * C + c;
  sycl::atomic_ref<float, sycl::memory_order::relaxed,
                   sycl::memory_scope::work_group>
      maxval(localmax);
  sycl::atomic_ref<float, sycl::memory_order::relaxed,
                   sycl::memory_scope::work_group>
      sum(localsum);

  // softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)

  float x = (float)input[index];
  if (input2 != nullptr) x += (float)input2[index];

  if (c == 0) {
    sum = 0;
    maxval = x;
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Get max across warp first, and then update across C dimension
  float warpmax = warpMax(x, item_ct1);
  int sg_mask = sg_size - 1;
  if ((c & sg_mask) == 0) maxval.fetch_max(warpmax);

  item_ct1.barrier(sycl::access::fence_space::local_space);

  float ex = sycl::exp(x - maxval);

  // compute warp wide sums first
  float val = warpReduce(ex, item_ct1);

  // update shared memory sum across C dimension
  if ((c & sg_mask) == 0) sum.fetch_add(val);

  item_ct1.barrier(sycl::access::fence_space::local_space);

  float op = ex / sum;

  output[index] = (T)op;
}

template <typename T>
void Softmax(int N, int C, T* output, const T* input, const T* input2,
             sycl::queue& sycl_queue) {
  if (C == 64) {
    int sg_size = GetSubGroupSize(sycl_queue);
    int size = N * (64 / sg_size) * sg_size;  // Total no of threads needed
    const int kBlockSize = 256;
    int blocks = DivUp(size, kBlockSize);
    {
      sycl_queue.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
              sycl::range<3>(1, 1, kBlockSize)),
          [=](sycl::nd_item<3> item_ct1) {
            softmax_opt_64_kernel<T>(output, input, input2, size, item_ct1,
                                     sg_size);
          });
    }
  } else {
    /*
    DPCT1049:15: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    int sg_size = GetSubGroupSize(sycl_queue);
    sycl_queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 0> sum_acc_ct1(cgh);
      sycl::local_accessor<float, 0> maxval_acc_ct1(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                            sycl::range<3>(1, 1, C)),
          [=](sycl::nd_item<3> item_ct1) {
            softmax_kernel<T>(output, input, input2, item_ct1, sum_acc_ct1,
                              maxval_acc_ct1, sg_size);
          });
    });
  }
}

[[gnu::always_inline]]
inline float shared_sum_for_layer_norm(float x,
                                       const sycl::nd_item<1>& item_ct1,
                                       sycl::local_accessor<float, 1> sum) {
  auto sg = item_ct1.get_sub_group();
  float s = warpReduce(x, item_ct1);

  if (sg.get_local_linear_id() == 0) {
    sum[sg.get_group_linear_id()] = s;
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  if (item_ct1.get_local_linear_id() == 0) {
    float cSum = 0;
    for (uint32_t j = 0; j < sg.get_group_linear_range(); j++) cSum += sum[j];
    sum[0] = cSum;
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  float result = sum[0];
  item_ct1.barrier(sycl::access::fence_space::local_space);

  return result;
}

// 1. Perform Bias add, and skip add
// 2. Perform layer norm (normalize across C dimension)
template <typename T>
void layer_norm_kernel(int N, int C, T* output, const T* input, const T* bias,
                       const T* skip, const T* gammas, const T* betas, float ep,
                       float alpha, ActivationFunction act,
                       const sycl::nd_item<1>& item_ct1,
                       sycl::local_accessor<float, 1> sum) {
  int n = item_ct1.get_group(0);  // Batch index
  if (n >= N) return;
  int local_id = item_ct1.get_local_id(0);
  int local_range = item_ct1.get_local_range(0);

  const bool fp16 = std::is_same<sycl::half, T>::value;

  // 1. Compute mean
  float s = 0;
  for (int c = local_id * 16; c < C; c += local_range * 16) {
    int tensorIndex = n * C + c;
    int biasIndex = c;
    float val[16] = {0};
    float oth[16] = {0};

    if (fp16) {
      sycl::half inp[8];
      copyAs<sycl::uint4>(&inp[0], &input[tensorIndex]);
      for (int i = 0; i < 8; i++) val[i] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &input[tensorIndex + 8]);
      for (int i = 0; i < 8; i++) val[i + 8] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &bias[biasIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &bias[biasIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
      for (int i = 0; i < 16; i++) val[i] += oth[i];
    } else {
      copyAs<sycl::uint4>(&val[0], &input[tensorIndex]);
      copyAs<sycl::uint4>(&val[4], &input[tensorIndex + 4]);
      copyAs<sycl::uint4>(&val[8], &input[tensorIndex + 8]);
      copyAs<sycl::uint4>(&val[12], &input[tensorIndex + 12]);
      copyAs<sycl::uint4>(&oth[0], &bias[biasIndex]);
      copyAs<sycl::uint4>(&oth[4], &bias[biasIndex + 4]);
      copyAs<sycl::uint4>(&oth[8], &bias[biasIndex + 8]);
      copyAs<sycl::uint4>(&oth[12], &bias[biasIndex + 12]);
      for (int i = 0; i < 16; i++) val[i] += oth[i];
    }

    if (skip != nullptr) {
      if (fp16) {
        sycl::half inp[8];
        copyAs<sycl::uint4>(&inp[0], &skip[tensorIndex]);
        for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
        copyAs<sycl::uint4>(&inp[0], &skip[tensorIndex + 8]);
        for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
      } else {
        copyAs<sycl::uint4>(&oth[0], &skip[tensorIndex]);
        copyAs<sycl::uint4>(&oth[4], &skip[tensorIndex + 4]);
        copyAs<sycl::uint4>(&oth[8], &skip[tensorIndex + 8]);
        copyAs<sycl::uint4>(&oth[12], &skip[tensorIndex + 12]);
      }
    }

    if (skip != nullptr) {
      for (int i = 0; i < 16; i++) {
        val[i] = activate(val[i], act) * alpha + oth[i];
        s += val[i];
      }
    } else {
      for (int i = 0; i < 16; i++) {
        val[i] = activate(val[i], act) * alpha;
        s += val[i];
      }
    }
  }

  s = shared_sum_for_layer_norm(s, item_ct1, sum);
  float mean = s / C;

  // 2. Compute variance
  s = 0;
  for (int c = local_id * 16; c < C; c += local_range * 16) {
    int tensorIndex = n * C + c;
    int biasIndex = c;
    float val[16] = {0};
    float oth[16] = {0};

    if (fp16) {
      sycl::half inp[8];
      copyAs<sycl::uint4>(&inp[0], &input[tensorIndex]);
      for (int i = 0; i < 8; i++) val[i] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &input[tensorIndex + 8]);
      for (int i = 0; i < 8; i++) val[i + 8] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &bias[biasIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &bias[biasIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
      for (int i = 0; i < 16; i++) val[i] += oth[i];
    } else {
      copyAs<sycl::uint4>(&val[0], &input[tensorIndex]);
      copyAs<sycl::uint4>(&val[4], &input[tensorIndex + 4]);
      copyAs<sycl::uint4>(&val[8], &input[tensorIndex + 8]);
      copyAs<sycl::uint4>(&val[12], &input[tensorIndex + 12]);
      copyAs<sycl::uint4>(&oth[0], &bias[biasIndex]);
      copyAs<sycl::uint4>(&oth[4], &bias[biasIndex + 4]);
      copyAs<sycl::uint4>(&oth[8], &bias[biasIndex + 8]);
      copyAs<sycl::uint4>(&oth[12], &bias[biasIndex + 12]);
      for (int i = 0; i < 16; i++) val[i] += oth[i];
    }

    if (skip != nullptr) {
      if (fp16) {
        sycl::half inp[8];
        copyAs<sycl::uint4>(&inp[0], &skip[tensorIndex]);
        for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
        copyAs<sycl::uint4>(&inp[0], &skip[tensorIndex + 8]);
        for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
      } else {
        copyAs<sycl::uint4>(&oth[0], &skip[tensorIndex]);
        copyAs<sycl::uint4>(&oth[4], &skip[tensorIndex + 4]);
        copyAs<sycl::uint4>(&oth[8], &skip[tensorIndex + 8]);
        copyAs<sycl::uint4>(&oth[12], &skip[tensorIndex + 12]);
      }
    }

    if (skip != nullptr) {
      for (int i = 0; i < 16; i++) {
        val[i] = activate(val[i], act) * alpha + oth[i];
      }
    } else {
      for (int i = 0; i < 16; i++) {
        val[i] = activate(val[i], act) * alpha;
      }
    }

    for (int i = 0; i < 16; i++) {
      float d = val[i] - mean;
      float d_sq = d * d;
      s += d_sq;
    }
  }

  s = shared_sum_for_layer_norm(s, item_ct1, sum);
  float var = s / C;

  // 3. Normalize
  for (int c = local_id * 16; c < C; c += local_range * 16) {
    int tensorIndex = n * C + c;
    int biasIndex = c;
    float val[16] = {0};
    float oth[16] = {0};

    if (fp16) {
      sycl::half inp[8];
      copyAs<sycl::uint4>(&inp[0], &input[tensorIndex]);
      for (int i = 0; i < 8; i++) val[i] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &input[tensorIndex + 8]);
      for (int i = 0; i < 8; i++) val[i + 8] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &bias[biasIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &bias[biasIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
      for (int i = 0; i < 16; i++) val[i] += oth[i];
    } else {
      copyAs<sycl::uint4>(&val[0], &input[tensorIndex]);
      copyAs<sycl::uint4>(&val[4], &input[tensorIndex + 4]);
      copyAs<sycl::uint4>(&val[8], &input[tensorIndex + 8]);
      copyAs<sycl::uint4>(&val[12], &input[tensorIndex + 12]);
      copyAs<sycl::uint4>(&oth[0], &bias[biasIndex]);
      copyAs<sycl::uint4>(&oth[4], &bias[biasIndex + 4]);
      copyAs<sycl::uint4>(&oth[8], &bias[biasIndex + 8]);
      copyAs<sycl::uint4>(&oth[12], &bias[biasIndex + 12]);
      for (int i = 0; i < 16; i++) val[i] += oth[i];
    }

    if (skip != nullptr) {
      if (fp16) {
        sycl::half inp[8];
        copyAs<sycl::uint4>(&inp[0], &skip[tensorIndex]);
        for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
        copyAs<sycl::uint4>(&inp[0], &skip[tensorIndex + 8]);
        for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
      } else {
        copyAs<sycl::uint4>(&oth[0], &skip[tensorIndex]);
        copyAs<sycl::uint4>(&oth[4], &skip[tensorIndex + 4]);
        copyAs<sycl::uint4>(&oth[8], &skip[tensorIndex + 8]);
        copyAs<sycl::uint4>(&oth[12], &skip[tensorIndex + 12]);
      }
    }

    if (skip != nullptr) {
      for (int i = 0; i < 16; i++) {
        val[i] = activate(val[i], act) * alpha + oth[i];
      }
    } else {
      for (int i = 0; i < 16; i++) {
        val[i] = activate(val[i], act) * alpha;
      }
    }

    if (fp16) {
      sycl::half inp[8];
      copyAs<sycl::uint4>(&inp[0], &gammas[biasIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &gammas[biasIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
    } else {
      copyAs<sycl::uint4>(&oth[0], &gammas[biasIndex]);
      copyAs<sycl::uint4>(&oth[4], &gammas[biasIndex + 4]);
      copyAs<sycl::uint4>(&oth[8], &gammas[biasIndex + 8]);
      copyAs<sycl::uint4>(&oth[12], &gammas[biasIndex + 12]);
    }

    for (int i = 0; i < 16; i++) {
      float d = val[i] - mean;
      float norm = d / sycl::sqrt(var + ep);
      float op = norm * oth[i];
      val[i] = op;
    }

    if (fp16) {
      sycl::half inp[8];
      copyAs<sycl::uint4>(&inp[0], &betas[biasIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<sycl::uint4>(&inp[0], &betas[biasIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
    } else {
      copyAs<sycl::uint4>(&oth[0], &betas[biasIndex]);
      copyAs<sycl::uint4>(&oth[4], &betas[biasIndex + 4]);
      copyAs<sycl::uint4>(&oth[8], &betas[biasIndex + 8]);
      copyAs<sycl::uint4>(&oth[12], &betas[biasIndex + 12]);
    }

    for (int i = 0; i < 16; i++) {
      val[i] += oth[i];
    }

    if (fp16) {
      sycl::half op[8];
      for (int i = 0; i < 8; i++) op[i] = (sycl::half)val[i];
      copyAs<sycl::uint4>(&output[tensorIndex], &op[0]);
      for (int i = 0; i < 8; i++) op[i] = (sycl::half)val[i + 8];
      copyAs<sycl::uint4>(&output[tensorIndex + 8], &op[0]);
    } else {
      copyAs<sycl::uint4>(&output[tensorIndex], &val[0]);
      copyAs<sycl::uint4>(&output[tensorIndex + 4], &val[4]);
      copyAs<sycl::uint4>(&output[tensorIndex + 8], &val[8]);
      copyAs<sycl::uint4>(&output[tensorIndex + 12], &val[12]);
    }
  }
}

// add (optional) skip connection to input, and then perform Layer normalization
// normalization is done across C dimension (i.e, sums and std deviations taken
// over elements in C dim)
template <typename T>
void LayerNorm(int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act, sycl::queue& sycl_queue) {
  // process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 16 != 0) throw Exception("unsupported filter size");
  if (C > 8192) throw Exception("unsupported filter size");

  // Max 256 threads per batch to respect max_work_group_size on most Intel
  // devices.
  int threads_per_batch = std::min(C / 16, 256);
  sycl::range<1> blockDim(threads_per_batch);
  sycl::range<1> gridDim(N * threads_per_batch);

  sycl_queue.submit([&](sycl::handler& cgh) {
    // Need enough space for max subgroups (256 / 8 = 32 max subgroups)
    sycl::local_accessor<float, 1> sum_acc_ct1(sycl::range<1>(64), cgh);

    cgh.parallel_for(sycl::nd_range<1>(gridDim, blockDim),
                     [=](sycl::nd_item<1> item_ct1)
                         [[intel::reqd_sub_group_size(SYCL_SUB_GROUP_SIZE)]] {
                           layer_norm_kernel<T>(N, C, output, input, bias, skip,
                                                gammas, betas, ep, alpha, act,
                                                item_ct1, sum_acc_ct1);
                         });
  });
}

// Compute promotion logits in a single kernel
// keys matrix is of N * 64 * C (but we use only last 8 from the 'rows'
// dimension, so N * 8 * C)
// ppo matrix is 4 * C (weights for dense layer / matrix multiplication)
// policy_attn_logits matrix is N * 64 * 64, but we use only 8x8 part of it
// from each batch dimension (so, N * 8 * 8)
// output matrix (promotion logits) is of N * 8 * 24 size
template <typename T>
void promotion_logits_kernel(int C, T* output, const T* keys, const T* ppo,
                             const T* policy_attn_logits,
                             const sycl::nd_item<3>& item_ct1,
                             sycl::local_accessor<float, 2> promotion_offsets) {
  constexpr int output_stride = 64 * 64 + 8 * 24;
  int n = item_ct1.get_group(2);     // [0..N)
  int y = item_ct1.get_local_id(1);  // [0..8)
  int x = item_ct1.get_local_id(2);  // [0..24)     // Can split into 8 * 3

  int threadInGroup = item_ct1.get_local_id(1) * 24 + item_ct1.get_local_id(2);

  // phase 1 : compute promotion_offsets by multiplying keys and ppo matrices
  // Each of the 32 matrix elements is computed cooperatively by all 32 threads:
  // thread `t` accumulates partial sums from positions i = t, t+32, t+64, ...
  // then a group reduce collapses them. This ensures fully coalesced reads
  // across adjacent threads (stride-1 access over i).
  const T* keys_start =
      keys + n * 64 * C + C * 56;  // we are interested only in last 8 out of 64
                                   // 'rows' of keys matrix

  if (threadInGroup < 32) {
    int elem_x = threadInGroup % 4;  // which ppo row (0..3)
    int elem_y = threadInGroup / 4;  // which keys row (0..7)

    // Each thread accumulates partial sums for its own unique (elem_x, elem_y)
    // element. We stride the inner loop by 32 so that adjacent threads (with
    // consecutive threadInGroup values) access consecutive memory addresses on
    // each iteration, yielding fully coalesced global loads.
    float S = 0;
    for (int i = threadInGroup; i < C; i += 32) {
      float a = (float)keys_start[elem_y * C + i];
      float b = (float)ppo[elem_x * C + i];  // weight matrix is col-major
      S += a * b;
    }

    // write the accumulated dot product into shared memory
    promotion_offsets[elem_x][elem_y] = S;
  }

  /*
  DPCT1065:69: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // phase 2: add the last "row" to the other 3
  // #knight offset is added to the other three
  // promotion_offsets = promotion_offsets[:, :3, :] + promotion_offsets[:, 3:4,
  // :]
  // Only 24 threads in the group are active in this phase
  if (threadInGroup < 32) {
    int x = threadInGroup % 4;
    int y = threadInGroup / 4;
    if (x < 3) {
      promotion_offsets[x][y] += promotion_offsets[3][y];
    }
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // phase 3: add 8x8 chunk of policy_attn_logits matrix to promotion offsets
  //          the output is 3x8x8 (written as 8 * 24)
  // All threads are active in this phase and they compute one element each
  int w = x / 3;
  int c = x % 3;

  // n_promo_logits = matmul_qk[:, -16:-8, -8:]  # default traversals from rank
  // 7 to rank 8
  float n_promo_logit =
      (float)policy_attn_logits[n * output_stride + (48 + y) * 64 + (56 + w)];
  float promo_offset = promotion_offsets[c][w];

  float op = n_promo_logit + promo_offset;

  output[n * output_stride + threadInGroup] = (T)op;
}

template <typename T>
void ComputePromotionLogits(int N, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits,
                            sycl::queue& sycl_queue) {
  // N blocks
  // 8 * 24 threads
  // Each thread computes a single output element
  sycl::range<3> blockDim(1, 8, 24);
  sycl_queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 2> promotion_offsets_acc_ct1(
        sycl::range<2>(4, 8), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, N) * blockDim, blockDim),
        [=](sycl::nd_item<3> item_ct1) {
          promotion_logits_kernel<T>(C, output, keys, ppo, policy_attn_logits,
                                     item_ct1, promotion_offsets_acc_ct1);
        });
  });
}

template <typename T>
void preprocess_for_attention_body_kernel(T* output, const T* input,
                                          const T* encoding, int input_size,
                                          int encoding_size,
                                          bool is_pe_dense_embedding,
                                          const sycl::nd_item<3>& item_ct1) {
  int n = item_ct1.get_group(2);
  int hw = item_ct1.get_group(1);
  int c = item_ct1.get_local_id(2) +
          item_ct1.get_local_range(2) * item_ct1.get_group(0);
  if (c >= input_size + encoding_size) return;

  T op;
  if (c >= input_size) {
    // concatenate from position encoding array
    if (is_pe_dense_embedding) {
      op = (T)(encoding[n * 64 * encoding_size + hw * encoding_size +
                        (c - input_size)]);
    } else {
      op = (T)(encoding[64 * hw + (c - input_size)]);
    }
  } else {
    op = input[n * input_size * 64 + c * 64 + hw];  // nchw
  }

  int outputC = input_size + encoding_size;

  // convert to nhwc
  output[n * 64 * outputC + hw * outputC + c] = op;
}

template <typename T>
void inputPreprocessForAttentionBody(T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size,
                                     bool is_pe_dense_embedding,
                                     sycl::queue& sycl_queue) {
  // N * 64 blocks
  // (kInputPlanes + kNumPosEncodingChannels) threads
  // Each thread computes a single output element
  sycl::range<3> gridSize = sycl::range<3>(1, 64, N);
  sycl::range<3> blockSize(1, 1, 1);
  blockSize[2] = sycl::min(input_size + encoding_size, 512);
  blockSize[1] = 1;
  blockSize[0] = 1;
  gridSize[0] = DivUp(input_size + encoding_size, blockSize[2]);

  sycl_queue.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<3> item_ct1) {
                            preprocess_for_attention_body_kernel<T>(
                                output, input, encoding, input_size,
                                encoding_size, is_pe_dense_embedding, item_ct1);
                          });
}

template <typename T>
void input_gating_kernel(T* output, const T* input, const T* mult, const T* add,
                         int HW, int C, const sycl::nd_item<3>& item_ct1) {
  int n_offset = item_ct1.get_group(0) * HW * C;
  int idx = item_ct1.get_local_id(1) * C +
            item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);  // index in input
  int idxT = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2)) *
                 HW +
             item_ct1.get_local_id(
                 1);  // index in transposed weights arrays mult and add.

  if (idx < HW * C) {
    // Combine multiply gating, add gating and weights transpose.
    float op =
        (float)input[n_offset + idx] * (float)mult[idxT] + (float)add[idxT];
    output[n_offset + idx] = (T)op;
  }
}

template <typename T>
void applyInputGating(T* output, const T* input, const T* mult, const T* add,
                      int N, int HW, int C, sycl::queue& sycl_queue) {
  // Multiple blocks to fit into each input area / volume
  // Block x position indicates horizontal section of area
  // Block y position indicates batch
  // Each thread computes a single output element
  sycl::range<3> blockSize(1, 1, 1), gridSize(1, 1, 1);
  blockSize[2] = DivUp(512, HW);
  blockSize[1] = HW;
  blockSize[0] = 1;
  gridSize[2] = DivUp(C, blockSize[2]);
  gridSize[1] = 1;
  gridSize[0] = N;

  sycl_queue.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize),
                          [=](sycl::nd_item<3> item_ct1) {
                            input_gating_kernel<T>(output, input, mult, add, HW,
                                                   C, item_ct1);
                          });
}

template <typename T, int kWorkPerThread>
static void genOffsetPointers_kernel(T** offsets, int heads, int block_size,
                                     int depth, int d_model, T* k, T* q, T* b1,
                                     T* v, T* b2,
                                     const sycl::nd_item<1>& item_ct) {
  const int i = item_ct.get_global_id(0) * kWorkPerThread;
  if (i >= block_size) return;
  const int h = i % heads;
  const int n = i / heads;
  int w;
  T* res[kWorkPerThread];
  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = k + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = q + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = b1 + i * 64 * 64 + w * 64 * 64;
    offsets[i + w + 2 * block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = v + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + 3 * block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = b2 + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + 4 * block_size] = res[w];
  }
}

template <typename T>
void genOffsetPointers(T** offsets, int heads, int max_batch, int depth,
                       int d_model, T* k, T* q, T* b1, T* v, T* b2,
                       sycl::queue& sycl_queue) {
  const int block_size = heads * max_batch;
  // Process two elements per thread to use 128 bit store instructions.
  constexpr int kWorkPerThread = 2;
  constexpr int kWorkGroupSize = 128;
  if (block_size % kWorkPerThread != 0) {
    // Handle odd block sizes.
    sycl::range<1> global(DivUp(block_size, kWorkGroupSize));
    sycl::range<1> local(kWorkGroupSize);
    sycl_queue.parallel_for(sycl::nd_range<1>(global * local, local),
                            [=](sycl::nd_item<1> item_ct) {
                              genOffsetPointers_kernel<T, 1>(
                                  offsets, heads, block_size, depth, d_model, k,
                                  q, b1, v, b2, item_ct);
                            });
  } else {
    // Handle even block size
    sycl::range<1> global(DivUp(block_size, kWorkGroupSize * kWorkPerThread));
    sycl::range<1> local(kWorkGroupSize);
    sycl_queue.parallel_for(sycl::nd_range<1>(global * local, local),
                            [=](sycl::nd_item<1> item_ct) {
                              genOffsetPointers_kernel<T, kWorkPerThread>(
                                  offsets, heads, block_size, depth, d_model, k,
                                  q, b1, v, b2, item_ct);
                            });
  }
}

// Template instantiation.
template void copyTypeConverted<sycl::half, float>(sycl::half* op, float* ip,
                                                   int N,
                                                   sycl::queue& sycl_queue);
template void copyTypeConverted<float, sycl::half>(float* op, sycl::half* ip,
                                                   int N,
                                                   sycl::queue& sycl_queue);
template void copyTypeConverted<float, float>(float* op, float* ip, int N,
                                              sycl::queue& sycl_queue);
template void copyTypeConverted<sycl::half, sycl::half>(
    sycl::half* op, sycl::half* ip, int N, sycl::queue& sycl_queue);

template void batchNorm<float>(float* output, const float* input,
                               const float* skipInput, int N, int C, int H,
                               int W, float* means, float* var_multipliers,
                               ActivationFunction activation,
                               sycl::queue& sycl_queue);

template void batchNorm<sycl::half>(sycl::half* output, const sycl::half* input,
                                    const sycl::half* skipInput, int N, int C,
                                    int H, int W, float* means,
                                    float* var_multipliers,
                                    ActivationFunction activation,
                                    sycl::queue& sycl_queue);

template void addVectors<float>(float* c, float* a, float* b, int size,
                                int asize, int bsize, ActivationFunction act,
                                sycl::queue& sycl_queue);

template void addVectors<sycl::half>(sycl::half* c, sycl::half* a,
                                     sycl::half* b, int size, int asize,
                                     int bsize, ActivationFunction act,
                                     sycl::queue& sycl_queue);

template void addVectorsHNC_NHC<float>(float* a, float* b, int N, int H, int C,
                                       sycl::queue& sycl_queue);
template void addVectorsHNC_NHC<sycl::half>(sycl::half* a, sycl::half* b, int N,
                                            int H, int C,
                                            sycl::queue& sycl_queue);

template void addBiasBatched<float>(float* output, const float* input,
                                    const float* bias, int Batch, int N, int C,
                                    ActivationFunction activation,
                                    sycl::queue& sycl_queue);

template void addBiasBatched<sycl::half>(sycl::half* output,
                                         const sycl::half* input,
                                         const sycl::half* bias, int Batch,
                                         int N, int C,
                                         ActivationFunction activation,
                                         sycl::queue& sycl_queue);

template void addBiasBatched<float>(float* output, const float* input,
                                    const float* bias, int Batch, int N, int C,
                                    int Nstride, ActivationFunction activation,
                                    sycl::queue& sycl_queue);

template void addBiasBatched<sycl::half>(sycl::half* output,
                                         const sycl::half* input,
                                         const sycl::half* bias, int Batch,
                                         int N, int C, int Nstride,
                                         ActivationFunction activation,
                                         sycl::queue& sycl_queue);

template void addBias_NCHW<float>(float* c, float* a, float* b, int N, int C,
                                  int H, int W, ActivationFunction activation,
                                  sycl::queue& sycl_queue);

template void addBias_NCHW<sycl::half>(sycl::half* c, sycl::half* a,
                                       sycl::half* b, int N, int C, int H,
                                       int W, ActivationFunction activation,
                                       sycl::queue& sycl_queue);

template void globalAvgPool<float>(int N, int C, float* output,
                                   const float* input,
                                   const float* prevLayerBias, bool nhwc,
                                   sycl::queue& sycl_queue);

template void globalAvgPool<sycl::half>(int N, int C, sycl::half* output,
                                        const sycl::half* input,
                                        const sycl::half* prevLayerBias,
                                        bool nhwc, sycl::queue& sycl_queue);

template void globalScale<float>(int N, int C, float* output,
                                 const float* input, const float* scaleBias,
                                 const float* prevLayerBias, bool nhwc,
                                 ActivationFunction activation,
                                 sycl::queue& sycl_queue);

template void globalScale<sycl::half>(int N, int C, sycl::half* output,
                                      const sycl::half* input,
                                      const sycl::half* scaleBias,
                                      const sycl::half* prevLayerBias,
                                      bool nhwc, ActivationFunction activation,
                                      sycl::queue& sycl_queue);

template void PolicyMap<float>(int N, float* output, const float* input,
                               const short* indices, int inputSize,
                               int usedSize, int outputSize,
                               sycl::queue& sycl_queue);

template void PolicyMap<sycl::half>(int N, sycl::half* output,
                                    const sycl::half* input,
                                    const short* indices, int inputSize,
                                    int usedSize, int outputSize,
                                    sycl::queue& sycl_queue);

template void FilterTransform<float>(int N, int C, float* transformedFilter,
                                     const float* filter,
                                     sycl::queue& sycl_queue);

template void InputTransform<float, true>(int N, int C,
                                          float* transformed_input,
                                          const float* input,
                                          sycl::queue& sycl_queue);

template void InputTransform<float, false>(int N, int C,
                                           float* transformed_input,
                                           const float* input,
                                           sycl::queue& sycl_queue);

template void OutputTransform<float, true, ACTIVATION_RELU, true, true, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue& sycl_queue);

template void
OutputTransform<float, false, ACTIVATION_RELU, true, true, false, false>(

    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, true, ACTIVATION_RELU, true, true, true,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, false, ACTIVATION_RELU, true, true, true,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, false, ACTIVATION_RELU, true, false, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, false, ACTIVATION_RELU, true, false, false,
                              true>(int N, int C, int se_K, float* output,
                                    const float* input, const float* skip,
                                    const float* bias, const float* w1,
                                    const float* b1, const float* w2,
                                    const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, true, ACTIVATION_RELU, true, true, true,
                              true>(int N, int C, int se_K, float* output,
                                    const float* input, const float* skip,
                                    const float* bias, const float* w1,
                                    const float* b1, const float* w2,
                                    const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, true, ACTIVATION_MISH, true, true, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, false, ACTIVATION_MISH, true, true, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, true, ACTIVATION_MISH, true, true, true,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, false, ACTIVATION_MISH, true, true, true,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, false, ACTIVATION_MISH, true, false, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, false, ACTIVATION_MISH, true, false, false,
                              true>(int N, int C, int se_K, float* output,
                                    const float* input, const float* skip,
                                    const float* bias, const float* w1,
                                    const float* b1, const float* w2,
                                    const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, true, ACTIVATION_MISH, true, true, true,
                              true>(int N, int C, int se_K, float* output,
                                    const float* input, const float* skip,
                                    const float* bias, const float* w1,
                                    const float* b1, const float* w2,
                                    const float* b2, sycl::queue& sycl_queue);

template void OutputTransform<float, false, ACTIVATION_NONE, true, false, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, sycl::queue& sycl_queue);

template void OutputInputTransform<float, true, ACTIVATION_RELU, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue& sycl_queue);

template void OutputInputTransform<float, false, ACTIVATION_RELU, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue& sycl_queue);

template void OutputInputTransform<float, false, ACTIVATION_RELU, true, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue& sycl_queue);

template void OutputInputTransform<float, true, ACTIVATION_MISH, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue& sycl_queue);

template void OutputInputTransform<float, false, ACTIVATION_MISH, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue& sycl_queue);

template void OutputInputTransform<float, false, ACTIVATION_MISH, true, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, sycl::queue& sycl_queue);

template void Softmax<sycl::half>(int N, int C, sycl::half* output,
                                  const sycl::half* input,
                                  const sycl::half* input2,
                                  sycl::queue& sycl_queue);

template void Softmax<float>(int N, int C, float* output, const float* input,
                             const float* input2, sycl::queue& sycl_queue);

template void LayerNorm<sycl::half>(
    int N, int C, sycl::half* output, const sycl::half* input,
    const sycl::half* bias, const sycl::half* skip, const sycl::half* gammas,
    const sycl::half* betas, float ep, float alpha, ActivationFunction act,
    sycl::queue& sycl_queue);

template void LayerNorm<float>(int N, int C, float* output, const float* input,
                               const float* bias, const float* skip,
                               const float* gammas, const float* betas,
                               float ep, float alpha, ActivationFunction act,
                               sycl::queue& sycl_queue);

template void ComputePromotionLogits<sycl::half>(
    int N, int C, sycl::half* output, const sycl::half* keys,
    const sycl::half* ppo, const sycl::half* policy_attn_logits,
    sycl::queue& sycl_queue);

template void ComputePromotionLogits<float>(int N, int C, float* output,
                                            const float* keys, const float* ppo,
                                            const float* policy_attn_logits,
                                            sycl::queue& sycl_queue);

template void convertNCHWtoNHWC<sycl::half, float>(sycl::half* output_tensor,
                                                   const float* input_tensor,
                                                   int Nin, int Cin, int Nout,
                                                   int Cout, int H, int W,
                                                   sycl::queue& sycl_queue);

template void convertNCHWtoNHWC<float, float>(float* output_tensor,
                                              const float* input_tensor,
                                              int Nin, int Cin, int Nout,
                                              int Cout, int H, int W,
                                              sycl::queue& sycl_queue);

template void convertNCHWtoNHWC<sycl::half, sycl::half>(
    sycl::half* output_tensor, const sycl::half* input_tensor, int Nin, int Cin,
    int Nout, int Cout, int H, int W, sycl::queue& sycl_queue);

template void inputPreprocessForAttentionBody<sycl::half>(
    sycl::half* output, const sycl::half* input, const sycl::half* encoding,
    int N, int input_size, int encoding_size, bool is_pe_dense_embedding,
    sycl::queue& sycl_queue);

template void inputPreprocessForAttentionBody<float>(
    float* output, const float* input, const float* encoding, int N,
    int input_size, int encoding_size, bool is_pe_dense_embedding,
    sycl::queue& sycl_queue);

template void applyInputGating<sycl::half>(sycl::half* output,
                                           const sycl::half* input,
                                           const sycl::half* mult,
                                           const sycl::half* add, int N, int C,
                                           int output_size,
                                           sycl::queue& sycl_queue);

template void applyInputGating<float>(float* output, const float* input,
                                      const float* mult, const float* add,
                                      int N, int C, int output_size,
                                      sycl::queue& sycl_queue);

template void genOffsetPointers<float>(float** offsets, int heads,
                                       int max_batch, int depth, int d_model,
                                       float* k, float* q, float* b1, float* v,
                                       float* b2, sycl::queue& sycl_queue);

template void genOffsetPointers<sycl::half>(sycl::half** offsets, int heads,
                                            int max_batch, int depth,
                                            int d_model, sycl::half* k,
                                            sycl::half* q, sycl::half* b1,
                                            sycl::half* v, sycl::half* b2,
                                            sycl::queue& sycl_queue);

template <typename T>
void kdaRecurrenceValueParallel(
    int N, int heads, int key_dim, int value_dim, int direction_count,
    const std::array<int, 16>& directions, float log_decay_floor, const T* qkv,
    int qkv_stride, const T* q, const T* k, const T* v, const T* raw_decay,
    const T* dt_bias, const T* a_log, const T* beta, T* mixed,
    sycl::queue& sycl_queue) {
  sycl::range<2> global_range(N * heads, value_dim);
  sycl::range<2> local_range(1, value_dim);

  sycl_queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> p_q(sycl::range<1>(key_dim), cgh);
    sycl::local_accessor<float, 1> p_k(sycl::range<1>(key_dim), cgh);
    sycl::local_accessor<float, 1> p_decay(sycl::range<1>(key_dim), cgh);

    cgh.parallel_for(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> item) {
          const int local_id = static_cast<int>(item.get_local_id(1));
          const int global_batch_head = static_cast<int>(item.get_global_id(0));
          const int batch = global_batch_head / heads;
          const int head = global_batch_head % heads;

          const int direction_index = head / (heads / direction_count);
          const int direction = directions[direction_index];
          // Resolved once per work-item for the whole kernel invocation --
          // direction is invariant across the token loop below, so the
          // square-order row is looked up here instead of re-branching on
          // `direction` on every one of the loop's 64 iterations. Direction
          // is validated to be in [1, 8] at layer construction (see
          // EncoderBlock's constructor), so no bounds check is needed here.
          const int* const square_order = kKdaDirectionOrder[direction - 1];

          const float scale = 1.0f / sycl::sqrt(static_cast<float>(key_dim));
          const float decay_scale = sycl::exp(static_cast<float>(a_log[head]));
          const int key_depth = heads * key_dim;
          const int value_depth = heads * value_dim;

          float state[32];
          for (int i = 0; i < key_dim && i < 32; ++i) {
            state[i] = 0.0f;
          }

          for (int token = 0; token < 64; ++token) {
            const int square = square_order[token];

            const int token_idx = batch * 64 + square;
            const T* q_ptr;
            const T* k_ptr;
            const T* v_ptr;

            if (qkv != nullptr) {
              q_ptr = qkv + token_idx * qkv_stride + head * key_dim;
              k_ptr = qkv + token_idx * qkv_stride + key_depth + head * key_dim;
              v_ptr = qkv + token_idx * qkv_stride + 2 * key_depth +
                      head * value_dim;
            } else {
              q_ptr = q + token_idx * key_depth + head * key_dim;
              k_ptr = k + token_idx * key_depth + head * key_dim;
              v_ptr = v + token_idx * value_depth + head * value_dim;
            }

            const int raw_decay_offset = token_idx * key_depth + head * key_dim;
            const int value_offset = token_idx * value_depth + head * value_dim;

            // Strided rather than a single `local_id < key_dim` guard: the
            // work-group only has value_dim lanes, so when key_dim >
            // value_dim a single pass would leave p_q/p_k/p_decay[value_dim,
            // key_dim) unwritten and later read as garbage.
            for (int i = local_id; i < key_dim; i += value_dim) {
              p_q[i] = static_cast<float>(q_ptr[i]);
              p_k[i] = static_cast<float>(k_ptr[i]);

              const float decay_input =
                  static_cast<float>(raw_decay[raw_decay_offset + i]) +
                  static_cast<float>(dt_bias[head * key_dim + i]);
              const float softplus =
                  (decay_input > 0.0f ? decay_input : 0.0f) +
                  sycl::log1p(sycl::exp(-sycl::fabs(decay_input)));
              const float log_decay =
                  sycl::fmax(-decay_scale * softplus, log_decay_floor);
              p_decay[i] = sycl::exp(log_decay);
            }

            item.barrier(sycl::access::fence_space::local_space);

            float q_norm_sq = 0.0f;
            float k_norm_sq = 0.0f;
            for (int key = 0; key < key_dim; ++key) {
              q_norm_sq += p_q[key] * p_q[key];
              k_norm_sq += p_k[key] * p_k[key];
            }
            const float q_norm =
                1.0f / sycl::sqrt(q_norm_sq > 1.0e-12f ? q_norm_sq : 1.0e-12f);
            const float k_norm =
                1.0f / sycl::sqrt(k_norm_sq > 1.0e-12f ? k_norm_sq : 1.0e-12f);

            for (int key = 0; key < key_dim; ++key) {
              state[key] *= p_decay[key];
            }

            const float beta_value =
                static_cast<float>(beta[token_idx * heads + head]);
            const float update_rate = 1.0f / (1.0f + sycl::exp(-beta_value));

            float prediction = 0.0f;
            for (int key = 0; key < key_dim; ++key) {
              prediction += p_k[key] * k_norm * state[key];
            }

            const float delta =
                update_rate *
                (static_cast<float>(v_ptr[local_id]) - prediction);

            float output = 0.0f;
            for (int key = 0; key < key_dim; ++key) {
              state[key] += p_k[key] * k_norm * delta;
              output += p_q[key] * q_norm * scale * state[key];
            }

            mixed[value_offset + local_id] = static_cast<T>(output);

            // p_q/p_k/p_decay are read by every work-item above (the
            // q_norm_sq/k_norm_sq/prediction/output loops over key_dim), but
            // only written by the first key_dim work-items at the top of the
            // next iteration. Without this barrier, a writer thread can loop
            // back and overwrite them before a slower reader thread has
            // finished consuming this token's values -- nothing upstream
            // guarantees the work-items in this group stay in lockstep.
            item.barrier(sycl::access::fence_space::local_space);
          }
        });
  });
}

template void kdaRecurrenceValueParallel<float>(
    int N, int heads, int key_dim, int value_dim, int direction_count,
    const std::array<int, 16>& directions, float log_decay_floor,
    const float* qkv, int qkv_stride, const float* q, const float* k,
    const float* v, const float* raw_decay, const float* dt_bias,
    const float* a_log, const float* beta, float* mixed,
    sycl::queue& sycl_queue);

template void kdaRecurrenceValueParallel<sycl::half>(
    int N, int heads, int key_dim, int value_dim, int direction_count,
    const std::array<int, 16>& directions, float log_decay_floor,
    const sycl::half* qkv, int qkv_stride, const sycl::half* q,
    const sycl::half* k, const sycl::half* v, const sycl::half* raw_decay,
    const sycl::half* dt_bias, const sycl::half* a_log, const sycl::half* beta,
    sycl::half* mixed, sycl::queue& sycl_queue);

// Applies the sigmoid output gate to the mixer output. Must run after the
// optional RMS norm (EvalKda), not before: the trainer and BLAS reference
// both normalize first and gate second, and the two do not commute since the
// gate rescales each element before the norm would divide by the resulting
// RMS.
template <typename T>
void applyKdaOutputGate(int N, T* mixed, const T* gate,
                        sycl::queue& sycl_queue) {
  sycl_queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
    const int i = static_cast<int>(idx[0]);
    const float gate_value = static_cast<float>(gate[i]);
    const float sigmoid = 1.0f / (1.0f + sycl::exp(-gate_value));
    mixed[i] = static_cast<T>(static_cast<float>(mixed[i]) * sigmoid);
  });
}

template void applyKdaOutputGate<float>(int N, float* mixed,
                                        const float* gate,
                                        sycl::queue& sycl_queue);
template void applyKdaOutputGate<sycl::half>(int N, sycl::half* mixed,
                                             const sycl::half* gate,
                                             sycl::queue& sycl_queue);

// Applies a 3x3 same-padded depthwise convolution over the 8x8 board and adds
// the result to the input (residual).  Matches the training Python
//   DepthwiseConv2D(kernel_size=3, padding="same")
// which is applied before the KDA Q/K/V projections when kda_local_conv=true.
//
// Tensor layout: [N * 64, emb_size]  (token-major, rank*8+file order).
// Kernel layout from proto: [emb_size, 1, 3, 3] flattened as
//   w[c * 9 + kr * 3 + kf]  for channel c, kernel row offset kr in {0,1,2},
//   kernel file offset kf in {0,1,2}.  The center element is at kr=1, kf=1.
// conv_b may be nullptr (no bias).
// scratch: temporary buffer of at least N*64*emb_size elements.
// output and input may alias (the kernel writes to scratch first, then adds).
template <typename T>
void applyKdaLocalDepthwiseConv(int N, int emb_size, const T* input,
                                const T* conv_w, const T* conv_b, T* scratch,
                                T* output, sycl::queue& sycl_queue) {
  const int tokens = N * 64;
  sycl_queue.parallel_for(
      sycl::range<2>(tokens, emb_size), [=](sycl::id<2> idx) {
        const int token = static_cast<int>(idx[0]);
        const int c = static_cast<int>(idx[1]);

        // Board coordinates: token = rank * 8 + file.
        const int rank = token / 8;
        const int file = token % 8;
        // Batch index; the local 8x8 grid starts at batch * 64.
        const int batch_base = (token / 64) * 64;

        float sum = 0.0f;
        // 3x3 neighbourhood with same (zero) padding.
        for (int kr = 0; kr < 3; ++kr) {
          const int nr = rank + kr - 1;  // neighbour rank
          if (nr < 0 || nr > 7) continue;
          for (int kf = 0; kf < 3; ++kf) {
            const int nf = file + kf - 1;  // neighbour file
            if (nf < 0 || nf > 7) continue;
            const int neighbour_token = batch_base + nr * 8 + nf;
            const float w =
                static_cast<float>(conv_w[c * 9 + kr * 3 + kf]);
            const float x =
                static_cast<float>(input[neighbour_token * emb_size + c]);
            sum += w * x;
          }
        }
        if (conv_b != nullptr) {
          sum += static_cast<float>(conv_b[c]);
        }
        scratch[token * emb_size + c] = static_cast<T>(sum);
      });

  // Add convolution output to original input (residual connection).
  sycl_queue.parallel_for(
      sycl::range<1>(tokens * emb_size), [=](sycl::id<1> idx) {
        const int i = static_cast<int>(idx[0]);
        output[i] = static_cast<T>(static_cast<float>(input[i]) +
                                   static_cast<float>(scratch[i]));
      });
}

template void applyKdaLocalDepthwiseConv<float>(int N, int emb_size,
                                                const float* input,
                                                const float* conv_w,
                                                const float* conv_b,
                                                float* scratch, float* output,
                                                sycl::queue& sycl_queue);

template void applyKdaLocalDepthwiseConv<sycl::half>(
    int N, int emb_size, const sycl::half* input, const sycl::half* conv_w,
    const sycl::half* conv_b, sycl::half* scratch, sycl::half* output,
    sycl::queue& sycl_queue);

}  // namespace sycldnn_backend
}  // namespace lczero
