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

#include "layers.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <sycl/sycl.hpp>
#include <vector>

#include "sycl_subgroup_winograd.h"

#ifdef USE_HIPBLAS
#include "cuBlasContext.h"
#include "hipblas/hipblas.h"

#elif defined(USE_CUBLAS)
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <sycl/backend/cuda.hpp>

#include "cuBlasContext.h"

#else
#include "oneapi/mkl.hpp"
#include "oneapi/mkl/blas.hpp"
#endif

#include <cmath>

#include "kernels.h"
#include "neural/network.h"
#include "neural/tables/attention_policy_map.h"
#include "sycl_common.h"
#include "utils/fp16_utils.h"

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

// ============================================================================
// High-Performance Adaptive Tiled SYCL GEMM helpers (Optimized for Intel iGPU
// EUs). Dynamically adjusts tile_m (e.g. tile_m=4 for M<=4 Winograd GEMMs) to
// eliminate idle EU threads, and uses padded Local Shared Memory [16][17] to
// prevent bank conflicts. C = alpha * op(A) * op(B) + beta * C, column-major
// layout.
// ============================================================================

template <typename DataType>
static void syclGemm(sycl::queue& q, transpose_type transa,
                     transpose_type transb, int m, int n, int k, float alpha,
                     const DataType* A, int lda, const DataType* B, int ldb,
                     float beta, DataType* C, int ldc) {
  bool transA = (transa == transpose_type_transpose);
  bool transB = (transb == transpose_type_transpose);

  if (m <= 16) {
    // Small-M path for Winograd input GEMMs (M = N_batch * 4 tiles, often <=
    // 16). TN=32: covers one SIMD-32 lane group across the N dimension. TK=16:
    // SLM tile depth balances reuse vs. SLM pressure; +1 padding on TileB
    //        (allocated as [TK][TN+1]) avoids 4-byte bank conflicts on Intel
    //        EUs.
    constexpr int TN = 32;
    constexpr int TK = 16;

    sycl::range<2> global_range((n + TN - 1) / TN * 16, 1);
    sycl::range<2> local_range(16, 1);

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 2> tileB(sycl::range<2>(TK, TN + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<2>(global_range, local_range),
          [=](sycl::nd_item<2> item) {
            int l_col = item.get_local_id(0);
            int group_col = item.get_group(0);

            int row = l_col;
            int col_base = group_col * TN;

            float sum[32] = {0.0f};
            int num_tiles = (k + TK - 1) / TK;

            for (int t = 0; t < num_tiles; ++t) {
#pragma unroll
              for (int i = 0; i < 32; ++i) {
                int idx = i * 16 + l_col;
                int b_r, b_c;
                if (transB) {
                  b_r = idx / 32;
                  b_c = idx % 32;
                } else {
                  b_c = idx / 16;
                  b_r = idx % 16;
                }
                int b_k = t * TK + b_r;
                int b_n = col_base + b_c;
                tileB[b_r][b_c] =
                    (b_k < k && b_n < n)
                        ? (transB ? static_cast<float>(B[b_k * ldb + b_n])
                                  : static_cast<float>(B[b_n * ldb + b_k]))
                        : 0.0f;
              }

              item.barrier(sycl::access::fence_space::local_space);

              float a_reg[16];
#pragma unroll
              for (int p = 0; p < TK; ++p) {
                int a_k = t * TK + p;
                a_reg[p] =
                    (row < m && a_k < k)
                        ? (transA ? static_cast<float>(A[row * lda + a_k])
                                  : static_cast<float>(A[a_k * lda + row]))
                        : 0.0f;
              }

#pragma unroll
              for (int p = 0; p < TK; ++p) {
                float a_val = a_reg[p];
#pragma unroll
                for (int c = 0; c < 32; ++c) {
                  sum[c] += a_val * tileB[p][c];
                }
              }

              item.barrier(sycl::access::fence_space::local_space);
            }

            if (row < m) {
#pragma unroll
              for (int c = 0; c < 32; ++c) {
                int c_idx = col_base + c;
                if (c_idx < n) {
                  float c_old = (beta != 0.0f)
                                    ? static_cast<float>(C[c_idx * ldc + row])
                                    : 0.0f;
                  C[c_idx * ldc + row] =
                      static_cast<DataType>(alpha * sum[c] + beta * c_old);
                }
              }
            }
          });
    });
  } else {
    constexpr int WM = 64;
    constexpr int WN = 32;
    constexpr int WK = 16;

    sycl::range<2> global_range((n + WN - 1) / WN * 16, (m + WM - 1) / WM * 4);
    sycl::range<2> local_range(16, 4);

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 2> tileB(sycl::range<2>(WK, WN + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<2>(global_range, local_range),
          [=](sycl::nd_item<2> item) {
            int l_col = item.get_local_id(0);
            int l_row = item.get_local_id(1);
            int wg_col = item.get_group(0);
            int wg_row = item.get_group(1);

            int row = wg_row * WM + l_row * 16 + l_col;
            int col_base = wg_col * WN;

            float sum[32] = {0.0f};
            int num_tiles = (k + WK - 1) / WK;
            int flat_id = l_row * 16 + l_col;

            for (int t = 0; t < num_tiles; ++t) {
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                int idx = i * 64 + flat_id;
                int b_r, b_c;
                if (transB) {
                  b_r = idx / 32;
                  b_c = idx % 32;
                } else {
                  b_c = idx / 16;
                  b_r = idx % 16;
                }
                int b_k = t * WK + b_r;
                int b_n = col_base + b_c;
                tileB[b_r][b_c] =
                    (b_k < k && b_n < n)
                        ? (transB ? static_cast<float>(B[b_k * ldb + b_n])
                                  : static_cast<float>(B[b_n * ldb + b_k]))
                        : 0.0f;
              }

              item.barrier(sycl::access::fence_space::local_space);

              float a_reg[16];
#pragma unroll
              for (int p = 0; p < WK; ++p) {
                int a_k = t * WK + p;
                a_reg[p] =
                    (row < m && a_k < k)
                        ? (transA ? static_cast<float>(A[row * lda + a_k])
                                  : static_cast<float>(A[a_k * lda + row]))
                        : 0.0f;
              }

#pragma unroll
              for (int p = 0; p < WK; ++p) {
                float a_val = a_reg[p];
#pragma unroll
                for (int c = 0; c < 32; ++c) {
                  sum[c] += a_val * tileB[p][c];
                }
              }

              item.barrier(sycl::access::fence_space::local_space);
            }

            if (row < m) {
#pragma unroll
              for (int c = 0; c < 32; ++c) {
                int c_idx = col_base + c;
                if (c_idx < n) {
                  float c_old = (beta != 0.0f)
                                    ? static_cast<float>(C[c_idx * ldc + row])
                                    : 0.0f;
                  C[c_idx * ldc + row] =
                      static_cast<DataType>(alpha * sum[c] + beta * c_old);
                }
              }
            }
          });
    });
  }
}

template <typename DataType>
static void syclGemmStridedBatched(sycl::queue& q, transpose_type transa,
                                   transpose_type transb, int m, int n, int k,
                                   float alpha, const DataType* A, int lda,
                                   long long int strideA, const DataType* B,
                                   int ldb, long long int strideB, float beta,
                                   DataType* C, int ldc, long long int strideC,
                                   int batchCount) {
  bool transA = (transa == transpose_type_transpose);
  bool transB = (transb == transpose_type_transpose);

  if (m <= 16) {
    constexpr int TN = 32;
    constexpr int TK = 16;

    sycl::range<3> global_range(batchCount, (n + TN - 1) / TN * 16, 1);
    sycl::range<3> local_range(1, 16, 1);

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 2> tileB(sycl::range<2>(TK, TN + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(global_range, local_range),
          [=](sycl::nd_item<3> item) {
            int batch = item.get_global_id(0);
            int l_col = item.get_local_id(1);
            int group_col = item.get_group(1);

            int row = l_col;
            int col_base = group_col * TN;

            const DataType* Ab = A + batch * strideA;
            const DataType* Bb = B + batch * strideB;
            DataType* Cb = C + batch * strideC;

            float sum[32] = {0.0f};
            int num_tiles = (k + TK - 1) / TK;

            for (int t = 0; t < num_tiles; ++t) {
#pragma unroll
              for (int i = 0; i < 32; ++i) {
                int idx = i * 16 + l_col;
                int b_r, b_c;
                if (transB) {
                  b_r = idx / 32;
                  b_c = idx % 32;
                } else {
                  b_c = idx / 16;
                  b_r = idx % 16;
                }
                int b_k = t * TK + b_r;
                int b_n = col_base + b_c;
                tileB[b_r][b_c] =
                    (b_k < k && b_n < n)
                        ? (transB ? static_cast<float>(Bb[b_k * ldb + b_n])
                                  : static_cast<float>(Bb[b_n * ldb + b_k]))
                        : 0.0f;
              }

              item.barrier(sycl::access::fence_space::local_space);

              float a_reg[16];
#pragma unroll
              for (int p = 0; p < TK; ++p) {
                int a_k = t * TK + p;
                a_reg[p] =
                    (row < m && a_k < k)
                        ? (transA ? static_cast<float>(Ab[row * lda + a_k])
                                  : static_cast<float>(Ab[a_k * lda + row]))
                        : 0.0f;
              }

#pragma unroll
              for (int p = 0; p < TK; ++p) {
                float a_val = a_reg[p];
#pragma unroll
                for (int c = 0; c < 32; ++c) {
                  sum[c] += a_val * tileB[p][c];
                }
              }

              item.barrier(sycl::access::fence_space::local_space);
            }

            if (row < m) {
#pragma unroll
              for (int c = 0; c < 32; ++c) {
                int c_idx = col_base + c;
                if (c_idx < n) {
                  float c_old = (beta != 0.0f)
                                    ? static_cast<float>(Cb[c_idx * ldc + row])
                                    : 0.0f;
                  Cb[c_idx * ldc + row] =
                      static_cast<DataType>(alpha * sum[c] + beta * c_old);
                }
              }
            }
          });
    });
  } else {
    constexpr int WM = 64;
    constexpr int WN = 32;
    constexpr int WK = 16;

    sycl::range<3> global_range(batchCount, (n + WN - 1) / WN * 16,
                                (m + WM - 1) / WM * 4);
    sycl::range<3> local_range(1, 16, 4);

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 2> tileB(sycl::range<2>(WK, WN + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(global_range, local_range),
          [=](sycl::nd_item<3> item) {
            int batch = item.get_global_id(0);
            int l_col = item.get_local_id(1);
            int l_row = item.get_local_id(2);
            int wg_col = item.get_group(1);
            int wg_row = item.get_group(2);

            int row = wg_row * WM + l_row * 16 + l_col;
            int col_base = wg_col * WN;

            const DataType* Ab = A + batch * strideA;
            const DataType* Bb = B + batch * strideB;
            DataType* Cb = C + batch * strideC;

            float sum[32] = {0.0f};
            int num_tiles = (k + WK - 1) / WK;
            int flat_id = l_row * 16 + l_col;

            for (int t = 0; t < num_tiles; ++t) {
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                int idx = i * 64 + flat_id;
                int b_r, b_c;
                if (transB) {
                  b_r = idx / 32;
                  b_c = idx % 32;
                } else {
                  b_c = idx / 16;
                  b_r = idx % 16;
                }
                int b_k = t * WK + b_r;
                int b_n = col_base + b_c;
                tileB[b_r][b_c] =
                    (b_k < k && b_n < n)
                        ? (transB ? static_cast<float>(Bb[b_k * ldb + b_n])
                                  : static_cast<float>(Bb[b_n * ldb + b_k]))
                        : 0.0f;
              }

              item.barrier(sycl::access::fence_space::local_space);

              float a_reg[16];
#pragma unroll
              for (int p = 0; p < WK; ++p) {
                int a_k = t * WK + p;
                a_reg[p] =
                    (row < m && a_k < k)
                        ? (transA ? static_cast<float>(Ab[row * lda + a_k])
                                  : static_cast<float>(Ab[a_k * lda + row]))
                        : 0.0f;
              }

#pragma unroll
              for (int p = 0; p < WK; ++p) {
                float a_val = a_reg[p];
#pragma unroll
                for (int c = 0; c < 32; ++c) {
                  sum[c] += a_val * tileB[p][c];
                }
              }

              item.barrier(sycl::access::fence_space::local_space);
            }

            if (row < m) {
#pragma unroll
              for (int c = 0; c < 32; ++c) {
                int c_idx = col_base + c;
                if (c_idx < n) {
                  float c_old = (beta != 0.0f)
                                    ? static_cast<float>(Cb[c_idx * ldc + row])
                                    : 0.0f;
                  Cb[c_idx * ldc + row] =
                      static_cast<DataType>(alpha * sum[c] + beta * c_old);
                }
              }
            }
          });
    });
  }
}

template <typename DataType>
static void syclGemmBatched(sycl::queue& q, transpose_type transa,
                            transpose_type transb, int m, int n, int k,
                            float alpha, DataType** A, int lda, DataType** B,
                            int ldb, float beta, DataType** C, int ldc,
                            int batchCount) {
  bool transA = (transa == transpose_type_transpose);
  bool transB = (transb == transpose_type_transpose);

  if (m <= 16) {
    constexpr int TN = 32;
    constexpr int TK = 16;

    sycl::range<3> global_range(batchCount, (n + TN - 1) / TN * 16, 1);
    sycl::range<3> local_range(1, 16, 1);

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 2> tileB(sycl::range<2>(TK, TN + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(global_range, local_range),
          [=](sycl::nd_item<3> item) {
            int batch = item.get_global_id(0);
            int l_col = item.get_local_id(1);
            int group_col = item.get_group(1);

            int row = l_col;
            int col_base = group_col * TN;

            const DataType* Ab = A[batch];
            const DataType* Bb = B[batch];
            DataType* Cb = C[batch];

            float sum[32] = {0.0f};
            int num_tiles = (k + TK - 1) / TK;

            for (int t = 0; t < num_tiles; ++t) {
#pragma unroll
              for (int i = 0; i < 32; ++i) {
                int idx = i * 16 + l_col;
                int b_r, b_c;
                if (transB) {
                  b_r = idx / 32;
                  b_c = idx % 32;
                } else {
                  b_c = idx / 16;
                  b_r = idx % 16;
                }
                int b_k = t * TK + b_r;
                int b_n = col_base + b_c;
                tileB[b_r][b_c] =
                    (b_k < k && b_n < n)
                        ? (transB ? static_cast<float>(Bb[b_k * ldb + b_n])
                                  : static_cast<float>(Bb[b_n * ldb + b_k]))
                        : 0.0f;
              }

              item.barrier(sycl::access::fence_space::local_space);

              float a_reg[16];
#pragma unroll
              for (int p = 0; p < TK; ++p) {
                int a_k = t * TK + p;
                a_reg[p] =
                    (row < m && a_k < k)
                        ? (transA ? static_cast<float>(Ab[row * lda + a_k])
                                  : static_cast<float>(Ab[a_k * lda + row]))
                        : 0.0f;
              }

#pragma unroll
              for (int p = 0; p < TK; ++p) {
                float a_val = a_reg[p];
#pragma unroll
                for (int c = 0; c < 32; ++c) {
                  sum[c] += a_val * tileB[p][c];
                }
              }

              item.barrier(sycl::access::fence_space::local_space);
            }

            if (row < m) {
#pragma unroll
              for (int c = 0; c < 32; ++c) {
                int c_idx = col_base + c;
                if (c_idx < n) {
                  float c_old = (beta != 0.0f)
                                    ? static_cast<float>(Cb[c_idx * ldc + row])
                                    : 0.0f;
                  Cb[c_idx * ldc + row] =
                      static_cast<DataType>(alpha * sum[c] + beta * c_old);
                }
              }
            }
          });
    });
  } else {
    constexpr int WM = 64;
    constexpr int WN = 32;
    constexpr int WK = 16;

    sycl::range<3> global_range(batchCount, (n + WN - 1) / WN * 16,
                                (m + WM - 1) / WM * 4);
    sycl::range<3> local_range(1, 16, 4);

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 2> tileB(sycl::range<2>(WK, WN + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(global_range, local_range),
          [=](sycl::nd_item<3> item) {
            int batch = item.get_global_id(0);
            int l_col = item.get_local_id(1);
            int l_row = item.get_local_id(2);
            int wg_col = item.get_group(1);
            int wg_row = item.get_group(2);

            int row = wg_row * WM + l_row * 16 + l_col;
            int col_base = wg_col * WN;

            const DataType* Ab = A[batch];
            const DataType* Bb = B[batch];
            DataType* Cb = C[batch];

            float sum[32] = {0.0f};
            int num_tiles = (k + WK - 1) / WK;
            int flat_id = l_row * 16 + l_col;

            for (int t = 0; t < num_tiles; ++t) {
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                int idx = i * 64 + flat_id;
                int b_r, b_c;
                if (transB) {
                  b_r = idx / 32;
                  b_c = idx % 32;
                } else {
                  b_c = idx / 16;
                  b_r = idx % 16;
                }
                int b_k = t * WK + b_r;
                int b_n = col_base + b_c;
                tileB[b_r][b_c] =
                    (b_k < k && b_n < n)
                        ? (transB ? static_cast<float>(Bb[b_k * ldb + b_n])
                                  : static_cast<float>(Bb[b_n * ldb + b_k]))
                        : 0.0f;
              }

              item.barrier(sycl::access::fence_space::local_space);

              float a_reg[16];
#pragma unroll
              for (int p = 0; p < WK; ++p) {
                int a_k = t * WK + p;
                a_reg[p] =
                    (row < m && a_k < k)
                        ? (transA ? static_cast<float>(Ab[row * lda + a_k])
                                  : static_cast<float>(Ab[a_k * lda + row]))
                        : 0.0f;
              }

#pragma unroll
              for (int p = 0; p < WK; ++p) {
                float a_val = a_reg[p];
#pragma unroll
                for (int c = 0; c < 32; ++c) {
                  sum[c] += a_val * tileB[p][c];
                }
              }

              item.barrier(sycl::access::fence_space::local_space);
            }

            if (row < m) {
#pragma unroll
              for (int c = 0; c < 32; ++c) {
                int c_idx = col_base + c;
                if (c_idx < n) {
                  float c_old = (beta != 0.0f)
                                    ? static_cast<float>(Cb[c_idx * ldc + row])
                                    : 0.0f;
                  Cb[c_idx * ldc + row] =
                      static_cast<DataType>(alpha * sum[c] + beta * c_old);
                }
              }
            }
          });
    });
  }
}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc,
                               sycl::queue& sycl_queue)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), sycl_queue_(sycl_queue) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip,
                               sycl::queue& sycl_queue)
    : input_(ip),
      C(c),
      H(h),
      W(w),
      nhwc_(ip ? ip->nhwc_ : false),
      sycl_queue_(sycl_queue) {}

template <typename DataType>
SELayer<DataType>::SELayer(BaseLayer<DataType>* ip, int fc1Outputs,
                           bool addPrevLayerBias, ActivationFunction activation,
                           sycl::queue& sycl_queue)
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
    w2_t_ = (DataType*)sycl::malloc_device(
        2 * C * numFc1Out_ * sizeof(DataType), sycl_queue_);
  }

  b1_ = (DataType*)sycl::malloc_device(numFc1Out_ * sizeof(DataType),
                                       sycl_queue_);
  b2_ = (DataType*)sycl::malloc_device(2 * C * sizeof(DataType), sycl_queue_);

  bPrev_ = (DataType*)sycl::malloc_device(C * sizeof(DataType), sycl_queue_);
}

template <typename DataType>
SELayer<DataType>::~SELayer() {
  sycl::free(w1_, sycl_queue_);
  if (w1_t_) sycl::free(w1_t_, sycl_queue_);
  sycl::free(w2_, sycl_queue_);
  if (w2_t_) sycl::free(w2_t_, sycl_queue_);
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
void SELayer<sycl::half>::LoadWeights(float* w1, float* b1, float* w2,
                                      float* b2, float* prevLayerBias,
                                      void* scratch) {
  const size_t num_weights1 = C * numFc1Out_;
  size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t num_weights2 = 2 * num_weights1;
  size_t weight_size2 = 2 * weight_size1;

  // Transpose the weight matrices for the fused path.
  std::vector<float> temp(num_weights2);

  // Weight for the first FC layer.

  sycl_queue_.memcpy(scratch, w1, weight_size1).wait();

  copyTypeConverted((sycl::half*)w1_, (float*)scratch, (int)num_weights1,
                    sycl_queue_);

  if (kUseFusedSELayer && nhwc_) {
    // transposed copy for fused SE kernel
    cpuTranspose(temp.data(), w1, numFc1Out_, C);

    sycl_queue_.memcpy(scratch, temp.data(), weight_size1).wait();

    copyTypeConverted((sycl::half*)w1_t_, (float*)scratch, (int)num_weights1,
                      sycl_queue_);
  }

  // Weight for the second FC layer.
  sycl_queue_.memcpy(scratch, w2, weight_size2).wait();

  copyTypeConverted((sycl::half*)w2_, (float*)scratch, (int)num_weights2,
                    sycl_queue_);
  if (kUseFusedSELayer && nhwc_) {
    cpuTranspose(temp.data(), w2, 2 * C, numFc1Out_);

    sycl_queue_.memcpy(scratch, temp.data(), weight_size2).wait();
    copyTypeConverted((sycl::half*)w2_t_, (float*)scratch, (int)num_weights2,
                      sycl_queue_);
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
                          size_t scratch_size, sycl::queue& sycl_queue,
                          float***) {
  // CERR << "SELayer<float>::Eval. ";
  //  Ping-pong between 'op1' and 'op2' (parts of scratch memory).
  float* op1 = (float*)scratch;
  float* op2 = (float*)scratch + scratch_size / sizeof(float) / 2;

  // 1. Global avg pooling (also adds previous layer bias before computing
  // averages).
  globalAvgPool(N, C, op2, input, bPrev_, false, sycl_queue);

  // 2. First fully connected layer.
  float alpha = 1.0f, beta = 0.0f;

#ifdef USE_CUBLAS
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();

  sycl_queue.submit([&](sycl::handler& cgh) {
    // auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);

    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto cudaStreamHandle =
          ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
      cublasSetStream(handle, cudaStreamHandle);

      ReportCUBLASErrors(cublasSgemm(
          handle, transpose_type_transpose, transpose_type_notranspose,
          numFc1Out_, N, C, &alpha, w1_, C, op2, C, &beta, op1, numFc1Out_));
    });
  });
#elif defined(USE_HIPBLAS)
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();

  sycl_queue.submit([&](sycl::handler& cgh) {
    // auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);

    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto hipStreamHandle =
          ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

      hipblasSetStream(handle, hipStreamHandle);

      hipblasSgemm(handle, transpose_type_transpose, transpose_type_notranspose,
                   numFc1Out_, N, C, &alpha, w1_, C, op2, C, &beta, op1,
                   numFc1Out_);
    });
  });
#else
  syclGemm<float>(sycl_queue, transpose_type_transpose,
                  transpose_type_notranspose, numFc1Out_, N, C, alpha, w1_, C,
                  op2, C, beta, op1, numFc1Out_);
#endif

  addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, act_,
             sycl_queue);

  // 3. Second fully connected layer.

#ifdef USE_CUBLAS
  sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto cudaStreamHandle =
          ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
      cublasSetStream(handle, cudaStreamHandle);

      ReportCUBLASErrors(cublasSgemm(handle, transpose_type_transpose,
                                     transpose_type_notranspose, 2 * C, N,
                                     numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                     numFc1Out_, &beta, op2, 2 * C));
    });
  });

#elif defined(USE_HIPBLAS)
  sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto hipStreamHandle =
          ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

      hipblasSetStream(handle, hipStreamHandle);

      hipblasSgemm(handle, transpose_type_transpose, transpose_type_notranspose,
                   2 * C, N, numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                   numFc1Out_, &beta, op2, 2 * C);
    });
  });
#else
  syclGemm<float>(sycl_queue, transpose_type_transpose,
                  transpose_type_notranspose, 2 * C, N, numFc1Out_, alpha, w2_,
                  numFc1Out_, op1, numFc1Out_, beta, op2, 2 * C);
#endif

  addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, ACTIVATION_NONE,
             sycl_queue);

  // 4. (Optional prev layer bias add), Global scale, residual add, relu and
  // bias.
  globalScale(N, C, output, input, op2, bPrev_, false, act_, sycl_queue);
}

template <>
void SELayer<sycl::half>::Eval(int N, sycl::half* output,
                               const sycl::half* input,
                               const sycl::half* input2, void* scratch,
                               size_t scratch_size, sycl::queue& sycl_queue,
                               sycl::half***) {
  // CERR << "SELayer<sycl::half>::Eval. ";

  bool se_done = false;
  if (kUseFusedSELayer && nhwc_) {
    se_done = Se_Fp16_NHWC(N, C, numFc1Out_, output, input2, input, w1_t_, b1_,
                           w2_t_, b2_, bPrev_, act_, sycl_queue);
  }
  if (!se_done) {
    assert(output == input2);
    // Ping-pong between 'op1' and 'op2' (parts of scratch memory).
    sycl::half* op1 = (sycl::half*)scratch;
    sycl::half* op2 =
        (sycl::half*)scratch + scratch_size / sizeof(sycl::half) / 2;

    // 1. Global avg pooling (also adds previous layer bias before computing
    // averages).
    globalAvgPool(N, C, op2, input, bPrev_, nhwc_, sycl_queue);

    // 2. First fully connected layer.
    // half_raw one_h{0x3C00};
    // half_raw zero_h{0};

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

    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);

        ReportCUBLASErrors(cublasHgemm(
            handle, transpose_type_transpose, transpose_type_notranspose,
            numFc1Out_, N, C, &alpha, ((const half*)w1_), C, ((const half*)op2),
            C, &beta, ((half*)op1), numFc1Out_));
      });
    });

#elif defined(USE_HIPBLAS)
    hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();

    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);

        hipblasHgemm(handle, transpose_type_transpose,
                     transpose_type_notranspose, numFc1Out_, N, C, &alpha,
                     ((const hipblasHalf*)w1_), C, ((const hipblasHalf*)op2), C,
                     &beta, ((hipblasHalf*)op1), numFc1Out_);
      });
    });
#else
    syclGemm<sycl::half>(sycl_queue, transpose_type_transpose,
                         transpose_type_notranspose, numFc1Out_, N, C, alpha,
                         (const sycl::half*)w1_, C, (const sycl::half*)op2, C,
                         beta, (sycl::half*)op1, numFc1Out_);
#endif

    addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, act_,
               sycl_queue);

#ifdef USE_CUBLAS

    // Submitted to the sycl_queue parameter, not the layer's sycl_queue_
    // member: everything else in this function (the first gemm, the
    // addVectors calls) runs on the caller's queue, and under
    // multi_stream=true the two differ per thread. Mixing them here put the
    // second gemm on a different in-order queue than the addVectors that
    // produces its input and consumes its output, with no ordering between
    // the queues.
    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);

        // 3. Second fully connected layer.
        ReportCUBLASErrors(cublasHgemm(
            handle, transpose_type_transpose, transpose_type_notranspose, 2 * C,
            N, numFc1Out_, &alpha, ((const half*)w2_), numFc1Out_,
            ((const half*)op1), numFc1Out_, &beta, ((half*)op2), 2 * C));
      });
    });

#elif defined(USE_HIPBLAS)
    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
        hipblasSetStream(handle, hipStreamHandle);

        hipblasHgemm(handle, transpose_type_transpose,
                     transpose_type_notranspose, 2 * C, N, numFc1Out_, &alpha,
                     ((const hipblasHalf*)w2_), numFc1Out_,
                     ((const hipblasHalf*)op1), numFc1Out_, &beta,
                     ((hipblasHalf*)op2), 2 * C);
      });
    });
#else
    syclGemm<sycl::half>(
        sycl_queue, transpose_type_transpose, transpose_type_notranspose, 2 * C,
        N, numFc1Out_, alpha, (const sycl::half*)w2_, numFc1Out_,
        (const sycl::half*)op1, numFc1Out_, beta, (sycl::half*)op2, 2 * C);
#endif

    addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, ACTIVATION_NONE,
               sycl_queue);

    // 4. (Optional prev layer bias add), Global scale, residual add, relu and
    // bias.
    globalScale(N, C, output, input, op2, bPrev_, nhwc_, act_, sycl_queue);
  }
}

template <typename DataType>
FCLayer<DataType>::FCLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           bool bias, ActivationFunction activation,
                           sycl::queue& sycl_queue)
    : BaseLayer<DataType>(C, H, W, ip, sycl_queue),
      use_bias_(bias),
      act_(activation) {
  const size_t weight_size =
      sizeof(DataType) * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  const size_t bias_size = sizeof(DataType) * C * H * W;

  weights_ = (DataType*)sycl::malloc_device(weight_size, sycl_queue_);

  if (use_bias_) {
    biases_ = (DataType*)sycl::malloc_device(bias_size, sycl_queue_);
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
    copyTypeConverted((sycl::half*)weights_, (float*)scratch, (int)num_weights,
                      sycl_queue_);
  }

  if (cpuBias) {
    sycl_queue_.memcpy(scratch, cpuBias, bias_size).wait();
    copyTypeConverted((sycl::half*)biases_, (float*)scratch, (int)num_biases,
                      sycl_queue_);
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

  // sycl_queue_.wait();
}

template <>
void FCLayer<sycl::half>::Eval(int N, sycl::half* output_tensor,
                               const sycl::half* input_tensor,
                               const sycl::half* /*input2*/, void* /*scratch*/,
                               size_t /*scratch_size*/, sycl::queue& sycl_queue,
                               sycl::half***) {
  // CERR << "FCLayer<sycl::half>::Eval. ";

  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  // sycl::half alpha = float2half_rn(1.0f),
  // beta = float2half_rn(0.0f);

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

  sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto cudaStreamHandle =
          ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
      cublasSetStream(handle, cudaStreamHandle);

      ReportCUBLASErrors(cublasHgemm(
          handle, transpose_type_transpose, transpose_type_notranspose,
          num_outputs, N, num_inputs, &alpha, ((const half*)weights_),
          num_inputs, ((const half*)input_tensor), num_inputs, &beta,
          ((half*)output_tensor), num_outputs));
    });
  });
#elif defined(USE_HIPBLAS)
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto hipStreamHandle =
          ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
      hipblasSetStream(handle, hipStreamHandle);

      hipblasHgemm(handle, transpose_type_transpose, transpose_type_notranspose,
                   num_outputs, N, num_inputs, &alpha,
                   ((const hipblasHalf*)weights_), num_inputs,
                   ((const hipblasHalf*)input_tensor), num_inputs, &beta,
                   ((hipblasHalf*)output_tensor), num_outputs);
    });
  });
#else
  oneapi::mkl::blas::column_major::gemm(
      sycl_queue, transpose_type_transpose, transpose_type_notranspose,
      static_cast<std::int64_t>(num_outputs), static_cast<std::int64_t>(N),
      static_cast<std::int64_t>(num_inputs), alpha,
      ((const sycl::half*)weights_), static_cast<std::int64_t>(num_inputs),
      ((const sycl::half*)input_tensor), static_cast<std::int64_t>(num_inputs),
      beta, ((sycl::half*)output_tensor),
      static_cast<std::int64_t>(num_outputs));
#endif

  if (use_bias_ || (act_ != ACTIVATION_NONE)) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, act_, sycl_queue);
  }
}

template <>
void FCLayer<float>::Eval(int N, float* output_tensor,
                          const float* input_tensor, const float* /*input2*/,
                          void* /*scratch*/, size_t /*scratch_size*/,
                          sycl::queue& sycl_queue, float***) {
  // CERR << "FCLayer<float>::Eval. ";

  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  float alpha = 1.0f, beta = 0.0f;
  // CERR << "FCLayer<float>::Eval - 1. " << num_inputs << " " << num_outputs;

#ifdef USE_CUBLAS
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();

  sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto cudaStreamHandle =
          ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
      cublasSetStream(handle, cudaStreamHandle);

      ReportCUBLASErrors(cublasSgemm(
          handle, transpose_type_transpose, transpose_type_notranspose,
          num_outputs, N, num_inputs, &alpha, weights_, num_inputs,
          input_tensor, num_inputs, &beta, output_tensor, num_outputs));
    });
  });
#elif defined(USE_HIPBLAS)
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto hipStreamHandle =
          ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

      hipblasSetStream(handle, hipStreamHandle);

      hipblasSgemm(handle, transpose_type_transpose, transpose_type_notranspose,
                   num_outputs, N, num_inputs, &alpha, weights_, num_inputs,
                   input_tensor, num_inputs, &beta, output_tensor, num_outputs);
    });
  });
#else
  oneapi::mkl::blas::column_major::gemm(
      sycl_queue, transpose_type_transpose, transpose_type_notranspose,
      static_cast<std::int64_t>(num_outputs), static_cast<std::int64_t>(N),
      static_cast<std::int64_t>(num_inputs), alpha, weights_,
      static_cast<std::int64_t>(num_inputs), input_tensor,
      static_cast<std::int64_t>(num_inputs), beta, output_tensor,
      static_cast<std::int64_t>(num_outputs));
#endif

  if (use_bias_ || (act_ != ACTIVATION_NONE)) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, act_, sycl_queue);
  }
}

template <typename DataType>
FCLayer<DataType>::~FCLayer() {
  sycl::free(weights_, sycl_queue_);
  if (use_bias_ && biases_) {
    sycl::free(biases_, sycl_queue_);
  }
}

template <typename DataType>
PolicyMapLayer<DataType>::PolicyMapLayer(BaseLayer<DataType>* ip, int C, int H,
                                         int W, int usedSize, bool attention,
                                         sycl::queue& sycl_queue)
    : BaseLayer<DataType>(C, H, W, ip, sycl_queue),
      used_size_(usedSize),
      attention_map_(attention) {
  size_t weight_size = sizeof(short) * this->input_->GetC() * 64;

  if (attention) weight_size = sizeof(short) * usedSize;

  weights_ = (short*)sycl::malloc_device(weight_size, sycl_queue_);
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
    sycl_queue_
        .memcpy(weights_, convertedWeights.data(), used_size_ * sizeof(short))
        .wait();
  } else {
    sycl_queue_.memcpy(weights_, cpuWeight, weight_size).wait();
  }
}

template <typename DataType>
void PolicyMapLayer<DataType>::Eval(int N, DataType* output_tensor,
                                    const DataType* input_tensor,
                                    const DataType* /*input2*/,
                                    void* /*scratch*/, size_t /*scratch_size*/,
                                    sycl::queue& sycl_queue, DataType***) {
  // CERR << "PolicyMapLayer<DataType>::Eval. ";

  int inputSize =
      this->input_->GetC() * this->input_->GetH() * this->input_->GetW();
  if (attention_map_) inputSize = used_size_;
  int outputSize = this->C * this->H * this->W;

  PolicyMap(N, output_tensor, input_tensor, weights_, inputSize, used_size_,
            outputSize, sycl_queue);
}

template <typename DataType>
PolicyMapLayer<DataType>::~PolicyMapLayer() {
  sycl::free(weights_, sycl_queue_);
}

template <typename DataType>
FusedWinogradConvSELayer<DataType>::FusedWinogradConvSELayer(
    BaseLayer<DataType>* ip, int C, int H, int W, int Cin,
    ActivationFunction activation, bool bias, bool skip_add, bool se, int se_k,
    sycl::queue& sycl_queue, bool op_nhcw)
    : BaseLayer<DataType>(C, H, W, ip, false, sycl_queue),
      c_input_(Cin),
      act_(activation),
      use_bias_(bias),
      skip_add_(skip_add),
      has_se_(se),
      se_k_(se_k),
      op_nhcw_(op_nhcw) {
  if (act_ != ACTIVATION_RELU && act_ != ACTIVATION_MISH &&
      act_ != ACTIVATION_NONE) {
    throw Exception("Unsupported activation for fused winograd conv SE layer.");
  }

  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * c_input_ * C * 3 * 3;

  if (use_bias_) {
    const size_t bias_size = sizeof(DataType) * C;
    biases_ = (DataType*)sycl::malloc_device(bias_size, sycl_queue_);
  }

  // 6x6 transformed filter size, for 3x3 convolution
  transformed_weights_ =
      (DataType*)sycl::malloc_device(weight_size * 4, sycl_queue_);

  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = sizeof(DataType) * num_weights1;
    const size_t weight_size2 = sizeof(DataType) * num_weights2;
    const size_t biases_size1 = sizeof(DataType) * num_biases1;
    const size_t biases_size2 = sizeof(DataType) * num_biases2;

    w1_ = (DataType*)sycl::malloc_device(weight_size1, sycl_queue_);
    w2_ = (DataType*)sycl::malloc_device(weight_size2, sycl_queue_);
    b1_ = (DataType*)sycl::malloc_device(biases_size1, sycl_queue_);
    b2_ = (DataType*)sycl::malloc_device(biases_size2, sycl_queue_);
  }
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::LoadWeights(float* pfilter,
                                                     float* pBias,
                                                     void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size + bias_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  // sycl_queue_.memcpy(scratch, pfilter, weight_size).wait_and_throw();
  sycl_queue_.memcpy(scratch, pfilter, weight_size).wait();
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3,
                    sycl_queue_);

  if (pBias) {
    sycl_queue_.memcpy(scratch, pBias, bias_size).wait();

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
  sycl_queue_
      .memcpy(scratch, temp_transposed.data(), num_weights1 * sizeof(float))
      .wait();

  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1,
                    sycl_queue_);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);

  sycl_queue_
      .memcpy(scratch, temp_transposed.data(), num_weights2 * sizeof(float))
      .wait();
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2,
                    sycl_queue_);

  sycl_queue_.memcpy(scratch, b1, num_biases1 * sizeof(float)).wait();
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1,
                    sycl_queue_);

  sycl_queue_.memcpy(scratch, b2, num_biases2 * sizeof(float)).wait();
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2,
                    sycl_queue_);
}

template <>
void BaseLayer<sycl::half>::cublasRowMajorMatrixMul(const sycl::half* A,
                                                    const sycl::half* B,
                                                    sycl::half* Out, int M,
                                                    int N, int K, int batchSize,
                                                    sycl::queue& sycl_queue) {
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

  sycl_queue.submit([&](sycl::handler& cgh) {
    // auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto cudaStreamHandle =
          ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
      cublasSetStream(handle, cudaStreamHandle);

      ReportCUBLASErrors(cublasGemmStridedBatchedEx(
          handle, transpose_type_notranspose, transpose_type_notranspose, N, M,
          K, &one_h, B, CUDA_R_16F, N, N * K, A, CUDA_R_16F, K, K * M, &zero_h,
          Out, CUDA_R_16F, N, N * M, batchSize, CUDA_R_16F,
          CUBLAS_GEMM_DEFAULT));
    });
  });

#elif defined(USE_HIPBLAS)
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();

  sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto hipStreamHandle =
          ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
      hipblasSetStream(handle, hipStreamHandle);

      hipblasGemmStridedBatchedEx(
          handle, transpose_type_notranspose, transpose_type_notranspose, N, M,
          K, &alpha, B, HIPBLAS_R_16F, N, N * K, A, HIPBLAS_R_16F, K, K * M,
          &beta, Out, HIPBLAS_R_16F, N, N * M, batchSize, HIPBLAS_COMPUTE_16F,
          HIPBLAS_GEMM_DEFAULT);
    });
  });
#else
  syclGemmStridedBatched<sycl::half>(
      sycl_queue, transpose_type_notranspose, transpose_type_notranspose, N, M,
      K, 1.0f, B, N, N * K, A, K, K * M, 0.0f, Out, N, N * M, batchSize);
#endif
}

template <>
void BaseLayer<float>::cublasRowMajorMatrixMul(const float* A, const float* B,
                                               float* Out, int M, int N, int K,
                                               int batchSize,
                                               sycl::queue& sycl_queue) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;

  int64_t M_ = M;
  int64_t N_ = N;
  int64_t K_ = K;

#ifdef USE_CUBLAS
  // static cublasHandle_t handle;
  // ReportCUBLASErrors(cublasCreate(&handle));
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
#endif

#ifdef USE_HIPBLAS
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
#endif

  {
#ifdef USE_CUBLAS
    sycl_queue.submit([&](sycl::handler& cgh) {
      // auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);

        ReportCUBLASErrors(cublasGemmStridedBatchedEx(
            handle, transpose_type_notranspose, transpose_type_notranspose, N,
            M, K, &floatOne, B, CUDA_R_32F, N, N * K, A, CUDA_R_32F, K, K * M,
            &floatZero, Out, CUDA_R_32F, N, N * M, batchSize, CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT));
      });
    });
#elif defined(USE_HIPBLAS)
    sycl_queue.submit([&](sycl::handler& cgh) {
      // auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
        hipblasSetStream(handle, hipStreamHandle);

        hipblasGemmStridedBatchedEx(
            handle, transpose_type_notranspose, transpose_type_notranspose, N,
            M, K, &floatOne, B, HIPBLAS_R_32F, N, N * K, A, HIPBLAS_R_32F, K,
            K * M, &floatZero, Out, HIPBLAS_R_32F, N, N * M, batchSize,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
      });
    });
#else
    syclGemmStridedBatched<float>(sycl_queue, transpose_type_notranspose,
                                  transpose_type_notranspose, N_, M_, K_,
                                  floatOne, B, N_, N_ * K_, A, K_, K_ * M_,
                                  floatZero, Out, N_, N_ * M_, batchSize);
#endif
  }
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* input2,
    void* scratch, size_t scratch_size, sycl::queue& sycl_queue, DataType***) {
  /*CERR << "[SYCL TRACE] FusedWinogradConvSELayer::Eval (N=" << N
       << ", C=" << C << ", c_input=" << c_input_
       << ", has_se=" << (has_se_ ? "true" : "false")
       << ", skip_add=" << (skip_add_ ? "true" : "false") << ")";*/

  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.

  DataType* transformed_input = (DataType*)scratch;
  DataType* transformed_output =
      transformed_input + scratch_size / (2 * sizeof(DataType));

  InputTransform<DataType, false>(N, c_input_, transformed_input, input,
                                  sycl_queue);
  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights_, transformed_output, N * 4, C,
      c_input_, 36, sycl_queue);

  if (act_ == ACTIVATION_NONE) {
    if (!has_se_ && use_bias_ && !skip_add_)
      OutputTransform<DataType, false, ACTIVATION_NONE, true, false, false,
                      false>(N, C, 0, output, transformed_output, nullptr,
                             biases_, nullptr, nullptr, nullptr, nullptr,
                             sycl_queue);
    else
      throw Exception("unsupported network type!");
  } else if (act_ == ACTIVATION_RELU) {
    if (has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, true, ACTIVATION_RELU, true, true, false,
                      false>(N, C, se_k_, output, transformed_output, input2,
                             biases_, w1_, b1_, w2_, b2_, sycl_queue);
    else if (!has_se_ && use_bias_ && !skip_add_) {
      if (op_nhcw_)
        OutputTransform<DataType, false, ACTIVATION_RELU, true, false, false,
                        true>(N, C, 0, output, transformed_output, nullptr,
                              biases_, nullptr, nullptr, nullptr, nullptr,
                              sycl_queue);
      else
        OutputTransform<DataType, false, ACTIVATION_RELU, true, false, false,
                        false>(N, C, 0, output, transformed_output, nullptr,
                               biases_, nullptr, nullptr, nullptr, nullptr,
                               sycl_queue);
    } else if (!has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, false, ACTIVATION_RELU, true, true, false,
                      false>(N, C, 0, output, transformed_output, input2,
                             biases_, nullptr, nullptr, nullptr, nullptr,
                             sycl_queue);
    else
      throw Exception("unsupported network type!");
  } else if (act_ == ACTIVATION_MISH) {
    if (has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, true, ACTIVATION_MISH, true, true, false,
                      false>(N, C, se_k_, output, transformed_output, input2,
                             biases_, w1_, b1_, w2_, b2_, sycl_queue);
    else if (!has_se_ && use_bias_ && !skip_add_) {
      if (op_nhcw_)
        OutputTransform<DataType, false, ACTIVATION_MISH, true, false, false,
                        true>(N, C, 0, output, transformed_output, nullptr,
                              biases_, nullptr, nullptr, nullptr, nullptr,
                              sycl_queue);
      else
        OutputTransform<DataType, false, ACTIVATION_MISH, true, false, false,
                        false>(N, C, 0, output, transformed_output, nullptr,
                               biases_, nullptr, nullptr, nullptr, nullptr,
                               sycl_queue);
    } else if (!has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, false, ACTIVATION_MISH, true, true, false,
                      false>(N, C, 0, output, transformed_output, input2,
                             biases_, nullptr, nullptr, nullptr, nullptr,
                             sycl_queue);
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
  weights_ = (DataType*)sycl::malloc_device(weight_size, sycl_queue_);

  if (use_bias_) {
    const size_t bias_size = sizeof(DataType) * C;
    // CERR << "Conv1Layer using bias " << bias_size;
    biases_ = (DataType*)sycl::malloc_device(bias_size, sycl_queue_);
  }
}

template <typename DataType>
void Conv1Layer<DataType>::LoadWeights(float* pfilter, float* pBias,
                                       void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 1 * 1;
  const size_t bias_size = sizeof(float) * C;

  assert(scratch);

  sycl_queue_.memcpy(scratch, pfilter, weight_size).wait();
  copyTypeConverted((DataType*)weights_, (float*)scratch, C * c_input_ * 1 * 1,
                    sycl_queue_);

  if (pBias) {
    sycl_queue_.memcpy(scratch, pBias, bias_size).wait();
    copyTypeConverted((DataType*)biases_, (float*)scratch, C, sycl_queue_);
  }
}

template <>
void Conv1Layer<sycl::half>::cublasSpecialMatrixMul(const sycl::half* A,
                                                    const sycl::half* B,
                                                    sycl::half* Out, int M,
                                                    int N, int K, int batchSize,
                                                    sycl::queue& sycl_queue) {
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
  sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto cudaStreamHandle =
          ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
      cublasSetStream(handle, cudaStreamHandle);

      ReportCUBLASErrors(cublasGemmStridedBatchedEx(
          handle, transpose_type_notranspose, transpose_type_notranspose, N, M,
          K, &one_h, B, CUDA_R_16F, N, N * K, A, CUDA_R_16F, K, 0, &zero_h, Out,
          CUDA_R_16F, N, N * M, batchSize, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
    });
  });
#elif defined(USE_HIPBLAS)
  sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
      auto hipStreamHandle =
          ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
      hipblasSetStream(handle, hipStreamHandle);
      hipblasGemmStridedBatchedEx(
          handle, transpose_type_notranspose, transpose_type_notranspose, N, M,
          K, &alpha, B, HIPBLAS_R_16F, N, N * K, A, HIPBLAS_R_16F, K, 0, &beta,
          Out, HIPBLAS_R_16F, N, N * M, batchSize, HIPBLAS_COMPUTE_16F,
          HIPBLAS_GEMM_DEFAULT);
    });
  });
#else
  syclGemmStridedBatched<sycl::half>(
      sycl_queue, transpose_type_notranspose, transpose_type_notranspose, N, M,
      K, alpha, B, N, N * K, A, K, 0, beta, Out, N, N * M, batchSize);
#endif
}

template <>
void Conv1Layer<float>::cublasSpecialMatrixMul(const float* A, const float* B,
                                               float* Out, int M, int N, int K,
                                               int batchSize,
                                               sycl::queue& sycl_queue) {
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
    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);

        ReportCUBLASErrors(cublasGemmStridedBatchedEx(
            handle, transpose_type_notranspose, transpose_type_notranspose, N,
            M, K, &floatOne, B, CUDA_R_32F, N, N * K, A, CUDA_R_32F, K, 0,
            &floatZero, Out, CUDA_R_32F, N, N * M, batchSize, CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT));
      });
    });
#elif defined(USE_HIPBLAS)
    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);

        hipblasGemmStridedBatchedEx(
            handle, transpose_type_notranspose, transpose_type_notranspose, N,
            M, K, &floatOne, B, HIPBLAS_R_32F, N, N * K, A, HIPBLAS_R_32F, K, 0,
            &floatZero, Out, HIPBLAS_R_32F, N, N * M, batchSize,
            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
      });
    });
#else
    syclGemmStridedBatched<float>(sycl_queue, transpose_type_notranspose,
                                  transpose_type_notranspose, N_, M_, K_,
                                  floatOne, B, N_, N_ * K_, A, K_, 0, floatZero,
                                  Out, N_, N_ * M_, batchSize);
#endif
  }
}

template <typename DataType>
void Conv1Layer<DataType>::Eval(int N, DataType* output, const DataType* input,
                                const DataType* /*input2*/, void* /*scratch*/,
                                size_t /*scratch_size*/,
                                sycl::queue& sycl_queue, DataType***) {
  // CERR << "Conv1Layer<DataType>::Eval. ";

  cublasSpecialMatrixMul(weights_, input, output, C, H * W, c_input_, N,
                         sycl_queue);
  // CERR << "cublasSpecialMatrixMul. ";

  if (use_bias_) {
    // CERR << "addBias. " << N << " " << C << " " << H << " " << W;
    addBias_NCHW(output, output, biases_, N, C, H, W, act_, sycl_queue);
  } else if (act_ != ACTIVATION_NONE) {
    addVectors(output, output, (DataType*)nullptr, N * C * H * W, N * C * H * W,
               0, act_, sycl_queue);
    //  CERR << "addVectors. ";
  }
}

template <typename DataType>
Conv1Layer<DataType>::~Conv1Layer() {
  sycl::free(weights_, sycl_queue_);
  if (use_bias_) sycl::free(biases_, sycl_queue_);
}

template <typename DataType>
ResidualBlock<DataType>::ResidualBlock(BaseLayer<DataType>* ip, int C, bool se,
                                       int se_k, bool first, bool last,
                                       ActivationFunction activation,
                                       int shared_mem_size,
                                       size_t max_work_group_size,
                                       sycl::queue& sycl_queue)
    : BaseLayer<DataType>(C, 8, 8, ip, ip->isNHWC(), sycl_queue),
      has_se_(se),
      se_k_(se_k),
      c_input_(C),
      first_block_(first),
      last_block_(last),
      shared_mem_size_(shared_mem_size),
      max_wg_size_(max_work_group_size),
      act_(activation) {
  if (act_ != ACTIVATION_RELU && act_ != ACTIVATION_MISH) {
    throw Exception("Unsupported activation for residual block.");
  }

  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * C * C * 3 * 3;

  const size_t bias_size = sizeof(DataType) * C;
  biases0_ = (DataType*)sycl::malloc_device(bias_size, sycl_queue_);
  biases1_ = (DataType*)sycl::malloc_device(bias_size, sycl_queue_);

  // 6x6 transformed filter size, for 3x3 convolution
  transformed_weights0_ =
      (DataType*)sycl::malloc_device(weight_size * 4, sycl_queue_);
  transformed_weights1_ =
      (DataType*)sycl::malloc_device(weight_size * 4, sycl_queue_);

  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = sizeof(DataType) * num_weights1;
    const size_t weight_size2 = sizeof(DataType) * num_weights2;
    const size_t biases_size1 = sizeof(DataType) * num_biases1;
    const size_t biases_size2 = sizeof(DataType) * num_biases2;

    w1_ = (DataType*)sycl::malloc_device(weight_size1, sycl_queue_);
    w2_ = (DataType*)sycl::malloc_device(weight_size2, sycl_queue_);
    b1_ = (DataType*)sycl::malloc_device(biases_size1, sycl_queue_);
    b2_ = (DataType*)sycl::malloc_device(biases_size2, sycl_queue_);
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

  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3,
                    sycl_queue_);

  if (pBias) {
    sycl_queue_.memcpy(scratch, pBias, bias_size).wait();
    copyTypeConverted((DataType*)biases0_, (float*)scratch, C, sycl_queue_);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, c_input_, transformed_weights0_, weights, sycl_queue_);
}

template <typename DataType>
void ResidualBlock<DataType>::LoadWeights1(float* pfilter, float* pBias,
                                           void* scratch) {
  const size_t weight_size = sizeof(float) * C * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  sycl_queue_.memcpy(scratch, pfilter, weight_size).wait();

  copyTypeConverted((DataType*)weights, (float*)scratch, C * C * 3 * 3,
                    sycl_queue_);

  if (pBias) {
    sycl_queue_.memcpy(scratch, pBias, bias_size).wait();
    copyTypeConverted((DataType*)biases1_, (float*)scratch, C, sycl_queue_);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, C, transformed_weights1_, weights, sycl_queue_);
}

template <typename DataType>
void ResidualBlock<DataType>::LoadSEWeights(float* w1, float* b1, float* w2,
                                            float* b2, void* scratch) {
  const size_t num_weights1 = C * se_k_;
  const size_t num_weights2 = num_weights1 * 2;
  const size_t num_biases1 = se_k_;
  const size_t num_biases2 = 2 * C;

  // The shader uses transposed weight matrices.
  std::vector<float> temp_transposed(num_weights2);

  CpuTranspose(temp_transposed.data(), w1, se_k_, C);

  sycl_queue_
      .memcpy(scratch, temp_transposed.data(), num_weights1 * sizeof(float))
      .wait();
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1,
                    sycl_queue_);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);

  sycl_queue_
      .memcpy(scratch, temp_transposed.data(), num_weights2 * sizeof(float))
      .wait();
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2,
                    sycl_queue_);

  sycl_queue_.memcpy(scratch, b1, num_biases1 * sizeof(float)).wait();
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1,
                    sycl_queue_);

  sycl_queue_.memcpy(scratch, b2, num_biases2 * sizeof(float)).wait();
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2,
                    sycl_queue_);
}

template <typename DataType>
void ResidualBlock<DataType>::Eval(int N, DataType* output,
                                   const DataType* input,
                                   const DataType* /*input2*/, void* scratch,
                                   size_t scratch_size, sycl::queue& sycl_queue,
                                   DataType***) {
  /*CERR << "[SYCL TRACE] ResidualBlock::Eval (N=" << N << ", C=" << C
       << ", first_block=" << (first_block_ ? "true" : "false")
       << ", last_block=" << (last_block_ ? "true" : "false")
       << ", scratch=" << (scratch ? "provided" : "null/L2cache") << ")";*/
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
    // Sub-allocate all memory we need from "output" tensor (tensor_mem[2]).
    // The primary optimization for Intel GPUs is the sub-allocation memory
    // packing trick that enableCacheOpt = true enables:
    // It packs all intermediate transformed Winograd tensors into a single
    // contiguous block of tensor_mem[2].
    // Because res_block_mem <= device_cache_.l2_cache_size, this contiguous
    // layout gives the Intel GPU hardware maximum spatial and temporal
    // locality, naturally keeping the entire working set inside L2/L3 cache
    // during the residual tower evaluation without needing explicit driver
    // hints.
    transformed_input = output;  // This is true in normal cases too!
    transformed_output = transformed_input + (N * C * 8 * 8 * 36 / 16);
  } else {
    transformed_input = (DataType*)scratch;
    transformed_output =
        transformed_input + scratch_size / (2 * sizeof(DataType));
  }

  if (first_block_) {
    InputTransform<DataType, true>(N, c_input_, transformed_input, input,
                                   sycl_queue);
    BaseLayer<DataType>::cublasRowMajorMatrixMul(
        transformed_input, transformed_weights0_, transformed_output, N * 4, C,
        c_input_, 36, sycl_queue);
  } else {
    BaseLayer<DataType>::cublasRowMajorMatrixMul(output, transformed_weights0_,
                                                 transformed_output, N * 4, C,
                                                 c_input_, 36, sycl_queue);
  }

  if (act_ == ACTIVATION_RELU) {
    if constexpr (std::is_same_v<DataType, sycl::half>) {
      if (static_cast<size_t>(C) <= max_wg_size_) {
        SubGroupOutputInputTransform_NoSE<ACTIVATION_RELU, true, false>(
            N, C, (sycl::half*)transformed_input,
            (const sycl::half*)transformed_output, nullptr,
            (const sycl::half*)biases0_, sycl_queue);
      } else {
        OutputInputTransform<DataType, false, ACTIVATION_RELU, true, false>(
            N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
            nullptr, nullptr, nullptr, nullptr, sycl_queue);
      }
    } else {
      OutputInputTransform<DataType, false, ACTIVATION_RELU, true, false>(
          N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
          nullptr, nullptr, nullptr, nullptr, sycl_queue);
    }
  } else if (act_ == ACTIVATION_MISH) {
    if constexpr (std::is_same_v<DataType, sycl::half>) {
      if (static_cast<size_t>(C) <= max_wg_size_) {
        SubGroupOutputInputTransform_NoSE<ACTIVATION_MISH, true, false>(
            N, C, (sycl::half*)transformed_input,
            (const sycl::half*)transformed_output, nullptr,
            (const sycl::half*)biases0_, sycl_queue);
      } else {
        OutputInputTransform<DataType, false, ACTIVATION_MISH, true, false>(
            N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
            nullptr, nullptr, nullptr, nullptr, sycl_queue);
      }
    } else {
      OutputInputTransform<DataType, false, ACTIVATION_MISH, true, false>(
          N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
          nullptr, nullptr, nullptr, nullptr, sycl_queue);
    }
  }
  // "transformed_input" tensor now contains transformed input for the next
  // convolution

  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights1_, transformed_output, N * 4, C, C,
      36, sycl_queue);

  // allowFusing controls whether the SubGroup (warp-level) fused
  // SE+output+input transform kernels are used. These kernels are FP16-only;
  // for float (FP32), allowFusing is still computed for the channel-count
  // guard but the if-constexpr branch below will not dispatch SubGroup kernels,
  // falling through to the generic OutputInputTransform / OutputTransform path.
  // The decision itself lives in sycl_common.h so sycl_test.cc exercises it.
  const bool fp16 = std::is_same<sycl::half, DataType>::value;
  bool allowFusing =
      AllowSubGroupFusing(C, se_k_, shared_mem_size_, max_wg_size_, fp16);

  if (act_ == ACTIVATION_RELU) {
    if (last_block_) {
      if (has_se_) {
        if (allowFusing) {
          if constexpr (std::is_same_v<DataType, sycl::half>) {
            SubGroupOutputTransform<ACTIVATION_RELU, true, true, true, false>(
                N, C, se_k_, (sycl::half*)output,
                (const sycl::half*)transformed_output, (const sycl::half*)input,
                (const sycl::half*)biases1_, (const sycl::half*)w1_,
                (const sycl::half*)b1_, (const sycl::half*)w2_,
                (const sycl::half*)b2_, sycl_queue);
          } else {
            OutputTransform<DataType, true, ACTIVATION_RELU, true, true, true,
                            false>(N, C, se_k_, output, transformed_output,
                                   input, biases1_, w1_, b1_, w2_, b2_,
                                   sycl_queue);
          }
        } else {
          OutputTransform<DataType, true, ACTIVATION_RELU, true, true, true,
                          false>(N, C, se_k_, output, transformed_output, input,
                                 biases1_, w1_, b1_, w2_, b2_, sycl_queue);
        }
      } else {
        if constexpr (std::is_same_v<DataType, sycl::half>) {
          if (static_cast<size_t>(C) <= max_wg_size_) {
            SubGroupOutputTransform_NoSE<ACTIVATION_RELU, true, true, true,
                                        false>(
                N, C, (sycl::half*)output,
                (const sycl::half*)transformed_output,
                (const sycl::half*)input, (const sycl::half*)biases1_,
                sycl_queue);
          } else {
            OutputTransform<DataType, false, ACTIVATION_RELU, true, true, true,
                            false>(N, C, se_k_, output, transformed_output,
                                   input, biases1_, w1_, b1_, w2_, b2_,
                                   sycl_queue);
          }
        } else {
          OutputTransform<DataType, false, ACTIVATION_RELU, true, true, true,
                          false>(N, C, se_k_, output, transformed_output, input,
                                 biases1_, w1_, b1_, w2_, b2_, sycl_queue);
        }
      }
    } else {
      if (has_se_) {
        if (allowFusing) {
          if constexpr (std::is_same_v<DataType, sycl::half>) {
            SubGroupOutputInputTransform<ACTIVATION_RELU, true, true>(
                N, C, se_k_, (sycl::half*)output,
                (const sycl::half*)transformed_output, (const sycl::half*)input,
                (const sycl::half*)biases1_, (const sycl::half*)w1_,
                (const sycl::half*)b1_, (const sycl::half*)w2_,
                (const sycl::half*)b2_, sycl_queue);
          } else {
            OutputInputTransform<DataType, true, ACTIVATION_RELU, true, true>(
                N, C, se_k_, output, transformed_output, input, biases1_, w1_,
                b1_, w2_, b2_, sycl_queue);
          }
        } else {
          OutputTransform<DataType, true, ACTIVATION_RELU, true, true, true,
                          true>(N, C, se_k_, (DataType*)input,
                                transformed_output, input, biases1_, w1_, b1_,
                                w2_, b2_, sycl_queue);
          InputTransform<DataType, true>(N, C, output, (DataType*)input,
                                         sycl_queue);
        }
      } else {
        if constexpr (std::is_same_v<DataType, sycl::half>) {
          if (static_cast<size_t>(C) <= max_wg_size_) {
            SubGroupOutputInputTransform_NoSE<ACTIVATION_RELU, true, true>(
                N, C, (sycl::half*)output,
                (const sycl::half*)transformed_output,
                (const sycl::half*)input, (const sycl::half*)biases1_,
                sycl_queue);
          } else {
            OutputInputTransform<DataType, false, ACTIVATION_RELU, true, true>(
                N, C, se_k_, output, transformed_output, input, biases1_, w1_,
                b1_, w2_, b2_, sycl_queue);
          }
        } else {
          OutputInputTransform<DataType, false, ACTIVATION_RELU, true, true>(
              N, C, se_k_, output, transformed_output, input, biases1_, w1_,
              b1_, w2_, b2_, sycl_queue);
        }
      }
    }
  } else if (act_ == ACTIVATION_MISH) {
    if (last_block_) {
      if (has_se_) {
        if (allowFusing) {
          if constexpr (std::is_same_v<DataType, sycl::half>) {
            SubGroupOutputTransform<ACTIVATION_MISH, true, true, true, false>(
                N, C, se_k_, (sycl::half*)output,
                (const sycl::half*)transformed_output, (const sycl::half*)input,
                (const sycl::half*)biases1_, (const sycl::half*)w1_,
                (const sycl::half*)b1_, (const sycl::half*)w2_,
                (const sycl::half*)b2_, sycl_queue);
          } else {
            OutputTransform<DataType, true, ACTIVATION_MISH, true, true, true,
                            false>(N, C, se_k_, output, transformed_output,
                                   input, biases1_, w1_, b1_, w2_, b2_,
                                   sycl_queue);
          }
        } else {
          OutputTransform<DataType, true, ACTIVATION_MISH, true, true, true,
                          false>(N, C, se_k_, output, transformed_output, input,
                                 biases1_, w1_, b1_, w2_, b2_, sycl_queue);
        }
      } else {
        if constexpr (std::is_same_v<DataType, sycl::half>) {
          if (static_cast<size_t>(C) <= max_wg_size_) {
            SubGroupOutputTransform_NoSE<ACTIVATION_MISH, true, true, true,
                                        false>(
                N, C, (sycl::half*)output,
                (const sycl::half*)transformed_output,
                (const sycl::half*)input, (const sycl::half*)biases1_,
                sycl_queue);
          } else {
            OutputTransform<DataType, false, ACTIVATION_MISH, true, true, true,
                            false>(N, C, se_k_, output, transformed_output,
                                   input, biases1_, w1_, b1_, w2_, b2_,
                                   sycl_queue);
          }
        } else {
          OutputTransform<DataType, false, ACTIVATION_MISH, true, true, true,
                          false>(N, C, se_k_, output, transformed_output, input,
                                 biases1_, w1_, b1_, w2_, b2_, sycl_queue);
        }
      }
    } else {
      if (has_se_) {
        if (allowFusing) {
          if constexpr (std::is_same_v<DataType, sycl::half>) {
            SubGroupOutputInputTransform<ACTIVATION_MISH, true, true>(
                N, C, se_k_, (sycl::half*)output,
                (const sycl::half*)transformed_output, (const sycl::half*)input,
                (const sycl::half*)biases1_, (const sycl::half*)w1_,
                (const sycl::half*)b1_, (const sycl::half*)w2_,
                (const sycl::half*)b2_, sycl_queue);
          } else {
            OutputInputTransform<DataType, true, ACTIVATION_MISH, true, true>(
                N, C, se_k_, output, transformed_output, input, biases1_, w1_,
                b1_, w2_, b2_, sycl_queue);
          }
        } else {
          OutputTransform<DataType, true, ACTIVATION_MISH, true, true, true,
                          true>(N, C, se_k_, (DataType*)input,
                                transformed_output, input, biases1_, w1_, b1_,
                                w2_, b2_, sycl_queue);
          InputTransform<DataType, true>(N, C, output, (DataType*)input,
                                         sycl_queue);
        }
      } else {
        if constexpr (std::is_same_v<DataType, sycl::half>) {
          if (static_cast<size_t>(C) <= max_wg_size_) {
            SubGroupOutputInputTransform_NoSE<ACTIVATION_MISH, true, true>(
                N, C, (sycl::half*)output,
                (const sycl::half*)transformed_output,
                (const sycl::half*)input, (const sycl::half*)biases1_,
                sycl_queue);
          } else {
            OutputInputTransform<DataType, false, ACTIVATION_MISH, true, true>(
                N, C, se_k_, output, transformed_output, input, biases1_, w1_,
                b1_, w2_, b2_, sycl_queue);
          }
        } else {
          OutputInputTransform<DataType, false, ACTIVATION_MISH, true, true>(
              N, C, se_k_, output, transformed_output, input, biases1_, w1_,
              b1_, w2_, b2_, sycl_queue);
        }
      }
    }
  }
  // "output" tensor now contains transformed input for the next
  // convolution
}

template <typename DataType>
ResidualBlock<DataType>::~ResidualBlock() {
  sycl::free(transformed_weights0_, sycl_queue_);
  sycl::free(biases0_, sycl_queue_);
  sycl::free(transformed_weights1_, sycl_queue_);
  sycl::free(biases1_, sycl_queue_);
  if (has_se_) {
    sycl::free(w1_, sycl_queue_);
    sycl::free(w2_, sycl_queue_);
    sycl::free(b1_, sycl_queue_);
    sycl::free(b2_, sycl_queue_);
  }
}

template <typename DataType>
void allocAndUpload(DataType** gpu_dest, std::vector<float> cpu_src,
                    void* scratch, sycl::queue& sycl_queue) {
  size_t size = cpu_src.size() * sizeof(DataType);
  if (size == 0) {
    *gpu_dest = nullptr;
    return;
  }

  auto deleter = [&sycl_queue](DataType* ptr) { sycl::free(ptr, sycl_queue); };
  std::unique_ptr<DataType, decltype(deleter)> ptr_guard(
      (DataType*)sycl::malloc_device(size, sycl_queue), deleter);

  sycl_queue.memcpy(scratch, &cpu_src[0], cpu_src.size() * sizeof(float))
      .wait();

  copyTypeConverted((DataType*)ptr_guard.get(), (float*)scratch,
                    (int)cpu_src.size(), sycl_queue);

  *gpu_dest = ptr_guard.release();
}

template <typename DataType>
AttentionPolicyHead<DataType>::AttentionPolicyHead(
    BaseLayer<DataType>* ip, const MultiHeadWeights::PolicyHead& weights,
    void* scratch, bool attention_body, ActivationFunction act,
    int max_batch_size, sycl::queue& sycl_queue)
    : BaseLayer<DataType>(64 * 64 + 24 * 8, 1, 1, ip, sycl_queue),
      attention_body_(attention_body),
      // Old networks without attention body (e.g. T79) use hardcoded SELU
      // activations.
      act_(attention_body ? act : ACTIVATION_SELU) {
  embedding_op_size_ = weights.ip_pol_b.size();
  wq_op_size_ = weights.ip2_pol_w.size() / embedding_op_size_;
  wk_op_size_ = weights.ip3_pol_w.size() / embedding_op_size_;
  if (wq_op_size_ <= 0 || wk_op_size_ != wq_op_size_ ||
      static_cast<size_t>(wq_op_size_ * embedding_op_size_) !=
          weights.ip2_pol_w.size() ||
      static_cast<size_t>(wk_op_size_ * embedding_op_size_) !=
          weights.ip3_pol_w.size()) {
    throw Exception("Invalid attention policy projection dimensions.");
  }

  encoder_heads_ = weights.pol_encoder_head_count;
  policy_d_model_ = wq_op_size_;

  allocAndUpload<DataType>(&ip_pol_w_, weights.ip_pol_w, scratch, sycl_queue_);
  allocAndUpload<DataType>(&ip_pol_b_, weights.ip_pol_b, scratch, sycl_queue_);

  allocAndUpload<DataType>(&ip2_pol_w_, weights.ip2_pol_w, scratch,
                           sycl_queue_);
  allocAndUpload<DataType>(&ip2_pol_b_, weights.ip2_pol_b, scratch,
                           sycl_queue_);

  allocAndUpload<DataType>(&ip3_pol_w_, weights.ip3_pol_w, scratch,
                           sycl_queue_);
  allocAndUpload<DataType>(&ip3_pol_b_, weights.ip3_pol_b, scratch,
                           sycl_queue_);

  // big allocation to hold wq and wk weights one after the other
  {
    size_t elements = weights.ip2_pol_w.size();
    assert(elements == weights.ip3_pol_w.size());

    size_t size = elements * sizeof(DataType) * 2;
    wqk_w_ = (DataType*)sycl::malloc_device(size, sycl_queue_);
    sycl_queue_.memcpy(wqk_w_, ip2_pol_w_, size / 2);

    sycl_queue_.memcpy(wqk_w_ + elements, ip3_pol_w_, size / 2);

    elements = weights.ip2_pol_b.size();
    if (elements != 0) {
      if (elements != weights.ip3_pol_b.size()) {
        throw Exception("Attention policy Q/K bias dimensions differ.");
      }
      size = elements * sizeof(DataType) * 2;
      wqk_b_ = (DataType*)sycl::malloc_device(size, sycl_queue_);
      sycl_queue_.memcpy(wqk_b_, ip2_pol_b_, size / 2);
      sycl_queue_.memcpy(wqk_b_ + elements, ip3_pol_b_, size / 2);
    }
  }

  allocAndUpload<DataType>(&ip4_pol_w_, weights.ip4_pol_w, scratch,
                           sycl_queue_);

  for (const auto& enc : weights.pol_encoder) {
    EncoderBlock<DataType>* pW = new EncoderBlock<DataType>(
        enc, scratch, encoder_heads_, embedding_op_size_,
        1.0f,        // using alpha = 1 for now (TODO: may change?)
        nullptr, 0,  // smolgen weights not implemented in
                     // policy encoder heads yet.
        max_batch_size, ACTIVATION_SWISH, act_, 1e-6, {},
        sycl_queue_);  // attentionbody nets don't have policy encoders, so
                       // using old epsilon for backward compatibility with T78.
    encoder_weights_.emplace_back(std::unique_ptr<EncoderBlock<DataType>>(pW));
  }
}

template <typename DataType>
EncoderBlock<DataType>::EncoderBlock(
    const MultiHeadWeights::EncoderLayer& cpu_weights, void* scratch, int heads,
    int size, float alpha, DataType* smolgen_global_scratch,
    int smolgen_global_size, int max_batch_size, ActivationFunction smolgen_act,
    ActivationFunction ffn_act, float default_eps,
    const std::vector<int>& kda_directions, sycl::queue& sycl_queue)
    : embedding_op_size_(size),
      encoder_heads_(heads),
      is_kda_(cpu_weights.is_kda),
      kda_key_dim_(cpu_weights.kda.key_dim),
      kda_value_dim_(cpu_weights.kda.value_dim),
      kda_gate_rank_(cpu_weights.kda.gate_rank),
      kda_rms_norm_epsilon_(cpu_weights.kda.rms_norm_epsilon),
      kda_output_gate_(cpu_weights.kda.output_gate),
      kda_output_rms_norm_(cpu_weights.kda.output_rms_norm),
      kda_local_conv_(cpu_weights.kda.local_conv),
      kda_qkv_silu_(cpu_weights.kda.qkv_silu),
      kda_direction_count_(static_cast<int>(kda_directions.size())),
      alpha_(alpha),
      has_smolgen_(!cpu_weights.is_kda && cpu_weights.mha.has_smolgen),
      smolgen_activation_(smolgen_act),
      ffn_activation_(ffn_act),
      max_batch_size_(max_batch_size),
      default_eps_(default_eps),
      sycl_queue_(sycl_queue) {
  ffn_dense1_size_ = cpu_weights.ffn.dense1_b.size();
  ffn_dense2_size_ = cpu_weights.ffn.dense2_b.size();

  if (is_kda_) {
    if (kda_key_dim_ <= 0 || kda_value_dim_ <= 0 || kda_gate_rank_ <= 0) {
      throw Exception("Invalid KDA dimensions.");
    }
    if (kda_key_dim_ > 32) {
      // kdaRecurrenceValueParallel keeps the recurrence state in a
      // fixed-size private `float state[32]` array per work-item.
      throw Exception("KDA key_dim > 32 is not supported by the SYCL "
                      "backend.");
    }
    {
      // kdaRecurrenceValueParallel launches one work-group of kda_value_dim_
      // work-items per (batch, head). An oversized config would otherwise
      // fail at kernel launch with an opaque runtime error instead of here.
      const size_t max_wg_size =
          sycl_queue_.get_device()
              .get_info<sycl::info::device::max_work_group_size>();
      if (static_cast<size_t>(kda_value_dim_) > max_wg_size) {
        throw Exception(
            "KDA value_dim (" + std::to_string(kda_value_dim_) +
            ") exceeds the device's max work-group size (" +
            std::to_string(max_wg_size) + ").");
      }
    }
    if (kda_direction_count_ <= 0 || kda_direction_count_ > 16 ||
        encoder_heads_ % kda_direction_count_ != 0) {
      throw Exception("KDA directions must evenly divide the encoder heads.");
    }
    // EvalKda() feeds these KDA-derived widths to addBiasBatched
    // (vectorized over groups of 4 channels) and LayerNorm (over 16), which
    // throw at *eval* time on non-divisible widths -- i.e. after a full
    // successful-looking load. A valid proto can carry odd dims, so reject
    // them here, where the message can name the offending dimension.
    {
      const int key_depth_ld = encoder_heads_ * kda_key_dim_;
      const int value_depth_ld = encoder_heads_ * kda_value_dim_;
      auto require_divisible = [](const char* what, int value, int divisor) {
        if (value % divisor != 0) {
          throw Exception(
              std::string("KDA ") + what + " (" + std::to_string(value) +
              ") must be a multiple of " + std::to_string(divisor) +
              " for the SYCL backend's vectorized kernels.");
        }
      };
      require_divisible("key_depth (heads * key_dim)", key_depth_ld, 4);
      require_divisible("value_depth (heads * value_dim)", value_depth_ld, 4);
      // gate_rank % 4 covers both the fused decay/gate bias width
      // (2*gate_rank) and the unfused per-projection width (gate_rank).
      require_divisible("gate_rank", kda_gate_rank_, 4);
      require_divisible("encoder head count", encoder_heads_, 4);
      require_divisible("embedding size", embedding_op_size_, 16);
    }
    for (int i = 0; i < kda_direction_count_; ++i) {
      if (kda_directions[i] < 1 || kda_directions[i] > 16) {
        throw Exception("Unsupported KDA traversal direction.");
      }
      kda_directions_[i] = kda_directions[i];
    }
    if (kda_output_rms_norm_ && cpu_weights.kda.out_norm_gammas.empty()) {
      throw Exception("KDA output RMS norm requested but out_norm_gammas is "
                      "missing from the network.");
    }
    if (kda_local_conv_ &&
        encoder_heads_ * kda_key_dim_ < embedding_op_size_) {
      // EvalKda() reuses buffer2 -- sized max_tokens * key_depth for the
      // later gate computation -- as scratch for the local-conv output,
      // which needs max_tokens * embedding_op_size_. Without this check a
      // net with too few key dims would silently write past the buffer.
      throw Exception(
          "KDA local_conv requires encoder_heads * kda_key_dim >= "
          "embedding_op_size.");
    }

    // KDA's proto makes most biases optional (unlike MHA), so a malformed
    // net silently becomes an out-of-bounds gemm/kernel read instead of a
    // load-time error unless every weight is shape-checked here.
    const int key_depth = encoder_heads_ * kda_key_dim_;
    const int value_depth = encoder_heads_ * kda_value_dim_;
    auto require_exact = [](const char* name, const std::vector<float>& v,
                            size_t expected) {
      if (v.size() != expected) {
        throw Exception(
            "KDA weight '" + std::string(name) + "' has size " +
            std::to_string(v.size()) + ", expected " +
            std::to_string(expected) + ".");
      }
    };
    auto check_if_present = [](const char* name, const std::vector<float>& v,
                               size_t expected) {
      if (!v.empty() && v.size() != expected) {
        throw Exception(
            "KDA weight '" + std::string(name) + "' has size " +
            std::to_string(v.size()) + ", expected " +
            std::to_string(expected) + ".");
      }
    };
    require_exact("q_w", cpu_weights.kda.q_w,
                  static_cast<size_t>(embedding_op_size_) * key_depth);
    require_exact("k_w", cpu_weights.kda.k_w,
                  static_cast<size_t>(embedding_op_size_) * key_depth);
    require_exact("v_w", cpu_weights.kda.v_w,
                  static_cast<size_t>(embedding_op_size_) * value_depth);
    require_exact("decay_a_w", cpu_weights.kda.decay_a_w,
                  static_cast<size_t>(embedding_op_size_) * kda_gate_rank_);
    require_exact("decay_b_w", cpu_weights.kda.decay_b_w,
                  static_cast<size_t>(kda_gate_rank_) * key_depth);
    require_exact("beta_w", cpu_weights.kda.beta_w,
                  static_cast<size_t>(embedding_op_size_) * encoder_heads_);
    require_exact("dt_bias", cpu_weights.kda.dt_bias,
                  static_cast<size_t>(key_depth));
    require_exact("a_log", cpu_weights.kda.a_log,
                  static_cast<size_t>(encoder_heads_));
    require_exact("dense_w", cpu_weights.kda.dense_w,
                  static_cast<size_t>(value_depth) * embedding_op_size_);
    // dense_b is required, not optional: EvalKda's LN1 dereferences it
    // unconditionally (unlike q_b/k_b/v_b/etc., which are only applied if
    // present).
    require_exact("dense_b", cpu_weights.kda.dense_b,
                  static_cast<size_t>(embedding_op_size_));
    check_if_present("q_b", cpu_weights.kda.q_b,
                     static_cast<size_t>(key_depth));
    check_if_present("k_b", cpu_weights.kda.k_b,
                     static_cast<size_t>(key_depth));
    check_if_present("v_b", cpu_weights.kda.v_b,
                     static_cast<size_t>(value_depth));
    check_if_present("decay_a_b", cpu_weights.kda.decay_a_b,
                     static_cast<size_t>(kda_gate_rank_));
    check_if_present("decay_b_b", cpu_weights.kda.decay_b_b,
                     static_cast<size_t>(key_depth));
    check_if_present("beta_b", cpu_weights.kda.beta_b,
                     static_cast<size_t>(encoder_heads_));
    if (kda_output_rms_norm_) {
      require_exact("out_norm_gammas", cpu_weights.kda.out_norm_gammas,
                    static_cast<size_t>(value_depth));
    }
    if (kda_output_gate_) {
      // Required, not optional: EvalKda's non-fused branch gemms with
      // kda_gate_a_w/kda_gate_b_w unconditionally once output_gate is set.
      require_exact("gate_a_w", cpu_weights.kda.gate_a_w,
                    static_cast<size_t>(embedding_op_size_) * kda_gate_rank_);
      require_exact("gate_b_w", cpu_weights.kda.gate_b_w,
                    static_cast<size_t>(kda_gate_rank_) * value_depth);
      check_if_present("gate_a_b", cpu_weights.kda.gate_a_b,
                       static_cast<size_t>(kda_gate_rank_));
      check_if_present("gate_b_b", cpu_weights.kda.gate_b_b,
                       static_cast<size_t>(value_depth));
    }
    if (kda_local_conv_) {
      require_exact("local_conv_w", cpu_weights.kda.local_conv_w,
                    static_cast<size_t>(embedding_op_size_) * 9);
      check_if_present("local_conv_b", cpu_weights.kda.local_conv_b,
                       static_cast<size_t>(embedding_op_size_));
    }

    allocAndUpload<DataType>(&kda_q_w, cpu_weights.kda.q_w, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_q_b, cpu_weights.kda.q_b, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_k_w, cpu_weights.kda.k_w, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_k_b, cpu_weights.kda.k_b, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_v_w, cpu_weights.kda.v_w, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_v_b, cpu_weights.kda.v_b, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_decay_a_w, cpu_weights.kda.decay_a_w,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&kda_decay_a_b, cpu_weights.kda.decay_a_b,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&kda_decay_b_w, cpu_weights.kda.decay_b_w,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&kda_decay_b_b, cpu_weights.kda.decay_b_b,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&kda_beta_w, cpu_weights.kda.beta_w, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_beta_b, cpu_weights.kda.beta_b, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_a_log, cpu_weights.kda.a_log, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_dt_bias, cpu_weights.kda.dt_bias, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_gate_a_w, cpu_weights.kda.gate_a_w, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_gate_a_b, cpu_weights.kda.gate_a_b, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_gate_b_w, cpu_weights.kda.gate_b_w, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_gate_b_b, cpu_weights.kda.gate_b_b, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_out_norm_gammas,
                             cpu_weights.kda.out_norm_gammas, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_dense_w, cpu_weights.kda.dense_w, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&kda_dense_b, cpu_weights.kda.dense_b, scratch,
                             sycl_queue_);

    if (kda_local_conv_) {
      allocAndUpload<DataType>(&kda_local_conv_w, cpu_weights.kda.local_conv_w,
                               scratch, sycl_queue_);
      if (!cpu_weights.kda.local_conv_b.empty()) {
        allocAndUpload<DataType>(&kda_local_conv_b, cpu_weights.kda.local_conv_b,
                                 scratch, sycl_queue_);
      }
    }

    const int qkv_depth = 2 * key_depth + value_depth;
    size_t qkv_w_elements = static_cast<size_t>(qkv_depth) * embedding_op_size_;
    kda_qkv_w = static_cast<DataType*>(sycl::malloc_device(
        qkv_w_elements * sizeof(DataType), sycl_queue_));
    sycl_queue_.memcpy(kda_qkv_w, kda_q_w,
                       cpu_weights.kda.q_w.size() * sizeof(DataType));
    sycl_queue_.memcpy(kda_qkv_w + cpu_weights.kda.q_w.size(), kda_k_w,
                       cpu_weights.kda.k_w.size() * sizeof(DataType));
    sycl_queue_.memcpy(
        kda_qkv_w + cpu_weights.kda.q_w.size() + cpu_weights.kda.k_w.size(),
        kda_v_w, cpu_weights.kda.v_w.size() * sizeof(DataType));

    if (kda_q_b && kda_k_b && kda_v_b) {
      kda_qkv_b = static_cast<DataType*>(
          sycl::malloc_device(qkv_depth * sizeof(DataType), sycl_queue_));
      sycl_queue_.memcpy(kda_qkv_b, kda_q_b,
                         cpu_weights.kda.q_b.size() * sizeof(DataType));
      sycl_queue_.memcpy(kda_qkv_b + cpu_weights.kda.q_b.size(), kda_k_b,
                         cpu_weights.kda.k_b.size() * sizeof(DataType));
      sycl_queue_.memcpy(
          kda_qkv_b + cpu_weights.kda.q_b.size() + cpu_weights.kda.k_b.size(),
          kda_v_b, cpu_weights.kda.v_b.size() * sizeof(DataType));
    }

    if (kda_output_gate_ && !cpu_weights.kda.gate_a_w.empty()) {
      size_t decay_gate_a_elements =
          static_cast<size_t>(2 * kda_gate_rank_) * embedding_op_size_;
      kda_decay_gate_a_w = static_cast<DataType*>(sycl::malloc_device(
          decay_gate_a_elements * sizeof(DataType), sycl_queue_));
      sycl_queue_.memcpy(
          kda_decay_gate_a_w, kda_decay_a_w,
          cpu_weights.kda.decay_a_w.size() * sizeof(DataType));
      sycl_queue_.memcpy(
          kda_decay_gate_a_w + cpu_weights.kda.decay_a_w.size(), kda_gate_a_w,
          cpu_weights.kda.gate_a_w.size() * sizeof(DataType));

      if (kda_decay_a_b && kda_gate_a_b) {
        kda_decay_gate_a_b = static_cast<DataType*>(sycl::malloc_device(
            (2 * kda_gate_rank_) * sizeof(DataType), sycl_queue_));
        sycl_queue_.memcpy(
            kda_decay_gate_a_b, kda_decay_a_b,
            cpu_weights.kda.decay_a_b.size() * sizeof(DataType));
        sycl_queue_.memcpy(
            kda_decay_gate_a_b + cpu_weights.kda.decay_a_b.size(),
            kda_gate_a_b, cpu_weights.kda.gate_a_b.size() * sizeof(DataType));
      }
    }
  } else {
    mha_q_size_ = cpu_weights.mha.q_w.size() / embedding_op_size_;
    mha_k_size_ = cpu_weights.mha.k_w.size() / embedding_op_size_;
    mha_v_size_ = cpu_weights.mha.v_w.size() / embedding_op_size_;
    mha_dense_size_ = cpu_weights.mha.dense_b.size();
    if (mha_q_size_ <= 0 || mha_k_size_ != mha_q_size_ ||
      mha_v_size_ != mha_q_size_ ||
      static_cast<size_t>(mha_q_size_ * embedding_op_size_) !=
        cpu_weights.mha.q_w.size() ||
      static_cast<size_t>(mha_k_size_ * embedding_op_size_) !=
        cpu_weights.mha.k_w.size() ||
      static_cast<size_t>(mha_v_size_ * embedding_op_size_) !=
        cpu_weights.mha.v_w.size()) {
      throw Exception("Invalid MHA projection dimensions.");
    }

    allocAndUpload<DataType>(&mha_q_w, cpu_weights.mha.q_w, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&mha_q_b, cpu_weights.mha.q_b, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&mha_k_w, cpu_weights.mha.k_w, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&mha_k_b, cpu_weights.mha.k_b, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&mha_v_w, cpu_weights.mha.v_w, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&mha_v_b, cpu_weights.mha.v_b, scratch,
                             sycl_queue_);

    size_t elements = cpu_weights.mha.q_w.size();
    size_t size = elements * sizeof(DataType) * 3;
    mha_qkv_w = (DataType*)sycl::malloc_device(size, sycl_queue_);
    sycl_queue_.memcpy(mha_qkv_w, mha_q_w, size / 3);
    sycl_queue_.memcpy(mha_qkv_w + elements, mha_k_w, size / 3);
    sycl_queue_.memcpy(mha_qkv_w + elements * 2, mha_v_w, size / 3);

    elements = cpu_weights.mha.q_b.size();
    if (elements != 0) {
      if (elements != cpu_weights.mha.k_b.size() ||
          elements != cpu_weights.mha.v_b.size()) {
        throw Exception("MHA Q/K/V bias dimensions differ.");
      }
      size = elements * sizeof(DataType) * 3;
      mha_qkv_b = (DataType*)sycl::malloc_device(size, sycl_queue_);
      sycl_queue_.memcpy(mha_qkv_b, mha_q_b, size / 3);
      sycl_queue_.memcpy(mha_qkv_b + elements, mha_k_b, size / 3);
      sycl_queue_.memcpy(mha_qkv_b + elements * 2, mha_v_b, size / 3);
    }

    allocAndUpload<DataType>(&mha_dense_w, cpu_weights.mha.dense_w, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&mha_dense_b, cpu_weights.mha.dense_b, scratch,
                             sycl_queue_);
  }

  allocAndUpload<DataType>(&ln1_gammas, cpu_weights.ln1_gammas, scratch,
                           sycl_queue_);
  allocAndUpload<DataType>(&ln1_betas, cpu_weights.ln1_betas, scratch,
                           sycl_queue_);

  allocAndUpload<DataType>(&ffn_dense1_w, cpu_weights.ffn.dense1_w, scratch,
                           sycl_queue_);
  allocAndUpload<DataType>(&ffn_dense1_b, cpu_weights.ffn.dense1_b, scratch,
                           sycl_queue_);

  allocAndUpload<DataType>(&ffn_dense2_w, cpu_weights.ffn.dense2_w, scratch,
                           sycl_queue_);
  allocAndUpload<DataType>(&ffn_dense2_b, cpu_weights.ffn.dense2_b, scratch,
                           sycl_queue_);
  if (!ffn_dense1_b || !ffn_dense2_b) {
    // addBiasBatched() dereferences these unconditionally in Eval/EvalKda;
    // a net that omits them loads fine but crashes mid-eval instead of here.
    throw Exception("FFN dense1/dense2 bias is missing from the network.");
  }

  allocAndUpload<DataType>(&ln2_gammas, cpu_weights.ln2_gammas, scratch,
                           sycl_queue_);
  allocAndUpload<DataType>(&ln2_betas, cpu_weights.ln2_betas, scratch,
                           sycl_queue_);

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
                             cpu_weights.mha.smolgen.ln1_gammas, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&smol_ln1_betas, cpu_weights.mha.smolgen.ln1_betas,
                             scratch, sycl_queue_);
    allocAndUpload<DataType>(&smol_ln2_gammas,
                             cpu_weights.mha.smolgen.ln2_gammas, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&smol_ln2_betas, cpu_weights.mha.smolgen.ln2_betas,
                             scratch, sycl_queue_);

    // GPU memory already allocated in AttentionBody.
    smol_global = smolgen_global_scratch;
  }
}

template <typename DataType>
static void cublasXgemm(transpose_type transa, transpose_type transb, int m,
                        int n, int k, float alpha, const DataType* A, int lda,
                        const DataType* B, int ldb, float beta, DataType* C,
                        int ldc, sycl::queue& sycl_queue) {
  const bool fp16 = std::is_same<sycl::half, DataType>::value;

#ifdef USE_CUBLAS
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);
        ReportCUBLASErrors(cublasHgemm(handle, transa, transb, m, n, k,
                                       (const half*)&alpha_h, ((const half*)A),
                                       lda, ((const half*)B), ldb,
                                       (const half*)&beta_h, ((half*)C), ldc));
      });
    });
  } else {
    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);
        ReportCUBLASErrors(cublasSgemm(handle, transa, transb, m, n, k, &alpha,
                                       (const float*)A, lda, (const float*)B,
                                       ldb, &beta, (float*)C, ldc));
      });
    });
  }
#elif defined(USE_HIPBLAS)
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
        hipblasSetStream(handle, hipStreamHandle);
        hipblasHgemm(handle, transa, transb, m, n, k, &alpha_h,
                     (const hipblasHalf*)A, lda, (const hipblasHalf*)B, ldb,
                     &beta_h, (hipblasHalf*)C, ldc);
      });
    });
  } else {
    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
        hipblasSetStream(handle, hipStreamHandle);
        hipblasSgemm(handle, transa, transb, m, n, k, &alpha, (const float*)A,
                     lda, (const float*)B, ldb, &beta, (float*)C, ldc);
      });
    });
  }
#else
  const DataType alpha_t = static_cast<DataType>(alpha);
  const DataType beta_t = static_cast<DataType>(beta);
  oneapi::mkl::blas::column_major::gemm(
      sycl_queue, transa, transb, static_cast<std::int64_t>(m),
      static_cast<std::int64_t>(n), static_cast<std::int64_t>(k), alpha_t,
      (const DataType*)A, static_cast<std::int64_t>(lda), (const DataType*)B,
      static_cast<std::int64_t>(ldb), beta_t, (DataType*)C,
      static_cast<std::int64_t>(ldc));
#endif
}

template <typename DataType>
static void cublasXGemmStridedBatched(
    transpose_type transa, transpose_type transb, int m, int n, int k,
    float alpha, const void* A, int lda, long long int strideA, const void* B,
    int ldb, long long int strideB, float beta, void* C, int ldc,
    long long int strideC, int batchCount, sycl::queue& sycl_queue) {
  const bool fp16 = std::is_same<sycl::half, DataType>::value;

#ifdef USE_CUBLAS
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);

    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);

        ReportCUBLASErrors(cublasGemmStridedBatchedEx(
            handle, transa, transb, m, n, k, &alpha_h, A, CUDA_R_16F, lda,
            strideA, B, CUDA_R_16F, ldb, strideB, &beta_h, C, CUDA_R_16F, ldc,
            strideC, batchCount, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
      });
    });

  } else {
    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);

        ReportCUBLASErrors(cublasGemmStridedBatchedEx(
            handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda,
            strideA, B, CUDA_R_32F, ldb, strideB, &beta, C, CUDA_R_32F, ldc,
            strideC, batchCount, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
      });
    });
  }
#elif defined(USE_HIPBLAS)
  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);

    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);

        hipblasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha_h,
                                    A, HIPBLAS_R_16F, lda, strideA, B,
                                    HIPBLAS_R_16F, ldb, strideB, &beta_h, C,
                                    HIPBLAS_R_16F, ldc, strideC, batchCount,
                                    HIPBLAS_COMPUTE_16F, HIPBLAS_GEMM_DEFAULT);
      });
    });
  } else {
    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);

        hipblasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha, A,
                                    HIPBLAS_R_32F, lda, strideA, B,
                                    HIPBLAS_R_32F, ldb, strideB, &beta, C,
                                    HIPBLAS_R_32F, ldc, strideC, batchCount,
                                    HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
      });
    });
  }
#else
  const DataType alpha_t = static_cast<DataType>(alpha);
  const DataType beta_t = static_cast<DataType>(beta);
  oneapi::mkl::blas::column_major::gemm_batch(
      sycl_queue, transa, transb, static_cast<std::int64_t>(m),
      static_cast<std::int64_t>(n), static_cast<std::int64_t>(k), alpha_t,
      static_cast<const DataType*>(A), static_cast<std::int64_t>(lda),
      static_cast<std::int64_t>(strideA), static_cast<const DataType*>(B),
      static_cast<std::int64_t>(ldb), static_cast<std::int64_t>(strideB),
      beta_t, static_cast<DataType*>(C), static_cast<std::int64_t>(ldc),
      static_cast<std::int64_t>(strideC),
      static_cast<std::int64_t>(batchCount));
#endif
}

template <typename DataType>
static void cublasXGemmBatched(transpose_type transa, transpose_type transb,
                               int m, int n, int k, float alpha, DataType** A,
                               int lda, DataType** B, int ldb, float beta,
                               DataType** C, int ldc, int batchCount,
                               sycl::queue& sycl_queue) {
  const bool fp16 = std::is_same<sycl::half, DataType>::value;

#ifdef USE_CUBLAS
  cublasHandle_t handle = cuBlasContextManager::getcuBlasHandle_t();

  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);

    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);

        ReportCUBLASErrors(cublasHgemmBatched(
            handle, transa, transb, m, n, k, (const half*)&alpha_h, (half**)A,
            lda, (half**)B, ldb, (const half*)&beta_h, (half**)C, ldc,
            batchCount));
      });
    });

  } else {
    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto cudaStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cudaStreamHandle);

        ReportCUBLASErrors(cublasSgemmBatched(
            handle, transa, transb, m, n, k, &alpha, (float**)A, lda,
            (float**)B, ldb, &beta, (float**)C, ldc, batchCount));
      });
    });
  }

#elif defined(USE_HIPBLAS)

  hipblasHandle_t handle = hipBlasContextManager::gethipBlasHandle_t();

  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);

    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);

        hipblasHgemmBatched(handle, transa, transb, m, n, k,
                            (const hipblasHalf*)&alpha_h, (hipblasHalf**)A, lda,
                            (hipblasHalf**)B, ldb, (const hipblasHalf*)&beta_h,
                            (hipblasHalf**)C, ldc, batchCount);
      });
    });

  } else {
    sycl_queue.submit([&](sycl::handler& cgh) {
      cgh.ext_codeplay_enqueue_native_command([=](sycl::interop_handle ih) {
        auto hipStreamHandle =
            ih.get_native_queue<sycl::backend::ext_oneapi_hip>();

        hipblasSetStream(handle, hipStreamHandle);

        hipblasSgemmBatched(handle, transa, transb, m, n, k, &alpha, (float**)A,
                            lda, (float**)B, ldb, &beta, (float**)C, ldc,
                            batchCount);
      });
    });
  }

#else
  const DataType alpha_t = static_cast<DataType>(alpha);
  const DataType beta_t = static_cast<DataType>(beta);
  oneapi::mkl::transpose transa_arr[1] = {transa};
  oneapi::mkl::transpose transb_arr[1] = {transb};
  std::int64_t m_arr[1] = {static_cast<std::int64_t>(m)};
  std::int64_t n_arr[1] = {static_cast<std::int64_t>(n)};
  std::int64_t k_arr[1] = {static_cast<std::int64_t>(k)};
  DataType alpha_arr[1] = {alpha_t};
  DataType beta_arr[1] = {beta_t};
  std::int64_t lda_arr[1] = {static_cast<std::int64_t>(lda)};
  std::int64_t ldb_arr[1] = {static_cast<std::int64_t>(ldb)};
  std::int64_t ldc_arr[1] = {static_cast<std::int64_t>(ldc)};
  std::int64_t group_size[1] = {static_cast<std::int64_t>(batchCount)};
  std::int64_t group_count = 1;

  oneapi::mkl::blas::column_major::gemm_batch(
      sycl_queue, transa_arr, transb_arr, m_arr, n_arr, k_arr, alpha_arr,
      const_cast<const DataType**>(A), lda_arr, const_cast<const DataType**>(B),
      ldb_arr, beta_arr, C, ldc_arr, group_count, group_size);
#endif
}

template <typename DataType>
void EncoderBlock<DataType>::EvalKda(int N, DataType* in_out_tensor,
                                     DataType* scratch, DataType* buffer1,
                                     DataType* buffer2,
                                     sycl::queue& sycl_queue) {
  // Mirrors KDA_LOG_DECAY_FLOOR in the trainer, which clamps the per-token log
  // decay so its chunkwise-parallel recurrence stays in the float32 range.
  constexpr float kKdaLogDecayFloor = -10.0f;
  const int tokens = N * 64;
  const int max_tokens = max_batch_size_ * 64;
  const int key_depth = encoder_heads_ * kda_key_dim_;
  const int value_depth = encoder_heads_ * kda_value_dim_;

  // Apply the optional 3x3 depthwise board convolution before projections.
  // The convolved tensor feeds the projections only -- it must NOT overwrite
  // in_out_tensor, because the encoder residual added at LN1 below is the
  // pre-mixer layer input x, not x + conv(x). The trainer (kda() rebinds a
  // local `inputs`, encoder_layer still adds the original) and the BLAS
  // reference (conv writes a separate conv_input; encoder_buffer stays x)
  // both keep the original input as the skip.
  DataType* proj_input = in_out_tensor;
  if (kda_local_conv_) {
    // Persistent region past q/k/v and the non-fused gate_hidden; it has to
    // stay live until the last projection (beta) reads it. buffer2 is still
    // free here, so it serves as the conv kernel's transient intermediate.
    proj_input =
        scratch + max_tokens * (2 * key_depth + value_depth + kda_gate_rank_);
    applyKdaLocalDepthwiseConv<DataType>(
        N, embedding_op_size_, in_out_tensor, kda_local_conv_w,
        kda_local_conv_b, buffer2, proj_input, sycl_queue);
  }

  DataType* q = scratch;
  DataType* k = q + max_tokens * key_depth;
  DataType* v = k + max_tokens * key_depth;

  // The trainer's qkv_silu applies SiLU (= Swish) to the q/k/v projections.
  const ActivationFunction qkv_act =
      kda_qkv_silu_ ? ACTIVATION_SWISH : ACTIVATION_NONE;

  if (kda_qkv_w) {
    const int qkv_depth = 2 * key_depth + value_depth;
    cublasXgemm<DataType>(
        transpose_type_transpose, transpose_type_notranspose, qkv_depth,
        tokens, embedding_op_size_, 1.0f, kda_qkv_w, embedding_op_size_,
        proj_input, embedding_op_size_, 0.0f, q, qkv_depth, sycl_queue);
    if (kda_qkv_b) {
      addBiasBatched(q, q, kda_qkv_b, 1, tokens, qkv_depth, qkv_act,
                     sycl_queue);
    } else if (qkv_act != ACTIVATION_NONE) {
      // qkv_silu with no fused bias: the activation must still run -- the
      // BLAS reference applies it regardless of bias presence, and folding
      // it into addBiasBatched silently skipped it whenever the biases were
      // absent. addVectors with a null second operand is exactly
      // elementwise-activate here.
      addVectors(q, q, static_cast<DataType*>(nullptr), tokens * qkv_depth,
                 tokens * qkv_depth, 0, qkv_act, sycl_queue);
    }
  } else {
    cublasXgemm<DataType>(
        transpose_type_transpose, transpose_type_notranspose, key_depth,
        tokens, embedding_op_size_, 1.0f, kda_q_w, embedding_op_size_,
        proj_input, embedding_op_size_, 0.0f, q, key_depth, sycl_queue);
    cublasXgemm<DataType>(
        transpose_type_transpose, transpose_type_notranspose, key_depth,
        tokens, embedding_op_size_, 1.0f, kda_k_w, embedding_op_size_,
        proj_input, embedding_op_size_, 0.0f, k, key_depth, sycl_queue);
    cublasXgemm<DataType>(
        transpose_type_transpose, transpose_type_notranspose, value_depth,
        tokens, embedding_op_size_, 1.0f, kda_v_w, embedding_op_size_,
        proj_input, embedding_op_size_, 0.0f, v, value_depth, sycl_queue);
    if (kda_q_b) {
      addBiasBatched(q, q, kda_q_b, 1, tokens, key_depth, qkv_act,
                     sycl_queue);
    }
    if (kda_k_b) {
      addBiasBatched(k, k, kda_k_b, 1, tokens, key_depth, qkv_act,
                     sycl_queue);
    }
    if (kda_v_b) {
      addBiasBatched(v, v, kda_v_b, 1, tokens, value_depth, qkv_act,
                     sycl_queue);
    }
    if (qkv_act != ACTIVATION_NONE) {
      // See the fused-branch comment: apply the activation even where the
      // matching bias is missing.
      if (!kda_q_b) {
        addVectors(q, q, static_cast<DataType*>(nullptr), tokens * key_depth,
                   tokens * key_depth, 0, qkv_act, sycl_queue);
      }
      if (!kda_k_b) {
        addVectors(k, k, static_cast<DataType*>(nullptr), tokens * key_depth,
                   tokens * key_depth, 0, qkv_act, sycl_queue);
      }
      if (!kda_v_b) {
        addVectors(v, v, static_cast<DataType*>(nullptr),
                   tokens * value_depth, tokens * value_depth, 0, qkv_act,
                   sycl_queue);
      }
    }
  }

  DataType* decay_hidden = buffer1;
  DataType* gate_hidden = kda_output_gate_ ? (scratch + max_tokens * (2 * key_depth + value_depth)) : nullptr;
  // Distance between consecutive tokens in decay_hidden / gate_hidden. The
  // fused gemm below writes ONE interleaved row per token
  // ([decay_a | gate_a], width 2*gate_rank, because ldc = 2*gate_rank in
  // column-major), so its consumers have to stride by the full fused width
  // and find the gate half gate_rank into each row. The non-fused path
  // instead writes two separately packed buffers of width gate_rank each.
  // Getting this wrong silently feeds decay_b the previous token's gate_a.
  int decay_hidden_ld = kda_gate_rank_;
  int gate_hidden_ld = kda_gate_rank_;

  if (kda_output_gate_ && kda_decay_gate_a_w) {
    cublasXgemm<DataType>(
        transpose_type_transpose, transpose_type_notranspose,
        2 * kda_gate_rank_, tokens, embedding_op_size_, 1.0f,
        kda_decay_gate_a_w, embedding_op_size_, proj_input,
        embedding_op_size_, 0.0f, decay_hidden, 2 * kda_gate_rank_,
        sycl_queue);
    if (kda_decay_gate_a_b) {
      addBiasBatched(decay_hidden, decay_hidden, kda_decay_gate_a_b, 1,
                     tokens, 2 * kda_gate_rank_, ACTIVATION_NONE, sycl_queue);
    }
    decay_hidden_ld = 2 * kda_gate_rank_;
    gate_hidden_ld = 2 * kda_gate_rank_;
    gate_hidden = decay_hidden + kda_gate_rank_;
  } else {
    cublasXgemm<DataType>(
        transpose_type_transpose, transpose_type_notranspose, kda_gate_rank_,
        tokens, embedding_op_size_, 1.0f, kda_decay_a_w, embedding_op_size_,
        proj_input, embedding_op_size_, 0.0f, decay_hidden, kda_gate_rank_,
        sycl_queue);
    if (kda_decay_a_b) {
      addBiasBatched(decay_hidden, decay_hidden, kda_decay_a_b, 1, tokens,
                     kda_gate_rank_, ACTIVATION_NONE, sycl_queue);
    }
    if (kda_output_gate_) {
      cublasXgemm<DataType>(
          transpose_type_transpose, transpose_type_notranspose, kda_gate_rank_,
          tokens, embedding_op_size_, 1.0f, kda_gate_a_w, embedding_op_size_,
          proj_input, embedding_op_size_, 0.0f, gate_hidden,
          kda_gate_rank_, sycl_queue);
      if (kda_gate_a_b) {
        addBiasBatched(gate_hidden, gate_hidden, kda_gate_a_b, 1, tokens,
                       kda_gate_rank_, ACTIVATION_NONE, sycl_queue);
      }
    }
  }

  DataType* raw_decay = buffer2;
  cublasXgemm<DataType>(
      transpose_type_transpose, transpose_type_notranspose, key_depth, tokens,
      kda_gate_rank_, 1.0f, kda_decay_b_w, kda_gate_rank_, decay_hidden,
      decay_hidden_ld, 0.0f, raw_decay, key_depth, sycl_queue);
  if (kda_decay_b_b) {
    addBiasBatched(raw_decay, raw_decay, kda_decay_b_b, 1, tokens, key_depth,
                   ACTIVATION_NONE, sycl_queue);
  }

  // The gate gemm must run BEFORE the beta gemm. In the fused path
  // gate_hidden points into buffer1 (decay_hidden + gate_rank, spanning up to
  // tokens * 2*gate_rank), while beta is written at
  // buffer1 + max_tokens * value_depth. Those regions are disjoint only while
  // 2*gate_rank <= value_depth; consuming gate_hidden first makes the
  // ordering safe for any dims instead of relying on that inequality.
  DataType* gate = nullptr;
  if (kda_output_gate_) {
    gate = buffer2 + max_tokens * key_depth;
    cublasXgemm<DataType>(
        transpose_type_transpose, transpose_type_notranspose, value_depth,
        tokens, kda_gate_rank_, 1.0f, kda_gate_b_w, kda_gate_rank_,
        gate_hidden, gate_hidden_ld, 0.0f, gate, value_depth, sycl_queue);
    if (kda_gate_b_b) {
      addBiasBatched(gate, gate, kda_gate_b_b, 1, tokens, value_depth,
                     ACTIVATION_NONE, sycl_queue);
    }
  }

  DataType* beta = buffer1 + max_tokens * value_depth;
  cublasXgemm<DataType>(
      transpose_type_transpose, transpose_type_notranspose, encoder_heads_,
      tokens, embedding_op_size_, 1.0f, kda_beta_w, embedding_op_size_,
      proj_input, embedding_op_size_, 0.0f, beta, encoder_heads_,
      sycl_queue);
  if (kda_beta_b) {
    addBiasBatched(beta, beta, kda_beta_b, 1, tokens, encoder_heads_,
                   ACTIVATION_NONE, sycl_queue);
  }

  DataType* mixed = buffer1;
  const DataType* qkv = kda_qkv_w ? scratch : nullptr;
  const int qkv_stride = 2 * key_depth + value_depth;
  // q/k/v describe the non-fused layout and are meaningless once qkv
  // (fused) is set -- the kernel takes the qkv branch exclusively in that
  // case and never dereferences them, so pass nullptr rather than pointers
  // a future reader might mistake for the fused layout.
  kdaRecurrenceValueParallel<DataType>(
      N, encoder_heads_, kda_key_dim_, kda_value_dim_, kda_direction_count_,
      kda_directions_, kKdaLogDecayFloor, qkv, qkv_stride,
      qkv ? nullptr : q, qkv ? nullptr : k, qkv ? nullptr : v, raw_decay,
      kda_dt_bias, kda_a_log, beta, mixed, sycl_queue);

  if (kda_output_rms_norm_) {
    const DataType* norm_gammas = kda_out_norm_gammas;
    const float norm_epsilon = kda_rms_norm_epsilon_;
    sycl_queue.parallel_for(sycl::range<1>(tokens), [=](sycl::id<1> item) {
      const int token = static_cast<int>(item[0]);
      const int offset = token * value_depth;
      float sum_squares = 0.0f;
      for (int channel = 0; channel < value_depth; ++channel) {
        const float val = static_cast<float>(mixed[offset + channel]);
        sum_squares += val * val;
      }
      const float factor =
          1.0f / sycl::sqrt(sum_squares / value_depth + norm_epsilon);
      for (int channel = 0; channel < value_depth; ++channel) {
        const float value = static_cast<float>(mixed[offset + channel]) *
                            factor * static_cast<float>(norm_gammas[channel]);
        mixed[offset + channel] = static_cast<DataType>(value);
      }
    });
  }

  // The gate must apply after the (optional) RMS norm, not before: they do
  // not commute, since the gate rescales each element before the norm would
  // divide by the resulting RMS. The trainer and BLAS reference both
  // normalize first and gate second.
  if (kda_output_gate_) {
    applyKdaOutputGate<DataType>(tokens * value_depth, mixed, gate,
                                 sycl_queue);
  }

  cublasXgemm<DataType>(
      transpose_type_transpose, transpose_type_notranspose,
      embedding_op_size_, tokens, value_depth, 1.0f, kda_dense_w, value_depth,
      mixed, value_depth, 0.0f, buffer2, embedding_op_size_, sycl_queue);
  LayerNorm<DataType>(tokens, embedding_op_size_, scratch, buffer2,
                      kda_dense_b, in_out_tensor, ln1_gammas, ln1_betas,
                      default_eps_, alpha_, ACTIVATION_NONE, sycl_queue);

  cublasXgemm<DataType>(
      transpose_type_transpose, transpose_type_notranspose, ffn_dense1_size_,
      tokens, embedding_op_size_, 1.0f, ffn_dense1_w, embedding_op_size_,
      scratch, embedding_op_size_, 0.0f, in_out_tensor, ffn_dense1_size_,
      sycl_queue);
  addBiasBatched(in_out_tensor, in_out_tensor, ffn_dense1_b, 1, tokens,
                 ffn_dense1_size_, ffn_activation_, sycl_queue);

  cublasXgemm<DataType>(
      transpose_type_transpose, transpose_type_notranspose,
      embedding_op_size_, tokens, ffn_dense1_size_, 1.0f, ffn_dense2_w,
      ffn_dense1_size_, in_out_tensor, ffn_dense1_size_, 0.0f, buffer1,
      embedding_op_size_, sycl_queue);
  LayerNorm<DataType>(tokens, embedding_op_size_, in_out_tensor, buffer1,
                      ffn_dense2_b, scratch, ln2_gammas, ln2_betas,
                      default_eps_, alpha_, ACTIVATION_NONE, sycl_queue);
}

// input/output tensor is in_out_tensor, others are used as scratch.
template <typename DataType>
void EncoderBlock<DataType>::Eval(int N, DataType* in_out_tensor,
                                  DataType* scratch, DataType* buffer1,
                                  DataType* buffer2, sycl::queue& sycl_queue,
                                  DataType*** offset_pointers) {
  // CERR << "EncoderBlock<DataType>::Eval. ";

  if (is_kda_) {
    EvalKda(N, in_out_tensor, scratch, buffer1, buffer2, sycl_queue);
    return;
  }

  const int d_model = mha_q_size_;
  const int depth = d_model / encoder_heads_;

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
      cublasXgemm<DataType>(
          transpose_type_transpose, transpose_type_notranspose, num_outputs,
          batch, num_inputs, 1.0f, (const DataType*)smol_compress, num_inputs,
          in_out_tensor, num_inputs, 0.0f, scratch, num_outputs, sycl_queue);
    }

    {
      // Hidden 1 dense.
      // input shape: N, 64 * hidden_channels
      // output shape: N, hidden_sz
      const int num_inputs = 64 * smol_compress_size_;
      const int num_outputs = smol_dense_1_size_;
      const int batch = N;
      cublasXgemm<DataType>(
          transpose_type_transpose, transpose_type_notranspose, num_outputs,
          batch, num_inputs, 1.0f, (const DataType*)smol_dense1_w, num_inputs,
          scratch, num_inputs, 0.0f, buffer1, num_outputs, sycl_queue);

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
      cublasXgemm<DataType>(
          transpose_type_transpose, transpose_type_notranspose, num_outputs,
          batch, num_inputs, 1.0f, (const DataType*)smol_dense2_w, num_inputs,
          scratch, num_inputs, 0.0f, buffer1, num_outputs, sycl_queue);

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

      cublasXgemm<DataType>(
          transpose_type_transpose, transpose_type_notranspose, num_outputs,
          batch, num_inputs, 1.0f, (const DataType*)smol_global, num_inputs,
          scratch, num_inputs, 0.0f, buffer2, num_outputs, sycl_queue);
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

    cublasXGemmStridedBatched<DataType>(
        transpose_type_transpose, transpose_type_notranspose, num_outputs,
        batch, num_inputs, 1.0f, mha_qkv_w, num_inputs,
        num_inputs * num_outputs, in_out_tensor, num_inputs, 0, 0.0f, mha_q,
        num_outputs, num_outputs * max_batch, 3, sycl_queue);
    if (mha_qkv_b) {
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
          encoder_heads_ * max_batch_size_ * 5, sycl_queue_);
      genOffsetPointers(*offset_pointers, encoder_heads_, max_batch_size_,
                        depth, d_model, mha_k, mha_q, buffer1, mha_v, buffer2,
                        sycl_queue_);
    }

    cublasXGemmBatched<DataType>(
        transpose_type_transpose, transpose_type_notranspose, 64 /*M*/,
        64 /*N*/, depth /*K*/,  // A/B, and M/N are swapped for
                                // row-major to col-major transform
        factor,                 // to handle "/ tf.math.sqrt(dk)"
        *offset_pointers,       // mha_k + offset /*A*/,
        d_model /*LDA*/,  // (d_model = depth * encoder_heads_) to skip over
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
    Softmax(encoder_heads_ * N * 64, 64, buffer1, buffer1, buffer2,
            sycl_queue_);
  } else {
    Softmax(encoder_heads_ * N * 64, 64, buffer1, buffer1,
            (const DataType*)nullptr, sycl_queue_);
  }

  {
    cublasXGemmBatched<DataType>(
        transpose_type_notranspose, transpose_type_notranspose, depth /*M*/,
        64 /*N*/, 64 /*K*/, 1.0f,
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
    cublasXgemm(transpose_type_transpose, transpose_type_notranspose,
                num_outputs, batch, num_inputs, 1.0f,
                (const DataType*)mha_dense_w, num_inputs, buffer2, num_inputs,
                0.0f, buffer1, num_outputs, sycl_queue_);
  }

  // LN1: skip connection and layer normalization (also bias add of prev gemm)
  // buffer1/in_out_tensor -> scratch
  LayerNorm<DataType>(N * 64, embedding_op_size_, scratch, buffer1, mha_dense_b,
                      in_out_tensor, ln1_gammas, ln1_betas, default_eps_,
                      alpha_, ACTIVATION_NONE, sycl_queue_);

  // #FFN dense 1, scratch -> in_out_tensor
  {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = ffn_dense1_size_;  // encoder_dff
    const int batch = N * 64;
    cublasXgemm(transpose_type_transpose, transpose_type_notranspose,
                num_outputs, batch, num_inputs, 1.0f,
                (const DataType*)ffn_dense1_w, num_inputs, scratch, num_inputs,
                0.0f, in_out_tensor, num_outputs, sycl_queue_);
    addBiasBatched(in_out_tensor, in_out_tensor, ffn_dense1_b, 1, batch,
                   num_outputs, ffn_activation_, sycl_queue_);
  }

  // #FFN dense 2, in_out_tensor -> buffer1
  {
    const int num_inputs = ffn_dense1_size_;  // encoder_dff
    const int num_outputs = embedding_op_size_;
    const int batch = N * 64;
    cublasXgemm(transpose_type_transpose, transpose_type_notranspose,
                num_outputs, batch, num_inputs, 1.0f,
                (const DataType*)ffn_dense2_w, num_inputs, in_out_tensor,
                num_inputs, 0.0f, buffer1, num_outputs, sycl_queue_);
  }

  // LN2: skip connection and layer normilization (also bias add of prev gemm)
  // buffer1/scratch -> in_out_tensor
  LayerNorm<DataType>(N * 64, embedding_op_size_, in_out_tensor, buffer1,
                      ffn_dense2_b, scratch, ln2_gammas, ln2_betas,
                      default_eps_, alpha_, ACTIVATION_NONE, sycl_queue_);
}

template <typename DataType>
void AttentionPolicyHead<DataType>::Eval(int N, DataType* output,
                                         const DataType* input,
                                         const DataType* input2, void* scratch,
                                         size_t scratch_size,
                                         sycl::queue& sycl_queue,
                                         DataType*** offset_pointers) {
  // CERR << "AttentionPolicyHead<DataType>::Eval. ";

  DataType* input2_tensor = (DataType*)input2;
  DataType* buffer1 = output + scratch_size / (2 * sizeof(DataType));
  DataType* buffer2 = input2_tensor + scratch_size / (2 * sizeof(DataType));

  int inputC = this->input_->GetC();
  if (!attention_body_)
    convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, inputC, 8, 8,
                      sycl_queue);

  // 1. Policy embedding (fully connected layer)
  // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
  DataType* pol_embedding = input2_tensor;
  {
    const int num_outputs = embedding_op_size_;
    const int num_inputs = inputC;
    const int batch = N * 64;
    cublasXgemm<DataType>(
        transpose_type_transpose, transpose_type_notranspose, num_outputs,
        batch, num_inputs, 1.0f, (const DataType*)ip_pol_w_, num_inputs,
        attention_body_ ? input : (DataType*)scratch, num_inputs, 0.0f,
        pol_embedding, num_outputs, sycl_queue);

    addBiasBatched(pol_embedding, pol_embedding, ip_pol_b_, 1, batch,
                   num_outputs, act_, sycl_queue);
  }

  // 2. Encoder layers
  for (const auto& pEnc : encoder_weights_) {
    pEnc->Eval(N, input2_tensor, (DataType*)scratch, buffer1, buffer2,
               sycl_queue, offset_pointers);
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
        transpose_type_transpose, transpose_type_notranspose, num_outputs,
        batch, num_inputs, 1.0f, wqk_w_, num_inputs, num_inputs * num_outputs,
        input2_tensor, num_inputs, 0, 0.0f, wq, num_outputs,
        num_outputs * batch, 2, sycl_queue);

    if (wqk_b_) {
      addBiasBatched<DataType>(wq, wq, wqk_b_, 2, batch, num_outputs,
                               ACTIVATION_NONE, sycl_queue);
    }
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
        transpose_type_transpose, transpose_type_notranspose, 64 /*M*/,
        64 /*N*/, policy_d_model_ /*K*/,
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
  // encoder_weights_ is managed by std::unique_ptr
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
  // The SYCL spec requires sycl::free's pointer to come from a USM
  // allocation; pure-MHA layers leave every kda_* member null, so guard
  // each one rather than rely on it being a no-op in practice.
  if (kda_q_w) sycl::free(kda_q_w, sycl_queue_);
  if (kda_q_b) sycl::free(kda_q_b, sycl_queue_);
  if (kda_k_w) sycl::free(kda_k_w, sycl_queue_);
  if (kda_k_b) sycl::free(kda_k_b, sycl_queue_);
  if (kda_v_w) sycl::free(kda_v_w, sycl_queue_);
  if (kda_v_b) sycl::free(kda_v_b, sycl_queue_);
  if (kda_decay_a_w) sycl::free(kda_decay_a_w, sycl_queue_);
  if (kda_decay_a_b) sycl::free(kda_decay_a_b, sycl_queue_);
  if (kda_decay_b_w) sycl::free(kda_decay_b_w, sycl_queue_);
  if (kda_decay_b_b) sycl::free(kda_decay_b_b, sycl_queue_);
  if (kda_beta_w) sycl::free(kda_beta_w, sycl_queue_);
  if (kda_beta_b) sycl::free(kda_beta_b, sycl_queue_);
  if (kda_a_log) sycl::free(kda_a_log, sycl_queue_);
  if (kda_dt_bias) sycl::free(kda_dt_bias, sycl_queue_);
  if (kda_gate_a_w) sycl::free(kda_gate_a_w, sycl_queue_);
  if (kda_gate_a_b) sycl::free(kda_gate_a_b, sycl_queue_);
  if (kda_gate_b_w) sycl::free(kda_gate_b_w, sycl_queue_);
  if (kda_gate_b_b) sycl::free(kda_gate_b_b, sycl_queue_);
  if (kda_out_norm_gammas) sycl::free(kda_out_norm_gammas, sycl_queue_);
  if (kda_dense_w) sycl::free(kda_dense_w, sycl_queue_);
  if (kda_dense_b) sycl::free(kda_dense_b, sycl_queue_);
  if (kda_local_conv_w) sycl::free(kda_local_conv_w, sycl_queue_);
  if (kda_local_conv_b) sycl::free(kda_local_conv_b, sycl_queue_);
  if (kda_qkv_w) sycl::free(kda_qkv_w, sycl_queue_);
  if (kda_qkv_b) sycl::free(kda_qkv_b, sycl_queue_);
  if (kda_decay_gate_a_w) sycl::free(kda_decay_gate_a_w, sycl_queue_);
  if (kda_decay_gate_a_b) sycl::free(kda_decay_gate_a_b, sycl_queue_);
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
}

template <typename DataType>
EmbeddingLayer<DataType>::EmbeddingLayer(BaseLayer<DataType>* ip,
                                         const std::vector<float>& weights,
                                         const std::vector<float>& biases,
                                         void* scratch, ActivationFunction act,
                                         sycl::queue& sycl_queue)
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
void EmbeddingLayer<DataType>::Eval(int N, DataType* output,
                                    const DataType* input,
                                    const DataType* /*input2*/,
                                    void* /*scratch*/, size_t /*scratch_size*/,
                                    sycl::queue& sycl_queue, DataType***) {
  // CERR << "EmbeddingLayer<DataType>::Eval. ";

  const int num_outputs = this->GetC();
  const int num_inputs = this->input_->GetC();
  const int batch = N * 64;
  cublasXgemm<DataType>(transpose_type_transpose, transpose_type_notranspose,
                        num_outputs, batch, num_inputs, 1.0f, weights_,
                        num_inputs, input, num_inputs, 0.0f, output,
                        num_outputs, sycl_queue);
  addBiasBatched(output, output, biases_, 1, batch, num_outputs, act_,
                 sycl_queue);
}

template <typename DataType>
AttentionBody<DataType>::AttentionBody(const MultiHeadWeights& weights,
                                       void* scratch, Activations activations,
                                       int num_res_blocks, int input_c,
                                       int max_batch_size,
                                       bool is_pe_dense_embedding,
                                       const std::vector<int>& kda_directions,
                                       sycl::queue& sycl_queue)
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
    size_t element_count = 64 * kNumPosEncodingChannels;
    size_t dest_byte_count = element_count * sizeof(DataType);
    size_t float_byte_count = element_count * sizeof(float);
    pos_encoding_ =
        (DataType*)sycl::malloc_device(dest_byte_count, sycl_queue_);
    sycl_queue_.memcpy(scratch, kPosEncoding, float_byte_count);
    copyTypeConverted(pos_encoding_, (float*)scratch,
                      static_cast<int>(element_count), sycl_queue_);
  }

  if (has_gating_) {
    allocAndUpload<DataType>(&ip_mult_gate_, weights.ip_mult_gate, scratch,
                             sycl_queue_);
    allocAndUpload<DataType>(&ip_add_gate_, weights.ip_add_gate, scratch,
                             sycl_queue_);
  }

  if (has_smolgen_) {
    allocAndUpload<DataType>(&smolgen_global_, weights.smolgen_w, scratch,
                             sycl_queue_);
    smolgen_global_size_ = 64 * 64;
  }

  int num_encoders = weights.encoder.size();
  float alpha = (float)pow(2.0 * num_encoders, -0.25);
  for (const auto& enc : weights.encoder) {
    EncoderBlock<DataType>* pW = new EncoderBlock<DataType>(
        enc, scratch, encoder_head_count_, embedding_op_size_, alpha,
        smolgen_global_, smolgen_global_size_, max_batch_size,
        activations_.smolgen_activation, activations_.ffn_activation,
        is_pe_dense_embedding_ ? 1e-3 : 1e-6, kda_directions, sycl_queue_);

    encoder_weights_.emplace_back(std::unique_ptr<EncoderBlock<DataType>>(pW));
  }
}

template <typename DataType>
AttentionBody<DataType>::~AttentionBody() {
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
  // encoder_weights_ is managed by std::unique_ptr
}

template <typename DataType>
void AttentionBody<DataType>::Eval(int N, DataType* output,
                                   const DataType* input,
                                   const DataType* input2, void* scratch,
                                   size_t scratch_size, sycl::queue& sycl_queue,
                                   DataType*** offset_pointers) {
  // CERR << "AttentionBody<DataType>::Eval. ";

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

      convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, 12, 8, 8,
                        sycl_queue);
      cublasXgemm<DataType>(transpose_type_transpose,
                            transpose_type_notranspose, num_outputs, batch,
                            num_inputs, 1.0f, (const DataType*)ip_emb_pre_w_,
                            num_inputs, (const DataType*)scratch, num_inputs,
                            0.0f, buffer1, num_outputs, sycl_queue);

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
    convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, inputC, 8, 8,
                      sycl_queue);
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
      cublasXgemm<DataType>(
          transpose_type_transpose, transpose_type_notranspose, num_outputs,
          batch, num_inputs, 1.0f, (const DataType*)ip_emb_w_, num_inputs, temp,
          num_inputs, 0.0f, embedding, num_outputs, sycl_queue);
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
      cublasXgemm(transpose_type_transpose, transpose_type_notranspose,
                  num_outputs, batch, num_inputs, 1.0f,
                  (const DataType*)ip_emb_ffn_d1_w_, num_inputs, temp,
                  num_inputs, 0.0f, buffer1, num_outputs, sycl_queue);
      addBiasBatched(buffer1, buffer1, ip_emb_ffn_d1_b_, 1, batch, num_outputs,
                     activations_.ffn_activation, sycl_queue);
    }

    // embedding FFN dense 2
    {
      const int num_inputs = embedding_ffn_dff_;  // encoder_dff
      const int num_outputs = embedding_ffn_size_;
      const int batch = N * 64;
      cublasXgemm(transpose_type_transpose, transpose_type_notranspose,
                  num_outputs, batch, num_inputs, 1.0f,
                  (const DataType*)ip_emb_ffn_d2_w_, num_inputs, buffer1,
                  num_inputs, 0.0f, buffer2, num_outputs, sycl_queue);
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
      cublasXgemm<DataType>(transpose_type_transpose,
                            transpose_type_notranspose, num_outputs, batch,
                            num_inputs, 1.0f, (const DataType*)ip_emb_w_,
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
  for (const auto& pEnc : encoder_weights_) {
    pEnc->Eval(N, output_tensor, (DataType*)scratch, buffer1, buffer2,
               sycl_queue, offset_pointers);
  }  // End of encoder blocks
}

template <typename DataType>
ValueHead<DataType>::ValueHead(BaseLayer<DataType>* ip,
                               const MultiHeadWeights::ValueHead& weights,
                               void* scratch, bool attention_body, bool wdl,
                               ActivationFunction act, int max_batch_size,
                               sycl::queue& sycl_queue)
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
                               size_t scratch_size, sycl::queue& sycl_queue,
                               DataType***) {
  DataType* buffer = (DataType*)input2;
  {
    const int num_inputs = this->input_->GetC();
    const int num_outputs = embedding_size_;
    const int batch = N * 64;
    if (attention_body_) {
      cublasXgemm<DataType>(
          transpose_type_transpose, transpose_type_notranspose, num_outputs,
          batch, num_inputs, 1.0f, (const DataType*)ip_val_w_, num_inputs,
          input, num_inputs, 0.0f, buffer, num_outputs, sycl_queue);
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
    cublasXgemm<DataType>(transpose_type_transpose, transpose_type_notranspose,
                          num_outputs, batch, num_inputs, 1.0f,
                          (const DataType*)ip1_val_w_, num_inputs, buffer,
                          num_inputs, 0.0f, layer_out, num_outputs, sycl_queue);
    addBiasBatched<DataType>(layer_out, layer_out, ip1_val_b_, 1, batch,
                             num_outputs, act_, sycl_queue);
  }

  {
    // Value dense 2
    const int num_inputs = value_hidden_size_;
    const int num_outputs = wdl_ ? 3 : 1;
    const int batch = N;
    DataType* layer_out = (DataType*)output;
    cublasXgemm<DataType>(transpose_type_transpose, transpose_type_notranspose,
                          num_outputs, batch, num_inputs, 1.0f,
                          (const DataType*)ip2_val_w_, num_inputs,
                          (DataType*)scratch, num_inputs, 0.0f, layer_out,
                          num_outputs, sycl_queue);
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
