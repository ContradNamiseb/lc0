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
template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, sycl::queue &sycl_queue);

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

// Header-inline W8A8 INT8 quantization/dequantization kernels for Transformer
// EncoderBlock (header-inline avoids missing device-image symbol issues on
// Windows oneAPI 2026.1, matching copyTypeConverted above).
inline float activateScalarInt8Epilogue(float val, ActivationFunction act) {
  switch (act) {
    case ACTIVATION_RELU_2:
      return val > 0.0f ? val * val : 0.0f;
    case ACTIVATION_RELU:
      return val > 0.0f ? val : 0.0f;
    case ACTIVATION_MISH: {
      if (val > 20.0f) return val;
      float e = sycl::exp(val);
      float n = e * e + 2.0f * e;
      float d = val / (n + 2.0f);
      return val <= -0.6f ? n * d : val - 2.0f * d;
    }
    case ACTIVATION_SWISH:
      return val / (1.0f + sycl::exp(-val));
    case ACTIVATION_SELU:
      return val > 0.0f ? 1.05070098f * val
                        : 1.75809934f * (sycl::exp(val) - 1.0f);
    case ACTIVATION_TANH:
      return sycl::tanh(val);
    case ACTIVATION_SIGMOID:
      return 1.0f / (1.0f + sycl::exp(-val));
    case ACTIVATION_NONE:
    default:
      return val;
  }
}

template <typename SrcType>
void quantizeToU8(uint8_t* dst, const SrcType* src, int total_elems,
                  float inv_scale, int zp, sycl::queue& sycl_queue) {
  const int kBlockSize = 256;
  const float fzp = static_cast<float>(zp);
  if ((total_elems & 3) == 0) {
    const int n4 = total_elems >> 2;
    const int blocks = DivUp(n4, kBlockSize);
    auto* dst4 = reinterpret_cast<uint32_t*>(dst);
    const auto* src4 = reinterpret_cast<const sycl::vec<SrcType, 4>*>(src);
    sycl_queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                          sycl::range<1>(kBlockSize)),
        [=](sycl::nd_item<1> item) {
          const int tid = static_cast<int>(item.get_global_linear_id());
          if (tid >= n4) return;
          const auto v = src4[tid];
          auto q1 = [&](SrcType x) -> uint32_t {
            int qi = static_cast<int>(
                sycl::rint(static_cast<float>(x) * inv_scale + fzp));
            return static_cast<uint32_t>(qi < 0 ? 0 : (qi > 255 ? 255 : qi));
          };
          dst4[tid] = q1(v.x()) | (q1(v.y()) << 8) | (q1(v.z()) << 16) |
                      (q1(v.w()) << 24);
        });
    return;
  }
  const int blocks = DivUp(total_elems, kBlockSize);
  sycl_queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        const int tid = static_cast<int>(item.get_global_linear_id());
        if (tid >= total_elems) return;
        int qi = static_cast<int>(
            sycl::rint(static_cast<float>(src[tid]) * inv_scale + fzp));
        qi = qi < 0 ? 0 : (qi > 255 ? 255 : qi);
        dst[tid] = static_cast<uint8_t>(qi);
      });
}

template <typename DstType>
void dequantBiasUnpackQKV(DstType* q_out, DstType* k_out, DstType* v_out,
                          const int32_t* s32_in, const float* comb_scales_3d,
                          const DstType* biases_3d, int batch_tokens,
                          int d_model, sycl::queue& sycl_queue) {
  if ((d_model & 3) == 0) {
    const int vec_d = d_model >> 2;
    const int vec_3d = 3 * vec_d;
    const int total_vec = batch_tokens * vec_3d;
    const int kBlockSize = 256;
    const int blocks = DivUp(total_vec, kBlockSize);
    const auto* in4 = reinterpret_cast<const sycl::int4*>(s32_in);
    const auto* sc4 = reinterpret_cast<const sycl::float4*>(comb_scales_3d);
    const auto* b4 = reinterpret_cast<const sycl::vec<DstType, 4>*>(biases_3d);
    auto* q4 = reinterpret_cast<sycl::vec<DstType, 4>*>(q_out);
    auto* k4 = reinterpret_cast<sycl::vec<DstType, 4>*>(k_out);
    auto* v4 = reinterpret_cast<sycl::vec<DstType, 4>*>(v_out);
    sycl_queue.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(batch_tokens,
                                         DivUp(vec_3d, kBlockSize) * kBlockSize),
                          sycl::range<2>(1, kBlockSize)),
        [=](sycl::nd_item<2> item) {
          const int tok = static_cast<int>(item.get_global_id(0));
          const int vc = static_cast<int>(item.get_global_id(1));
          if (vc >= vec_3d) return;
          const int idx = tok * vec_3d + vc;
          const sycl::int4 s = in4[idx];
          const sycl::float4 sc = sc4[vc];
          const auto bv = b4[vc];
          sycl::vec<DstType, 4> out(
              static_cast<DstType>(static_cast<float>(s.x()) * sc.x() +
                                   static_cast<float>(bv.x())),
              static_cast<DstType>(static_cast<float>(s.y()) * sc.y() +
                                   static_cast<float>(bv.y())),
              static_cast<DstType>(static_cast<float>(s.z()) * sc.z() +
                                   static_cast<float>(bv.z())),
              static_cast<DstType>(static_cast<float>(s.w()) * sc.w() +
                                   static_cast<float>(bv.w())));
          if (vc < vec_d) {
            q4[tok * vec_d + vc] = out;
          } else if (vc < 2 * vec_d) {
            k4[tok * vec_d + (vc - vec_d)] = out;
          } else {
            v4[tok * vec_d + (vc - 2 * vec_d)] = out;
          }
        });
    (void)blocks;
    return;
  }
  const int stride3 = 3 * d_model;
  const int total_elems = batch_tokens * stride3;
  const int kBlockSize = 256;
  const int blocks = DivUp(total_elems, kBlockSize);
  sycl_queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        const int tid = static_cast<int>(item.get_global_linear_id());
        if (tid >= total_elems) return;
        const int tok = tid / stride3;
        const int rem = tid - tok * stride3;
        const int which = rem / d_model;
        const int ch = rem - which * d_model;
        float val = static_cast<float>(s32_in[tid]) * comb_scales_3d[rem] +
                    static_cast<float>(biases_3d[rem]);
        DstType* dst = (which == 0) ? q_out : ((which == 1) ? k_out : v_out);
        dst[tok * d_model + ch] = static_cast<DstType>(val);
      });
}

template <typename DstType>
void dequantS32ToDataType(DstType* dst, const int32_t* s32_in,
                          const float* comb_scales, int batch_tokens,
                          int num_outputs, sycl::queue& sycl_queue) {
  if ((num_outputs & 3) == 0) {
    const int vec_c = num_outputs >> 2;
    const int kBlockSize = std::min(vec_c, 256);
    const auto* in4 = reinterpret_cast<const sycl::int4*>(s32_in);
    const auto* sc4 = reinterpret_cast<const sycl::float4*>(comb_scales);
    auto* out4 = reinterpret_cast<sycl::vec<DstType, 4>*>(dst);
    sycl_queue.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(batch_tokens,
                           DivUp(vec_c, kBlockSize) * kBlockSize),
            sycl::range<2>(1, kBlockSize)),
        [=](sycl::nd_item<2> item) {
          const int tok = static_cast<int>(item.get_global_id(0));
          const int vc = static_cast<int>(item.get_global_id(1));
          if (vc >= vec_c) return;
          const int idx = tok * vec_c + vc;
          const sycl::int4 s = in4[idx];
          const sycl::float4 sc = sc4[vc];
          out4[idx] = sycl::vec<DstType, 4>(
              static_cast<DstType>(static_cast<float>(s.x()) * sc.x()),
              static_cast<DstType>(static_cast<float>(s.y()) * sc.y()),
              static_cast<DstType>(static_cast<float>(s.z()) * sc.z()),
              static_cast<DstType>(static_cast<float>(s.w()) * sc.w()));
        });
    return;
  }
  const int total_elems = batch_tokens * num_outputs;
  const int kBlockSize = 256;
  const int blocks = DivUp(total_elems, kBlockSize);
  sycl_queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        const int tid = static_cast<int>(item.get_global_linear_id());
        if (tid >= total_elems) return;
        const int ch = tid % num_outputs;
        dst[tid] = static_cast<DstType>(static_cast<float>(s32_in[tid]) *
                                        comb_scales[ch]);
      });
}

template <typename BiasType>
void dequantBiasActQuantizeU8(uint8_t* dst_u8, const int32_t* s32_in,
                              const float* comb_scales, const BiasType* bias,
                              int batch_tokens, int num_outputs,
                              ActivationFunction act, float next_inv_scale,
                              int next_zp, sycl::queue& sycl_queue) {
  const float fzp = static_cast<float>(next_zp);
  if ((num_outputs & 3) == 0) {
    const int vec_c = num_outputs >> 2;
    const int kBlockSize = std::min(vec_c, 256);
    const auto* in4 = reinterpret_cast<const sycl::int4*>(s32_in);
    const auto* sc4 = reinterpret_cast<const sycl::float4*>(comb_scales);
    const auto* b4 = reinterpret_cast<const sycl::vec<BiasType, 4>*>(bias);
    auto* out4 = reinterpret_cast<uint32_t*>(dst_u8);
    sycl_queue.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(batch_tokens,
                           DivUp(vec_c, kBlockSize) * kBlockSize),
            sycl::range<2>(1, kBlockSize)),
        [=](sycl::nd_item<2> item) {
          const int tok = static_cast<int>(item.get_global_id(0));
          const int vc = static_cast<int>(item.get_global_id(1));
          if (vc >= vec_c) return;
          const int idx = tok * vec_c + vc;
          const sycl::int4 s = in4[idx];
          const sycl::float4 sc = sc4[vc];
          const auto bv = b4[vc];
          auto fq = [&](int32_t sv, float scv, BiasType bval) -> uint32_t {
            float v = static_cast<float>(sv) * scv + static_cast<float>(bval);
            v = activateScalarInt8Epilogue(v, act);
            int qi = static_cast<int>(sycl::rint(v * next_inv_scale + fzp));
            return static_cast<uint32_t>(qi < 0 ? 0 : (qi > 255 ? 255 : qi));
          };
          out4[idx] = fq(s.x(), sc.x(), bv.x()) |
                      (fq(s.y(), sc.y(), bv.y()) << 8) |
                      (fq(s.z(), sc.z(), bv.z()) << 16) |
                      (fq(s.w(), sc.w(), bv.w()) << 24);
        });
    return;
  }
  const int total_elems = batch_tokens * num_outputs;
  const int kBlockSize = 256;
  const int blocks = DivUp(total_elems, kBlockSize);
  sycl_queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        const int tid = static_cast<int>(item.get_global_linear_id());
        if (tid >= total_elems) return;
        const int ch = tid % num_outputs;
        float val = static_cast<float>(s32_in[tid]) * comb_scales[ch] +
                    static_cast<float>(bias[ch]);
        val = activateScalarInt8Epilogue(val, act);
        int qi = static_cast<int>(sycl::rint(val * next_inv_scale + fzp));
        qi = qi < 0 ? 0 : (qi > 255 ? 255 : qi);
        dst_u8[tid] = static_cast<uint8_t>(qi);
      });
}

// Header-inline NCHW <-> NHWC helpers and fused FP32-accumulated NHWC SE + Skip + Act
// for Int8ResidualTower (oneDNN SYCL DP4A conv1+conv2 bridge).
template <typename DataType>
void transposeNCHWtoNHWC_Int8Tower(DataType* dst_nhwc, const DataType* src_nchw,
                                   int N_real, int N_pad, int C,
                                   sycl::queue& sycl_queue) {
  const int wg = C <= 256 ? C : 256;
  const int g1 = DivUp(C, wg) * wg;
  sycl_queue.parallel_for(
      sycl::nd_range<2>(sycl::range<2>(N_pad * 64, g1), sycl::range<2>(1, wg)),
      [=](sycl::nd_item<2> item) {
        const int tok = static_cast<int>(item.get_global_id(0));
        const int c = static_cast<int>(item.get_global_id(1));
        if (c >= C) return;
        const int n = tok >> 6;
        if (n >= N_real) {
          dst_nhwc[tok * C + c] = DataType(0);
          return;
        }
        const int hw = tok & 63;
        dst_nhwc[tok * C + c] = src_nchw[(n * C + c) * 64 + hw];
      });
}

template <typename DataType>
void transposeNHWCtoNCHW_Int8Tower(DataType* dst_nchw, const DataType* src_nhwc,
                                   int N, int C, sycl::queue& sycl_queue) {
  sycl_queue.parallel_for(
      sycl::nd_range<2>(sycl::range<2>(N * C, 64), sycl::range<2>(1, 64)),
      [=](sycl::nd_item<2> item) {
        const int nc = static_cast<int>(item.get_global_id(0));
        const int hw = static_cast<int>(item.get_global_id(1));
        const int n = nc / C;
        const int c = nc - n * C;
        dst_nchw[nc * 64 + hw] = src_nhwc[(n * 64 + hw) * C + c];
      });
}

template <ActivationFunction kAct>
inline float activateCompileTime(float val) {
  if constexpr (kAct == ACTIVATION_RELU) {
    return val > 0.0f ? val : 0.0f;
  } else if constexpr (kAct == ACTIVATION_MISH) {
    if (val > 20.0f) return val;
    float e = sycl::exp(val);
    float n = e * e + 2.0f * e;
    float d = val / (n + 2.0f);
    return val <= -0.6f ? n * d : val - 2.0f * d;
  } else {
    return activateScalarInt8Epilogue(val, kAct);
  }
}

template <ActivationFunction kAct>
void fusedSE_SkipAct_Fp16_NHWC_Fast(sycl::half* output, const sycl::half* skip,
                                    const sycl::half* conv2_out,
                                    const float* w1_t, const float* b1,
                                    const float* w2_t, const float* b2, int N,
                                    int C, int se_k, sycl::queue& sycl_queue) {
  const int c4 = C >> 2;
  sycl_queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> smem(sycl::range<1>(4 * C + se_k), cgh);
    cgh.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(N, C), sycl::range<2>(1, C)),
        [=](sycl::nd_item<2> item) {
          const int n = static_cast<int>(item.get_global_id(0));
          const int tid = static_cast<int>(item.get_local_id(1));
          const int chunk = tid / c4;       // 0..3 (16 squares per chunk)
          const int vc = tid - chunk * c4;  // 0..c4-1 (half4 channel group)
          const int hw0 = chunk * 16;

          const auto* in4 =
              reinterpret_cast<const sycl::half4*>(conv2_out) + (n * 64 * c4);

          // 1. Single-read register-cached global avg pool (half4 vectorized).
          sycl::half4 loc[16];
          sycl::float4 sum(0.0f);
          for (int j = 0; j < 16; ++j) {
            const sycl::half4 v = in4[(hw0 + j) * c4 + vc];
            loc[j] = v;
            sum += sycl::float4(static_cast<float>(v.x()),
                                static_cast<float>(v.y()),
                                static_cast<float>(v.z()),
                                static_cast<float>(v.w()));
          }
          float* dst = &smem[chunk * C + (vc << 2)];
          dst[0] = sum.x();
          dst[1] = sum.y();
          dst[2] = sum.z();
          dst[3] = sum.w();
          item.barrier(sycl::access::fence_space::local_space);

          smem[tid] = (smem[tid] + smem[C + tid] + smem[2 * C + tid] +
                       smem[3 * C + tid]) *
                      (1.0f / 64.0f);
          item.barrier(sycl::access::fence_space::local_space);

          // 2. SE FC1: [C] -> [se_k] + bias + activation.
          if (tid < se_k) {
            float acc = b1[tid];
            for (int i = 0; i < C; ++i) {
              acc += smem[i] * w1_t[i * se_k + tid];
            }
            smem[4 * C + tid] = activateCompileTime<kAct>(acc);
          }
          item.barrier(sycl::access::fence_space::local_space);

          // 3. SE FC2: [se_k] -> [2*C] + sigmoid(scale).
          const int two_c = 2 * C;
          float sc = b2[tid];
          float sh = b2[tid + C];
          for (int k = 0; k < se_k; ++k) {
            const float v = smem[4 * C + k];
            sc += v * w2_t[k * two_c + tid];
            sh += v * w2_t[k * two_c + C + tid];
          }
          smem[tid] = 1.0f / (1.0f + sycl::exp(-sc));
          smem[C + tid] = sh;
          item.barrier(sycl::access::fence_space::local_space);

          // 4. Scale + skip + activation using register-cached conv2_out (half4).
          const int c0 = vc << 2;
          const sycl::float4 vsc(smem[c0], smem[c0 + 1], smem[c0 + 2],
                                 smem[c0 + 3]);
          const sycl::float4 vsh(smem[C + c0], smem[C + c0 + 1],
                                 smem[C + c0 + 2], smem[C + c0 + 3]);
          const auto* sk4 =
              reinterpret_cast<const sycl::half4*>(skip) + (n * 64 * c4);
          auto* out4 = reinterpret_cast<sycl::half4*>(output) + (n * 64 * c4);
          for (int j = 0; j < 16; ++j) {
            const int idx = (hw0 + j) * c4 + vc;
            const sycl::half4 x = loc[j];
            const sycl::half4 s = sk4[idx];
            out4[idx] = sycl::half4(
                static_cast<sycl::half>(activateCompileTime<kAct>(
                    static_cast<float>(s.x()) +
                    static_cast<float>(x.x()) * vsc.x() + vsh.x())),
                static_cast<sycl::half>(activateCompileTime<kAct>(
                    static_cast<float>(s.y()) +
                    static_cast<float>(x.y()) * vsc.y() + vsh.y())),
                static_cast<sycl::half>(activateCompileTime<kAct>(
                    static_cast<float>(s.z()) +
                    static_cast<float>(x.z()) * vsc.z() + vsh.z())),
                static_cast<sycl::half>(activateCompileTime<kAct>(
                    static_cast<float>(s.w()) +
                    static_cast<float>(x.w()) * vsc.w() + vsh.w())));
          }
        });
  });
}

template <typename DataType>
void fusedSE_SkipAct_NHWC(DataType* output, const DataType* skip,
                          const DataType* conv2_out, const float* w1_t,
                          const float* b1, const float* w2_t, const float* b2,
                          int N, int C, int se_k, ActivationFunction act,
                          sycl::queue& sycl_queue) {
  if constexpr (std::is_same<DataType, sycl::half>::value) {
    if ((C & 3) == 0 && C <= 256 && se_k <= C) {
      if (act == ACTIVATION_RELU) {
        fusedSE_SkipAct_Fp16_NHWC_Fast<ACTIVATION_RELU>(
            output, skip, conv2_out, w1_t, b1, w2_t, b2, N, C, se_k,
            sycl_queue);
        return;
      }
      if (act == ACTIVATION_MISH) {
        fusedSE_SkipAct_Fp16_NHWC_Fast<ACTIVATION_MISH>(
            output, skip, conv2_out, w1_t, b1, w2_t, b2, N, C, se_k,
            sycl_queue);
        return;
      }
    }
  }
  const int wg = C <= 256 ? C : 256;
  sycl_queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> smem(sycl::range<1>(C + se_k), cgh);
    cgh.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(N, wg), sycl::range<2>(1, wg)),
        [=](sycl::nd_item<2> item) {
          const int n = static_cast<int>(item.get_global_id(0));
          const int tid = static_cast<int>(item.get_local_id(1));
          const int base = n * 64 * C;

          for (int c = tid; c < C; c += wg) {
            float sum = 0.0f;
            for (int hw = 0; hw < 64; ++hw) {
              sum += static_cast<float>(conv2_out[base + hw * C + c]);
            }
            smem[c] = sum * (1.0f / 64.0f);
          }
          item.barrier(sycl::access::fence_space::local_space);

          for (int k = tid; k < se_k; k += wg) {
            float acc = b1[k];
            for (int i = 0; i < C; ++i) {
              acc += smem[i] * w1_t[i * se_k + k];
            }
            smem[C + k] = activateScalarInt8Epilogue(acc, act);
          }
          item.barrier(sycl::access::fence_space::local_space);

          const int two_c = 2 * C;
          for (int c = tid; c < C; c += wg) {
            float sc = b2[c];
            float sh = b2[c + C];
            for (int k = 0; k < se_k; ++k) {
              const float v = smem[C + k];
              sc += v * w2_t[k * two_c + c];
              sh += v * w2_t[k * two_c + C + c];
            }
            sc = 1.0f / (1.0f + sycl::exp(-sc));
            for (int hw = 0; hw < 64; ++hw) {
              const int idx = base + hw * C + c;
              float y = static_cast<float>(skip[idx]) +
                        static_cast<float>(conv2_out[idx]) * sc + sh;
              y = activateScalarInt8Epilogue(y, act);
              output[idx] = static_cast<DataType>(y);
            }
          }
        });
  });
}

template <typename DataType>
void addSkipAct_NHWC(DataType* output, const DataType* skip,
                     const DataType* conv2_out, int total_elems,
                     ActivationFunction act, sycl::queue& sycl_queue) {
  const int kBlockSize = 256;
  const int blocks = DivUp(total_elems, kBlockSize);
  sycl_queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        const int tid = static_cast<int>(item.get_global_linear_id());
        if (tid >= total_elems) return;
        float y = static_cast<float>(skip[tid]) +
                  static_cast<float>(conv2_out[tid]);
        output[tid] = static_cast<DataType>(activateScalarInt8Epilogue(y, act));
      });
}
}  // namespace sycldnn_backend
}  // namespace lczero
