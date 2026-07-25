#pragma once

#include <sycl/sycl.hpp>
#include <cassert>
#include <algorithm>
#include "winograd_helper.h"
#include "neural/backends/shared/activation.h"

namespace lczero {
namespace sycldnn_backend {

/**
 * @brief SYCL device kernel: fused Winograd output transform + SE block + input transform.
 *
 * Performs in a single kernel:
 *   1. Inverse Winograd transform (6x6 tiles → 8x8 spatial board) per channel.
 *   2. Squeeze-and-Excitation (SE) MLP: global avg pool → FC1(activation) → FC2 →
 *      per-channel scale (sigmoid) and bias.
 *   3. Optional skip connection addition (layout: NHCW).
 *   4. Activation on the final output.
 *   5. Forward Winograd input transform (8x8 → four 6x6 tiles) ready for the next conv.
 *
 * Thread mapping: one work-item per channel (k = local_id(2)), one work-group per sample (n = group(2)).
 * Work-group size must equal C. shared_data[C] must be pre-allocated as local accessor by the caller.
 *
 * @tparam activation  Post-SE activation applied to each board element (ACTIVATION_RELU, ACTIVATION_MISH, etc.).
 * @tparam use_bias    If true, the per-channel bias from @p bias is added before the SE global avg pool.
 * @tparam use_skip    If true, the skip connection from @p skip (NHCW layout) is added and also updated in-place.
 * @param  N           Batch size.
 * @param  C           Number of channels. Must be <= device max_work_group_size.
 * @param  se_K        SE bottleneck width (number of neurons in the SE FC layers).
 * @param  output      [out] Winograd-transformed output tensor (TEMP_INDEX_HWNC layout).
 * @param  input       [in]  Winograd-transformed input tensor (TEMP_INDEX_HWNC layout).
 * @param  skip        [in/out] Spatial skip connection tensor (NHCW layout). Modified in-place if use_skip.
 * @param  bias        [in]  Per-channel output bias (length C), or nullptr if !use_bias.
 * @param  w1          [in]  SE FC1 weight matrix (se_K x C, transposed).
 * @param  b1          [in]  SE FC1 bias vector (length se_K).
 * @param  w2          [in]  SE FC2 weight matrix (2*C x se_K, transposed).
 * @param  b2          [in]  SE FC2 bias vector (length 2*C; first C = scale bias, next C = additive bias).
 * @param  item_ct1    SYCL nd_item for the 3D kernel.
 * @param  shared_data Pointer to local memory of size C (channel averages for SE reduction).
 */
template <ActivationFunction activation, bool use_bias, bool use_skip>
void OutputInputTransformKernel_SubGroup(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input,
    sycl::half* skip, const sycl::half* bias, const sycl::half* w1,
    const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
    const sycl::nd_item<3>& item_ct1, float* shared_data) {
  
  int k = item_ct1.get_local_id(2);
  int n = item_ct1.get_group(2);

  sycl::half board[8][8];
  sycl::half b = use_bias ? bias[k] : (sycl::half)0.0f;

#pragma unroll
  for (int hStart = 0; hStart < 8; hStart += 4) {
#pragma unroll
    for (int wStart = 0; wStart < 8; wStart += 4) {
      int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
      sycl::half outElTransformed[6][6];
#pragma unroll
      for (int y = 0; y < 6; y++) {
#pragma unroll
        for (int x = 0; x < 6; x++) {
          outElTransformed[y][x] = input[TEMP_INDEX_HWNC(y, x, shln, k)];
        }
      }

      sycl::half outEl[4][4];
      OutputTransform4x4(&outEl[0][0], &outElTransformed[0][0]);

#pragma unroll
      for (int y = 0; y < 4; y++) {
        copyAs<sycl::uint2>(&board[hStart + y][wStart], &outEl[y][0]);
      }
    }
  }

  float S = 0;
  float B = 0;

#pragma unroll
  for (int y = 0; y < 8; y++) {
#pragma unroll
    for (int x = 0; x < 8; x++) {
      if (use_bias) board[y][x] += b;
      S += (float)board[y][x];
    }
  }

  float avg = S / 64.0f;
  shared_data[k] = avg;

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // First FC layer
  float S_fc1 = 0;
  if (k < se_K) {
    S_fc1 = (float)b1[k];
    for (int i = 0; i < C; i++) {
      S_fc1 += shared_data[i] * (float)readw1(i, k);
    }
    S_fc1 = activate(S_fc1, activation);
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  if (k < se_K) {
    shared_data[k] = S_fc1;
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Second FC layer
  S = 0;
  B = 0;
  
  for (int i = 0; i < se_K; i++) {
    float val = shared_data[i];
    S += val * (float)readw2(i, k);
    B += val * (float)readw2(i, k + C);
  }

  S += (float)b2[k];
  B += (float)b2[k + C];

  // Sigmoid (only on the scale part).
  S = 1.0f / (1.0f + sycl::native::exp(-S));

  // Scale/bias, add skip connection, perform activation, and write to output.
#pragma unroll
  for (int h = 0; h < 8; h++) {
#pragma unroll
    for (int w = 0; w < 8; w++) {
      float board_val = (float)board[h][w];
      board_val = board_val * S + B;

      if (use_skip) {
        board_val += (float)skip[INDEX_NHCW(n, k, h, w)];
      }

      if (activation != ACTIVATION_NONE) {
        board_val = activate(board_val, activation);
      }

      if (use_skip) {
        skip[INDEX_NHCW(n, k, h, w)] = (sycl::half)board_val;
      }

      board[h][w] = (sycl::half)board_val;
    }
  }

  // Perform input transform
  int c = k;
  // top-left
  {
    sycl::half inEl[6][6] = {};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j + 1] = board[i][j];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
  }

  // top-right
  {
    sycl::half inEl[6][6] = {};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j] = board[i][j + 3];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
  }

  // bottom-left
  {
    sycl::half inEl[6][6] = {};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j + 1] = board[i + 3][j];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
  }

  // bottom-right
  {
    sycl::half inEl[6][6] = {};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j] = board[i + 3][j + 3];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
  }
}

/**
 * @brief SYCL device kernel: fused Winograd output transform + SE block (output only, no input re-transform).
 *
 * Same as OutputInputTransformKernel_SubGroup but writes the final spatial result to @p output
 * instead of re-applying the Winograd input transform. Used for the last residual block in the tower.
 *
 * @tparam activation      Post-SE activation (e.g., ACTIVATION_RELU, ACTIVATION_MISH).
 * @tparam use_bias        If true, adds per-channel @p bias before SE global avg pool.
 * @tparam use_skip        If true, adds skip connection from @p skip.
 * @tparam skipInput_nhcw  Layout of the @p skip tensor: true = NHCW, false = NCHW.
 * @tparam output_nhcw     Layout of the @p output tensor: true = NHCW, false = NCHW.
 * @param  N      Batch size.
 * @param  C      Number of channels. Must be <= device max_work_group_size.
 * @param  se_K   SE bottleneck width.
 * @param  output [out] Final spatial output tensor (NHCW or NCHW as per output_nhcw).
 * @param  input  [in]  Winograd-transformed input tensor (TEMP_INDEX_HWNC layout).
 * @param  skip   [in]  Skip connection tensor (layout per skipInput_nhcw).
 * @param  bias   [in]  Per-channel bias (length C), or nullptr if !use_bias.
 * @param  w1,b1  [in]  SE FC1 weights (se_K x C transposed) and biases (se_K).
 * @param  w2,b2  [in]  SE FC2 weights (2*C x se_K transposed) and biases (2*C).
 * @param  item_ct1   SYCL nd_item for the 3D kernel.
 * @param  shared_data  Local memory pointer of size C for SE channel averages.
 */
template <ActivationFunction activation, bool use_bias, bool use_skip,
          bool skipInput_nhcw, bool output_nhcw>
void OutputTransformKernel_SubGroup(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input,
    const sycl::half* skip, const sycl::half* bias, const sycl::half* w1,
    const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
    const sycl::nd_item<3>& item_ct1, float* shared_data) {
  
  int k = item_ct1.get_local_id(2);
  int n = item_ct1.get_group(2);

  sycl::half board[8][8];
  sycl::half b = use_bias ? bias[k] : (sycl::half)0.0f;

#pragma unroll
  for (int hStart = 0; hStart < 8; hStart += 4) {
#pragma unroll
    for (int wStart = 0; wStart < 8; wStart += 4) {
      int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
      sycl::half outElTransformed[6][6];
#pragma unroll
      for (int y = 0; y < 6; y++) {
#pragma unroll
        for (int x = 0; x < 6; x++) {
          outElTransformed[y][x] = input[TEMP_INDEX_HWNC(y, x, shln, k)];
        }
      }

      sycl::half outEl[4][4];
      OutputTransform4x4(&outEl[0][0], &outElTransformed[0][0]);

#pragma unroll
      for (int y = 0; y < 4; y++) {
        copyAs<sycl::uint2>(&board[hStart + y][wStart], &outEl[y][0]);
      }
    }
  }

  float S = 0;
  float B = 0;

#pragma unroll
  for (int y = 0; y < 8; y++) {
#pragma unroll
    for (int x = 0; x < 8; x++) {
      if (use_bias) board[y][x] += b;
      S += (float)board[y][x];
    }
  }

  float avg = S / 64.0f;
  shared_data[k] = avg;

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // First FC layer
  float S_fc1 = 0;
  if (k < se_K) {
    S_fc1 = (float)b1[k];
    for (int i = 0; i < C; i++) {
      S_fc1 += shared_data[i] * (float)readw1(i, k);
    }
    S_fc1 = activate(S_fc1, activation);
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  if (k < se_K) {
    shared_data[k] = S_fc1;
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Second FC layer
  S = 0;
  B = 0;
  
  for (int i = 0; i < se_K; i++) {
    float val = shared_data[i];
    S += val * (float)readw2(i, k);
    B += val * (float)readw2(i, k + C);
  }

  S += (float)b2[k];
  B += (float)b2[k + C];

  // Sigmoid (only on the scale part).
  S = 1.0f / (1.0f + sycl::native::exp(-S));

  // Scale/bias, add skip connection, perform activation, and write to output.
#pragma unroll
  for (int h = 0; h < 8; h++) {
#pragma unroll
    for (int w = 0; w < 8; w++) {
      float board_val = (float)board[h][w];
      board_val = board_val * S + B;

      if (use_skip) {
        if (skipInput_nhcw)
            board_val += (float)skip[INDEX_NHCW(n, k, h, w)];
        else
            board_val += (float)skip[INDEX_NCHW(n, k, h, w)];
      }

      if (activation != ACTIVATION_NONE) {
        board_val = activate(board_val, activation);
      }

      if (output_nhcw)
          output[INDEX_NHCW(n, k, h, w)] = (sycl::half)board_val;
      else
          output[INDEX_NCHW(n, k, h, w)] = (sycl::half)board_val;
    }
  }
}

/**
 * @brief Submits the SubGroup fused output+SE+input transform kernel to a SYCL queue.
 *
 * Launches OutputInputTransformKernel_SubGroup with nd_range(N*C, C), one work-group per
 * sample, C work-items per work-group. A local accessor of size C is allocated for SE reduction.
 *
 * @pre  C <= sycl_queue.get_device().get_info<sycl::info::device::max_work_group_size>()
 * @param  N, C, se_K   Batch size, channel count, SE bottleneck width.
 * @param  output       Winograd transform-domain output (TEMP_INDEX_HWNC).
 * @param  input        Winograd transform-domain input (TEMP_INDEX_HWNC).
 * @param  skip         Skip connection in NHCW layout (may be nullptr if !use_skip).
 * @param  bias         Per-channel bias (length C; may be nullptr if !use_bias).
 * @param  w1,b1        SE FC1 weights/biases.
 * @param  w2,b2        SE FC2 weights/biases.
 * @param  sycl_queue   Target in-order SYCL queue.
 */
template <ActivationFunction activation, bool use_bias, bool use_skip>
void SubGroupOutputInputTransform(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input,
    const sycl::half* skip, const sycl::half* bias, const sycl::half* w1,
    const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
    sycl::queue &sycl_queue) {
  assert(se_K <= C);
  const int local_storage_size = std::max(C, se_K);

  sycl_queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> shared_data_acc(sycl::range<1>(local_storage_size), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
            sycl::range<3>(1, 1, C)),
        [=](sycl::nd_item<3> item) {
          OutputInputTransformKernel_SubGroup<activation, use_bias, use_skip>(
              N, C, se_K, output, input, (sycl::half*)skip, (sycl::half*)bias,
              (sycl::half*)w1, (sycl::half*)b1, (sycl::half*)w2, (sycl::half*)b2,
              item, shared_data_acc.get_pointer());
        });
  });
}

/**
 * @brief Submits the SubGroup fused output transform + SE block kernel to a SYCL queue.
 *
 * Launches OutputTransformKernel_SubGroup — like SubGroupOutputInputTransform but writes
 * the final spatial result directly to @p output without re-applying the input transform.
 * Used for the last residual block in the tower.
 *
 * @tparam skipInput_nhcw  Layout of @p skip: true = NHCW (residual path), false = NCHW.
 * @tparam output_nhcw     Layout of @p output: true = NHCW, false = NCHW.
 * @pre  C <= sycl_queue.get_device().get_info<sycl::info::device::max_work_group_size>()
 */
template <ActivationFunction activation, bool use_bias, bool use_skip,
          bool skipInput_nhcw, bool output_nhcw>
void SubGroupOutputTransform(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input,
    const sycl::half* skip, const sycl::half* bias, const sycl::half* w1,
    const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
    sycl::queue &sycl_queue) {
  assert(se_K <= C);
  const int local_storage_size = std::max(C, se_K);

  sycl_queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> shared_data_acc(sycl::range<1>(local_storage_size), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
            sycl::range<3>(1, 1, C)),
        [=](sycl::nd_item<3> item) {
          OutputTransformKernel_SubGroup<activation, use_bias, use_skip, skipInput_nhcw, output_nhcw>(
              N, C, se_K, output, input, (sycl::half*)skip, (sycl::half*)bias,
              (sycl::half*)w1, (sycl::half*)b1, (sycl::half*)w2, (sycl::half*)b2,
              item, shared_data_acc.get_pointer());
        });
  });
}

/**
 * @brief SYCL device kernel: fused Winograd output transform + input transform (No SE).
 *
 * Performs in a single kernel:
 *   1. Inverse Winograd transform (6x6 tiles → 8x8 spatial board) per channel.
 *   2. Optional per-channel bias addition.
 *   3. Optional skip connection addition (layout: NHCW).
 *   4. Activation on the board.
 *   5. Forward Winograd input transform (8x8 → four 6x6 tiles) ready for the next conv.
 *
 * Thread mapping: one work-item per channel (k = local_id(2)), one work-group per sample (n = group(2)).
 * Work-group size must equal C. No shared memory or barriers required.
 *
 * @tparam activation  Activation applied to each board element (ACTIVATION_RELU, ACTIVATION_MISH, etc.).
 * @tparam use_bias    If true, per-channel bias is added.
 * @tparam use_skip    If true, skip connection from @p skip (NHCW layout) is added and updated in-place.
 * @param  N           Batch size.
 * @param  C           Number of channels. Must be <= device max_work_group_size.
 * @param  output      [out] Winograd-transformed output tensor (TEMP_INDEX_HWNC layout).
 * @param  input       [in]  Winograd-transformed input tensor (TEMP_INDEX_HWNC layout).
 * @param  skip        [in/out] Spatial skip connection tensor (NHCW layout). Modified in-place if use_skip.
 * @param  bias        [in]  Per-channel output bias (length C), or nullptr if !use_bias.
 * @param  item_ct1    SYCL nd_item for the 3D kernel.
 */
template <ActivationFunction activation, bool use_bias, bool use_skip>
void OutputInputTransformKernel_NoSE_SubGroup(
    int N, int C, sycl::half* output, const sycl::half* input,
    sycl::half* skip, const sycl::half* bias,
    const sycl::nd_item<3>& item_ct1) {
  int k = item_ct1.get_local_id(2);
  int n = item_ct1.get_group(2);

  sycl::half board[8][8];
  sycl::half b = use_bias ? bias[k] : (sycl::half)0.0f;

#pragma unroll
  for (int hStart = 0; hStart < 8; hStart += 4) {
#pragma unroll
    for (int wStart = 0; wStart < 8; wStart += 4) {
      int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
      sycl::half outElTransformed[6][6];
#pragma unroll
      for (int y = 0; y < 6; y++) {
#pragma unroll
        for (int x = 0; x < 6; x++) {
          outElTransformed[y][x] = input[TEMP_INDEX_HWNC(y, x, shln, k)];
        }
      }

      sycl::half outEl[4][4];
      OutputTransform4x4(&outEl[0][0], &outElTransformed[0][0]);

#pragma unroll
      for (int y = 0; y < 4; y++) {
        copyAs<sycl::uint2>(&board[hStart + y][wStart], &outEl[y][0]);
      }
    }
  }

#pragma unroll
  for (int h = 0; h < 8; h++) {
#pragma unroll
    for (int w = 0; w < 8; w++) {
      float board_val = (float)board[h][w];
      if (use_bias) {
        board_val += (float)b;
      }

      if (use_skip) {
        board_val += (float)skip[INDEX_NHCW(n, k, h, w)];
      }

      if (activation != ACTIVATION_NONE) {
        board_val = activate(board_val, activation);
      }

      if (use_skip) {
        skip[INDEX_NHCW(n, k, h, w)] = (sycl::half)board_val;
      }

      board[h][w] = (sycl::half)board_val;
    }
  }

  // Perform input transform
  int c = k;
  // top-left
  {
    sycl::half inEl[6][6] = {};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j + 1] = board[i][j];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
  }

  // top-right
  {
    sycl::half inEl[6][6] = {};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j] = board[i][j + 3];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
  }

  // bottom-left
  {
    sycl::half inEl[6][6] = {};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j + 1] = board[i + 3][j];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
  }

  // bottom-right
  {
    sycl::half inEl[6][6] = {};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j] = board[i + 3][j + 3];

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
  }
}

/**
 * @brief SYCL device kernel: fused Winograd output transform (No SE, output only).
 *
 * Same as OutputInputTransformKernel_NoSE_SubGroup but writes the final spatial result to @p output
 * instead of re-applying the Winograd input transform. Used for the last residual block in the tower.
 *
 * @tparam activation      Post-activation (e.g., ACTIVATION_RELU, ACTIVATION_MISH).
 * @tparam use_bias        If true, adds per-channel @p bias.
 * @tparam use_skip        If true, adds skip connection from @p skip.
 * @tparam skipInput_nhcw  Layout of the @p skip tensor: true = NHCW, false = NCHW.
 * @tparam output_nhcw     Layout of the @p output tensor: true = NHCW, false = NCHW.
 * @param  N      Batch size.
 * @param  C      Number of channels. Must be <= device max_work_group_size.
 * @param  output [out] Final spatial output tensor (NHCW or NCHW as per output_nhcw).
 * @param  input  [in]  Winograd-transformed input tensor (TEMP_INDEX_HWNC layout).
 * @param  skip   [in]  Skip connection tensor (layout per skipInput_nhcw).
 * @param  bias   [in]  Per-channel bias (length C), or nullptr if !use_bias.
 * @param  item_ct1   SYCL nd_item for the 3D kernel.
 */
template <ActivationFunction activation, bool use_bias, bool use_skip,
          bool skipInput_nhcw, bool output_nhcw>
void OutputTransformKernel_NoSE_SubGroup(
    int N, int C, sycl::half* output, const sycl::half* input,
    const sycl::half* skip, const sycl::half* bias,
    const sycl::nd_item<3>& item_ct1) {
  int k = item_ct1.get_local_id(2);
  int n = item_ct1.get_group(2);

  sycl::half board[8][8];
  sycl::half b = use_bias ? bias[k] : (sycl::half)0.0f;

#pragma unroll
  for (int hStart = 0; hStart < 8; hStart += 4) {
#pragma unroll
    for (int wStart = 0; wStart < 8; wStart += 4) {
      int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
      sycl::half outElTransformed[6][6];
#pragma unroll
      for (int y = 0; y < 6; y++) {
#pragma unroll
        for (int x = 0; x < 6; x++) {
          outElTransformed[y][x] = input[TEMP_INDEX_HWNC(y, x, shln, k)];
        }
      }

      sycl::half outEl[4][4];
      OutputTransform4x4(&outEl[0][0], &outElTransformed[0][0]);

#pragma unroll
      for (int y = 0; y < 4; y++) {
        copyAs<sycl::uint2>(&board[hStart + y][wStart], &outEl[y][0]);
      }
    }
  }

#pragma unroll
  for (int h = 0; h < 8; h++) {
#pragma unroll
    for (int w = 0; w < 8; w++) {
      float board_val = (float)board[h][w];
      if (use_bias) {
        board_val += (float)b;
      }

      if (use_skip) {
        if (skipInput_nhcw)
          board_val += (float)skip[INDEX_NHCW(n, k, h, w)];
        else
          board_val += (float)skip[INDEX_NCHW(n, k, h, w)];
      }

      if (activation != ACTIVATION_NONE) {
        board_val = activate(board_val, activation);
      }

      if (output_nhcw)
        output[INDEX_NHCW(n, k, h, w)] = (sycl::half)board_val;
      else
        output[INDEX_NCHW(n, k, h, w)] = (sycl::half)board_val;
    }
  }
}

/**
 * @brief Submits the SubGroup fused output+input transform kernel (No SE) to a SYCL queue.
 *
 * Launches OutputInputTransformKernel_NoSE_SubGroup with nd_range(N*C, C), one work-group per
 * sample, C work-items per work-group.
 *
 * @pre  C <= sycl_queue.get_device().get_info<sycl::info::device::max_work_group_size>()
 * @param  N, C         Batch size, channel count.
 * @param  output       Winograd transform-domain output (TEMP_INDEX_HWNC).
 * @param  input        Winograd transform-domain input (TEMP_INDEX_HWNC).
 * @param  skip         Skip connection in NHCW layout (may be nullptr if !use_skip).
 * @param  bias         Per-channel bias (length C; may be nullptr if !use_bias).
 * @param  sycl_queue   Target in-order SYCL queue.
 */
template <ActivationFunction activation, bool use_bias, bool use_skip>
void SubGroupOutputInputTransform_NoSE(
    int N, int C, sycl::half* output, const sycl::half* input,
    const sycl::half* skip, const sycl::half* bias,
    sycl::queue &sycl_queue) {
  sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
            sycl::range<3>(1, 1, C)),
        [=](sycl::nd_item<3> item) {
          OutputInputTransformKernel_NoSE_SubGroup<activation, use_bias, use_skip>(
              N, C, output, input, (sycl::half*)skip, (sycl::half*)bias, item);
        });
  });
}

/**
 * @brief Submits the SubGroup fused output transform kernel (No SE) to a SYCL queue.
 *
 * Launches OutputTransformKernel_NoSE_SubGroup — like SubGroupOutputInputTransform_NoSE but writes
 * the final spatial result directly to @p output without re-applying the input transform.
 * Used for the last residual block in the tower.
 *
 * @tparam skipInput_nhcw  Layout of @p skip: true = NHCW (residual path), false = NCHW.
 * @tparam output_nhcw     Layout of @p output: true = NHCW, false = NCHW.
 * @pre  C <= sycl_queue.get_device().get_info<sycl::info::device::max_work_group_size>()
 */
template <ActivationFunction activation, bool use_bias, bool use_skip,
          bool skipInput_nhcw, bool output_nhcw>
void SubGroupOutputTransform_NoSE(
    int N, int C, sycl::half* output, const sycl::half* input,
    const sycl::half* skip, const sycl::half* bias,
    sycl::queue &sycl_queue) {
  sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
            sycl::range<3>(1, 1, C)),
        [=](sycl::nd_item<3> item) {
          OutputTransformKernel_NoSE_SubGroup<activation, use_bias, use_skip,
                                              skipInput_nhcw, output_nhcw>(
              N, C, output, input, (const sycl::half*)skip, (const sycl::half*)bias, item);
        });
  });
}

} // namespace sycldnn_backend
} // namespace lczero

