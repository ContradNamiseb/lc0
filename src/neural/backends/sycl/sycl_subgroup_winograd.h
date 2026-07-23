#pragma once

#include <sycl/sycl.hpp>
#include "winograd_helper.h"
#include "neural/backends/shared/activation.h"

namespace lczero {
namespace sycldnn_backend {

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

template <ActivationFunction activation, bool use_bias, bool use_skip>
void SubGroupOutputInputTransform(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input,
    const sycl::half* skip, const sycl::half* bias, const sycl::half* w1,
    const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
    sycl::queue &sycl_queue) {
  
  sycl_queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> shared_data_acc(sycl::range<1>(C), cgh);

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

template <ActivationFunction activation, bool use_bias, bool use_skip,
          bool skipInput_nhcw, bool output_nhcw>
void SubGroupOutputTransform(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input,
    const sycl::half* skip, const sycl::half* bias, const sycl::half* w1,
    const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
    sycl::queue &sycl_queue) {
  
  sycl_queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> shared_data_acc(sycl::range<1>(C), cgh);

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

} // namespace sycldnn_backend
} // namespace lczero
