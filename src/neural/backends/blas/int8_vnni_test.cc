/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2026 The LCZero Authors

 Leela Chess is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Leela Chess is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "neural/backends/blas/int8_vnni.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#include "neural/backends/blas/int8_layers.h"

namespace lczero {
namespace blas_backend {
namespace {

constexpr size_t kSquares = 64;

void ScalarGemmRef(size_t rows, size_t outputs, size_t inputs, const uint8_t* a,
                   const int8_t* w, const float* w_scales, float in_scale,
                   uint8_t in_zp, const float* biases, ActivationFunction act,
                   float* out) {
  for (size_t r = 0; r < rows; ++r) {
    for (size_t o = 0; o < outputs; ++o) {
      int32_t acc = 0;
      for (size_t k = 0; k < inputs; ++k) {
        acc += (static_cast<int32_t>(a[r * inputs + k]) -
                static_cast<int32_t>(in_zp)) *
               static_cast<int32_t>(w[o * inputs + k]);
      }
      float val = static_cast<float>(acc) * (in_scale * w_scales[o]);
      if (biases) val += biases[o];
      out[r * outputs + o] = Activate(val, act);
    }
  }
}

void ScalarConv3NhwcRef(size_t channels, const float* in_nhwc, const int8_t* w,
                        const float* w_scales, float in_scale, uint8_t in_zp,
                        const float* biases, ActivationFunction act,
                        float* out_nhwc) {
  // Quantize input first (using fmaf(x, 1/scale, zp) to match AVX-512 FMA).
  const float inv_scale = 1.0f / in_scale;
  std::vector<uint8_t> q(kSquares * channels);
  for (size_t i = 0; i < q.size(); ++i) {
    float v = std::min(std::max(std::fmaf(in_nhwc[i], inv_scale, in_zp), 0.0f),
                       255.0f);
    q[i] = static_cast<uint8_t>(std::rint(v));
  }
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      for (size_t oc = 0; oc < channels; ++oc) {
        int32_t acc = 0;
        for (size_t ic = 0; ic < channels; ++ic) {
          for (int dy = 0; dy < 3; ++dy) {
            for (int dx = 0; dx < 3; ++dx) {
              int ny = y + dy - 1;
              int nx = x + dx - 1;
              uint8_t a = (ny >= 0 && ny < 8 && nx >= 0 && nx < 8)
                              ? q[(ny * 8 + nx) * channels + ic]
                              : in_zp;
              int8_t wt = w[oc * (channels * 9) + ic * 9 + dy * 3 + dx];
              acc += (static_cast<int32_t>(a) - static_cast<int32_t>(in_zp)) *
                     static_cast<int32_t>(wt);
            }
          }
        }
        float val = biases
                        ? std::fmaf(static_cast<float>(acc),
                                    in_scale * w_scales[oc], biases[oc])
                        : static_cast<float>(acc) * (in_scale * w_scales[oc]);
        out_nhwc[(y * 8 + x) * channels + oc] = Activate(val, act);
      }
    }
  }
}

void ScalarResidualBlockNhwcRef(
    size_t channels, float* board_io, const int8_t* w1, const float* w1_scales,
    float in1_scale, uint8_t in1_zp, const float* b1, const int8_t* w2,
    const float* w2_scales, float in2_scale, uint8_t in2_zp, const float* b2,
    ActivationFunction act, const ResidualSeParams& se) {
  std::vector<float> mid(kSquares * channels);
  std::vector<float> conv2_out(kSquares * channels);
  ScalarConv3NhwcRef(channels, board_io, w1, w1_scales, in1_scale, in1_zp, b1,
                     act, mid.data());
  ScalarConv3NhwcRef(channels, mid.data(), w2, w2_scales, in2_scale, in2_zp, b2,
                     ACTIVATION_NONE, conv2_out.data());
  if (se.has_se) {
    std::vector<float> pool(channels, 0.0f);
    for (size_t sq = 0; sq < kSquares; ++sq) {
      for (size_t c = 0; c < channels; ++c) {
        pool[c] += conv2_out[sq * channels + c];
      }
    }
    for (size_t c = 0; c < channels; ++c) pool[c] /= 64.0f;

    std::vector<float> fc1(se.se_fc_outputs, 0.0f);
    for (size_t o = 0; o < se.se_fc_outputs; ++o) {
      float sum = se.b1[o];
      for (size_t c = 0; c < channels; ++c) {
        sum += se.w1[o * channels + c] * pool[c];
      }
      fc1[o] = Activate(sum, act);
    }
    std::vector<float> fc2(2 * channels, 0.0f);
    for (size_t o = 0; o < 2 * channels; ++o) {
      float sum = se.b2[o];
      for (size_t i = 0; i < se.se_fc_outputs; ++i) {
        sum += se.w2[o * se.se_fc_outputs + i] * fc1[i];
      }
      fc2[o] = sum;
    }
    for (size_t sq = 0; sq < kSquares; ++sq) {
      for (size_t c = 0; c < channels; ++c) {
        float gamma = 1.0f / (1.0f + std::exp(-fc2[c]));
        float beta = fc2[channels + c];
        float v = gamma * conv2_out[sq * channels + c] +
                  board_io[sq * channels + c] + beta;
        board_io[sq * channels + c] = Activate(v, act);
      }
    }
  } else {
    for (size_t i = 0; i < kSquares * channels; ++i) {
      board_io[i] = Activate(conv2_out[i] + board_io[i], act);
    }
  }
}

std::vector<Int8Kernel> AvailableVnniKernels() {
  std::vector<Int8Kernel> ks;
  if (CpuSupportsAvx512Vnni()) ks.push_back(Int8Kernel::kVnniAvx512);
  if (CpuSupportsAvxVnni()) ks.push_back(Int8Kernel::kVnniAvx);
  return ks;
}

TEST(BlasInt8VnniTest, SignedUnsignedSemanticsProbe) {
  for (Int8Kernel kernel : AvailableVnniKernels()) {
    std::vector<int8_t> weights = {-1};
    std::vector<float> scales = {1.0f};
    Int8Weights layer(weights, scales, 1.0f, 0, kernel);
    uint8_t input = 200;
    float bias = 0.0f;
    float out = 0.0f;
    Int8Scratch scratch;
    layer.Forward(1, &input, &bias, ACTIVATION_NONE, &out, scratch);
    EXPECT_FLOAT_EQ(out, -200.0f);
  }
}

TEST(BlasInt8VnniTest, QuantizeParity) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  const size_t n = 2013;
  std::vector<float> input(n);
  for (size_t i = 0; i < n; ++i) input[i] = dist(rng);
  // Explicit half-way ties-to-even values.
  input[0] = 0.0f;
  input[1] = 0.5f;
  input[2] = 1.5f;
  input[3] = 2.5f;
  input[4] = -100.0f;
  input[5] = 1000.0f;

  const float scale = 0.025f;
  const uint8_t zp = 131;
  std::vector<int8_t> dummy_w = {1};
  std::vector<float> dummy_s = {1.0f};

  std::vector<uint8_t> ref(n);
  for (size_t i = 0; i < n; ++i) {
    float v = std::min(std::max(input[i] * (1.0f / scale) + zp, 0.0f), 255.0f);
    ref[i] = static_cast<uint8_t>(std::rint(v));
  }

  for (Int8Kernel kernel : AvailableVnniKernels()) {
    Int8Weights layer(dummy_w, dummy_s, scale, zp, kernel);
    std::vector<uint8_t> got(n, 0);
    layer.QuantizeInput(n, input.data(), got.data());
    for (size_t i = 0; i < n; ++i) {
      EXPECT_EQ(got[i], ref[i]) << "Mismatch at index " << i;
    }
  }
}

TEST(BlasInt8VnniTest, GemmEdgeShapesAndActivations) {
  struct Shape {
    size_t m, n, k;
  };
  const std::vector<Shape> shapes = {
      {1, 1, 1},      {3, 17, 5},     {7, 31, 13},
      {8, 32, 64},    {13, 65, 129},  {16, 128, 256},
      {7, 1024, 256}, {64, 256, 256}, {71, 384, 256},
  };
  const std::vector<ActivationFunction> acts = {
      ACTIVATION_NONE, ACTIVATION_RELU,  ACTIVATION_RELU_2,
      ACTIVATION_MISH, ACTIVATION_SWISH,
  };

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> w_dist(-127, 127);
  std::uniform_int_distribution<int> a_dist(0, 255);
  std::uniform_real_distribution<float> f_dist(-1.0f, 1.0f);

  Int8ThreadPool pool(4);
  for (Int8Kernel kernel : AvailableVnniKernels()) {
    for (const auto& s : shapes) {
      std::vector<int8_t> w(s.n * s.k);
      for (auto& v : w) v = static_cast<int8_t>(w_dist(rng));
      std::vector<float> w_scales(s.n);
      for (auto& v : w_scales) v = 0.001f + std::abs(f_dist(rng)) * 0.01f;
      std::vector<float> biases(s.n);
      for (auto& v : biases) v = f_dist(rng) * 0.5f;
      std::vector<uint8_t> a(s.m * s.k);
      for (auto& v : a) v = static_cast<uint8_t>(a_dist(rng));

      const float in_scale = 0.015f;
      const uint8_t in_zp = 140;
      Int8Weights layer(w, w_scales, in_scale, in_zp, kernel);

      for (ActivationFunction act : acts) {
        std::vector<float> ref(s.m * s.n);
        ScalarGemmRef(s.m, s.n, s.k, a.data(), w.data(), w_scales.data(),
                      in_scale, in_zp, biases.data(), act, ref.data());

        for (int threads : {1, 4}) {
          Int8Scratch scratch;
          scratch.pool = (threads > 1) ? &pool : nullptr;
          std::vector<float> got(s.m * s.n, -999.0f);
          layer.Forward(s.m, a.data(), biases.data(), act, got.data(), scratch);

          for (size_t i = 0; i < ref.size(); ++i) {
            const float tol = 1e-4f * std::max(1.0f, std::abs(ref[i]));
            ASSERT_NEAR(got[i], ref[i], tol)
                << "Shape (" << s.m << "," << s.n << "," << s.k
                << ") act=" << act << " threads=" << threads << " idx=" << i;
          }
        }
      }
    }
  }
}

TEST(BlasInt8VnniTest, ResidualBlockNhwcParity) {
  std::mt19937 rng(777);
  std::uniform_int_distribution<int> w_dist(-64, 64);
  std::uniform_real_distribution<float> f_dist(-1.0f, 1.0f);

  Int8ThreadPool pool(4);
  for (Int8Kernel kernel : AvailableVnniKernels()) {
    for (size_t channels : {size_t{37}, size_t{128}, size_t{192}}) {
      for (bool has_se : {false, true}) {
        for (ActivationFunction act : {ACTIVATION_RELU, ACTIVATION_MISH}) {
          const size_t batch = 5;
          const size_t w_size = channels * channels * 9;
          std::vector<int8_t> w1(w_size), w2(w_size);
          for (auto& v : w1) v = static_cast<int8_t>(w_dist(rng));
          for (auto& v : w2) v = static_cast<int8_t>(w_dist(rng));
          std::vector<float> s1(channels), s2(channels), b1(channels),
              b2(channels);
          for (size_t c = 0; c < channels; ++c) {
            s1[c] = 0.0005f + std::abs(f_dist(rng)) * 0.001f;
            s2[c] = 0.0005f + std::abs(f_dist(rng)) * 0.001f;
            b1[c] = f_dist(rng) * 0.2f;
            b2[c] = f_dist(rng) * 0.2f;
          }
          const float in1_scale = 0.02f, in2_scale = 0.025f;
          const uint8_t in1_zp = 128, in2_zp = 135;

          const size_t se_fc = 16;
          std::vector<float> se_w1(se_fc * channels), se_b1(se_fc),
              se_w2(2 * channels * se_fc), se_b2(2 * channels);
          for (auto& v : se_w1) v = f_dist(rng) * 0.1f;
          for (auto& v : se_b1) v = f_dist(rng) * 0.1f;
          for (auto& v : se_w2) v = f_dist(rng) * 0.1f;
          for (auto& v : se_b2) v = f_dist(rng) * 0.1f;

          ResidualSeParams se;
          se.has_se = has_se;
          se.se_fc_outputs = se_fc;
          se.w1 = se_w1.data();
          se.b1 = se_b1.data();
          se.w2 = se_w2.data();
          se.b2 = se_b2.data();

          std::vector<float> input(batch * kSquares * channels);
          for (auto& v : input) v = f_dist(rng);

          std::vector<float> ref = input;
          for (size_t b = 0; b < batch; ++b) {
            ScalarResidualBlockNhwcRef(
                channels, ref.data() + b * kSquares * channels, w1.data(),
                s1.data(), in1_scale, in1_zp, b1.data(), w2.data(), s2.data(),
                in2_scale, in2_zp, b2.data(), act, se);
          }

          Int8Weights conv1(w1, s1, in1_scale, in1_zp, kernel, true, true);
          Int8Weights conv2(w2, s2, in2_scale, in2_zp, kernel, true, true);
          std::vector<float> got = input;
          Int8Scratch scratch;
          scratch.pool = &pool;
          Int8Weights::RunResidualBlockNhwc(batch, channels, got.data(), conv1,
                                            b1.data(), conv2, b2.data(), act,
                                            se, scratch);

          for (size_t i = 0; i < ref.size(); ++i) {
            // Two cascaded quantized layers with Mish can tip a single
            // mid-layer activation across a 0.5 rounding boundary from a 1-ULP
            // FMA / exp() difference, shifting one output by ~1e-3.
            const float tol = (act == ACTIVATION_MISH ? 2e-3f : 5e-4f) *
                              std::max(1.0f, std::abs(ref[i]));
            ASSERT_NEAR(got[i], ref[i], tol)
                << "channels=" << channels << " has_se=" << has_se
                << " act=" << act << " idx=" << i;
          }
        }
      }
    }
  }
}

TEST(BlasInt8VnniTest, NchwNhwcRoundtrip) {
  Int8ThreadPool pool(4);
  for (size_t channels : {size_t{37}, size_t{128}, size_t{192}}) {
    const size_t batch = 7;
    std::vector<float> orig(batch * channels * kSquares);
    for (size_t i = 0; i < orig.size(); ++i) orig[i] = static_cast<float>(i);
    std::vector<float> nhwc(orig.size(), -1.0f);
    std::vector<float> back(orig.size(), -1.0f);
    NchwToNhwcFast(batch, channels, orig.data(), nhwc.data(), &pool);
    NhwcToNchwFast(batch, channels, nhwc.data(), back.data(), &pool);
    EXPECT_EQ(orig, back);
  }
}

TEST(BlasInt8VnniTest, ThreadPoolConcurrentCallers) {
  Int8ThreadPool pool(4);
  std::atomic<int> failures{0};
  auto worker = [&](int seed) {
    for (int iter = 0; iter < 50; ++iter) {
      std::vector<int> out(128, 0);
      pool.ParallelFor(out.size(), 8, [&](size_t b, size_t e, size_t) {
        for (size_t i = b; i < e; ++i) out[i] = static_cast<int>(i) + seed;
      });
      for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] != static_cast<int>(i) + seed) failures.fetch_add(1);
      }
    }
  };
  std::thread t1(worker, 10);
  std::thread t2(worker, 20);
  t1.join();
  t2.join();
  EXPECT_EQ(failures.load(), 0);

  // Verify park-and-wake after workers finish spinning and enter atomic::wait.
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  std::vector<int> after_sleep(64, 0);
  pool.ParallelFor(after_sleep.size(), 8, [&](size_t b, size_t e, size_t) {
    for (size_t i = b; i < e; ++i) after_sleep[i] = static_cast<int>(i) + 99;
  });
  for (size_t i = 0; i < after_sleep.size(); ++i) {
    EXPECT_EQ(after_sleep[i], static_cast<int>(i) + 99);
  }
}

TEST(BlasInt8VnniTest, ThreadPoolAlternatingActiveStress) {
  // Reproducer for idle-worker (tid >= active) data race: alternates active=2
  // and active=4 on a 4-thread pool and checks every slice executes once.
  Int8ThreadPool pool(4);
  std::atomic<int> counts[4];
  for (int iter = 0; iter < 50000; ++iter) {
    const size_t items = (iter & 1) ? 4 : 2;
    for (size_t i = 0; i < items; ++i)
      counts[i].store(0, std::memory_order_relaxed);
    pool.ParallelFor(items, 1, [&](size_t b, size_t e, size_t) {
      for (size_t i = b; i < e; ++i) {
        counts[i].fetch_add(1, std::memory_order_relaxed);
      }
    });
    for (size_t i = 0; i < items; ++i) {
      ASSERT_EQ(counts[i].load(std::memory_order_relaxed), 1)
          << "iter=" << iter << " items=" << items << " slot=" << i;
    }
  }
}

TEST(BlasInt8VnniTest, Convolve3NchwVnniParity) {
  // Verifies single-layer NCHW Convolve3 routes through VNNI (used by partly
  // INT8 residual towers and non-MKL builds).
  std::mt19937 rng(999);
  std::uniform_int_distribution<int> w_dist(-64, 64);
  std::uniform_real_distribution<float> f_dist(-1.0f, 1.0f);

  Int8ThreadPool pool(4);
  for (Int8Kernel kernel : AvailableVnniKernels()) {
    const size_t batch = 3;
    const size_t channels = 40;
    std::vector<int8_t> weights(channels * channels * 9);
    for (auto& v : weights) v = static_cast<int8_t>(w_dist(rng));
    std::vector<float> scales(channels);
    for (auto& v : scales) v = 0.001f + std::abs(f_dist(rng)) * 0.002f;
    const float in_scale = 0.02f;
    const uint8_t in_zp = 130;

    std::vector<float> input_nchw(batch * channels * kSquares);
    for (auto& v : input_nchw) v = f_dist(rng);

    std::vector<float> input_nhwc(input_nchw.size());
    std::vector<float> ref_nhwc(input_nchw.size());
    std::vector<float> ref_nchw(input_nchw.size());
    NchwToNhwcFast(batch, channels, input_nchw.data(), input_nhwc.data());
    for (size_t b = 0; b < batch; ++b) {
      ScalarConv3NhwcRef(channels, input_nhwc.data() + b * kSquares * channels,
                         weights.data(), scales.data(), in_scale, in_zp,
                         nullptr, ACTIVATION_NONE,
                         ref_nhwc.data() + b * kSquares * channels);
    }
    NhwcToNchwFast(batch, channels, ref_nhwc.data(), ref_nchw.data());

    Int8Weights layer(weights, scales, in_scale, in_zp, kernel, true);
    Int8Scratch scratch;
    scratch.pool = &pool;
    std::vector<float> got_nchw(input_nchw.size(), -999.0f);
    layer.Convolve3(batch, input_nchw.data(), got_nchw.data(), scratch);

    for (size_t i = 0; i < ref_nchw.size(); ++i) {
      ASSERT_NEAR(got_nchw[i], ref_nchw[i], 1e-4f) << "idx=" << i;
    }
  }
}

#ifdef LC0_HAS_AVX512_VNNI
TEST(BlasInt8VnniTest, FusedLayerNormQuantizeAndSoftmaxAvx512) {
  if (!CpuSupportsAvx512Vnni()) GTEST_SKIP();
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> f_dist(-2.0f, 2.0f);

  // 1. LayerNorm2DWithSkipAndQuantizeAvx512 (both C=256 fast path & general C)
  for (size_t channels : {size_t{256}, size_t{37}, size_t{512}}) {
    for (bool with_skip : {false, true}) {
      for (bool with_quant : {false, true}) {
        const size_t rows = 17;
        const float alpha = 0.75f;
        const float eps = 1e-3f;
        const float inv_scale = 23.5f;
        const float zp = 128.0f;
        std::vector<float> data(rows * channels), skip(rows * channels);
        std::vector<float> gamma(channels), beta(channels);
        for (auto& v : data) v = f_dist(rng);
        for (auto& v : skip) v = f_dist(rng);
        for (auto& v : gamma) v = 0.5f + std::abs(f_dist(rng));
        for (auto& v : beta) v = f_dist(rng) * 0.25f;

        std::vector<float> ref = data;
        std::vector<uint8_t> ref_q(rows * channels, 0);
        for (size_t r = 0; r < rows; ++r) {
          float mean = 0.0f;
          for (size_t c = 0; c < channels; ++c) {
            float x = ref[r * channels + c] * alpha +
                      (with_skip ? skip[r * channels + c] : 0.0f);
            ref[r * channels + c] = x;
            mean += x;
          }
          mean /= static_cast<float>(channels);
          float var = 0.0f;
          for (size_t c = 0; c < channels; ++c) {
            float d = ref[r * channels + c] - mean;
            var += d * d;
          }
          var /= static_cast<float>(channels);
          const float inv_std = 1.0f / std::sqrt(var + eps);
          for (size_t c = 0; c < channels; ++c) {
            float y =
                beta[c] + gamma[c] * (ref[r * channels + c] - mean) * inv_std;
            ref[r * channels + c] = y;
            if (with_quant) {
              float s = std::clamp(y * inv_scale + zp, 0.0f, 255.0f);
              ref_q[r * channels + c] = static_cast<uint8_t>(std::nearbyint(s));
            }
          }
        }

        std::vector<float> got = data;
        std::vector<uint8_t> got_q(rows * channels, 0xA5);
        LayerNorm2DWithSkipAndQuantizeAvx512(
            rows, channels, got.data(), alpha,
            with_skip ? skip.data() : nullptr, gamma.data(), beta.data(), eps,
            with_quant ? got_q.data() : nullptr, inv_scale, zp);

        for (size_t i = 0; i < ref.size(); ++i) {
          ASSERT_NEAR(got[i], ref[i], 1e-5f * std::max(1.0f, std::abs(ref[i])))
              << "C=" << channels << " skip=" << with_skip << " idx=" << i;
          if (with_quant) {
            ASSERT_LE(std::abs(int(got_q[i]) - int(ref_q[i])), 1)
                << "C=" << channels << " idx=" << i;
          }
        }
      }
    }
  }

  // 2. Softmax64Avx512
  {
    const size_t rows = 33;
    std::vector<float> data(rows * 64);
    for (auto& v : data) v = f_dist(rng) * 3.0f;
    std::vector<float> ref = data;
    for (size_t r = 0; r < rows; ++r) {
      float mx = ref[r * 64];
      for (size_t c = 1; c < 64; ++c) mx = std::max(mx, ref[r * 64 + c]);
      float sum = 0.0f;
      for (size_t c = 0; c < 64; ++c) {
        ref[r * 64 + c] = std::exp(ref[r * 64 + c] - mx);
        sum += ref[r * 64 + c];
      }
      for (size_t c = 0; c < 64; ++c) ref[r * 64 + c] /= sum;
    }
    std::vector<float> got = data;
    Softmax64Avx512(rows, got.data());
    for (size_t i = 0; i < ref.size(); ++i) {
      ASSERT_NEAR(got[i], ref[i], 1e-6f) << "idx=" << i;
    }
  }

  // 3. ForwardQuantized (FFN1 + RELU_2 + Quantize(FFN2))
  {
    Int8ThreadPool pool(4);
    const size_t rows = 67, inputs = 256, outputs = 100;
    std::uniform_int_distribution<int> w_dist(-127, 127);
    std::vector<int8_t> w(outputs * inputs);
    for (auto& v : w) v = static_cast<int8_t>(w_dist(rng));
    std::vector<float> scales(outputs), bias(outputs);
    for (auto& v : scales) v = 0.002f + std::abs(f_dist(rng)) * 0.004f;
    for (auto& v : bias) v = f_dist(rng) * 0.2f;
    std::vector<float> x(rows * inputs);
    for (auto& v : x) v = f_dist(rng);

    Int8Weights layer(w, scales, 0.02f, 128, Int8Kernel::kVnniAvx512);
    Int8Scratch scratch;
    scratch.pool = &pool;
    std::vector<float> fp_out(rows * outputs);
    layer.Forward(rows, x.data(), bias.data(), ACTIVATION_RELU_2, fp_out.data(),
                  scratch);
    const float out_inv_scale = 18.0f;
    const uint8_t out_zp = 12;
    std::vector<uint8_t> ref_q(rows * outputs);
    QuantizeToUint8Avx512(fp_out.size(), fp_out.data(), out_inv_scale,
                          static_cast<float>(out_zp), ref_q.data());

    std::vector<uint8_t> in_q(rows * inputs);
    layer.QuantizeInput(x.size(), x.data(), in_q.data(), &pool);
    std::vector<uint8_t> got_q(rows * outputs, 0x5A);
    layer.ForwardQuantized(rows, in_q.data(), bias.data(), ACTIVATION_RELU_2,
                           out_inv_scale, out_zp, got_q.data(), scratch);
    EXPECT_EQ(got_q, ref_q);
  }

  // 4. DynamicInt8Weights (both row-parallel and n_block-parallel regimes)
  {
    Int8ThreadPool pool(4);
    for (auto [rows, inputs, outputs] :
         std::vector<std::tuple<size_t, size_t, size_t>>{{16, 256, 2048},
                                                         {16, 2048, 256},
                                                         {128, 256, 512},
                                                         {17, 192, 100}}) {
      std::vector<float> w(outputs * inputs), b(outputs), x(rows * inputs);
      for (auto& v : w) v = f_dist(rng) * 0.25f;
      for (auto& v : b) v = f_dist(rng) * 0.1f;
      for (auto& v : x) v = f_dist(rng);

      std::vector<float> ref(rows * outputs, 0.0f);
      for (size_t r = 0; r < rows; ++r) {
        for (size_t o = 0; o < outputs; ++o) {
          float acc = b[o];
          for (size_t k = 0; k < inputs; ++k) {
            acc += x[r * inputs + k] * w[o * inputs + k];
          }
          ref[r * outputs + o] = Activate(acc, ACTIVATION_SWISH);
        }
      }

      DynamicInt8Weights dyn(w.data(), outputs, inputs);
      Int8Scratch scratch;
      scratch.pool = &pool;
      std::vector<float> got(rows * outputs, -999.0f);
      dyn.Forward(rows, x.data(), b.data(), ACTIVATION_SWISH, got.data(),
                  scratch);

      double sum_sq_err = 0.0, sum_sq_ref = 0.0;
      for (size_t i = 0; i < ref.size(); ++i) {
        double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
        sum_sq_err += d * d;
        sum_sq_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
      }
      const double rel_rmse = std::sqrt(sum_sq_err / (sum_sq_ref + 1e-12));
      EXPECT_LT(rel_rmse, 0.02)
          << "Shape (" << rows << "," << outputs << "," << inputs << ")";
    }
  }

  // 5. Conv3x3Nhwc with fused bias and Mish activation
  {
    Int8ThreadPool pool(4);
    const size_t batch = 3, in_c = 112, out_c = 192;
    std::uniform_int_distribution<int> w_dist(-64, 64);
    std::vector<int8_t> w(out_c * in_c * 9);
    for (auto& v : w) v = static_cast<int8_t>(w_dist(rng));
    std::vector<float> scales(out_c), bias(out_c);
    for (auto& v : scales) v = 0.001f + std::abs(f_dist(rng)) * 0.002f;
    for (auto& v : bias) v = f_dist(rng) * 0.2f;
    std::vector<float> x_nhwc(batch * kSquares * in_c);
    for (auto& v : x_nhwc) v = (f_dist(rng) > 0.0f) ? 1.0f : 0.0f;

    std::vector<float> ref_nhwc(batch * kSquares * out_c, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
      const float* in_b = x_nhwc.data() + b * kSquares * in_c;
      float* out_b = ref_nhwc.data() + b * kSquares * out_c;
      for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
          for (size_t o = 0; o < out_c; ++o) {
            int32_t acc = 0;
            for (size_t c = 0; c < in_c; ++c) {
              for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                  int sy = y + dy, sx = x + dx;
                  uint8_t q = 0;
                  if (sy >= 0 && sy < 8 && sx >= 0 && sx < 8) {
                    q = static_cast<uint8_t>(
                        std::lround(in_b[(sy * 8 + sx) * in_c + c] * 255.0f));
                  }
                  int tap = (dy + 1) * 3 + (dx + 1);
                  acc += static_cast<int32_t>(q) *
                         static_cast<int32_t>(w[(o * in_c + c) * 9 + tap]);
                }
              }
            }
            float val =
                scales[o] * (1.0f / 255.0f) * static_cast<float>(acc) + bias[o];
            out_b[(y * 8 + x) * out_c + o] = Activate(val, ACTIVATION_MISH);
          }
        }
      }
    }

    Int8Weights layer(w, scales, 1.0f / 255.0f, uint8_t{0},
                      Int8Kernel::kVnniAvx512, /*is_conv3x3=*/true);
    Int8Scratch scratch;
    scratch.pool = &pool;
    std::vector<float> got_nhwc(batch * kSquares * out_c, -999.0f);
    layer.Convolve3Nhwc(batch, x_nhwc.data(), got_nhwc.data(), scratch,
                        bias.data(), ACTIVATION_MISH);
    for (size_t i = 0; i < ref_nhwc.size(); ++i) {
      ASSERT_NEAR(got_nhwc[i], ref_nhwc[i],
                  1e-3f * std::max(1.0f, std::abs(ref_nhwc[i])))
          << "idx=" << i;
    }
  }

  // 6. DynamicConv3x3Int8 (direct-NHWC policy1/policy path)
  {
    Int8ThreadPool pool(4);
    const size_t batch = 3, in_c = 128, out_c = 80;
    std::vector<float> w(out_c * in_c * 9), bias(out_c);
    for (auto& v : w) v = f_dist(rng) * 0.1f;
    for (auto& v : bias) v = f_dist(rng) * 0.2f;
    std::vector<float> x_nhwc(batch * kSquares * in_c);
    for (auto& v : x_nhwc) v = std::max(0.0f, f_dist(rng));

    std::vector<float> ref(batch * kSquares * out_c, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
      const float* in_b = x_nhwc.data() + b * kSquares * in_c;
      float* out_b = ref.data() + b * kSquares * out_c;
      for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
          for (size_t o = 0; o < out_c; ++o) {
            float acc = bias[o];
            for (size_t c = 0; c < in_c; ++c) {
              for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                  int sy = y + dy, sx = x + dx;
                  if (sy >= 0 && sy < 8 && sx >= 0 && sx < 8) {
                    int tap = (dy + 1) * 3 + (dx + 1);
                    acc += in_b[(sy * 8 + sx) * in_c + c] *
                           w[(o * in_c + c) * 9 + tap];
                  }
                }
              }
            }
            out_b[(y * 8 + x) * out_c + o] = acc;
          }
        }
      }
    }

    DynamicConv3x3Int8 layer(w.data(), out_c, in_c);
    Int8Scratch scratch;
    scratch.pool = &pool;
    std::vector<float> got(batch * kSquares * out_c, -999.0f);
    layer.ForwardNhwc(batch, x_nhwc.data(), bias.data(), ACTIVATION_NONE,
                      got.data(), scratch);
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
      double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
      sum_sq_err += d * d;
      sum_sq_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    EXPECT_LT(std::sqrt(sum_sq_err / (sum_sq_ref + 1e-12)), 0.02);
  }
}

TEST(BlasInt8VnniTest, SeluDynamicQuant192AndPreparedReuse) {
  std::mt19937 rng(791556);
  std::uniform_real_distribution<float> f_dist(-3.5f, 3.5f);

  const size_t rows = 128;
  const size_t inputs = 192;
  const size_t outputs = 192;
  std::vector<float> x(rows * inputs);
  for (auto& v : x) v = f_dist(rng);

  std::vector<float> w_q(outputs * inputs), b_q(outputs);
  std::vector<float> w_k(outputs * inputs), b_k(outputs);
  for (auto& v : w_q) v = f_dist(rng) * 0.1f;
  for (auto& v : w_k) v = f_dist(rng) * 0.1f;
  for (auto& v : b_q) v = f_dist(rng) * 0.2f;
  for (auto& v : b_k) v = f_dist(rng) * 0.2f;

  DynamicInt8Weights layer_q(w_q.data(), outputs, inputs);
  DynamicInt8Weights layer_k(w_k.data(), outputs, inputs);
  Int8ThreadPool pool(4);
  Int8Scratch scratch;
  scratch.pool = &pool;

  std::vector<float> out_selu(rows * outputs);
  layer_q.Forward(rows, x.data(), b_q.data(), ACTIVATION_SELU, out_selu.data(),
                  scratch);

  layer_q.PrepareDynamicRows(rows, x.data(), scratch);
  std::vector<float> out_q(rows * outputs);
  std::vector<float> out_k(rows * outputs);
  std::vector<float> ref_k(rows * outputs);
  layer_q.ForwardPrepared(rows, b_q.data(), ACTIVATION_NONE, out_q.data(),
                          scratch);
  layer_k.ForwardPrepared(rows, b_k.data(), ACTIVATION_NONE, out_k.data(),
                          scratch);
  layer_k.Forward(rows, x.data(), b_k.data(), ACTIVATION_NONE, ref_k.data(),
                  scratch);

  for (size_t i = 0; i < out_k.size(); ++i) {
    ASSERT_NEAR(out_k[i], ref_k[i], 1e-4f) << "idx=" << i;
  }
  for (size_t i = 0; i < out_selu.size(); ++i) {
    const float expected_selu = Activate(out_q[i], ACTIVATION_SELU);
    ASSERT_NEAR(out_selu[i], expected_selu,
                5e-4f * std::max(1.0f, std::abs(expected_selu)))
        << "idx=" << i;
  }
}

TEST(BlasInt8VnniTest, PreTransposedSeAndMishExpAccuracy) {
  std::mt19937 rng(202611);
  std::uniform_real_distribution<float> f_dist(-1.5f, 1.5f);
  std::uniform_int_distribution<int> w_dist(-25, 25);

  const size_t batch = 3;
  const size_t channels = 192;
  const size_t se_k = 32;
  std::vector<int8_t> w1(channels * channels * 9), w2(channels * channels * 9);
  for (auto& v : w1) v = static_cast<int8_t>(w_dist(rng));
  for (auto& v : w2) v = static_cast<int8_t>(w_dist(rng));
  std::vector<float> s1(channels), b1(channels), s2(channels), b2(channels);
  for (size_t o = 0; o < channels; ++o) {
    s1[o] = 0.008f + 0.001f * static_cast<float>(o % 7);
    s2[o] = 0.009f + 0.001f * static_cast<float>(o % 5);
    b1[o] = f_dist(rng) * 0.25f;
    b2[o] = f_dist(rng) * 0.25f;
  }

  std::vector<float> se_w1(se_k * channels), se_b1(se_k);
  std::vector<float> se_w2(2 * channels * se_k), se_b2(2 * channels);
  for (auto& v : se_w1) v = f_dist(rng) * 0.08f;
  for (auto& v : se_b1) v = f_dist(rng) * 0.2f;
  for (auto& v : se_w2) v = f_dist(rng) * 0.08f;
  for (auto& v : se_b2) v = f_dist(rng) * 0.2f;

  std::vector<float> se_w1_t(channels * se_k);
  for (size_t o = 0; o < se_k; ++o) {
    for (size_t c = 0; c < channels; ++c) {
      se_w1_t[c * se_k + o] = se_w1[o * channels + c];
    }
  }
  std::vector<float> se_w2_t(se_k * 2 * channels);
  for (size_t o = 0; o < 2 * channels; ++o) {
    for (size_t k = 0; k < se_k; ++k) {
      se_w2_t[k * 2 * channels + o] = se_w2[o * se_k + k];
    }
  }

  Int8Weights conv1_nr32(w1, s1, 1.0f / 64.0f, uint8_t{128},
                         Int8Kernel::kVnniAvx512, /*is_conv3x3=*/true,
                         /*allow_nr48=*/false);
  Int8Weights conv2_nr32(w2, s2, 1.0f / 64.0f, uint8_t{128},
                         Int8Kernel::kVnniAvx512, /*is_conv3x3=*/true,
                         /*allow_nr48=*/false);
  Int8Weights conv1_nr48(w1, s1, 1.0f / 64.0f, uint8_t{128},
                         Int8Kernel::kVnniAvx512, /*is_conv3x3=*/true,
                         /*allow_nr48=*/true);
  Int8Weights conv2_nr48(w2, s2, 1.0f / 64.0f, uint8_t{128},
                         Int8Kernel::kVnniAvx512, /*is_conv3x3=*/true,
                         /*allow_nr48=*/true);

  ResidualSeParams se_plain;
  se_plain.has_se = true;
  se_plain.se_fc_outputs = se_k;
  se_plain.w1 = se_w1.data();
  se_plain.b1 = se_b1.data();
  se_plain.w2 = se_w2.data();
  se_plain.b2 = se_b2.data();

  ResidualSeParams se_transposed = se_plain;
  se_transposed.w1_t = se_w1_t.data();
  se_transposed.w2_t = se_w2_t.data();

  std::vector<float> io_plain(batch * kSquares * channels);
  for (auto& v : io_plain) v = f_dist(rng);
  std::vector<float> io_transposed = io_plain;

  Int8ThreadPool pool(4);
  Int8Scratch scratch;
  scratch.pool = &pool;

  Int8Weights::RunResidualBlockNhwc(
      batch, channels, io_plain.data(), conv1_nr32, b1.data(), conv2_nr32,
      b2.data(), ACTIVATION_MISH, se_plain, scratch);
  Int8Weights::RunResidualBlockNhwc(
      batch, channels, io_transposed.data(), conv1_nr48, b1.data(), conv2_nr48,
      b2.data(), ACTIVATION_MISH, se_transposed, scratch);

  for (size_t i = 0; i < io_plain.size(); ++i) {
    ASSERT_NEAR(io_transposed[i], io_plain[i],
                2e-3f * std::max(1.0f, std::abs(io_plain[i])))
        << "idx=" << i;
  }
}

TEST(BlasInt8VnniTest, GemmFp32Avx512AndConvolve3NchwToNhwc) {
  std::mt19937 rng(98765);
  std::uniform_real_distribution<float> f_dist(-4.0f, 4.0f);
  std::uniform_real_distribution<float> wide_dist(-20.0f, 20.0f);
  Int8ThreadPool pool(4);

  struct ShapeCase {
    size_t rows;
    size_t outputs;
    size_t inputs;
    ActivationFunction act;
  };
  const ShapeCase cases[] = {
      {1, 1, 128, ACTIVATION_RELU},     {7, 15, 33, ACTIVATION_NONE},
      {16, 128, 2048, ACTIVATION_MISH}, {64, 48, 192, ACTIVATION_SELU},
      {448, 32, 192, ACTIVATION_MISH},
  };
  for (const auto& sc : cases) {
    std::vector<float> a(sc.rows * sc.inputs);
    std::vector<float> w_t(sc.inputs * sc.outputs);
    std::vector<float> bias(sc.outputs);
    for (auto& v : a)
      v = (sc.act == ACTIVATION_MISH ? wide_dist : f_dist)(rng) * 0.1f;
    for (auto& v : w_t) v = f_dist(rng) * 0.05f;
    for (auto& v : bias) v = f_dist(rng) * 0.5f;

    std::vector<float> got(sc.rows * sc.outputs, -999.0f);
    RunGemmFp32Avx512(sc.rows, sc.outputs, sc.inputs, a.data(), w_t.data(),
                      bias.data(), sc.act, got.data(), &pool);

    for (size_t r = 0; r < sc.rows; ++r) {
      for (size_t o = 0; o < sc.outputs; ++o) {
        double acc = bias[o];
        for (size_t k = 0; k < sc.inputs; ++k) {
          acc += static_cast<double>(a[r * sc.inputs + k]) *
                 static_cast<double>(w_t[k * sc.outputs + o]);
        }
        const float expected = Activate(static_cast<float>(acc), sc.act);
        ASSERT_NEAR(got[r * sc.outputs + o], expected,
                    1e-3f * std::max(1.0f, std::abs(expected)))
            << "rows=" << sc.rows << " out=" << sc.outputs << " r=" << r
            << " o=" << o;
      }
    }
  }

  // Convolve3NchwToNhwc (112 -> 192, conv_nr=48, plane109_scale=0.01f) parity.
  const size_t batch = 4;
  const size_t in_c = 112;
  const size_t out_c = 192;
  std::uniform_int_distribution<int> w_dist(-30, 30);
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  std::vector<int8_t> w(out_c * in_c * 9);
  for (auto& v : w) v = static_cast<int8_t>(w_dist(rng));
  std::vector<float> scales(out_c), bias(out_c);
  for (size_t o = 0; o < out_c; ++o) {
    scales[o] = 0.01f + 0.002f * static_cast<float>(o % 5);
    bias[o] = f_dist(rng) * 0.2f;
  }
  std::vector<float> x_nchw(batch * in_c * kSquares);
  for (auto& v : x_nchw) v = u01(rng);
  for (size_t b = 0; b < batch; ++b) {
    for (size_t sq = 0; sq < kSquares; ++sq) {
      x_nchw[(b * in_c + 109) * kSquares + sq] = 50.0f * u01(rng);
    }
  }

  Int8Weights conv(w, scales, 1.0f / 255.0f, uint8_t{0},
                   Int8Kernel::kVnniAvx512, /*is_conv3x3=*/true,
                   /*allow_nr48=*/true);
  Int8Scratch scratch;
  scratch.pool = &pool;

  std::vector<float> fused_nhwc(batch * kSquares * out_c, -999.0f);
  conv.Convolve3NchwToNhwc(batch, x_nchw.data(), 0.01f, fused_nhwc.data(),
                           scratch, bias.data(), ACTIVATION_MISH);

  std::vector<float> staged_in(batch * kSquares * in_c);
  NchwToNhwcFast(batch, in_c, x_nchw.data(), staged_in.data(), &pool);
  for (size_t sq = 0; sq < batch * kSquares; ++sq) {
    staged_in[sq * in_c + 109] *= 0.01f;
  }
  std::vector<float> ref_nhwc(batch * kSquares * out_c, -888.0f);
  conv.Convolve3Nhwc(batch, staged_in.data(), ref_nhwc.data(), scratch,
                     bias.data(), ACTIVATION_MISH);

  for (size_t i = 0; i < ref_nhwc.size(); ++i) {
    ASSERT_FLOAT_EQ(fused_nhwc[i], ref_nhwc[i]) << "idx=" << i;
  }
}

TEST(BlasInt8VnniTest, AttentionPrepGatingAndPromoParity) {
  std::mt19937 rng(20261128);
  std::uniform_real_distribution<float> f_dist(-2.0f, 2.0f);

  const size_t batch = 3;
  const size_t in_c = 112;
  const size_t pe_c = 64;
  const size_t emb_in = in_c + pe_c;
  std::vector<float> planes(batch * kSquares * in_c);
  std::vector<float> pe(kSquares * pe_c);
  for (auto& v : planes) v = f_dist(rng);
  for (auto& v : pe) v = f_dist(rng);

  std::vector<float> got_nhwc(batch * kSquares * emb_in, -999.0f);
  PrepAttentionInputPeMapAvx512(batch, in_c, pe_c, planes.data(), pe.data(),
                                got_nhwc.data());
  for (size_t b = 0; b < batch; ++b) {
    for (size_t sq = 0; sq < kSquares; ++sq) {
      const float* row = got_nhwc.data() + (b * kSquares + sq) * emb_in;
      for (size_t c = 0; c < in_c; ++c) {
        ASSERT_FLOAT_EQ(row[c],
                        planes[b * kSquares * in_c + c * kSquares + sq]);
      }
      for (size_t c = 0; c < pe_c; ++c) {
        ASSERT_FLOAT_EQ(row[in_c + c], pe[sq * pe_c + c]);
      }
    }
  }

  const size_t emb = 256;
  std::vector<float> flow(batch * kSquares * emb);
  std::vector<float> mult_nchw(emb * kSquares);
  std::vector<float> add_nchw(emb * kSquares);
  for (auto& v : flow) v = f_dist(rng);
  for (auto& v : mult_nchw) v = f_dist(rng);
  for (auto& v : add_nchw) v = f_dist(rng);
  std::vector<float> flow_ref = flow;
  for (size_t b = 0; b < batch; ++b) {
    for (size_t sq = 0; sq < kSquares; ++sq) {
      for (size_t c = 0; c < emb; ++c) {
        const size_t idx = (b * kSquares + sq) * emb + c;
        flow_ref[idx] = flow_ref[idx] * mult_nchw[c * kSquares + sq] +
                        add_nchw[c * kSquares + sq];
      }
    }
  }
  std::vector<float> mult_nhwc(kSquares * emb), add_nhwc(kSquares * emb);
  for (size_t sq = 0; sq < kSquares; ++sq) {
    for (size_t c = 0; c < emb; ++c) {
      mult_nhwc[sq * emb + c] = mult_nchw[c * kSquares + sq];
      add_nhwc[sq * emb + c] = add_nchw[c * kSquares + sq];
    }
  }
  ApplyInputGatingAvx512(batch, emb, flow.data(), mult_nhwc.data(),
                         add_nhwc.data());
  for (size_t i = 0; i < flow.size(); ++i) {
    ASSERT_NEAR(flow[i], flow_ref[i],
                1e-5f * std::max(1.0f, std::abs(flow_ref[i])));
  }

  const size_t pol_d = 256;
  std::vector<float> k_board(kSquares * pol_d);
  std::vector<float> ip4_w(4 * pol_d);
  std::vector<float> logits(64 * 64 + 8 * 24);
  for (auto& v : k_board) v = f_dist(rng);
  for (auto& v : ip4_w) v = f_dist(rng);
  for (auto& v : logits) v = f_dist(rng);
  std::vector<float> logits_ref = logits;

  float promo[4][8];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 8; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < pol_d; ++k) {
        sum += k_board[(56 + j) * pol_d + k] * ip4_w[i * pol_d + k];
      }
      promo[i][j] = sum;
    }
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 8; ++j) promo[i][j] += promo[3][j];
  }
  for (int k = 0; k < 8; ++k) {
    for (int j = 0; j < 8; ++j) {
      for (int i = 0; i < 3; ++i) {
        logits_ref[64 * 64 + 24 * k + 3 * j + i] =
            logits_ref[(48 + k) * 64 + 56 + j] + promo[i][j];
      }
    }
  }
  AttentionPolicyPromoAvx512(pol_d, k_board.data(), ip4_w.data(),
                             logits.data());
  for (size_t i = 0; i < logits.size(); ++i) {
    ASSERT_NEAR(logits[i], logits_ref[i],
                1e-4f * std::max(1.0f, std::abs(logits_ref[i])));
  }
}

TEST(BlasInt8VnniTest, MhaAttentionHead32Avx512Parity) {
  std::mt19937 rng(20261129);
  std::uniform_real_distribution<float> f_dist(-1.5f, 1.5f);

  const size_t d_model = 256;
  const size_t heads = 8;
  const size_t depth = 32;
  const float scaling = 1.0f / std::sqrt(static_cast<float>(depth));

  std::vector<float> q(kSquares * d_model), k(kSquares * d_model),
      v(kSquares * d_model), smol(heads * kSquares * kSquares);
  for (auto& x : q) x = f_dist(rng);
  for (auto& x : k) x = f_dist(rng);
  for (auto& x : v) x = f_dist(rng);
  for (auto& x : smol) x = f_dist(rng) * 0.5f;

  auto run_ref = [&](bool with_smolgen) {
    std::vector<float> out(kSquares * d_model, 0.0f);
    for (size_t h = 0; h < heads; ++h) {
      for (size_t r = 0; r < kSquares; ++r) {
        double logits[kSquares];
        double mx = -1e30;
        for (size_t c = 0; c < kSquares; ++c) {
          double dot = 0.0;
          for (size_t d = 0; d < depth; ++d) {
            dot += static_cast<double>(q[r * d_model + h * depth + d]) *
                   static_cast<double>(k[c * d_model + h * depth + d]);
          }
          logits[c] =
              dot * static_cast<double>(scaling) +
              (with_smolgen ? static_cast<double>(smol[h * 4096 + r * 64 + c])
                            : 0.0);
          if (logits[c] > mx) mx = logits[c];
        }
        double sum = 0.0;
        for (size_t c = 0; c < kSquares; ++c) {
          logits[c] = std::exp(logits[c] - mx);
          sum += logits[c];
        }
        for (size_t d = 0; d < depth; ++d) {
          double acc = 0.0;
          for (size_t c = 0; c < kSquares; ++c) {
            acc += (logits[c] / sum) *
                   static_cast<double>(v[c * d_model + h * depth + d]);
          }
          out[r * d_model + h * depth + d] = static_cast<float>(acc);
        }
      }
    }
    return out;
  };

  for (bool with_smolgen : {false, true}) {
    const std::vector<float> ref = run_ref(with_smolgen);
    std::vector<float> out_aliasing = q;  // in-place q -> attn
    for (size_t h = 0; h < heads; ++h) {
      MhaAttentionHead32Avx512(
          d_model, scaling, out_aliasing.data(), k.data(), v.data(),
          with_smolgen ? &smol[h * kSquares * kSquares] : nullptr, h,
          out_aliasing.data());
    }
    for (size_t i = 0; i < ref.size(); ++i) {
      ASSERT_NEAR(out_aliasing[i], ref[i], 2e-4f)
          << "with_smolgen=" << with_smolgen << " idx=" << i;
    }
  }
}

TEST(BlasInt8VnniTest, SmolgenFusionAndSmallMGemmParity) {
  std::mt19937 rng(20261130);
  std::uniform_real_distribution<float> f_dist(-2.0f, 2.0f);

  // 1. LayerNormAndDynamicQuantizeAvx512 parity for both (rows=7, c=256, g=256)
  // and (rows=7, c=2048, g=256) against separate QuantizeDynamicRowsAvx512.
  for (size_t channels : {size_t{256}, size_t{2048}}) {
    const size_t rows = 7;
    const size_t group = 256;
    const size_t num_groups = channels / group;
    std::vector<float> x(rows * channels), gamma(channels), beta(channels);
    for (auto& v : x) v = f_dist(rng);
    for (auto& v : gamma) v = 0.5f + 0.5f * f_dist(rng);
    for (auto& v : beta) v = f_dist(rng) * 0.25f;

    std::vector<uint8_t> got_q(rows * channels);
    std::vector<float> got_s(rows * num_groups), got_m(rows * num_groups);
    LayerNormAndDynamicQuantizeAvx512(rows, channels, group, x.data(),
                                      gamma.data(), beta.data(), 1e-3f,
                                      got_q.data(), got_s.data(), got_m.data());

    std::vector<float> ref_ln = x;
    LayerNorm2DWithSkipAndQuantizeAvx512(rows, channels, ref_ln.data(), 1.0f,
                                         nullptr, gamma.data(), beta.data(),
                                         1e-3f, nullptr, 0.0f, 0.0f);
    std::vector<uint8_t> ref_q(rows * channels);
    std::vector<float> ref_s(rows * num_groups), ref_m(rows * num_groups);
    QuantizeDynamicRowsAvx512(rows * num_groups, group, ref_ln.data(),
                              ref_q.data(), ref_s.data(), ref_m.data());

    for (size_t i = 0; i < ref_s.size(); ++i) {
      ASSERT_NEAR(got_s[i], ref_s[i], 1e-6f);
      ASSERT_NEAR(got_m[i], ref_m[i], 1e-5f);
    }
    for (size_t i = 0; i < ref_q.size(); ++i) {
      ASSERT_NEAR(static_cast<int>(got_q[i]), static_cast<int>(ref_q[i]), 1)
          << "channels=" << channels << " idx=" << i;
    }
  }

  // 2. LayerNorm2DWithSkipAndQuantizeAvx512 fused static + dynamic quant
  // parity.
  {
    const size_t rows = 16;
    const size_t channels = 256;
    std::vector<float> x(rows * channels), skip(rows * channels),
        gamma(channels), beta(channels);
    for (auto& v : x) v = f_dist(rng);
    for (auto& v : skip) v = f_dist(rng);
    for (auto& v : gamma) v = 0.8f + 0.2f * f_dist(rng);
    for (auto& v : beta) v = f_dist(rng) * 0.2f;

    std::vector<float> x_fused = x;
    std::vector<uint8_t> static_q(rows * channels), dyn_q(rows * channels);
    std::vector<float> dyn_s(rows), dyn_m(rows);
    LayerNorm2DWithSkipAndQuantizeAvx512(
        rows, channels, x_fused.data(), 0.7f, skip.data(), gamma.data(),
        beta.data(), 1e-3f, static_q.data(), 20.0f, 128.0f, dyn_q.data(),
        dyn_s.data(), dyn_m.data());

    std::vector<uint8_t> ref_dyn_q(rows * channels);
    std::vector<float> ref_dyn_s(rows), ref_dyn_m(rows);
    QuantizeDynamicRowsAvx512(rows, channels, x_fused.data(), ref_dyn_q.data(),
                              ref_dyn_s.data(), ref_dyn_m.data());
    for (size_t i = 0; i < rows; ++i) {
      ASSERT_FLOAT_EQ(dyn_s[i], ref_dyn_s[i]);
      ASSERT_FLOAT_EQ(dyn_m[i], ref_dyn_m[i]);
    }
    for (size_t i = 0; i < ref_dyn_q.size(); ++i) {
      ASSERT_EQ(dyn_q[i], ref_dyn_q[i]) << "idx=" << i;
    }
  }
}
#endif

}  // namespace
}  // namespace blas_backend
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
