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
      {1, 1, 1},     {3, 17, 5},     {7, 31, 13},    {8, 32, 64},
      {13, 65, 129}, {64, 256, 256}, {71, 384, 256},
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

          Int8Weights conv1(w1, s1, in1_scale, in1_zp, kernel, true);
          Int8Weights conv2(w2, s2, in2_scale, in2_zp, kernel, true);
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

}  // namespace
}  // namespace blas_backend
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
