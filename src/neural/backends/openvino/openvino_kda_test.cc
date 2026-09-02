/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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
*/

// Verifies KdaScanOp::evaluate (the CPU reference implementation of the
// fused KDA recurrence) against an independent plain-C++ recurrence written
// from the EmitKdaLayer definition in onnx/converter.cc. The GPU kernel
// (kda_scan_kernel_source.h) claims bit-for-bit parity with the same math,
// so this pins the semantics both implementations must match. Covers the
// direction-table traversal too: each case picks different direction sets,
// so a wrong token->square mapping shows up as a mismatch.

#include <gtest/gtest.h>

#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

#include "neural/backends/openvino/kda_scan_op.h"
#include "neural/kda_directions.h"

namespace lczero {
namespace openvino_backend {
namespace {

struct KdaCase {
  int batch;
  int heads;
  int key_dim;
  int value_dim;
  int direction_count;
  std::vector<int> directions;
};

// Deterministic pseudo-random floats in [-1, 1] (LCG; good enough here).
class Lcg {
 public:
  explicit Lcg(uint64_t seed) : state_(seed) {}
  float Next() {
    state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
    const uint32_t bits = static_cast<uint32_t>(state_ >> 33);
    return (static_cast<float>(bits) / 4294967295.0f) * 2.0f - 1.0f;
  }

 private:
  uint64_t state_;
};

inline float Softplus(float x) {
  return (x > 0.0f ? x : 0.0f) + std::log1p(std::exp(-std::fabs(x)));
}
inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Reference recurrence, structured as the mathematical definition (state
// matrix decayed row-wise, prediction, gated delta update, query readout)
// rather than as a copy of kda_scan_op.cc's loop nest. Inputs/outputs are
// square-major [N, 64, H, *]; the scan walks tokens in the direction's
// traversal order via KdaSquareForToken, exactly as converter.cc's Scan
// body does over its pre-permuted inputs.
std::vector<float> ReferenceKdaScan(const KdaCase& c,
                                    const std::vector<float>& q,
                                    const std::vector<float>& k,
                                    const std::vector<float>& v,
                                    const std::vector<float>& raw_decay,
                                    const std::vector<float>& beta,
                                    const std::vector<float>& dt_bias,
                                    const std::vector<float>& neg_decay_scale) {
  const int N = c.batch, H = c.heads, K = c.key_dim, V = c.value_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(K));
  const int heads_per_direction = H / c.direction_count;
  std::vector<float> mixed(static_cast<size_t>(N) * 64 * H * V, 0.0f);

  auto qk_idx = [&](int n, int sq, int h, int key) {
    return (static_cast<size_t>(n) * 64 + sq) * H * K +
           static_cast<size_t>(h) * K + key;
  };
  auto v_idx = [&](int n, int sq, int h, int val) {
    return (static_cast<size_t>(n) * 64 + sq) * H * V +
           static_cast<size_t>(h) * V + val;
  };

  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < H; ++h) {
      const int dir = c.directions[h / heads_per_direction];
      const float neg_scale = neg_decay_scale[h];

      std::vector<std::vector<float>> state(K, std::vector<float>(V, 0.0f));

      for (int t = 0; t < 64; ++t) {
        const int sq = KdaSquareForToken(dir, t);

        float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
        for (int key = 0; key < K; ++key) {
          q_norm_sq += q[qk_idx(n, sq, h, key)] * q[qk_idx(n, sq, h, key)];
          k_norm_sq += k[qk_idx(n, sq, h, key)] * k[qk_idx(n, sq, h, key)];
        }
        const float q_norm =
            1.0f / std::sqrt(q_norm_sq > 1e-12f ? q_norm_sq : 1e-12f);
        const float k_norm =
            1.0f / std::sqrt(k_norm_sq > 1e-12f ? k_norm_sq : 1e-12f);

        std::vector<float> decay(K);
        for (int key = 0; key < K; ++key) {
          const float x =
              raw_decay[qk_idx(n, sq, h, key)] + dt_bias[h * K + key];
          float log_decay = neg_scale * Softplus(x);
          if (log_decay < KdaScanOp::kLogDecayFloor) {
            log_decay = KdaScanOp::kLogDecayFloor;
          }
          decay[key] = std::exp(log_decay);
        }
        for (int key = 0; key < K; ++key) {
          for (int val = 0; val < V; ++val) state[key][val] *= decay[key];
        }

        const float update_rate =
            Sigmoid(beta[(static_cast<size_t>(n) * 64 + sq) * H + h]);

        std::vector<float> prediction(V, 0.0f);
        for (int val = 0; val < V; ++val) {
          for (int key = 0; key < K; ++key) {
            prediction[val] +=
                k[qk_idx(n, sq, h, key)] * k_norm * state[key][val];
          }
        }

        std::vector<float> delta(V);
        for (int val = 0; val < V; ++val) {
          delta[val] =
              update_rate * (v[v_idx(n, sq, h, val)] - prediction[val]);
        }
        for (int key = 0; key < K; ++key) {
          const float k_normed = k[qk_idx(n, sq, h, key)] * k_norm;
          for (int val = 0; val < V; ++val) {
            state[key][val] += k_normed * delta[val];
          }
        }

        for (int val = 0; val < V; ++val) {
          float out = 0.0f;
          for (int key = 0; key < K; ++key) {
            out += q[qk_idx(n, sq, h, key)] * q_norm * scale *
                   state[key][val];
          }
          mixed[v_idx(n, sq, h, val)] = out;
        }
      }
    }
  }
  return mixed;
}

void RunCase(const KdaCase& c) {
  const int N = c.batch, H = c.heads, K = c.key_dim, V = c.value_dim;
  ASSERT_EQ(H % c.direction_count, 0);
  ASSERT_EQ(static_cast<int>(c.directions.size()), c.direction_count);

  Lcg rng(0x5eedu * (N + 31 * H + 977 * K + 13 * V + 7 * c.direction_count));
  auto fill = [&](size_t count) {
    std::vector<float> data(count);
    for (auto& x : data) x = rng.Next();
    return data;
  };

  const std::vector<float> q = fill(static_cast<size_t>(N) * 64 * H * K);
  const std::vector<float> k = fill(static_cast<size_t>(N) * 64 * H * K);
  const std::vector<float> v = fill(static_cast<size_t>(N) * 64 * H * V);
  const std::vector<float> raw_decay = fill(static_cast<size_t>(N) * 64 * H * K);
  const std::vector<float> beta = fill(static_cast<size_t>(N) * 64 * H);
  const std::vector<float> dt_bias = fill(static_cast<size_t>(H) * K);
  const std::vector<float> neg_decay_scale = [&] {
    std::vector<float> data(H);
    // Negative scale factor (= -exp(a_log)); keep magnitudes moderate so
    // the recurrence stays in a numerically interesting but stable range.
    for (auto& x : data) x = -(0.5f + 0.5f * std::fabs(rng.Next()));
    return data;
  }();

  // Build the op over Parameters with the case's static shapes.
  auto param = [](const ov::Shape& shape) {
    return std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
  };
  const size_t n = static_cast<size_t>(N), h = static_cast<size_t>(H),
             kd = static_cast<size_t>(K), vd = static_cast<size_t>(V);
  auto dt_bias_c = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, h, kd}, dt_bias);
  auto nds_c = std::make_shared<ov::op::v0::Constant>(
      ov::element::f32, ov::Shape{1, h, 1}, neg_decay_scale);
  KdaScanOp op(param({n, 64, h, kd}), param({n, 64, h, kd}),
               param({n, 64, h, vd}), param({n, 64, h, kd}),
               param({n, 64, h, 1}), dt_bias_c, nds_c, c.direction_count,
               c.directions);
  ASSERT_TRUE(op.has_evaluate());
  EXPECT_EQ(op.heads(), H);
  EXPECT_EQ(op.key_dim(), K);
  EXPECT_EQ(op.value_dim(), V);

  auto tensor = [](const ov::Shape& shape, const std::vector<float>& data) {
    return ov::Tensor(ov::element::f32, shape,
                      const_cast<float*>(data.data()));
  };
  ov::TensorVector inputs = {
      tensor({n, 64, h, kd}, q),      tensor({n, 64, h, kd}, k),
      tensor({n, 64, h, vd}, v),      tensor({n, 64, h, kd}, raw_decay),
      tensor({n, 64, h, 1}, beta),    tensor({1, h, kd}, dt_bias),
      tensor({1, h, 1}, neg_decay_scale)};
  ov::TensorVector outputs = {ov::Tensor(ov::element::f32, {n, 64, h, vd})};
  ASSERT_TRUE(op.evaluate(outputs, inputs));

  const std::vector<float> expected =
      ReferenceKdaScan(c, q, k, v, raw_decay, beta, dt_bias, neg_decay_scale);
  const float* actual = outputs[0].data<float>();
  ASSERT_EQ(outputs[0].get_size(), expected.size());

  double max_abs_err = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double err = std::fabs(static_cast<double>(expected[i]) - actual[i]);
    if (err > max_abs_err) max_abs_err = err;
  }
  // fp32 with mildly different accumulation orders between the two
  // implementations; anything above this is a real semantic divergence.
  EXPECT_LT(max_abs_err, 1e-4);
}

TEST(OpenVinoKdaScanTest, MatchesReferenceStandardDirections) {
  RunCase({/*batch=*/2, /*heads=*/8, /*key_dim=*/4, /*value_dim=*/6,
           /*direction_count=*/8, {1, 2, 3, 4, 5, 6, 7, 8}});
}

TEST(OpenVinoKdaScanTest, MatchesReferenceSerpentineSingleDirection) {
  RunCase({/*batch=*/3, /*heads=*/4, /*key_dim=*/4, /*value_dim=*/8,
           /*direction_count=*/1, {9}});
}

TEST(OpenVinoKdaScanTest, MatchesReferenceMixedDirections) {
  RunCase({/*batch=*/1, /*heads=*/16, /*key_dim=*/5, /*value_dim=*/3,
           /*direction_count=*/2, {2, 15}});
}

}  // namespace
}  // namespace openvino_backend
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
