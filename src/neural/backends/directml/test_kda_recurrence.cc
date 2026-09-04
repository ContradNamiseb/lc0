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

// Runs the KDA recurrence HLSL shader on this machine's real D3D12 adapter
// against synthetic input at this network's production shape (key_dim =
// value_dim = 32, heads = 16, 8 directions -- see docs/kda_split.textproto),
// and checks it against an independent scalar CPU port of the identical
// formula. This is the correctness check the DirectML backend's status
// notes point to: it is the one piece of that backend that is actually
// tested, as opposed to the rest, which is not yet built.
//
// The CPU reference below is deliberately written straight from the math,
// not copy-pasted from the HLSL or SYCL kernels -- the point is to catch a
// transcription error in either, not to validate them against each other's
// bugs. If this test and the SYCL kernel ever disagree, trust neither until
// a third, from-spec implementation breaks the tie.

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "neural/backends/directml/inputs_outputs.h"
#include "neural/backends/directml/layers.h"
#include "neural/kda_directions.h"

namespace lczero {
namespace directml_backend {
namespace {

using Microsoft::WRL::ComPtr;

constexpr float kLogDecayFloor = -10.0f;  // KDA_LOG_DECAY_FLOOR

// Straight port of the recurrence formula (see kda_recurrence.hlsl's
// comment block for the derivation and why state/norms stay float).
// direction: 1=rank_forward, 2=rank_reverse, 3=file_forward, 4=file_reverse,
// 5..8=diagonals (not exercised here -- see the "not covered" note below).
std::vector<float> CpuReferenceKdaRecurrence(
    int N, int heads, int key_dim, int value_dim,
    const std::vector<int>& directions,  // one per head-group, length = direction_count
    int direction_count, const std::vector<float>& q,
    const std::vector<float>& k, const std::vector<float>& v,
    const std::vector<float>& raw_decay, const std::vector<float>& dt_bias,
    const std::vector<float>& a_log, const std::vector<float>& beta) {
  const int key_depth = heads * key_dim;
  const int value_depth = heads * value_dim;
  std::vector<float> mixed(static_cast<size_t>(N) * 64 * value_depth, 0.0f);

  for (int batch = 0; batch < N; ++batch) {
    for (int head = 0; head < heads; ++head) {
      const int direction_index = head / (heads / direction_count);
      const int direction = directions[direction_index];
      const float scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
      const float decay_scale = std::exp(a_log[head]);

      // One state vector PER output lane, not one shared across lanes: each
      // of the value_dim work-items in the real kernel owns a private
      // state[key_dim] and updates it from its own delta (which depends on
      // that lane's own v[local_id]), so the recurrence genuinely
      // diverges per lane, not just per (batch, head).
      std::vector<std::vector<float>> state(
          value_dim, std::vector<float>(key_dim, 0.0f));

      for (int token = 0; token < 64; ++token) {
        int square = token;
        if (direction == 2) {
          square = 63 - token;
        } else if (direction == 3) {
          square = (token % 8) * 8 + token / 8;
        } else if (direction == 4) {
          int reverse = 63 - token;
          square = (reverse % 8) * 8 + reverse / 8;
        }
        // Diagonal directions (5-8) are not exercised by this test.

        const int token_idx = batch * 64 + square;
        const int q_off = token_idx * key_depth + head * key_dim;
        const int k_off = q_off;
        const int v_off = token_idx * value_depth + head * value_dim;
        const int raw_decay_off = q_off;

        std::vector<float> pq(key_dim), pk(key_dim), pdecay(key_dim);
        for (int i = 0; i < key_dim; ++i) {
          pq[i] = q[q_off + i];
          pk[i] = k[k_off + i];
          const float decay_input =
              raw_decay[raw_decay_off + i] + dt_bias[head * key_dim + i];
          const float softplus = std::max(decay_input, 0.0f) +
                                 std::log1p(std::exp(-std::fabs(decay_input)));
          const float log_decay =
              std::max(-decay_scale * softplus, kLogDecayFloor);
          pdecay[i] = std::exp(log_decay);
        }

        float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
        for (int i = 0; i < key_dim; ++i) {
          q_norm_sq += pq[i] * pq[i];
          k_norm_sq += pk[i] * pk[i];
        }
        const float q_norm = 1.0f / std::sqrt(std::max(q_norm_sq, 1.0e-12f));
        const float k_norm = 1.0f / std::sqrt(std::max(k_norm_sq, 1.0e-12f));

        const float beta_value = beta[token_idx * heads + head];
        const float update_rate = 1.0f / (1.0f + std::exp(-beta_value));

        // Each lane owns an independent state[key_dim] (see the comment on
        // the `state` declaration above), so decay/predict/update/output all
        // happen per-lane, in that order, exactly like the real kernel's
        // per-work-item sequence -- nothing here is shared across lanes
        // except the q/k/decay staging computed once above.
        for (int lane = 0; lane < value_dim; ++lane) {
          for (int i = 0; i < key_dim; ++i) state[lane][i] *= pdecay[i];

          float prediction = 0.0f;
          for (int i = 0; i < key_dim; ++i) {
            prediction += pk[i] * k_norm * state[lane][i];
          }
          const float delta = update_rate * (v[v_off + lane] - prediction);

          for (int i = 0; i < key_dim; ++i) {
            state[lane][i] += pk[i] * k_norm * delta;
          }

          float output = 0.0f;
          for (int i = 0; i < key_dim; ++i) {
            output += pq[i] * q_norm * scale * state[lane][i];
          }
          mixed[static_cast<size_t>(token_idx) * value_depth +
               head * value_dim + lane] = output;
        }
      }
    }
  }
  return mixed;
}

// Minimal D3D12 device + one-shot command submission helper, scoped to
// this test -- not shared production code, so it stays a plain function
// rather than growing into a class other files would depend on.
struct TestDevice {
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12Fence> fence;
  uint64_t fence_value = 0;

  static bool TryCreate(TestDevice* out) {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return false;
    ComPtr<IDXGIAdapter1> adapter;
    bool found = false;
    for (UINT i = 0;
         factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
      DXGI_ADAPTER_DESC1 desc;
      adapter->GetDesc1(&desc);
      if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
      if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                      IID_PPV_ARGS(&out->device)))) {
        found = true;
        break;
      }
    }
    if (!found) return false;
    D3D12_COMMAND_QUEUE_DESC qdesc = {};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    out->device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&out->queue));
    out->device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                             IID_PPV_ARGS(&out->fence));
    return true;
  }

  void RunAndWait(ID3D12GraphicsCommandList* list) {
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence.Get(), ++fence_value);
    while (fence->GetCompletedValue() < fence_value) {
      // Spin-wait; this is a test, not the hot path.
    }
  }
};

ComPtr<ID3D12Resource> UploadBuffer(ID3D12Device* device,
                                    const std::vector<float>& data) {
  return detail::CreateBuffer(
      device, data.empty() ? 4 : data.size() * sizeof(float),
      D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
}

TEST(KdaRecurrence, MatchesCpuReferenceAtProductionShape) {
  TestDevice test_device;
  if (!TestDevice::TryCreate(&test_device)) {
    GTEST_SKIP() << "No hardware D3D12 adapter available on this machine.";
  }

  constexpr int kN = 2;
  constexpr int kHeads = 16;
  constexpr int kKeyDim = 32;
  constexpr int kValueDim = 32;
  constexpr int kDirectionCount = 8;
  const std::vector<int> directions = {1, 2, 3, 4, 1, 2, 3, 4};  // skip 5-8: diagonals untested here

  const int key_depth = kHeads * kKeyDim;
  const int value_depth = kHeads * kValueDim;
  const size_t tokens = static_cast<size_t>(kN) * 64;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  auto random_vec = [&](size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
  };

  const auto q = random_vec(tokens * key_depth);
  const auto k = random_vec(tokens * key_depth);
  const auto v = random_vec(tokens * value_depth);
  const auto raw_decay = random_vec(tokens * key_depth);
  const auto dt_bias = random_vec(key_depth);
  const auto a_log = random_vec(kHeads);
  const auto beta = random_vec(tokens * kHeads);

  const auto expected = CpuReferenceKdaRecurrence(
      kN, kHeads, kKeyDim, kValueDim, directions, kDirectionCount, q, k, v,
      raw_decay, dt_bias, a_log, beta);

  KdaRecurrenceLayer layer(test_device.device.Get(), /*fp16=*/false, kKeyDim,
                           kValueDim);

  auto upload_and_get = [&](const std::vector<float>& data) {
    auto buf = UploadBuffer(test_device.device.Get(), data);
    void* mapped = nullptr;
    buf->Map(0, nullptr, &mapped);
    std::memcpy(mapped, data.data(), data.size() * sizeof(float));
    buf->Unmap(0, nullptr);
    return buf;
  };

  auto q_buf = upload_and_get(q);
  auto k_buf = upload_and_get(k);
  auto v_buf = upload_and_get(v);
  auto raw_decay_buf = upload_and_get(raw_decay);
  auto dt_bias_buf = upload_and_get(dt_bias);
  auto a_log_buf = upload_and_get(a_log);
  auto beta_buf = upload_and_get(beta);
  // The traversal table the kernel indexes, straight from the shared header.
  std::vector<uint32_t> order(16 * 64);
  for (int d = 0; d < 16; ++d) {
    for (int t = 0; t < 64; ++t) {
      order[d * 64 + t] = static_cast<uint32_t>(kKdaDirectionOrder[d][t]);
    }
  }
  auto order_buf = detail::CreateBuffer(
      test_device.device.Get(), order.size() * sizeof(uint32_t),
      D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
  {
    void* mapped = nullptr;
    order_buf->Map(0, nullptr, &mapped);
    std::memcpy(mapped, order.data(), order.size() * sizeof(uint32_t));
    order_buf->Unmap(0, nullptr);
  }
  auto mixed_buf = detail::CreateBuffer(
      test_device.device.Get(), expected.size() * sizeof(float),
      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  auto readback_buf = detail::CreateBuffer(
      test_device.device.Get(), expected.size() * sizeof(float),
      D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> command_list;
  test_device.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             IID_PPV_ARGS(&allocator));
  test_device.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                        allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&command_list));

  KdaRecurrenceLayer::Params params{};
  params.batch_size = kN;
  params.heads = kHeads;
  params.key_dim = kKeyDim;
  params.value_dim = kValueDim;
  params.direction_count = kDirectionCount;
  for (size_t i = 0; i < directions.size() && i < 16; ++i) {
    params.directions[i] = directions[i];
  }
  params.use_fused_qkv = false;
  params.qkv_stride = 0;
  params.log_decay_floor = kLogDecayFloor;
  params.fp16 = false;

  layer.Record(command_list.Get(), params, /*qkv=*/DmlPtr(),
               DmlPtr(q_buf.Get(), 0), DmlPtr(k_buf.Get(), 0),
               DmlPtr(v_buf.Get(), 0), DmlPtr(raw_decay_buf.Get(), 0),
               DmlPtr(dt_bias_buf.Get(), 0), DmlPtr(a_log_buf.Get(), 0),
               DmlPtr(beta_buf.Get(), 0), DmlPtr(order_buf.Get(), 0),
               DmlPtr(mixed_buf.Get(), 0));

  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = mixed_buf.Get();
  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  command_list->ResourceBarrier(1, &barrier);
  command_list->CopyBufferRegion(readback_buf.Get(), 0, mixed_buf.Get(), 0,
                                 expected.size() * sizeof(float));
  command_list->Close();

  test_device.RunAndWait(command_list.Get());

  float* readback_ptr = nullptr;
  readback_buf->Map(0, nullptr, reinterpret_cast<void**>(&readback_ptr));

  int mismatches = 0;
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < expected.size(); ++i) {
    const float diff = std::fabs(readback_ptr[i] - expected[i]);
    max_abs_diff = std::max(max_abs_diff, diff);
    if (diff > 1.0e-3f) {
      if (mismatches < 5) {
        ADD_FAILURE() << "mismatch at index " << i << ": gpu=" << readback_ptr[i]
                      << " cpu=" << expected[i] << " diff=" << diff;
      }
      ++mismatches;
    }
  }
  readback_buf->Unmap(0, nullptr);

  EXPECT_EQ(mismatches, 0) << mismatches << "/" << expected.size()
                           << " elements mismatched, max abs diff = "
                           << max_abs_diff;
}

}  // namespace
}  // namespace directml_backend
}  // namespace lczero

// Not linked against gtest_main (this target only pulls in `gtest`, not
// `gtest_main`, to keep its meson dependency list matching the other
// backend-specific bits it needs) -- so it supplies its own entry point.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
