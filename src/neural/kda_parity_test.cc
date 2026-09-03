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

// KDA parity test: runs one synthetic KDA-hybrid network through two
// independent implementations (the BLAS/CPU reference and the SYCL kernels)
// and asserts the outputs agree.
//
// This is the test the review history kept asking for: every critical KDA
// bug found so far (gate/norm ordering, fused decay/gate layout, the
// gate-gemm-vs-beta-gemm buffer overwrite) was a divergence between
// implementations, found by inspection only. A whole-network parity check
// catches that entire class, including in the parts of the dispatch that
// pure kernel-level tests miss.
//
// The synthetic net is tiny (embedding 32, 8 heads, one KDA encoder layer
// with key/value dim 4 and gate rank 4, attention policy) but exercises the
// full KDA-hybrid path: input embedding + positional encoding, the KDA
// recurrence with qkv_silu, output RMS norm, output gate, FFN + LN2, the
// attention policy head, and the WDL value head.
//
// The SYCL half runs on whatever device the SYCL runtime exposes (an Intel
// CPU device is enough) and is skipped when no SYCL backend is compiled in
// or no usable device is found.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "chess/board.h"
#include "chess/position.h"
#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "proto/net.pb.h"
#include "utils/optionsdict.h"

namespace lczero {
namespace {

// Fills a Layer message with raw float data (LayerAdapter reads it back).
void FillLayer(pblczero::Weights::Layer* layer,
               const std::vector<float>& values) {
  layer->set_encoding(pblczero::Weights::Layer::FLOAT32);
  layer->set_params(std::string(reinterpret_cast<const char*>(values.data()),
                                values.size() * sizeof(float)));
}

struct NetDims {
  int embedding = 32;   // ip_emb_b
  int heads = 8;        // encoder headcount
  int key_dim = 4;      // per-head KDA key rank
  int value_dim = 4;    // per-head KDA value rank
  int gate_rank = 4;    // KDA decay/gate rank
  int dff = 64;         // encoder FFN width
  int pol_emb = 32;     // policy head embedding (ip_pol_b)
  int pol_dmodel = 32;  // policy attention d_model
  int val_planes = 32;  // value head embedding (ip_val_b)
  int val_channels = 32;  // ip1_val_b
};

std::vector<float> RandomVec(std::mt19937& rng, size_t n, float scale) {
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

// Builds a tiny but complete KDA-hybrid net. Shapes follow
// network_blas.cc's attention-body path (and the mirrored SYCL one):
// input embedding over 112 planes + 64 positional-encoding channels, one
// KDA encoder layer, attention policy head, WDL value head, no moves-left.
pblczero::Net MakeKdaHybridNet() {
  const NetDims d;
  const int key_depth = d.heads * d.key_dim;      // 32
  const int value_depth = d.heads * d.value_dim;  // 32
  const int input_size = kInputPlanes + 64;       // planes + pos encoding

  std::mt19937 rng(42);
  pblczero::Net file;
  auto* weights = file.mutable_weights();
  auto* nf = file.mutable_format()->mutable_network_format();
  using NF = pblczero::NetworkFormat;
  nf->set_input(NF::INPUT_CLASSICAL_112_PLANE);
  nf->set_network(NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT);
  nf->set_policy(NF::POLICY_ATTENTION);
  nf->set_value(NF::VALUE_WDL);
  nf->set_moves_left(NF::MOVES_LEFT_NONE);
  nf->set_input_embedding(NF::INPUT_EMBEDDING_NONE);
  nf->set_default_activation(NF::DEFAULT_ACTIVATION_RELU);
  nf->set_ffn_activation(NF::ACTIVATION_DEFAULT);
  nf->set_smolgen_activation(NF::ACTIVATION_DEFAULT);
  // Serpentine directions 1..4; heads % count == 0.
  for (int dir : {1, 2, 3, 4}) {
    nf->add_kda_directions(static_cast<NF::KdaDirection>(dir));
  }

  // Input embedding.
  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);

  // One KDA encoder layer.
  auto* enc = weights->add_encoder();
  enc->set_mixer(pblczero::Weights::EncoderLayer::MIXER_KDA);
  auto* kda = enc->mutable_kda();
  FillLayer(kda->mutable_q_w(),
            RandomVec(rng, d.embedding * key_depth, 0.1f));
  FillLayer(kda->mutable_q_b(), RandomVec(rng, key_depth, 0.05f));
  FillLayer(kda->mutable_k_w(),
            RandomVec(rng, d.embedding * key_depth, 0.1f));
  FillLayer(kda->mutable_k_b(), RandomVec(rng, key_depth, 0.05f));
  FillLayer(kda->mutable_v_w(),
            RandomVec(rng, d.embedding * value_depth, 0.1f));
  FillLayer(kda->mutable_v_b(), RandomVec(rng, value_depth, 0.05f));
  FillLayer(kda->mutable_decay_a_w(),
            RandomVec(rng, d.embedding * d.gate_rank, 0.1f));
  FillLayer(kda->mutable_decay_a_b(), RandomVec(rng, d.gate_rank, 0.05f));
  FillLayer(kda->mutable_decay_b_w(),
            RandomVec(rng, d.gate_rank * key_depth, 0.1f));
  FillLayer(kda->mutable_decay_b_b(), RandomVec(rng, key_depth, 0.05f));
  FillLayer(kda->mutable_beta_w(), RandomVec(rng, d.embedding * d.heads, 0.1f));
  FillLayer(kda->mutable_beta_b(), RandomVec(rng, d.heads, 0.05f));
  FillLayer(kda->mutable_a_log(), RandomVec(rng, d.heads, 0.05f));
  FillLayer(kda->mutable_dt_bias(), RandomVec(rng, key_depth, 0.05f));
  FillLayer(kda->mutable_gate_a_w(),
            RandomVec(rng, d.embedding * d.gate_rank, 0.1f));
  FillLayer(kda->mutable_gate_a_b(), RandomVec(rng, d.gate_rank, 0.05f));
  FillLayer(kda->mutable_gate_b_w(),
            RandomVec(rng, d.gate_rank * value_depth, 0.1f));
  FillLayer(kda->mutable_gate_b_b(), RandomVec(rng, value_depth, 0.05f));
  FillLayer(kda->mutable_out_norm_gammas(), RandomVec(rng, value_depth, 0.1f));
  FillLayer(kda->mutable_dense_w(),
            RandomVec(rng, value_depth * d.embedding, 0.1f));
  FillLayer(kda->mutable_dense_b(), RandomVec(rng, d.embedding, 0.05f));
  kda->set_key_dim(d.key_dim);
  kda->set_value_dim(d.value_dim);
  kda->set_gate_rank(d.gate_rank);
  kda->set_rms_norm_epsilon(1e-6f);
  kda->set_output_gate(true);
  kda->set_output_rms_norm(true);
  kda->set_local_conv(false);
  kda->set_qkv_silu(true);
  FillLayer(enc->mutable_ln1_gammas(), RandomVec(rng, d.embedding, 0.1f));
  FillLayer(enc->mutable_ln1_betas(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(enc->mutable_ffn()->mutable_dense1_w(),
            RandomVec(rng, d.embedding * d.dff, 0.1f));
  FillLayer(enc->mutable_ffn()->mutable_dense1_b(), RandomVec(rng, d.dff, 0.05f));
  FillLayer(enc->mutable_ffn()->mutable_dense2_w(),
            RandomVec(rng, d.dff * d.embedding, 0.1f));
  FillLayer(enc->mutable_ffn()->mutable_dense2_b(),
            RandomVec(rng, d.embedding, 0.05f));
  FillLayer(enc->mutable_ln2_gammas(), RandomVec(rng, d.embedding, 0.1f));
  FillLayer(enc->mutable_ln2_betas(), RandomVec(rng, d.embedding, 0.05f));

  // Policy head (attention policy, no policy encoders, no smolgen).
  auto* ph = weights->mutable_policy_heads();
  FillLayer(ph->mutable_ip_pol_w(), RandomVec(rng, d.pol_emb * d.embedding, 0.1f));
  FillLayer(ph->mutable_ip_pol_b(), RandomVec(rng, d.pol_emb, 0.05f));
  auto* vanilla = ph->mutable_vanilla();
  FillLayer(vanilla->mutable_ip2_pol_w(),
            RandomVec(rng, d.pol_dmodel * d.pol_emb, 0.1f));
  FillLayer(vanilla->mutable_ip2_pol_b(), RandomVec(rng, d.pol_dmodel, 0.05f));
  FillLayer(vanilla->mutable_ip3_pol_w(),
            RandomVec(rng, d.pol_dmodel * d.pol_emb, 0.1f));
  FillLayer(vanilla->mutable_ip3_pol_b(), RandomVec(rng, d.pol_dmodel, 0.05f));
  FillLayer(vanilla->mutable_ip4_pol_w(), RandomVec(rng, 4 * d.pol_dmodel, 0.1f));

  // Value head (WDL).
  auto* winner = weights->mutable_value_heads()->mutable_winner();
  FillLayer(winner->mutable_ip_val_w(),
            RandomVec(rng, d.val_planes * d.embedding, 0.1f));
  FillLayer(winner->mutable_ip_val_b(), RandomVec(rng, d.val_planes, 0.05f));
  FillLayer(winner->mutable_ip1_val_w(),
            RandomVec(rng, d.val_channels * d.val_planes * 64, 0.05f));
  FillLayer(winner->mutable_ip1_val_b(), RandomVec(rng, d.val_channels, 0.05f));
  FillLayer(winner->mutable_ip2_val_w(), RandomVec(rng, 3 * d.val_channels, 0.05f));
  FillLayer(winner->mutable_ip2_val_b(), RandomVec(rng, 3, 0.05f));

  return file;
}

InputPlanes EncodeStartPos() {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);
  return EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE, history, 8,
      FillEmptyHistory::NO, nullptr);
}

struct Outputs {
  float q = 0.0f;
  float d = 0.0f;
  std::vector<float> policy;
};

Outputs RunNetwork(const std::string& backend, const WeightsFile& weights,
                   const InputPlanes& planes) {
  // Backend options are read straight off the dict the factory receives
  // (same as NetworkFactory::LoadNetwork after AddSubdictFromString), so
  // nothing to set here -- every option keeps its default.
  OptionsDict options;
  auto network = NetworkFactory::Get()->Create(backend, weights, options);
  auto computation = network->NewComputation();
  computation->AddInput(InputPlanes(planes));
  computation->ComputeBlocking();
  Outputs out;
  out.q = computation->GetQVal(0);
  out.d = computation->GetDVal(0);
  out.policy.reserve(1858);
  for (int i = 0; i < 1858; ++i) out.policy.push_back(computation->GetPVal(0, i));
  return out;
}

bool HasBackend(const std::string& name) {
  const auto& backends = NetworkFactory::Get()->GetBackendsList();
  return std::find(backends.begin(), backends.end(), name) != backends.end();
}

TEST(KdaParity, SyclMatchesBlasOnKdaHybridNet) {
  pblczero::Net net = MakeKdaHybridNet();
  const InputPlanes planes = EncodeStartPos();

  // BLAS is the CPU reference; it must be present in any build this test
  // links (it is on by default).
  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  const Outputs reference = RunNetwork("blas", net, planes);

  if (!HasBackend("sycl")) {
    GTEST_SKIP() << "sycl backend not compiled in; nothing to compare";
  }

  Outputs sycl;
  try {
    sycl = RunNetwork("sycl", net, planes);
  } catch (const Exception& e) {
    GTEST_SKIP() << "no usable SYCL device: " << e.what();
  }

  // The two implementations accumulate in different orders (and the SYCL
  // kernels vectorize), so compare with a tolerance rather than exactly.
  // The recurrence itself is sequential-float in both, so the dominant
  // error source is the surrounding GEMMs; 2e-4 leaves two orders of
  // magnitude of headroom over what an actual math divergence produces
  // (a wrong traversal order or dropped gate produces errors ~1e-1+).
  constexpr float kTol = 2e-4f;
  EXPECT_NEAR(sycl.q, reference.q, kTol)
      << "WDL value Q diverges between sycl and blas";
  EXPECT_NEAR(sycl.d, reference.d, kTol)
      << "WDL draw probability diverges between sycl and blas";

  float worst = 0.0f;
  int worst_move = -1;
  for (int i = 0; i < 1858; ++i) {
    const float diff = std::fabs(sycl.policy[i] - reference.policy[i]);
    if (diff > worst) {
      worst = diff;
      worst_move = i;
    }
  }
  EXPECT_LT(worst, kTol)
      << "policy diverges between sycl and blas (worst move " << worst_move
      << ", diff " << worst << ")";
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
