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

// DirectML parity test: runs synthetic KDA-hybrid networks through the
// BLAS/CPU reference and the DirectML backend and asserts the outputs
// agree -- the same harness and tolerance the SYCL kda_parity_test uses
// (src/neural/kda_parity_test.cc), with three nets:
//
//  1. the identical net the SYCL parity test builds (one KDA encoder with
//     qkv_silu, attention policy, WDL, no moves-left), so all three
//     backends are compared against the same reference;
//  2. a wider net adding an MHA encoder and the moves-left head, covering
//     the multi-head attention path and the head the first net omits;
//  3. a no-encoder net, bisecting embedding/head divergence from the
//     encoder stacks.
//
// Skipped when no DirectML backend is compiled in or no hardware D3D12
// adapter exists (the same skip-with-reason convention as the SYCL test).

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "chess/board.h"
#include "chess/position.h"
#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "utils/weights_adapter.h"
#include "neural/network.h"
#include "proto/net.pb.h"
#include "utils/optionsdict.h"

namespace lczero {
namespace {

// Every synthetic net here stores weights as FLOAT32 while every real
// trained net is LINEAR16 (min/max plus uint16 thetas, decoded by
// LayerAdapter as min * (1 - theta) + max * theta). Set LC0_TEST_LINEAR16=1
// to quantise instead, so the same nets can be run through the encoding real
// nets actually use.
void FillLayer(pblczero::Weights::Layer* layer,
               const std::vector<float>& values) {
  if (getenv("LC0_TEST_LINEAR16") && !values.empty()) {
    float mn = values[0], mx = values[0];
    for (float v : values) {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    if (mx == mn) mx = mn + 1.0f;  // avoid a zero range
    layer->set_min_val(mn);
    layer->set_max_val(mx);
    layer->set_encoding(pblczero::Weights::Layer::LINEAR16);
    std::string params(values.size() * sizeof(uint16_t), '\0');
    uint16_t* q = reinterpret_cast<uint16_t*>(&params[0]);
    for (size_t i = 0; i < values.size(); ++i) {
      const float theta = (values[i] - mn) / (mx - mn);
      q[i] = static_cast<uint16_t>(std::lround(theta * 65535.0f));
    }
    layer->set_params(params);
    return;
  }
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
  int mlh_hidden = 8;     // ip1_mov_b
};

std::vector<float> RandomVec(std::mt19937& rng, size_t n, float scale) {
  // Synthetic weights are tiny (0.05-0.1) next to trained ones, and the KDA
  // decay and the softmax both exponentiate, so magnitude is its own axis.
  if (const char* mul = getenv("LC0_TEST_WEIGHT_SCALE")) {
    scale *= static_cast<float>(atof(mul));
  }
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

void FillKdaEncoder(pblczero::Net* net, std::mt19937& rng, const NetDims& d) {
  const int key_depth = d.heads * d.key_dim;
  const int value_depth = d.heads * d.value_dim;
  auto* enc = net->mutable_weights()->add_encoder();
  enc->set_mixer(pblczero::Weights::EncoderLayer::MIXER_KDA);
  auto* kda = enc->mutable_kda();
  FillLayer(kda->mutable_q_w(), RandomVec(rng, d.embedding * key_depth, 0.1f));
  FillLayer(kda->mutable_q_b(), RandomVec(rng, key_depth, 0.05f));
  FillLayer(kda->mutable_k_w(), RandomVec(rng, d.embedding * key_depth, 0.1f));
  FillLayer(kda->mutable_k_b(), RandomVec(rng, key_depth, 0.05f));
  FillLayer(kda->mutable_v_w(), RandomVec(rng, d.embedding * value_depth, 0.1f));
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
  FillLayer(kda->mutable_dense_w(), RandomVec(rng, value_depth * d.embedding, 0.1f));
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
  FillLayer(enc->mutable_ffn()->mutable_dense2_b(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(enc->mutable_ln2_gammas(), RandomVec(rng, d.embedding, 0.1f));
  FillLayer(enc->mutable_ln2_betas(), RandomVec(rng, d.embedding, 0.05f));
}

void FillMhaEncoder(pblczero::Net* net, std::mt19937& rng, const NetDims& d) {
  auto* enc = net->mutable_weights()->add_encoder();
  enc->set_mixer(pblczero::Weights::EncoderLayer::MIXER_MHA);
  auto* mha = enc->mutable_mha();
  FillLayer(mha->mutable_q_w(), RandomVec(rng, d.embedding * d.embedding, 0.1f));
  FillLayer(mha->mutable_q_b(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(mha->mutable_k_w(), RandomVec(rng, d.embedding * d.embedding, 0.1f));
  FillLayer(mha->mutable_k_b(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(mha->mutable_v_w(), RandomVec(rng, d.embedding * d.embedding, 0.1f));
  FillLayer(mha->mutable_v_b(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(mha->mutable_dense_w(),
            RandomVec(rng, d.embedding * d.embedding, 0.1f));
  FillLayer(mha->mutable_dense_b(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(enc->mutable_ln1_gammas(), RandomVec(rng, d.embedding, 0.1f));
  FillLayer(enc->mutable_ln1_betas(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(enc->mutable_ffn()->mutable_dense1_w(),
            RandomVec(rng, d.embedding * d.dff, 0.1f));
  FillLayer(enc->mutable_ffn()->mutable_dense1_b(), RandomVec(rng, d.dff, 0.05f));
  FillLayer(enc->mutable_ffn()->mutable_dense2_w(),
            RandomVec(rng, d.dff * d.embedding, 0.1f));
  FillLayer(enc->mutable_ffn()->mutable_dense2_b(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(enc->mutable_ln2_gammas(), RandomVec(rng, d.embedding, 0.1f));
  FillLayer(enc->mutable_ln2_betas(), RandomVec(rng, d.embedding, 0.05f));
}

void FillPolicyAndValueHeads(pblczero::Net* net, std::mt19937& rng,
                             const NetDims& d, bool moves_left, int mlh) {
  auto* weights = net->mutable_weights();
  auto* ph = weights->mutable_policy_heads();
  FillLayer(ph->mutable_ip_pol_w(),
            RandomVec(rng, d.pol_emb * d.embedding, 0.1f));
  FillLayer(ph->mutable_ip_pol_b(), RandomVec(rng, d.pol_emb, 0.05f));
  auto* vanilla = ph->mutable_vanilla();
  FillLayer(vanilla->mutable_ip2_pol_w(),
            RandomVec(rng, d.pol_dmodel * d.pol_emb, 0.1f));
  FillLayer(vanilla->mutable_ip2_pol_b(),
            RandomVec(rng, d.pol_dmodel, 0.05f));
  FillLayer(vanilla->mutable_ip3_pol_w(),
            RandomVec(rng, d.pol_dmodel * d.pol_emb, 0.1f));
  FillLayer(vanilla->mutable_ip3_pol_b(),
            RandomVec(rng, d.pol_dmodel, 0.05f));
  FillLayer(vanilla->mutable_ip4_pol_w(),
            RandomVec(rng, 4 * d.pol_dmodel, 0.1f));

  auto* winner = weights->mutable_value_heads()->mutable_winner();
  FillLayer(winner->mutable_ip_val_w(),
            RandomVec(rng, d.val_planes * d.embedding, 0.1f));
  FillLayer(winner->mutable_ip_val_b(), RandomVec(rng, d.val_planes, 0.05f));
  FillLayer(winner->mutable_ip1_val_w(),
            RandomVec(rng, d.val_channels * d.val_planes * 64, 0.05f));
  FillLayer(winner->mutable_ip1_val_b(),
            RandomVec(rng, d.val_channels, 0.05f));
  FillLayer(winner->mutable_ip2_val_w(),
            RandomVec(rng, 3 * d.val_channels, 0.05f));
  FillLayer(winner->mutable_ip2_val_b(), RandomVec(rng, 3, 0.05f));

  if (moves_left) {
    FillLayer(weights->mutable_ip_mov_w(),
              RandomVec(rng, mlh * d.embedding, 0.1f));
    FillLayer(weights->mutable_ip_mov_b(), RandomVec(rng, mlh, 0.05f));
    FillLayer(weights->mutable_ip1_mov_w(),
              RandomVec(rng, d.mlh_hidden * mlh * 64, 0.05f));
    FillLayer(weights->mutable_ip1_mov_b(),
              RandomVec(rng, d.mlh_hidden, 0.05f));
    FillLayer(weights->mutable_ip2_mov_w(),
              RandomVec(rng, d.mlh_hidden, 0.05f));
    FillLayer(weights->mutable_ip2_mov_b(), RandomVec(rng, 1, 0.05f));
  }
}

// Net 1: byte-for-byte the SYCL parity test's MakeKdaHybridNet() so the
// DirectML numbers land on exactly the same reference.
pblczero::Net MakeKdaHybridNet() {
  const NetDims d;
  const int input_size = kInputPlanes + 64;
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
  for (int dir : {1, 2, 3, 4}) {
    nf->add_kda_directions(static_cast<NF::KdaDirection>(dir));
  }

  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);

  FillKdaEncoder(&file, rng, d);
  FillPolicyAndValueHeads(&file, rng, d, false, 0);
  return file;
}

// Net 2: one KDA encoder + one MHA encoder + moves-left head.
pblczero::Net MakeKdaMhaNet() {
  const NetDims d;
  const int input_size = kInputPlanes + 64;
  const int mlh = 4;
  std::mt19937 rng(1234);
  pblczero::Net file;
  auto* weights = file.mutable_weights();
  auto* nf = file.mutable_format()->mutable_network_format();
  using NF = pblczero::NetworkFormat;
  nf->set_input(NF::INPUT_CLASSICAL_112_PLANE);
  nf->set_network(NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT);
  nf->set_policy(NF::POLICY_ATTENTION);
  nf->set_value(NF::VALUE_WDL);
  nf->set_moves_left(NF::MOVES_LEFT_V1);
  nf->set_input_embedding(NF::INPUT_EMBEDDING_NONE);
  nf->set_default_activation(NF::DEFAULT_ACTIVATION_RELU);
  nf->set_ffn_activation(NF::ACTIVATION_DEFAULT);
  nf->set_smolgen_activation(NF::ACTIVATION_DEFAULT);
  for (int dir : {1, 2}) {
    nf->add_kda_directions(static_cast<NF::KdaDirection>(dir));
  }

  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);

  FillKdaEncoder(&file, rng, d);
  FillMhaEncoder(&file, rng, d);
  FillPolicyAndValueHeads(&file, rng, d, true, mlh);
  return file;
}

// Net 3: no encoders at all -- bisects divergence between the
// embedding/heads and the encoder stacks.
pblczero::Net MakeNoEncoderNet() {
  const NetDims d;
  const int input_size = kInputPlanes + 64;
  std::mt19937 rng(7);
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
  nf->add_kda_directions(static_cast<NF::KdaDirection>(1));

  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);
  FillPolicyAndValueHeads(&file, rng, d, false, 0);
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
  float m = 0.0f;
  std::vector<float> policy;
};

Outputs RunNetwork(const std::string& backend, const WeightsFile& weights,
                   const InputPlanes& planes) {
  OptionsDict options;
  auto network = NetworkFactory::Get()->Create(backend, weights, options);
  auto computation = network->NewComputation();
  computation->AddInput(InputPlanes(planes));
  computation->ComputeBlocking();
  Outputs out;
  out.q = computation->GetQVal(0);
  out.d = computation->GetDVal(0);
  out.m = computation->GetMVal(0);
  out.policy.reserve(1858);
  for (int i = 0; i < 1858; ++i) {
    out.policy.push_back(computation->GetPVal(0, i));
  }
  return out;
}

bool HasBackend(const std::string& name) {
  const auto& backends = NetworkFactory::Get()->GetBackendsList();
  return std::find(backends.begin(), backends.end(), name) != backends.end();
}

// Same tolerance rationale as the SYCL parity test: the recurrence is
// sequential-float in both backends, the surrounding GEMMs accumulate in
// different orders, and an actual math divergence (wrong traversal, dropped
// gate, wrong softmax axis) produces errors ~1e-1+, two orders of magnitude
// above this bar.
constexpr float kTol = 2e-4f;

void CompareBackends(const pblczero::Net& net) {
  const InputPlanes planes = EncodeStartPos();

  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  const Outputs reference = RunNetwork("blas", net, planes);

  if (!HasBackend("directml")) {
    GTEST_SKIP() << "directml backend not compiled in; nothing to compare";
  }
  Outputs dml;
  try {
    dml = RunNetwork("directml", net, planes);
  } catch (const Exception& e) {
    GTEST_SKIP() << "no usable DirectML device: " << e.what();
  }

  EXPECT_NEAR(dml.q, reference.q, kTol)
      << "WDL value Q diverges between directml and blas";
  EXPECT_NEAR(dml.d, reference.d, kTol)
      << "WDL draw probability diverges between directml and blas";
  EXPECT_NEAR(dml.m, reference.m, kTol)
      << "moves-left diverges between directml and blas";

  if (getenv("LC0_DIAG_OUTPUTS")) {
    float dml_absmax = 0.0f, ref_absmax = 0.0f;
    for (int i = 0; i < 1858; ++i) {
      dml_absmax = std::max(dml_absmax, std::fabs(dml.policy[i]));
      ref_absmax = std::max(ref_absmax, std::fabs(reference.policy[i]));
    }
    CERR << "[diag] dml q=" << dml.q << " d=" << dml.d << " m=" << dml.m
         << " policy|max|=" << dml_absmax;
    CERR << "[diag] ref q=" << reference.q << " d=" << reference.d
         << " m=" << reference.m << " policy|max|=" << ref_absmax;
  }
  float worst = 0.0f;
  int worst_move = -1;
  for (int i = 0; i < 1858; ++i) {
    const float diff = std::fabs(dml.policy[i] - reference.policy[i]);
    if (diff > worst) {
      worst = diff;
      worst_move = i;
    }
  }
  EXPECT_LT(worst, kTol)
      << "policy diverges between directml and blas (worst move "
      << worst_move << ", diff " << worst << ")";
}

// KDA + moves-left head, no MHA encoder: covers the MLH path
// against the BLAS reference.
pblczero::Net MakeKdaMlhNet() {
  const NetDims d;
  const int input_size = kInputPlanes + 64;
  const int mlh = 4;
  std::mt19937 rng(55);
  pblczero::Net file;
  auto* weights = file.mutable_weights();
  auto* nf = file.mutable_format()->mutable_network_format();
  using NF = pblczero::NetworkFormat;
  nf->set_input(NF::INPUT_CLASSICAL_112_PLANE);
  nf->set_network(NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT);
  nf->set_policy(NF::POLICY_ATTENTION);
  nf->set_value(NF::VALUE_WDL);
  nf->set_moves_left(NF::MOVES_LEFT_V1);
  nf->set_input_embedding(NF::INPUT_EMBEDDING_NONE);
  nf->set_default_activation(NF::DEFAULT_ACTIVATION_RELU);
  nf->set_ffn_activation(NF::ACTIVATION_DEFAULT);
  nf->set_smolgen_activation(NF::ACTIVATION_DEFAULT);
  for (int dir : {1, 2, 3, 4}) nf->add_kda_directions(static_cast<NF::KdaDirection>(dir));
  FillLayer(weights->mutable_ip_emb_w(), RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);
  FillKdaEncoder(&file, rng, d);
  FillPolicyAndValueHeads(&file, rng, d, true, mlh);
  return file;
}

TEST(DirectMlKdaParity, MatchesBlasOnKdaMlhNet) { CompareBackends(MakeKdaMlhNet()); }

// Smolgen dimensions, named as the BLAS reference names them
// (network_blas.cc EncodeLayer): compress is [hidden_channels, embedding],
// dense1 is [hidden_sz, 64 * hidden_channels], dense2 is
// [gen_outputs, hidden_sz], and the shared global table is
// [64 * 64, gen_outputs / heads].
struct SmolgenDims {
  int hidden_channels = 8;
  int hidden_sz = 16;
  int gen_outputs = 64;  // heads * per-head generated width
};

// Adds smolgen to the net's last encoder plus the shared global table.
//
// No parity net had smolgen before, which is why two bugs lived in that path
// unnoticed: the attention graph added an unbound bias tensor whenever a
// block had NO smolgen, and the generated-bias matmul ran as a hand-written
// kernel that was never checked against BLAS.
void FillSmolgen(pblczero::Net* net, std::mt19937& rng, const NetDims& d,
                 const SmolgenDims& sd) {
  auto* weights = net->mutable_weights();
  auto* enc = weights->mutable_encoder(weights->encoder_size() - 1);
  auto* smol = enc->mutable_mha()->mutable_smolgen();
  FillLayer(smol->mutable_compress(),
            RandomVec(rng, sd.hidden_channels * d.embedding, 0.1f));
  FillLayer(smol->mutable_dense1_w(),
            RandomVec(rng, sd.hidden_sz * 64 * sd.hidden_channels, 0.05f));
  FillLayer(smol->mutable_dense1_b(), RandomVec(rng, sd.hidden_sz, 0.05f));
  FillLayer(smol->mutable_ln1_gammas(), RandomVec(rng, sd.hidden_sz, 0.1f));
  FillLayer(smol->mutable_ln1_betas(), RandomVec(rng, sd.hidden_sz, 0.05f));
  FillLayer(smol->mutable_dense2_w(),
            RandomVec(rng, sd.gen_outputs * sd.hidden_sz, 0.05f));
  FillLayer(smol->mutable_dense2_b(), RandomVec(rng, sd.gen_outputs, 0.05f));
  FillLayer(smol->mutable_ln2_gammas(), RandomVec(rng, sd.gen_outputs, 0.1f));
  FillLayer(smol->mutable_ln2_betas(), RandomVec(rng, sd.gen_outputs, 0.05f));
  FillLayer(weights->mutable_smolgen_w(),
            RandomVec(rng, 64 * 64 * (sd.gen_outputs / d.heads), 0.05f));
}

// The same net at a real net's dimensions. Every other synthetic net here is
// tiny (embedding 32, 8 heads, policy d_model 32) while every trained net is
// embedding 128 with 16 heads and policy d_model 128, and the backend is
// wrong on real nets while passing all the small ones -- so the dimensions
// are the variable worth isolating.
pblczero::Net MakeNetWithDims(const NetDims& d, unsigned seed) {
  const int input_size = kInputPlanes + 64;
  const int mlh = 4;
  std::mt19937 rng(seed);
  pblczero::Net file;
  auto* weights = file.mutable_weights();
  auto* nf = file.mutable_format()->mutable_network_format();
  using NF = pblczero::NetworkFormat;
  nf->set_input(NF::INPUT_CLASSICAL_112_PLANE);
  nf->set_network(NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT);
  nf->set_policy(NF::POLICY_ATTENTION);
  nf->set_value(NF::VALUE_WDL);
  nf->set_moves_left(NF::MOVES_LEFT_V1);
  nf->set_input_embedding(NF::INPUT_EMBEDDING_NONE);
  nf->set_default_activation(NF::DEFAULT_ACTIVATION_RELU);
  nf->set_ffn_activation(NF::ACTIVATION_DEFAULT);
  nf->set_smolgen_activation(NF::ACTIVATION_DEFAULT);
  nf->add_kda_directions(static_cast<NF::KdaDirection>(1));
  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);
  FillKdaEncoder(&file, rng, d);
  FillPolicyAndValueHeads(&file, rng, d, true, mlh);
  return file;
}

// Matches kda-native-935532: embedding 128, 16 heads, KDA key/value dim 32,
// gate rank 32, encoder DFF 256, policy d_model 128.
NetDims RealisticDims() {
  NetDims d;
  d.embedding = 128;
  d.heads = 16;
  d.key_dim = 32;
  d.value_dim = 32;
  d.gate_rank = 32;
  d.dff = 256;
  d.pol_emb = 128;
  d.pol_dmodel = 128;
  d.val_planes = 128;
  d.val_channels = 128;
  d.mlh_hidden = 128;
  return d;
}

TEST(DirectMlKdaParity, MatchesBlasOnRealisticDimsNet) {
  CompareBackends(MakeNetWithDims(RealisticDims(), 2024));
}

// Same, but with only the policy head widened, to separate a policy-head
// size problem from a body/encoder one.
TEST(DirectMlKdaParity, MatchesBlasOnWidePolicyHeadNet) {
  NetDims d;
  d.pol_emb = 128;
  d.pol_dmodel = 128;
  CompareBackends(MakeNetWithDims(d, 2025));
}

// One dimension at a time, to name the one that breaks it.
TEST(DirectMlKdaParity, DimBisect_Embedding128) {
  NetDims d; d.embedding = 128;
  CompareBackends(MakeNetWithDims(d, 3001));
}
TEST(DirectMlKdaParity, DimBisect_Heads16) {
  NetDims d; d.heads = 16;
  CompareBackends(MakeNetWithDims(d, 3002));
}
TEST(DirectMlKdaParity, DimBisect_KeyValue32) {
  NetDims d; d.key_dim = 32; d.value_dim = 32;
  CompareBackends(MakeNetWithDims(d, 3003));
}
TEST(DirectMlKdaParity, DimBisect_GateRank32) {
  NetDims d; d.gate_rank = 32;
  CompareBackends(MakeNetWithDims(d, 3004));
}
TEST(DirectMlKdaParity, DimBisect_Dff256) {
  NetDims d; d.dff = 256;
  CompareBackends(MakeNetWithDims(d, 3005));
}
// Pairs. KD = heads * key_dim and VD = heads * value_dim scale
// multiplicatively, so those are the ones that can blow a per-token budget
// while each factor alone looks harmless.
TEST(DirectMlKdaParity, DimBisect_Heads16_KeyValue32) {
  NetDims d; d.heads = 16; d.key_dim = 32; d.value_dim = 32;
  CompareBackends(MakeNetWithDims(d, 3006));
}
TEST(DirectMlKdaParity, DimBisect_Emb128_Heads16) {
  NetDims d; d.embedding = 128; d.heads = 16;
  CompareBackends(MakeNetWithDims(d, 3007));
}
TEST(DirectMlKdaParity, DimBisect_Emb128_KeyValue32) {
  NetDims d; d.embedding = 128; d.key_dim = 32; d.value_dim = 32;
  CompareBackends(MakeNetWithDims(d, 3008));
}
TEST(DirectMlKdaParity, DimBisect_Heads16_Key32Only) {
  NetDims d; d.heads = 16; d.key_dim = 32;
  CompareBackends(MakeNetWithDims(d, 3009));
}
TEST(DirectMlKdaParity, DimBisect_Heads16_Value32Only) {
  NetDims d; d.heads = 16; d.value_dim = 32;
  CompareBackends(MakeNetWithDims(d, 3010));
}
// Delta-debug from the failing side: RealisticDims with one field put back
// to its small default. Whichever revert makes it pass is required for the
// bug.
TEST(DirectMlKdaParity, DeltaRevert_Embedding) {
  NetDims d = RealisticDims(); d.embedding = 32;
  CompareBackends(MakeNetWithDims(d, 4001));
}
TEST(DirectMlKdaParity, DeltaRevert_Heads) {
  NetDims d = RealisticDims(); d.heads = 8;
  CompareBackends(MakeNetWithDims(d, 4002));
}
TEST(DirectMlKdaParity, DeltaRevert_KeyValue) {
  NetDims d = RealisticDims(); d.key_dim = 4; d.value_dim = 4;
  CompareBackends(MakeNetWithDims(d, 4003));
}
TEST(DirectMlKdaParity, DeltaRevert_GateRank) {
  NetDims d = RealisticDims(); d.gate_rank = 4;
  CompareBackends(MakeNetWithDims(d, 4004));
}
TEST(DirectMlKdaParity, DeltaRevert_Dff) {
  NetDims d = RealisticDims(); d.dff = 64;
  CompareBackends(MakeNetWithDims(d, 4005));
}
TEST(DirectMlKdaParity, DeltaRevert_HeadSizes) {
  NetDims d = RealisticDims();
  d.pol_emb = 32; d.pol_dmodel = 32; d.val_planes = 32; d.val_channels = 32;
  CompareBackends(MakeNetWithDims(d, 4006));
}
// Head sizes are the necessary ingredient; split policy from value.
TEST(DirectMlKdaParity, DeltaRevert_PolicyHeadOnly) {
  NetDims d = RealisticDims(); d.pol_emb = 32; d.pol_dmodel = 32;
  CompareBackends(MakeNetWithDims(d, 4007));
}
TEST(DirectMlKdaParity, DeltaRevert_ValueHeadOnly) {
  NetDims d = RealisticDims(); d.val_planes = 32; d.val_channels = 32;
  CompareBackends(MakeNetWithDims(d, 4008));
}
TEST(DirectMlKdaParity, WideValueHeadSmallBody) {
  NetDims d; d.val_planes = 128; d.val_channels = 128;
  CompareBackends(MakeNetWithDims(d, 4009));
}

// The full shape of a real trained net at real dimensions: PE_DENSE input
// embedding, three KDA encoders plus one MHA encoder carrying smolgen, MLH,
// and 128-wide heads. Everything a real net has except LINEAR16 weights.
pblczero::Net MakeFullRealisticNet(bool pe_dense, bool smolgen, bool mha) {
  const NetDims d = RealisticDims();
  const int dense_size = 32;
  const int input_size =
      kInputPlanes + (pe_dense ? dense_size : 64);
  std::mt19937 rng(5150);
  pblczero::Net file;
  auto* weights = file.mutable_weights();
  auto* nf = file.mutable_format()->mutable_network_format();
  using NF = pblczero::NetworkFormat;
  nf->set_input(NF::INPUT_CLASSICAL_112_PLANE);
  nf->set_network(NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT);
  nf->set_policy(NF::POLICY_ATTENTION);
  nf->set_value(NF::VALUE_WDL);
  nf->set_moves_left(NF::MOVES_LEFT_V1);
  nf->set_input_embedding(pe_dense ? NF::INPUT_EMBEDDING_PE_DENSE
                                   : NF::INPUT_EMBEDDING_NONE);
  nf->set_default_activation(NF::DEFAULT_ACTIVATION_MISH);
  nf->set_ffn_activation(NF::ACTIVATION_DEFAULT);
  nf->set_smolgen_activation(NF::ACTIVATION_SWISH);
  for (int dir : {9, 10, 11, 12}) {
    nf->add_kda_directions(static_cast<NF::KdaDirection>(dir));
  }

  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  if (pe_dense) {
    FillLayer(weights->mutable_ip_emb_preproc_w(),
              RandomVec(rng, 64 * dense_size * 64 * 12, 0.05f));
    FillLayer(weights->mutable_ip_emb_preproc_b(),
              RandomVec(rng, 64 * dense_size, 0.05f));
    FillLayer(weights->mutable_ip_emb_ln_gammas(),
              RandomVec(rng, d.embedding, 0.1f));
    FillLayer(weights->mutable_ip_emb_ln_betas(),
              RandomVec(rng, d.embedding, 0.05f));
    auto* ffn = weights->mutable_ip_emb_ffn();
    FillLayer(ffn->mutable_dense1_w(),
              RandomVec(rng, d.embedding * d.embedding, 0.1f));
    FillLayer(ffn->mutable_dense1_b(), RandomVec(rng, d.embedding, 0.05f));
    FillLayer(ffn->mutable_dense2_w(),
              RandomVec(rng, d.embedding * d.embedding, 0.1f));
    FillLayer(ffn->mutable_dense2_b(), RandomVec(rng, d.embedding, 0.05f));
    FillLayer(weights->mutable_ip_emb_ffn_ln_gammas(),
              RandomVec(rng, d.embedding, 0.1f));
    FillLayer(weights->mutable_ip_emb_ffn_ln_betas(),
              RandomVec(rng, d.embedding, 0.05f));
  }
  weights->set_headcount(d.heads);
  FillKdaEncoder(&file, rng, d);
  FillKdaEncoder(&file, rng, d);
  FillKdaEncoder(&file, rng, d);
  if (mha) {
    FillMhaEncoder(&file, rng, d);
    if (smolgen) {
      SmolgenDims sd;
      sd.hidden_channels = 32;
      sd.hidden_sz = 256;
      sd.gen_outputs = d.heads * 16;
      FillSmolgen(&file, rng, d, sd);
    }
  }
  FillPolicyAndValueHeads(&file, rng, d, true, 32);
  return file;
}

TEST(DirectMlKdaParity, FullRealistic_NoPeDenseNoMha) {
  CompareBackends(MakeFullRealisticNet(false, false, false));
}
TEST(DirectMlKdaParity, FullRealistic_PeDenseOnly) {
  CompareBackends(MakeFullRealisticNet(true, false, false));
}
TEST(DirectMlKdaParity, FullRealistic_MhaNoSmolgen) {
  CompareBackends(MakeFullRealisticNet(false, false, true));
}
TEST(DirectMlKdaParity, FullRealistic_MhaSmolgen) {
  CompareBackends(MakeFullRealisticNet(false, true, true));
}
TEST(DirectMlKdaParity, FullRealistic_Everything) {
  CompareBackends(MakeFullRealisticNet(true, true, true));
}

// The same KDA encoder driven by the four serpentine (boustrophedon)
// traversals, directions 13/14/15/16 alongside 9 and 11.
//
// These were not covered anywhere, and the DirectML recurrence kernel used to
// resolve the square order with a branch chain that stopped at direction 8.
// Anything above it fell through to `square = token`, so a net trained with a
// serpentine direction loaded, ran, and returned quietly wrong evaluations --
// no error, no warning. The kernel now indexes the shared table from
// neural/kda_directions.h, and the backend rejects anything outside 1-16 at
// load rather than guessing.
pblczero::Net MakeKdaSerpentineNet() {
  const NetDims d;
  const int input_size = kInputPlanes + 64;
  const int mlh = 4;
  std::mt19937 rng(99);
  pblczero::Net file;
  auto* weights = file.mutable_weights();
  auto* nf = file.mutable_format()->mutable_network_format();
  using NF = pblczero::NetworkFormat;
  nf->set_input(NF::INPUT_CLASSICAL_112_PLANE);
  nf->set_network(NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT);
  nf->set_policy(NF::POLICY_ATTENTION);
  nf->set_value(NF::VALUE_WDL);
  nf->set_moves_left(NF::MOVES_LEFT_V1);
  nf->set_input_embedding(NF::INPUT_EMBEDDING_NONE);
  nf->set_default_activation(NF::DEFAULT_ACTIVATION_RELU);
  nf->set_ffn_activation(NF::ACTIVATION_DEFAULT);
  nf->set_smolgen_activation(NF::ACTIVATION_DEFAULT);
  for (int dir : {9, 11, 13, 15}) {
    nf->add_kda_directions(static_cast<NF::KdaDirection>(dir));
  }
  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);
  FillKdaEncoder(&file, rng, d);
  FillPolicyAndValueHeads(&file, rng, d, true, mlh);
  return file;
}

TEST(DirectMlKdaParity, MatchesBlasOnKdaSerpentineNet) {
  CompareBackends(MakeKdaSerpentineNet());
}

// The reverse serpentine traversals, directions 10/12/14/16.
pblczero::Net MakeKdaSerpentineReverseNet() {
  pblczero::Net file = MakeKdaSerpentineNet();
  auto* nf = file.mutable_format()->mutable_network_format();
  nf->mutable_kda_directions()->clear();
  using NF = pblczero::NetworkFormat;
  for (int dir : {10, 12, 14, 16}) {
    nf->add_kda_directions(static_cast<NF::KdaDirection>(dir));
  }
  return file;
}

TEST(DirectMlKdaParity, MatchesBlasOnKdaSerpentineReverseNet) {
  CompareBackends(MakeKdaSerpentineReverseNet());
}

// End-to-end check against a real trained net, if one is pointed at. Set
// LC0_TEST_REAL_NET to a .pb.gz path to run it; skipped otherwise so the
// suite stays self-contained and hermetic.
TEST(DirectMlKdaParity, MatchesBlasOnRealNetFromEnv) {
  const char* path = getenv("LC0_TEST_REAL_NET");
  if (!path) GTEST_SKIP() << "set LC0_TEST_REAL_NET to a .pb.gz to run";
  pblczero::Net net = LoadWeightsFromFile(path);
  // Bisection knob: keep only the first N encoders. Both backends see the
  // same truncated net, so the comparison stays fair.
  if (const char* keep = getenv("LC0_TEST_REAL_NET_ENCODERS")) {
    const int n = atoi(keep);
    auto* w = net.mutable_weights();
    while (w->encoder_size() > n) w->mutable_encoder()->pop_back();
    CERR << "[bisect] encoders kept: " << w->encoder_size();
  }
  // Keep only encoders of one mixer type, to see which diverges.
  if (const char* mixer = getenv("LC0_TEST_KEEP_MIXER")) {
    const bool want_kda = std::string(mixer) == "kda";
    auto* w = net.mutable_weights();
    std::vector<pblczero::Weights::EncoderLayer> kept;
    for (size_t i = 0; i < w->encoder_size(); ++i) {
      const auto& enc = w->encoder(i);
      const bool is_kda =
          enc.mixer() == pblczero::Weights::EncoderLayer::MIXER_KDA;
      if (is_kda == want_kda) kept.push_back(enc);
    }
    w->mutable_encoder()->clear();
    for (const auto& e : kept) *w->add_encoder() = e;
    CERR << "[bisect] kept " << w->encoder_size() << " " << mixer
         << " encoders";
  }
  // Replace one group of a real net's weights with random values of the same
  // length. Both backends see the same mutated net, so the comparison stays
  // fair; if a group's real values are what trigger a divergence, replacing
  // that group makes the test pass.
  if (const char* group = getenv("LC0_TEST_RANDOMIZE")) {
    std::mt19937 rng(7777);
    const std::string g = group;
    auto* w = net.mutable_weights();
    auto reroll = [&](pblczero::Weights::Layer* layer) {
      const size_t n = LayerAdapter(*layer).size();
      if (n == 0) return;
      FillLayer(layer, RandomVec(rng, n, 0.05f));
    };
    if (g == "embedding") {
      reroll(w->mutable_ip_emb_w());
      reroll(w->mutable_ip_emb_b());
      reroll(w->mutable_ip_emb_preproc_w());
      reroll(w->mutable_ip_emb_preproc_b());
      reroll(w->mutable_ip_emb_ln_gammas());
      reroll(w->mutable_ip_emb_ln_betas());
      reroll(w->mutable_ip_emb_ffn()->mutable_dense1_w());
      reroll(w->mutable_ip_emb_ffn()->mutable_dense1_b());
      reroll(w->mutable_ip_emb_ffn()->mutable_dense2_w());
      reroll(w->mutable_ip_emb_ffn()->mutable_dense2_b());
      reroll(w->mutable_ip_emb_ffn_ln_gammas());
      reroll(w->mutable_ip_emb_ffn_ln_betas());
    } else if (g == "encoders") {
      for (size_t i = 0; i < w->encoder_size(); ++i) {
        auto* e = w->mutable_encoder(i);
        reroll(e->mutable_ln1_gammas());
        reroll(e->mutable_ln1_betas());
        reroll(e->mutable_ln2_gammas());
        reroll(e->mutable_ln2_betas());
        reroll(e->mutable_ffn()->mutable_dense1_w());
        reroll(e->mutable_ffn()->mutable_dense1_b());
        reroll(e->mutable_ffn()->mutable_dense2_w());
        reroll(e->mutable_ffn()->mutable_dense2_b());
        auto* m = e->mutable_mha();
        reroll(m->mutable_q_w());
        reroll(m->mutable_q_b());
        reroll(m->mutable_k_w());
        reroll(m->mutable_k_b());
        reroll(m->mutable_v_w());
        reroll(m->mutable_v_b());
        reroll(m->mutable_dense_w());
        reroll(m->mutable_dense_b());
      }
    } else if (g == "kda") {
      for (size_t i = 0; i < w->encoder_size(); ++i) {
        auto* k = w->mutable_encoder(i)->mutable_kda();
        reroll(k->mutable_q_w());
        reroll(k->mutable_q_b());
        reroll(k->mutable_k_w());
        reroll(k->mutable_k_b());
        reroll(k->mutable_v_w());
        reroll(k->mutable_v_b());
        reroll(k->mutable_a_log());
        reroll(k->mutable_dt_bias());
        reroll(k->mutable_beta_w());
        reroll(k->mutable_beta_b());
        reroll(k->mutable_dense_w());
        reroll(k->mutable_dense_b());
      }
    } else if (g == "heads") {
      auto* ph = w->mutable_policy_heads();
      reroll(ph->mutable_ip_pol_w());
      reroll(ph->mutable_ip_pol_b());
      auto* v = ph->mutable_vanilla();
      reroll(v->mutable_ip2_pol_w());
      reroll(v->mutable_ip2_pol_b());
      reroll(v->mutable_ip3_pol_w());
      reroll(v->mutable_ip3_pol_b());
      reroll(v->mutable_ip4_pol_w());
    }
    CERR << "[bisect] randomized group: " << g;
  }
  if (getenv("LC0_TEST_STRIP_MLH")) {
    net.mutable_format()->mutable_network_format()->set_moves_left(
        pblczero::NetworkFormat::MOVES_LEFT_NONE);
    CERR << "[bisect] mlh stripped";
  }
  CompareBackends(net);
}

// MHA + moves-left head, no KDA encoder: covers the MHA encoder and
// MLH paths together against the BLAS reference.
pblczero::Net MakeMhaMlhNet() {
  const NetDims d;
  const int input_size = kInputPlanes + 64;
  const int mlh = 4;
  std::mt19937 rng(66);
  pblczero::Net file;
  auto* weights = file.mutable_weights();
  auto* nf = file.mutable_format()->mutable_network_format();
  using NF = pblczero::NetworkFormat;
  nf->set_input(NF::INPUT_CLASSICAL_112_PLANE);
  nf->set_network(NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT);
  nf->set_policy(NF::POLICY_ATTENTION);
  nf->set_value(NF::VALUE_WDL);
  nf->set_moves_left(NF::MOVES_LEFT_V1);
  nf->set_input_embedding(NF::INPUT_EMBEDDING_NONE);
  nf->set_default_activation(NF::DEFAULT_ACTIVATION_RELU);
  nf->set_ffn_activation(NF::ACTIVATION_DEFAULT);
  nf->set_smolgen_activation(NF::ACTIVATION_DEFAULT);
  nf->add_kda_directions(static_cast<NF::KdaDirection>(1));
  FillLayer(weights->mutable_ip_emb_w(), RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);
  FillMhaEncoder(&file, rng, d);
  FillPolicyAndValueHeads(&file, rng, d, true, mlh);
  return file;
}

TEST(DirectMlKdaParity, MatchesBlasOnMhaMlhNet) { CompareBackends(MakeMhaMlhNet()); }

pblczero::Net MakeSmolgenMhaNet() {
  const NetDims d;
  const SmolgenDims sd;
  const int input_size = kInputPlanes + 64;
  const int mlh = 4;
  std::mt19937 rng(4242);
  pblczero::Net file;
  auto* weights = file.mutable_weights();
  auto* nf = file.mutable_format()->mutable_network_format();
  using NF = pblczero::NetworkFormat;
  nf->set_input(NF::INPUT_CLASSICAL_112_PLANE);
  nf->set_network(NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT);
  nf->set_policy(NF::POLICY_ATTENTION);
  nf->set_value(NF::VALUE_WDL);
  nf->set_moves_left(NF::MOVES_LEFT_V1);
  nf->set_input_embedding(NF::INPUT_EMBEDDING_NONE);
  nf->set_default_activation(NF::DEFAULT_ACTIVATION_RELU);
  nf->set_ffn_activation(NF::ACTIVATION_DEFAULT);
  nf->set_smolgen_activation(NF::ACTIVATION_SWISH);
  nf->add_kda_directions(static_cast<NF::KdaDirection>(1));
  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);
  FillMhaEncoder(&file, rng, d);
  FillSmolgen(&file, rng, d, sd);
  FillPolicyAndValueHeads(&file, rng, d, true, mlh);
  return file;
}

// Exercises smolgen end to end: the compress GEMM, both MLP stages with
// their LayerNorms and activations, the shared-table bias GEMM, and the bias
// reaching the attention logits. Nothing covered any of that before, which
// is why several defects lived there -- including a graph input that was
// bound with a byte count computed from the wrong strides, so the bias was
// read as though it were absent.
TEST(DirectMlKdaParity, MatchesBlasOnSmolgenMhaNet) {
  CompareBackends(MakeSmolgenMhaNet());
}



// PE_DENSE input-embedding net (no encoders): the real trained nets use
// INPUT_EMBEDDING_PE_DENSE, a body branch the earlier synthetic nets never
// exercised.
pblczero::Net MakePeDenseNet() {
  const NetDims d;
  const int dense_size = 32;
  std::mt19937 rng(88);
  pblczero::Net file;
  auto* weights = file.mutable_weights();
  auto* nf = file.mutable_format()->mutable_network_format();
  using NF = pblczero::NetworkFormat;
  nf->set_input(NF::INPUT_CLASSICAL_112_PLANE);
  nf->set_network(NF::NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT);
  nf->set_policy(NF::POLICY_ATTENTION);
  nf->set_value(NF::VALUE_WDL);
  nf->set_moves_left(NF::MOVES_LEFT_NONE);
  nf->set_input_embedding(NF::INPUT_EMBEDDING_PE_DENSE);
  nf->set_default_activation(NF::DEFAULT_ACTIVATION_RELU);
  nf->set_ffn_activation(NF::ACTIVATION_DEFAULT);
  nf->set_smolgen_activation(NF::ACTIVATION_DEFAULT);
  nf->add_kda_directions(static_cast<NF::KdaDirection>(1));

  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * (kInputPlanes + dense_size), 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  // PE_DENSE preprocess: dense gemm over the flattened 12-channel slice.
  FillLayer(weights->mutable_ip_emb_preproc_w(),
            RandomVec(rng, 64 * dense_size * 64 * 12, 0.05f));
  FillLayer(weights->mutable_ip_emb_preproc_b(),
            RandomVec(rng, 64 * dense_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_ln_gammas(),
            RandomVec(rng, d.embedding, 0.1f));
  FillLayer(weights->mutable_ip_emb_ln_betas(),
            RandomVec(rng, d.embedding, 0.05f));
  auto* ffn = weights->mutable_ip_emb_ffn();
  FillLayer(ffn->mutable_dense1_w(),
            RandomVec(rng, d.embedding * d.embedding, 0.1f));
  FillLayer(ffn->mutable_dense1_b(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(ffn->mutable_dense2_w(),
            RandomVec(rng, d.embedding * d.embedding, 0.1f));
  FillLayer(ffn->mutable_dense2_b(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(weights->mutable_ip_emb_ffn_ln_gammas(),
            RandomVec(rng, d.embedding, 0.1f));
  FillLayer(weights->mutable_ip_emb_ffn_ln_betas(),
            RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);
  FillPolicyAndValueHeads(&file, rng, d, false, 0);
  return file;
}

TEST(DirectMlKdaParity, MatchesBlasOnPeDenseNet) {
  CompareBackends(MakePeDenseNet());
}

// PE_DENSE feeding an actual encoder stack, which is the shape every real
// trained net has and which no synthetic net covered: MakePeDenseNet has no
// encoder, and every net that does have one uses INPUT_EMBEDDING_NONE.
pblczero::Net MakePeDenseWithEncodersNet() {
  const NetDims d;
  std::mt19937 rng(4711);
  pblczero::Net file = MakePeDenseNet();
  auto* weights = file.mutable_weights();
  auto* nf = file.mutable_format()->mutable_network_format();
  nf->set_moves_left(pblczero::NetworkFormat::MOVES_LEFT_V1);
  FillKdaEncoder(&file, rng, d);
  FillMhaEncoder(&file, rng, d);
  weights->set_headcount(d.heads);
  FillPolicyAndValueHeads(&file, rng, d, true, 4);
  return file;
}

TEST(DirectMlKdaParity, MatchesBlasOnPeDenseWithEncodersNet) {
  CompareBackends(MakePeDenseWithEncodersNet());
}

TEST(DirectMlKdaParity, MatchesBlasOnKdaHybridNet) {
  CompareBackends(MakeKdaHybridNet());
}

TEST(DirectMlKdaParity, MatchesBlasOnKdaMhaNet) {
  CompareBackends(MakeKdaMhaNet());
}

TEST(DirectMlKdaParity, MatchesBlasOnNoEncoderNet) {
  CompareBackends(MakeNoEncoderNet());
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
