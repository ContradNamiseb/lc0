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
#include "neural/backends/directml/dml_common.h"

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

// Trained LayerNorm gammas cluster near 1.0; every synthetic gamma here is
// uniform around ZERO. That is the one distribution axis the parity suite
// never varied -- shape, encoding and magnitude are all matched to real
// nets, but not the distribution -- and a near-zero gamma scales a
// normalisation's output toward zero, which would mask an indexing or
// broadcast bug inside it. LC0_TEST_GAMMA_ONE=1 centres them on 1.0.
std::vector<float> GammaVec(std::mt19937& rng, size_t n) {
  std::vector<float> v = RandomVec(rng, n, 0.1f);
  if (getenv("LC0_TEST_GAMMA_ONE")) {
    for (auto& x : v) x += 1.0f;
  }
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
  FillLayer(kda->mutable_out_norm_gammas(), GammaVec(rng, value_depth));
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
  FillLayer(enc->mutable_ln1_gammas(), GammaVec(rng, d.embedding));
  FillLayer(enc->mutable_ln1_betas(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(enc->mutable_ffn()->mutable_dense1_w(),
            RandomVec(rng, d.embedding * d.dff, 0.1f));
  FillLayer(enc->mutable_ffn()->mutable_dense1_b(), RandomVec(rng, d.dff, 0.05f));
  FillLayer(enc->mutable_ffn()->mutable_dense2_w(),
            RandomVec(rng, d.dff * d.embedding, 0.1f));
  FillLayer(enc->mutable_ffn()->mutable_dense2_b(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(enc->mutable_ln2_gammas(), GammaVec(rng, d.embedding));
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
  FillLayer(enc->mutable_ln1_gammas(), GammaVec(rng, d.embedding));
  FillLayer(enc->mutable_ln1_betas(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(enc->mutable_ffn()->mutable_dense1_w(),
            RandomVec(rng, d.embedding * d.dff, 0.1f));
  FillLayer(enc->mutable_ffn()->mutable_dense1_b(), RandomVec(rng, d.dff, 0.05f));
  FillLayer(enc->mutable_ffn()->mutable_dense2_w(),
            RandomVec(rng, d.dff * d.embedding, 0.1f));
  FillLayer(enc->mutable_ffn()->mutable_dense2_b(), RandomVec(rng, d.embedding, 0.05f));
  FillLayer(enc->mutable_ln2_gammas(), GammaVec(rng, d.embedding));
  FillLayer(enc->mutable_ln2_betas(), RandomVec(rng, d.embedding, 0.05f));
}

// Adds one MHA encoder to the POLICY head. Every other net here leaves
// pol_encoder empty, so AttentionPolicyHead's encoder path and the scratch
// terms that must cover it were entirely untested: the head's q/k/v need
// 3 * pol_d_model per token and its buffer1 carve-up another 5, none of which
// the wq/wk/scores term (2 * ip2_pol_b + 64) accounts for once pol_d_model
// exceeds 64.
void FillPolicyEncoder(pblczero::Net* net, std::mt19937& rng, int pol_emb,
                       int pol_d_model, int heads, int dff) {
  // pol_encoder and pol_headcount live on the individual head (proto
  // PolicyHead fields 8 and 9), not on the PolicyHeads container.
  auto* ph = net->mutable_weights()->mutable_policy_heads()->mutable_vanilla();
  ph->set_pol_headcount(heads);
  auto* enc = ph->add_pol_encoder();
  enc->set_mixer(pblczero::Weights::EncoderLayer::MIXER_MHA);
  auto* mha = enc->mutable_mha();
  // q_w is [pol_emb, pol_d_model]: the sizing derives d_model from
  // q_w.size() / pol_emb, so this is what sets the requirement under test.
  FillLayer(mha->mutable_q_w(), RandomVec(rng, pol_emb * pol_d_model, 0.1f));
  FillLayer(mha->mutable_q_b(), RandomVec(rng, pol_d_model, 0.05f));
  FillLayer(mha->mutable_k_w(), RandomVec(rng, pol_emb * pol_d_model, 0.1f));
  FillLayer(mha->mutable_k_b(), RandomVec(rng, pol_d_model, 0.05f));
  FillLayer(mha->mutable_v_w(), RandomVec(rng, pol_emb * pol_d_model, 0.1f));
  FillLayer(mha->mutable_v_b(), RandomVec(rng, pol_d_model, 0.05f));
  FillLayer(mha->mutable_dense_w(),
            RandomVec(rng, pol_d_model * pol_emb, 0.1f));
  FillLayer(mha->mutable_dense_b(), RandomVec(rng, pol_emb, 0.05f));
  FillLayer(enc->mutable_ln1_gammas(), GammaVec(rng, pol_emb));
  FillLayer(enc->mutable_ln1_betas(), RandomVec(rng, pol_emb, 0.05f));
  FillLayer(enc->mutable_ffn()->mutable_dense1_w(),
            RandomVec(rng, pol_emb * dff, 0.1f));
  FillLayer(enc->mutable_ffn()->mutable_dense1_b(), RandomVec(rng, dff, 0.05f));
  FillLayer(enc->mutable_ffn()->mutable_dense2_w(),
            RandomVec(rng, dff * pol_emb, 0.1f));
  FillLayer(enc->mutable_ffn()->mutable_dense2_b(),
            RandomVec(rng, pol_emb, 0.05f));
  FillLayer(enc->mutable_ln2_gammas(), GammaVec(rng, pol_emb));
  FillLayer(enc->mutable_ln2_betas(), RandomVec(rng, pol_emb, 0.05f));
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


// Distinct positions. A batch of identical inputs would hide any per-sample
// indexing bug, because every row would hold the same numbers.
std::vector<InputPlanes> EncodeDistinctPositions(int count) {
  static const char* kFens[] = {
      ChessBoard::kStartposFen,
      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
      "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
      "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "rnbq1rk1/pp2bppp/4pn2/2pp4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 1",
      "4rrk1/pp1n1ppp/2pb4/q2p4/3P4/1QPB1N2/PP3PPP/R4RK1 w - - 0 1",
  };
  const int kCount = static_cast<int>(sizeof(kFens) / sizeof(kFens[0]));
  std::vector<InputPlanes> out;
  for (int i = 0; i < count; ++i) {
    ChessBoard board;
    PositionHistory history;
    board.SetFromFen(kFens[i % kCount]);
    history.Reset(board, 0, 1);
    out.push_back(EncodePositionForNN(
        pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE, history, 8,
        FillEmptyHistory::NO, nullptr));
  }
  return out;
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

// Runs a whole batch and returns EVERY sample, not just the first.
std::vector<Outputs> RunNetworkBatch(const std::string& backend,
                                     const WeightsFile& weights,
                                     const std::vector<InputPlanes>& planes) {
  OptionsDict options;
  auto network = NetworkFactory::Get()->Create(backend, weights, options);
  auto computation = network->NewComputation();
  for (const auto& p : planes) computation->AddInput(InputPlanes(p));
  computation->ComputeBlocking();
  std::vector<Outputs> out(planes.size());
  for (size_t n = 0; n < planes.size(); ++n) {
    out[n].q = computation->GetQVal(static_cast<int>(n));
    out[n].d = computation->GetDVal(static_cast<int>(n));
    out[n].m = computation->GetMVal(static_cast<int>(n));
    out[n].policy.reserve(1858);
    for (int i = 0; i < 1858; ++i) {
      out[n].policy.push_back(computation->GetPVal(static_cast<int>(n), i));
    }
  }
  return out;
}

bool HasBackend(const std::string& name) {
  const auto& backends = NetworkFactory::Get()->GetBackendsList();
  return std::find(backends.begin(), backends.end(), name) != backends.end();
}

// Hardware availability, decided ONCE and independently of any net.
//
// This used to be a catch-to-GTEST_SKIP wrapped around the whole subject
// network's Create/Compute path, which conflated two unrelated things: "this
// machine has no DirectML device" and "the backend REJECTED this net". The
// second is a defect, and folding it into a skip made every such defect exit
// the suite green -- the policy-encoder scratch guard fired for both new
// wide-encoder tests and the run still reported success, which is how a
// regression test that never bit shipped as if it had.
//
// Availability is a property of the MACHINE, so it is settled here, by
// bringing the device up on its own with no weights involved. Deliberately
// NOT a known-good probe network: a regression in the probe would put backend
// defects straight back into suite-wide skips, which is the failure this
// replaces.
struct DmlAvailability {
  bool available = false;
  std::string reason;
};

const DmlAvailability& DirectMlAvailability() {
  static const DmlAvailability cached = [] {
    DmlAvailability r;
    try {
      OptionsDict options;
      // The skip path has to be testable on a machine whose DirectML works,
      // or it is itself untested -- the same one-sided-coverage trap this
      // whole change exists to close. Point this at an adapter index that
      // does not resolve and Init throws before any net exists.
      if (const char* gpu = getenv("LC0_TEST_DML_GPU")) {
        options.Set<int>("gpu", std::atoi(gpu));
      }
      directml_backend::DmlDeviceContext ctx;
      ctx.Init(options);
      r.available = true;
    } catch (const Exception& e) {
      r.reason = e.what();
    }
    return r;
  }();
  return cached;
}

// Same tolerance rationale as the SYCL parity test: the recurrence is
// sequential-float in both backends, the surrounding GEMMs accumulate in
// different orders, and an actual math divergence (wrong traversal, dropped
// gate, wrong softmax axis) produces errors ~1e-1+, two orders of magnitude
// above this bar.
constexpr float kTol = 2e-4f;

// Every sample of a multi-position batch, against BLAS.
//
// The single-position CompareBackends validates exactly one (batch, sample) =
// (1, 0), and every HLSL kernel here indexes by sample. A wrong per-row stride
// is invisible at batch 1, because row 0 starts at offset 0 whichever stride
// is used -- which is exactly how policy_finalize shipped ROW_STRIDE 4168
// against a 4288-wide reader, scrambling the policy of every sample after the
// first in every real search.
void CompareBackendsBatch(const pblczero::Net& net, int batch) {
  const std::vector<InputPlanes> planes = EncodeDistinctPositions(batch);

  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  const std::vector<Outputs> reference =
      RunNetworkBatch("blas", net, planes);

  const char* backend_env = getenv("LC0_TEST_BACKEND");
  const std::string test_backend = backend_env ? backend_env : "directml";
  if (!HasBackend(test_backend)) {
    GTEST_SKIP() << test_backend << " backend not compiled in";
  }
  // Availability was settled above, for the machine. Past this point every
  // backend exception -- CreateOperator, a sizing guard, dispatch, output --
  // is a real failure and must fail the test, not skip it. eigen is CPU, so
  // no hardware preflight applies to it.
  if (test_backend == "directml" && !DirectMlAvailability().available) {
    GTEST_SKIP() << "no usable directml device: "
                 << DirectMlAvailability().reason;
  }
  const std::vector<Outputs> dml =
      RunNetworkBatch(test_backend, net, planes);

  constexpr float kRelTol = 5e-5f;
  auto bound = [](float reference_value) {
    return kTol + kRelTol * std::fabs(reference_value);
  };
  for (int n = 0; n < batch; ++n) {
    EXPECT_NEAR(dml[n].q, reference[n].q, bound(reference[n].q))
        << "sample " << n << ": WDL value Q diverges";
    EXPECT_NEAR(dml[n].d, reference[n].d, bound(reference[n].d))
        << "sample " << n << ": WDL draw probability diverges";
    EXPECT_NEAR(dml[n].m, reference[n].m, bound(reference[n].m))
        << "sample " << n << ": moves-left diverges";

    float ref_absmax = 0.0f, worst = 0.0f;
    int worst_move = -1;
    for (int i = 0; i < 1858; ++i) {
      ref_absmax = std::max(ref_absmax, std::fabs(reference[n].policy[i]));
      const float diff =
          std::fabs(dml[n].policy[i] - reference[n].policy[i]);
      if (diff > worst) {
        worst = diff;
        worst_move = i;
      }
    }
    EXPECT_LT(worst, bound(ref_absmax))
        << "sample " << n << ": policy diverges (worst move " << worst_move
        << ", diff " << worst << ")";
  }
}

void CompareBackends(const pblczero::Net& net) {
  const InputPlanes planes = EncodeStartPos();

  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  const Outputs reference = RunNetwork("blas", net, planes);

  // LC0_TEST_BACKEND swaps what is compared against the BLAS reference.
  // Setting it to "eigen" measures the reference against ITSELF: eigen runs
  // the same network_blas.cc code with a different GEMM, so the difference is
  // purely accumulation order. That is the noise floor this suite's tolerance
  // should sit just above -- a bar derived from a measurement rather than
  // from the size of the bugs already caught, which is survivorship.
  const char* backend_env = getenv("LC0_TEST_BACKEND");
  const std::string test_backend = backend_env ? backend_env : "directml";
  if (!HasBackend(test_backend)) {
    GTEST_SKIP() << test_backend << " backend not compiled in";
  }
  // Availability was settled above, for the machine. Past this point every
  // backend exception -- CreateOperator, a sizing guard, dispatch, output --
  // is a real failure and must fail the test, not skip it. eigen is CPU, so
  // no hardware preflight applies to it.
  if (test_backend == "directml" && !DirectMlAvailability().available) {
    GTEST_SKIP() << "no usable directml device: "
                 << DirectMlAvailability().reason;
  }
  const Outputs dml = RunNetwork(test_backend, net, planes);

  // kTol alone is an ABSOLUTE bar, which is the right shape for q and d --
  // both bounded in [-1, 1] -- but wrong for moves-left and for policy
  // logits, neither of which is bounded. Moves-left is a ply count: on
  // kda-native-825532 it is 17.455, so 2e-4 absolute demands 1.1e-5
  // relative, tighter than two fp32 GEMM orderings agree. That net's body
  // matches BLAS to 4e-6 relative at every stage, and q, d and policy all
  // pass; only m missed, by 2.5e-4 on 17.455 (1.4e-5 relative).
  //
  // So allow atol + rtol * |reference|, with rtol anchored to a MEASURED
  // noise floor rather than to the size of the bugs already caught. That
  // latter argument is survivorship: the binding-overrun bug left the policy
  // head bit-exact while the value head was garbage, so partial wrongness is
  // this backend's real signature, and a smaller overrun or a single wrong
  // broadcast element lands at O(1/N) relative -- under any bar calibrated on
  // 39% errors.
  //
  // The floor is measured by running the reference against ITSELF under a
  // different accumulation order: LC0_TEST_BACKEND=eigen compares MKL BLAS
  // against Eigen through the same network_blas.cc code. Worst relative
  // difference over the real nets, position 0:
  //   blas vs eigen     m 1.3e-6 .. 1.9e-6,  policy 9.4e-7 .. 2.8e-6
  //   blas vs directml  m 1.1e-7 .. 1.4e-5,  policy 7.8e-6 .. 9.4e-6
  // DirectML sits within ~10x of the reference's own floor, which is what a
  // different device with different tiling and FMA contraction should cost --
  // not the ~100x that would mean something systematic is left.
  //
  // rtol is set just above the worst observed (1.4e-5), not an order above:
  // slack is where the first small systematic error hides.
  constexpr float kRelTol = 5e-5f;
  auto bound = [](float reference_value) {
    return kTol + kRelTol * std::fabs(reference_value);
  };
  EXPECT_NEAR(dml.q, reference.q, bound(reference.q))
      << "WDL value Q diverges between directml and blas";
  EXPECT_NEAR(dml.d, reference.d, bound(reference.d))
      << "WDL draw probability diverges between directml and blas";
  EXPECT_NEAR(dml.m, reference.m, bound(reference.m))
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
    float pol_worst = 0.0f;
    for (int i = 0; i < 1858; ++i) {
      pol_worst =
          std::max(pol_worst, std::fabs(dml.policy[i] - reference.policy[i]));
    }
    auto rel = [](float diff, float ref) {
      return ref != 0.0f ? diff / std::fabs(ref) : 0.0f;
    };
    const float mdiff = std::fabs(dml.m - reference.m);
    const float qdiff = std::fabs(dml.q - reference.q);
    CERR << std::scientific << std::setprecision(3)
         << "[diff] q " << qdiff << " (rel " << rel(qdiff, reference.q)
         << ")  m " << mdiff << " (rel " << rel(mdiff, reference.m)
         << ")  policy " << pol_worst << " (rel "
         << rel(pol_worst, ref_absmax) << ")";
  }
  // Policy logits are unbounded too -- 6.5 on a real net, and hundreds once
  // weights are scaled up -- so scale the bar by the largest reference logit
  // rather than comparing each move against a flat 2e-4.
  float ref_absmax = 0.0f;
  for (int i = 0; i < 1858; ++i) {
    ref_absmax = std::max(ref_absmax, std::fabs(reference.policy[i]));
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
  EXPECT_LT(worst, bound(ref_absmax))
      << "policy diverges between directml and blas (worst move "
      << worst_move << ", diff " << worst << ")";

  // Parity of decisions, not just of tensors: a net a little off everywhere
  // that never changes its best move is a better result than one closer in
  // norm that flips the move it plays. Nearly free once both outputs are in
  // hand, and it is the property the engine actually depends on.
  int dml_best = 0, ref_best = 0;
  for (int i = 1; i < 1858; ++i) {
    if (dml.policy[i] > dml.policy[dml_best]) dml_best = i;
    if (reference.policy[i] > reference.policy[ref_best]) ref_best = i;
  }
  EXPECT_EQ(dml_best, ref_best)
      << "policy argmax differs: directml picks move index " << dml_best
      << " (logit " << dml.policy[dml_best] << "), blas picks " << ref_best
      << " (logit " << reference.policy[ref_best] << ")";
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
  FillLayer(smol->mutable_ln1_gammas(), GammaVec(rng, sd.hidden_sz));
  FillLayer(smol->mutable_ln1_betas(), RandomVec(rng, sd.hidden_sz, 0.05f));
  FillLayer(smol->mutable_dense2_w(),
            RandomVec(rng, sd.gen_outputs * sd.hidden_sz, 0.05f));
  FillLayer(smol->mutable_dense2_b(), RandomVec(rng, sd.gen_outputs, 0.05f));
  FillLayer(smol->mutable_ln2_gammas(), GammaVec(rng, sd.gen_outputs));
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

// Input gating populated, in the [channels][64] layout real nets use. Every
// other synthetic net here leaves ip_mult_gate/ip_add_gate empty, so
// has_gating_ is false and the gating path is dead under test -- which is
// exactly how a per-channel broadcast of a per-square matrix survived 36
// passing tests and was only caught by a real net.
//
// The general rule, and worth applying to the other optional features: a
// feature flag should be exercised in BOTH states somewhere in the suite. A
// guard that is always true hides its else-branch just as thoroughly as one
// that is always false.
pblczero::Net MakeGatedNet(const NetDims& d, unsigned seed) {
  pblczero::Net file = MakeNetWithDims(d, seed);
  std::mt19937 rng(seed ^ 0x9e37u);
  auto* w = file.mutable_weights();
  // Values must vary across squares, or a per-channel broadcast would still
  // agree with a per-square read and the test would prove nothing.
  FillLayer(w->mutable_ip_mult_gate(),
            RandomVec(rng, static_cast<size_t>(d.embedding) * 64, 0.3f));
  FillLayer(w->mutable_ip_add_gate(),
            RandomVec(rng, static_cast<size_t>(d.embedding) * 64, 0.1f));
  return file;
}

// Batch tests. These are the ones that cover per-sample indexing in every
// HLSL kernel; the single-position cases above cannot.
TEST(DirectMlKdaParity, MatchesBlasOnBatchOfTwo) {
  CompareBackendsBatch(MakeKdaMlhNet(), 2);
}

TEST(DirectMlKdaParity, MatchesBlasOnBatchOfEight) {
  CompareBackendsBatch(MakeKdaMlhNet(), 8);
}

TEST(DirectMlKdaParity, MatchesBlasOnBatchOfEightRealisticDims) {
  CompareBackendsBatch(MakeNetWithDims(RealisticDims(), 8001), 8);
}

TEST(DirectMlKdaParity, MatchesBlasOnGatedEmbeddingNet) {
  CompareBackends(MakeGatedNet(NetDims(), 6001));
}

TEST(DirectMlKdaParity, MatchesBlasOnGatedEmbeddingRealisticDims) {
  CompareBackends(MakeGatedNet(RealisticDims(), 6002));
}

// A policy encoder wide enough that its own scratch requirement dominates.
//
// The shape matters: the body must stay SMALL. A realistic KDA body pushes
// scratch_elems to ~2720, which would swamp a policy encoder's 3 * d_model and
// prove nothing. With the default tiny body the old sizing offered 320
// elements per token -- max(2 * ip2_pol_b + 64, 2 * ip_pol_b, KDA terms) --
// against the 3 * d_model = 384 the encoder's q/k/v actually need.
//
// 128 is the smallest power-of-two width that trips it: at 64 the encoder
// needs 3 * 64 = 192, which still fits under the ~196 the KDA body reserves
// anyway, so a narrower net would pass with or without the fix and prove
// nothing. Verified as a real regression test by disabling the policy fold in
// network_directml.cc: the guard then fires with "needs 25165824 bytes of
// scratch for q/k/v but only 20971520 are sized (policy d_model 128)", and
// both tests below pass with it restored.
pblczero::Net MakeWidePolicyEncoderNet(int pol_d_model) {
  NetDims d;
  // The encoder's embedding and d_model must match: every encoder in this
  // backend projects q/k/v from the embedding width, and DirectML rejects
  // the graph outright (CreateOperator throws) when they differ.
  d.pol_emb = pol_d_model;
  d.pol_dmodel = pol_d_model;
  std::mt19937 rng(9100 + pol_d_model);
  pblczero::Net file = MakeNetWithDims(d, 9100 + pol_d_model);
  FillPolicyEncoder(&file, rng, d.pol_emb, pol_d_model, /*heads=*/8,
                    /*dff=*/64);
  return file;
}

TEST(DirectMlKdaParity, MatchesBlasOnWidePolicyEncoderNet) {
  CompareBackends(MakeWidePolicyEncoderNet(128));
}

TEST(DirectMlKdaParity, MatchesBlasOnWidePolicyEncoderBatch) {
  CompareBackendsBatch(MakeWidePolicyEncoderNet(128), 4);
}

// The BLAS reference sizes buffer1/buffer2/buffer3 from max_channels, a BODY
// quantity, but the attention policy head writes into all three with its OWN
// widths. Either width crossing max_channels overruns the allocation and
// smashes the process heap; the fault then surfaces at some later unrelated
// allocation, pointing nowhere near the cause. This is a defect in the
// REFERENCE, so it takes the whole parity harness down with it rather than
// failing the backend under test.
//
// max_channels for these synthetic nets is the input width, kInputPlanes(112)
// + 64 positional-encoding channels = 176, which is why 192 is the trip width
// and 128 the safe one.
//
// The two hazards are separated deliberately. MakeWidePolicyEncoderNet above
// ties pol_emb == pol_dmodel, because an encoder projects q/k/v from the
// embedding width and DirectML rejects the graph when they differ -- but that
// coupling would leave it unknown WHICH of the two writes overflowed. These
// nets carry no policy encoder, so the widths are free, and each one isolates
// a single write:
//
//   embedding-wide: exercises the policy-embedding Forward1D into buffer2
//   d_model-wide:   exercises the Q/K writes into buffer1/buffer3 and the
//                   policy_d_model strides that index them afterwards
pblczero::Net MakePolicyEmbeddingWiderThanBodyNet() {
  NetDims d;
  d.pol_emb = 192;     // > max_channels 176
  d.pol_dmodel = 128;  // <= max_channels, so only the embedding write trips
  return MakeNetWithDims(d, 9301);
}

pblczero::Net MakePolicyDModelWiderThanBodyNet() {
  NetDims d;
  d.pol_emb = 128;     // <= max_channels, so the embedding write is safe
  d.pol_dmodel = 192;  // > max_channels 176
  return MakeNetWithDims(d, 9302);
}

TEST(DirectMlKdaParity, MatchesBlasOnPolicyEmbeddingWiderThanBody) {
  CompareBackends(MakePolicyEmbeddingWiderThanBodyNet());
}

// Batch > 1 on at least one of the pair: the Q/K path is indexed per sample
// with a policy_d_model stride, so a batch case covers strides the
// single-position case cannot reach.
TEST(DirectMlKdaParity, MatchesBlasOnPolicyDModelWiderThanBodyBatch) {
  CompareBackendsBatch(MakePolicyDModelWiderThanBodyNet(), 4);
}

TEST(DirectMlKdaParity, MatchesBlasOnPolicyDModelWiderThanBody) {
  CompareBackends(MakePolicyDModelWiderThanBodyNet());
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
              GammaVec(rng, d.embedding));
    FillLayer(weights->mutable_ip_emb_ln_betas(),
              RandomVec(rng, d.embedding, 0.05f));
    auto* ffn = weights->mutable_ip_emb_ffn();
    FillLayer(ffn->mutable_dense1_w(),
              GammaVec(rng, d.embedding * d.embedding));
    FillLayer(ffn->mutable_dense1_b(), RandomVec(rng, d.embedding, 0.05f));
    FillLayer(ffn->mutable_dense2_w(),
              GammaVec(rng, d.embedding * d.embedding));
    FillLayer(ffn->mutable_dense2_b(), RandomVec(rng, d.embedding, 0.05f));
    FillLayer(weights->mutable_ip_emb_ffn_ln_gammas(),
              GammaVec(rng, d.embedding));
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
    if (g == "all" || g == "embedding") {
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
    }
    if (g == "all" || g == "encoders") {
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
    }
    // Narrower than "kda": only the two parameters the decay path
    // exponentiates. decay_scale = exp(a_log[head]) turns a trained value
    // into a multiplier directly, so a large trained a_log lands somewhere a
    // near-zero synthetic one never reaches.
    if (g == "decay") {
      for (size_t i = 0; i < w->encoder_size(); ++i) {
        auto* k = w->mutable_encoder(i)->mutable_kda();
        reroll(k->mutable_a_log());
        reroll(k->mutable_dt_bias());
      }
    }
    if (g == "beta") {
      for (size_t i = 0; i < w->encoder_size(); ++i) {
        auto* k = w->mutable_encoder(i)->mutable_kda();
        reroll(k->mutable_beta_w());
        reroll(k->mutable_beta_b());
      }
    }
    if (g == "all" || g == "kda") {
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
    }
    if (g == "all" || g == "heads") {
      auto* ph = w->mutable_policy_heads();
      reroll(ph->mutable_ip_pol_w());
      reroll(ph->mutable_ip_pol_b());
      auto* v = ph->mutable_vanilla();
      reroll(v->mutable_ip2_pol_w());
      reroll(v->mutable_ip2_pol_b());
      reroll(v->mutable_ip3_pol_w());
      reroll(v->mutable_ip3_pol_b());
      reroll(v->mutable_ip4_pol_w());
      // The value and moves-left heads, which this group used to miss.
      auto* vh = w->mutable_value_heads()->mutable_winner();
      reroll(vh->mutable_ip_val_w());
      reroll(vh->mutable_ip_val_b());
      reroll(vh->mutable_ip1_val_w());
      reroll(vh->mutable_ip1_val_b());
      reroll(vh->mutable_ip2_val_w());
      reroll(vh->mutable_ip2_val_b());
      reroll(w->mutable_ip_mov_w());
      reroll(w->mutable_ip_mov_b());
      reroll(w->mutable_ip1_mov_w());
      reroll(w->mutable_ip1_mov_b());
      reroll(w->mutable_ip2_mov_w());
      reroll(w->mutable_ip2_mov_b());
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
            GammaVec(rng, d.embedding));
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
            GammaVec(rng, d.embedding));
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
