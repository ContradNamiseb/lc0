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
#include "neural/network_legacy.h"
#include "utils/exception.h"
#include "utils/files.h"
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
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

// Behavioural switches for the KDA encoder, kept SEPARATE from NetDims.
// Dimensions and feature flags are different axes: a width is a number to
// vary, a flag is a branch that has to be taken both ways somewhere in the
// suite. Keeping them in one struct is how four of these came to be
// hardcoded to a single value in every net in this file.
//
// The defaults reproduce exactly what every existing net already used, so
// introducing this changes no existing test.
struct KdaFeatures {
  bool output_gate = true;
  bool output_rms_norm = true;
  bool local_conv = false;
  bool qkv_silu = true;
};

void FillKdaEncoder(pblczero::Net* net, std::mt19937& rng, const NetDims& d,
                    const KdaFeatures& feat = KdaFeatures()) {
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
  kda->set_output_gate(feat.output_gate);
  kda->set_output_rms_norm(feat.output_rms_norm);
  kda->set_local_conv(feat.local_conv);
  kda->set_qkv_silu(feat.qkv_silu);
  // Setting the flag alone would prove nothing: BOTH backends additionally
  // guard execution on the weights being non-empty (network_blas.cc:323
  // requires local_conv && !local_conv_w.empty(); the DirectML layer uploads
  // local_conv_w/b at layers.cc:1718-1719 before dispatching). A flag-only
  // net would run the same code as local_conv=false and pass trivially.
  //
  // local_conv_w is [emb, 1, 3, 3] flattened, read as
  // local_conv_w[c * 9 + kr * 3 + kf] (network_blas.cc:339). The values must
  // vary ACROSS THE KERNEL, not merely across channels: a spatially uniform
  // 3x3 kernel is a scaled blur whose output barely depends on the row/file
  // indexing, and the board-edge clamping and per-sample batch_base in the
  // HLSL path are exactly what these tests exist to check.
  if (feat.local_conv) {
    std::vector<float> w(static_cast<size_t>(d.embedding) * 9);
    for (size_t i = 0; i < w.size(); ++i) {
      const size_t c = i / 9, tap = i % 9;
      // Deliberately STRONG and sign-varying, and fixture-specific: these
      // values exist only on this net. A gentle kernel was measurably
      // insufficient -- bypassing the dispatch moved q by 3% relative but
      // only 1.07e-04 absolute, under the suite's 2e-4 absolute floor, so
      // the tests exercised the feature without protecting it.
      //
      // Uniform scaling would not have fixed that: the KDA mixer's output
      // RMS norm is scale-invariant, so inflating the kernel uniformly is
      // normalised straight back out. What has to change is the spatial
      // PATTERN, so that removing the convolution alters the direction of
      // the mixer input rather than only its length. Hence taps that change
      // sign across the 3x3 and a channel term that shifts the pattern
      // rather than merely rescaling it.
      const float tap_term = 0.9f * (static_cast<float>(tap % 3) - 1.0f);
      const float chan_term = 0.35f * (static_cast<float>(c % 5) - 2.0f) *
                              (static_cast<float>(tap / 3) - 1.0f);
      w[i] = tap_term + chan_term;
    }
    FillLayer(kda->mutable_local_conv_w(), w);
    // Non-zero bias too: network_blas.cc:343 guards the bias separately, so
    // leaving it empty would keep that branch untested even with weights.
    std::vector<float> b(static_cast<size_t>(d.embedding));
    for (size_t i = 0; i < b.size(); ++i) {
      b[i] = 0.25f * (static_cast<float>(i % 7) - 3.0f);
    }
    FillLayer(kda->mutable_local_conv_b(), b);
  }
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

// head_scale is a FIXTURE-SPECIFIC amplifier, default 1.0 so every existing
// net is untouched. It scales only the three projections that set OUTPUT
// magnitude -- the policy Q/K and the WDL logits.
//
// It was introduced under the OLD additive bar, where it was necessary for
// detectability: at this fixture's unamplified policy peak of 0.0374 that bar
// was 2.019e-4, i.e. 0.54% relative, and the measured local-conv control
// defect of 1.278e-4 (0.34% relative) fell under it and passed.
//
// Under the calibrated rule it is NO LONGER required for detectability. The
// floor at that same peak is 5e-5, i.e. 0.134% relative, and the same 0.34%
// control defect is rejected without any amplification. What the amplifier
// still buys is margin: it moves the fixture from a 2.6x rejection to 86x,
// which is worth keeping so the test is not sitting just over the line. It
// does NOT change the bar.
void FillPolicyAndValueHeads(pblczero::Net* net, std::mt19937& rng,
                             const NetDims& d, bool moves_left, int mlh,
                             float head_scale = 1.0f) {
  auto* weights = net->mutable_weights();
  auto* ph = weights->mutable_policy_heads();
  FillLayer(ph->mutable_ip_pol_w(),
            RandomVec(rng, d.pol_emb * d.embedding, 0.1f));
  FillLayer(ph->mutable_ip_pol_b(), RandomVec(rng, d.pol_emb, 0.05f));
  auto* vanilla = ph->mutable_vanilla();
  FillLayer(vanilla->mutable_ip2_pol_w(),
            RandomVec(rng, d.pol_dmodel * d.pol_emb, 0.1f * head_scale));
  FillLayer(vanilla->mutable_ip2_pol_b(),
            RandomVec(rng, d.pol_dmodel, 0.05f));
  FillLayer(vanilla->mutable_ip3_pol_w(),
            RandomVec(rng, d.pol_dmodel * d.pol_emb, 0.1f * head_scale));
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
            RandomVec(rng, 3 * d.val_channels, 0.05f * head_scale));
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

// Comparison bounds, calibrated by MEASUREMENT rather than by the size of
// bugs already caught. The previous rationale claimed real divergences run
// ~1e-1+; that is false by three orders of magnitude, and believing it cost
// real coverage. Under the old additive rule (2e-4 + 5e-5*|ref|), removing
// the KDA local convolution outright moved policy by 1.278e-4 on a net whose
// policy peaks at 0.037 -- 3.4e-3 relative -- and PASSED. The feature shipped
// with a test that exercised it and could not fail.
//
// MEASURED NOISE, 2026-09-05, this machine:
//   synthetic single-position, 12 fixtures, blas~eigen and dml~blas:
//     q, d and policy absolute max 2.98e-08. That is exactly one float32 ULP
//     at ~0.33, the magnitude the WDL probabilities sit at; the floor is one
//     last-bit rounding difference, not a distribution with a tail.
//   synthetic batch, 5 fixtures, every sample: same 2.98e-08 ceiling, and
//     m max 9.3e-10.
//   real nets kda-native-935532 and kda-native-825532, dml~blas:
//     q 7.86781e-06, d 8.00192e-06, m 1.421e-05 relative, policy 9.408e-06
//     relative. Real nets are two orders looser than synthetic ones, which
//     is why a bound calibrated only on synthetic fixtures would reject
//     healthy real nets.
//   DirectML is bit-identical run to run over the complete 1858-element
//     policy vector, single and batch, so none of the above is
//     nondeterminism. (That rules out nondeterminism as a source; it does
//     not by itself prove every remaining delta is accumulation order.)
//
// SHAPE OF THE BOUND. q and d are bounded in [-1,1] and get a pure ABSOLUTE
// bar. A relative bar is meaningless for them: one measured batch sample has
// q = 0.00015, where a single ULP is already 2e-4 relative. m and policy are
// unbounded -- m is a ply count reaching 34.6, policy logits reach 6.5 -- so
// their bars scale with the reference's own magnitude.
//
// max(), NOT addition. The old rule added the absolute and relative terms,
// which makes it looser than a pure relative bar at every scale above 4: at
// m=17.4553 it allowed 1.073e-3 where 5e-5*|ref| allows 8.728e-4. Taking the
// max is uniformly tighter and still never falls below the absolute floor.
//
// MARGINS over the measured maxima: q 6.4x, d 6.2x, policy 5.3x, m 3.5x.
// The m margin is the WEAKEST of the four, it rests on two real nets alone,
// and it has no rejection evidence behind it at all -- every synthetic
// fixture here returns m=0, so no control defect has ever moved m. It is
// justified by noise headroom only, and it is the first number to revisit
// when more real nets are available. Nothing here is proof about other
// devices or drivers; it is this machine, today.
constexpr float kAbsTol = 5e-5f;
constexpr float kScaleTol = 5e-5f;

// q and d: bounded in [-1,1], so absolute only. Deliberately takes no
// reference value -- passing one would invite reintroducing relative scaling.
inline float BoundedOutputBound() { return kAbsTol; }

// m and policy: unbounded, so the bar scales with the reference's own
// magnitude and never drops below the absolute floor.
inline float ScaledOutputBound(float reference_scale) {
  return std::max(kAbsTol, kScaleTol * std::fabs(reference_scale));
}

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

  for (int n = 0; n < batch; ++n) {
    EXPECT_NEAR(dml[n].q, reference[n].q, BoundedOutputBound())
        << "sample " << n << ": WDL value Q diverges";
    EXPECT_NEAR(dml[n].d, reference[n].d, BoundedOutputBound())
        << "sample " << n << ": WDL draw probability diverges";
    EXPECT_NEAR(dml[n].m, reference[n].m, ScaledOutputBound(reference[n].m))
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
    EXPECT_LT(worst, ScaledOutputBound(ref_absmax))
        << "sample " << n << ": policy diverges (worst move " << worst_move
        << ", diff " << worst << ")";

    // Decision parity per SAMPLE, not only for sample 0. A wrong per-sample
    // stride can leave sample 0 correct and flip the move played in every
    // other position of the batch, which is exactly how the policy_finalize
    // ROW_STRIDE bug presented.
    int dml_best = 0, ref_best = 0;
    for (int i = 1; i < 1858; ++i) {
      if (dml[n].policy[i] > dml[n].policy[dml_best]) dml_best = i;
      if (reference[n].policy[i] > reference[n].policy[ref_best]) ref_best = i;
    }
    EXPECT_EQ(dml_best, ref_best)
        << "sample " << n << ": policy argmax differs, directml picks "
        << dml_best << " (logit " << dml[n].policy[dml_best] << "), blas picks "
        << ref_best << " (logit " << reference[n].policy[ref_best] << ")";
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

  // Bounds and their calibration live with the constants above.
  EXPECT_NEAR(dml.q, reference.q, BoundedOutputBound())
      << "WDL value Q diverges between directml and blas";
  EXPECT_NEAR(dml.d, reference.d, BoundedOutputBound())
      << "WDL draw probability diverges between directml and blas";
  EXPECT_NEAR(dml.m, reference.m, ScaledOutputBound(reference.m))
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
    // d was missing here, which meant an output the suite ASSERTS on was
    // never reported by its own diagnostic; the tolerance study had to infer
    // it by subtracting rounded text.
    const float ddiff = std::fabs(dml.d - reference.d);
    CERR << std::scientific << std::setprecision(3)
         << "[diff] q " << qdiff << " (rel " << rel(qdiff, reference.q)
         << ")  d " << ddiff << " (rel " << rel(ddiff, reference.d)
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
  EXPECT_LT(worst, ScaledOutputBound(ref_absmax))
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
pblczero::Net MakeNetWithDims(const NetDims& d, unsigned seed,
                              const KdaFeatures& feat = KdaFeatures(),
                              float head_scale = 1.0f) {
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
  FillKdaEncoder(&file, rng, d, feat);
  FillPolicyAndValueHeads(&file, rng, d, true, mlh, head_scale);
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

// The local 3x3 depthwise convolution before the KDA mixer (added in b3bc1c5)
// had ZERO parity coverage in either direction: every net in this suite
// hardcoded local_conv=false, so the feature was live in production and never
// once compared against BLAS. Same shape as the input-gating bug, whose guard
// was likewise false in all 36 tests of the day.
//
// It is its own handwritten HLSL path with board-edge clamping and an explicit
// per-sample batch_base, so a batch > 1 case is required as well: at batch 1
// every batch_base is 0 and a wrong sample stride cannot be seen.
pblczero::Net MakeLocalConvNet(unsigned seed) {
  KdaFeatures feat;
  feat.local_conv = true;
  // Measured, not guessed: at the default scale this net's policy peaks at
  // 0.037 and q at 0.0036, so bypassing the convolution moved policy by only
  // 1.9e-4 -- under the 2e-4 absolute floor, and the test passed while the
  // feature was demonstrably absent. This amplifier lifts the outputs into
  // the range the bar discriminates in.
  constexpr float kLocalConvHeadScale = 12.0f;
  return MakeNetWithDims(NetDims(), seed, feat, kLocalConvHeadScale);
}

TEST(DirectMlKdaParity, MatchesBlasOnLocalConvNet) {
  CompareBackends(MakeLocalConvNet(9401));
}

TEST(DirectMlKdaParity, MatchesBlasOnLocalConvBatch) {
  CompareBackendsBatch(MakeLocalConvNet(9402), 4);
}

// The other three KDA feature switches were pinned to a single value in every
// net in this file: output_gate and output_rms_norm always true, qkv_silu
// always true. Their disabled branches were therefore dead under test while
// live in production -- the same one-sided-coverage shape as the input-gating
// bug and the local convolution, found by auditing the switches rather than by
// anything failing.
//
// One flag per net, deliberately. A single all-off net would be cheaper but
// worse: combined toggles can cancel or mask an isolated branch defect, and
// when such a net failed it would not say which branch broke.
//
// Dimensions are RealisticDims() throughout, and every feature setting except
// the named toggle matches the existing passing MatchesBlasOnRealisticDimsNet.
// The WEIGHTS differ from it: that fixture seeds 2024 and these seed 9501-9504,
// which is deliberate seed diversity rather than an oversight. So these are not
// a controlled A/B against that fixture, and nothing here should be read as
// isolating the flag's effect on OUTPUT. What each parity comparison isolates
// is the flag's effect on AGREEMENT: BLAS and DirectML are handed byte-identical
// weights, so any divergence is an implementation difference in the branch the
// toggle selects, whatever the weights happen to be.
pblczero::Net MakeOutputGateDisabledNet(unsigned seed) {
  KdaFeatures feat;
  feat.output_gate = false;
  return MakeNetWithDims(RealisticDims(), seed, feat);
}

pblczero::Net MakeOutputRmsNormDisabledNet(unsigned seed) {
  KdaFeatures feat;
  feat.output_rms_norm = false;
  return MakeNetWithDims(RealisticDims(), seed, feat);
}

pblczero::Net MakeQkvSiluDisabledNet(unsigned seed) {
  KdaFeatures feat;
  feat.qkv_silu = false;
  return MakeNetWithDims(RealisticDims(), seed, feat);
}

TEST(DirectMlKdaParity, MatchesBlasOnOutputGateDisabled) {
  CompareBackends(MakeOutputGateDisabledNet(9501));
}

TEST(DirectMlKdaParity, MatchesBlasOnOutputRmsNormDisabled) {
  CompareBackends(MakeOutputRmsNormDisabledNet(9502));
}

TEST(DirectMlKdaParity, MatchesBlasOnQkvSiluDisabled) {
  CompareBackends(MakeQkvSiluDisabledNet(9503));
}

// Batch > 1 on one of the three. The disabled branches sit inside the KDA
// mixer, which indexes per sample, so a batch case covers strides the
// single-position tests cannot reach. Kept on one fixture rather than all
// three: the per-sample indexing is shared, so repeating it would add runtime
// without adding coverage.
TEST(DirectMlKdaParity, MatchesBlasOnOutputGateDisabledBatch) {
  CompareBackendsBatch(MakeOutputGateDisabledNet(9504), 4);
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

// ---------------------------------------------------------------------------
// Embedding LayerNorm weight validation. LOAD-TIME, no GPU.
//
// The validation is gated on CONSUMPTION, not on tensor presence, and these
// tests exercise the same format-aware entry point production uses:
// ValidateEmbeddingNormWeights(weights, consumes_pe_dense_embedding), called
// from network_blas.cc where is_pe_dense_embedding_ is derived and from
// network_directml.cc where is_pe_dense is.
//
// Presence is the wrong predicate in both directions, which is why an earlier
// version of this guard was wrong: both LayerNorm calls sit inside
// `if (is_pe_dense_embedding_)` and neither checks that the tensors exist, so
// a CONSUMING net with both absent still reached the unguarded reads, while a
// NON-consuming net carrying them would have been rejected over tensors
// nobody touches. Both cases are tested below.
namespace {

// A net on the consuming path: dense positional-encoding embedding, so it
// carries ip_emb_preproc and an embedding FFN, whose dense2_b width is what
// the SECOND LayerNorm normalises at -- a different quantity from ip_emb_b,
// which is what the first uses. Callers choose each tensor's length; 0 omits.
pblczero::Net MakeConsumingNet(int ln_gammas, int ln_betas, int ffn_gammas,
                               int ffn_betas, int ffn_out = -1) {
  NetDims d;
  pblczero::Net file = MakeNetWithDims(d, 7701);
  std::mt19937 rng(7702);
  // MakeNetWithDims sets INPUT_EMBEDDING_NONE, which is NOT the consuming
  // path: the backends gate the embedding LayerNorm on
  // input_embedding() == INPUT_EMBEDDING_PE_DENSE. Without this the net is
  // correctly skipped by the validation and loads fine, which made the
  // subprocess rejection test fail with the guard working exactly as
  // designed and the fixture simply not being what it claimed.
  file.mutable_format()->mutable_network_format()->set_input_embedding(
      pblczero::NetworkFormat::INPUT_EMBEDDING_PE_DENSE);
  auto* w = file.mutable_weights();
  FillLayer(w->mutable_ip_emb_preproc_w(),
            RandomVec(rng, 64 * d.embedding * 64 * 12, 0.05f));
  FillLayer(w->mutable_ip_emb_preproc_b(),
            RandomVec(rng, 64 * d.embedding, 0.05f));
  auto* ffn = w->mutable_ip_emb_ffn();
  FillLayer(ffn->mutable_dense1_w(), GammaVec(rng, d.embedding * d.embedding));
  FillLayer(ffn->mutable_dense1_b(), RandomVec(rng, d.embedding, 0.05f));
  // ffn_out defaults to the embedding width, which is what the FFN LayerNorm's
  // skip connection requires; tests override it to exercise that check.
  const int out = ffn_out < 0 ? d.embedding : ffn_out;
  FillLayer(ffn->mutable_dense2_w(), GammaVec(rng, d.embedding * out));
  FillLayer(ffn->mutable_dense2_b(), RandomVec(rng, out, 0.05f));
  if (ln_gammas > 0) {
    FillLayer(w->mutable_ip_emb_ln_gammas(), GammaVec(rng, ln_gammas));
  }
  if (ln_betas > 0) {
    FillLayer(w->mutable_ip_emb_ln_betas(), RandomVec(rng, ln_betas, 0.05f));
  }
  if (ffn_gammas > 0) {
    FillLayer(w->mutable_ip_emb_ffn_ln_gammas(), GammaVec(rng, ffn_gammas));
  }
  if (ffn_betas > 0) {
    FillLayer(w->mutable_ip_emb_ffn_ln_betas(),
              RandomVec(rng, ffn_betas, 0.05f));
  }
  return file;
}

// Requires the rejection AND that the message names the offending tensor. An
// earlier version of these tests asserted only that something was thrown, and
// every case "passed" by throwing "Could not find valid policy head weights"
// -- the right verdict for the wrong reason. A test that cannot say WHY it
// rejected is not evidence.
void ExpectRejectedNaming(const pblczero::Net& net, const char* tensor) {
  const MultiHeadWeights decoded{net.weights()};
  try {
    ValidateEmbeddingNormWeights(decoded, /*consumes=*/true);
    ADD_FAILURE() << "expected validation to throw, naming " << tensor;
  } catch (const Exception& e) {
    EXPECT_NE(std::string(e.what()).find(tensor), std::string::npos)
        << "threw, but not about " << tensor << ": " << e.what();
  }
}

constexpr int kEmb = 32;  // NetDims::embedding, what ip_emb_b is sized to.

}  // namespace

// The case that motivated this: gammas present, betas absent. Exactly
// kda-hybrid-512x8-transformer-3000, which faulted at 0xC0000005.
TEST(EmbeddingLnValidation, RejectsAbsentBeta) {
  ExpectRejectedNaming(MakeConsumingNet(kEmb, 0, kEmb, kEmb),
                       "ip_emb_ln_betas");
}

TEST(EmbeddingLnValidation, RejectsAbsentFfnBeta) {
  ExpectRejectedNaming(MakeConsumingNet(kEmb, kEmb, kEmb, 0),
                       "ip_emb_ffn_ln_betas");
}

// BOTH absent while the format says the path is taken. This is the hole the
// presence-based version of the guard left open: it returned early and let
// the net through to the unguarded reads.
TEST(EmbeddingLnValidation, RejectsBothAbsentOnConsumingPath) {
  ExpectRejectedNaming(MakeConsumingNet(0, 0, kEmb, kEmb),
                       "ip_emb_ln_gammas");
}

TEST(EmbeddingLnValidation, RejectsBothAbsentFfnOnConsumingPath) {
  ExpectRejectedNaming(MakeConsumingNet(kEmb, kEmb, 0, 0),
                       "ip_emb_ffn_ln_gammas");
}

// Present but wrong length, separate from absence: the bound is the width the
// consumer normalises at, not agreement between the two tensors.
TEST(EmbeddingLnValidation, RejectsWrongLengthBeta) {
  ExpectRejectedNaming(MakeConsumingNet(kEmb, kEmb / 2, kEmb, kEmb),
                       "ip_emb_ln_betas");
}

TEST(EmbeddingLnValidation, RejectsWrongLengthGamma) {
  ExpectRejectedNaming(MakeConsumingNet(kEmb / 2, kEmb, kEmb, kEmb),
                       "ip_emb_ln_gammas");
}

TEST(EmbeddingLnValidation, RejectsWrongLengthFfnBeta) {
  ExpectRejectedNaming(MakeConsumingNet(kEmb, kEmb, kEmb, kEmb / 2),
                       "ip_emb_ffn_ln_betas");
}

// Everything internally consistent at the WRONG width: the FFN output and both
// its norm tensors agree with each other at 16 while the embedding is 32.
// Sizing each tensor to its own consumer accepts this; only the residual
// check catches it. The FFN LayerNorm adds a skip from the embedding output,
// so the two strides must agree.
TEST(EmbeddingLnValidation, RejectsFfnWidthDisagreeingWithEmbedding) {
  ExpectRejectedNaming(
      MakeConsumingNet(kEmb, kEmb, kEmb / 2, kEmb / 2, kEmb / 2),
      "ip_emb_ffn.dense2_b");
}

TEST(EmbeddingLnValidation, AcceptsCompleteTensorsOnConsumingPath) {
  const pblczero::Net net = MakeConsumingNet(kEmb, kEmb, kEmb, kEmb);
  const MultiHeadWeights decoded{net.weights()};
  EXPECT_NO_THROW(ValidateEmbeddingNormWeights(decoded, /*consumes=*/true));
}

// Non-consuming architectures are preserved by FORMAT, not by tensor absence.
// Every other fixture in this file is one of these, so over-broad validation
// here would take the whole suite down with it.
TEST(EmbeddingLnValidation, AcceptsNonConsumingArchitecture) {
  const pblczero::Net net = MakeNetWithDims(NetDims(), 7703);
  const MultiHeadWeights decoded{net.weights()};
  EXPECT_NO_THROW(ValidateEmbeddingNormWeights(decoded, /*consumes=*/false));
}

// And a non-consuming net that happens to CARRY the tensors must also load:
// nothing reads them, so their sizes are not this validation's business.
// The format gate must be inert for every malformed shape, not just the one
// tried above. A guard that rejected any of these on a non-consuming net would
// break the other 60-odd fixtures in this file, all of which are non-consuming
// -- and it would do so for tensors nothing in the execution path reads.
TEST(EmbeddingLnValidation, NonConsumingIgnoresEveryMalformedShape) {
  struct Case {
    const char* name;
    int ln_g, ln_b, ffn_g, ffn_b, ffn_out;
  };
  const Case cases[] = {
      {"gammas without betas", kEmb, 0, kEmb, kEmb, -1},
      {"ffn gammas without betas", kEmb, kEmb, kEmb, 0, -1},
      {"both absent", 0, 0, kEmb, kEmb, -1},
      {"beta wrong length", kEmb, kEmb / 2, kEmb, kEmb, -1},
      {"gamma wrong length", kEmb / 2, kEmb, kEmb, kEmb, -1},
      {"ffn width disagrees with embedding", kEmb, kEmb, kEmb / 2, kEmb / 2,
       kEmb / 2},
  };
  for (const Case& c : cases) {
    const pblczero::Net net =
        MakeConsumingNet(c.ln_g, c.ln_b, c.ffn_g, c.ffn_b, c.ffn_out);
    const MultiHeadWeights decoded{net.weights()};
    EXPECT_NO_THROW(ValidateEmbeddingNormWeights(decoded, /*consumes=*/false))
        << "non-consuming net rejected over " << c.name
        << ", which nothing on that path reads";
  }
}

TEST(EmbeddingLnValidation, AcceptsNonConsumingWithTensorsPresent) {
  const pblczero::Net net = MakeConsumingNet(kEmb / 2, kEmb, 0, 0);
  const MultiHeadWeights decoded{net.weights()};
  EXPECT_NO_THROW(ValidateEmbeddingNormWeights(decoded, /*consumes=*/false));
}


// ---------------------------------------------------------------------------
// `lc0 benchmark` exit status on load failure.
//
// The benchmark printed the failure and returned normally, leaving the
// process status at 0, so a net that could not be loaded looked like a
// successful benchmark to anything checking exit codes -- including any CI
// step that runs one. Found while validating the embedding normalisation
// weights: the rejection message appeared and the shell still reported
// success.
//
// These drive the real binary as a subprocess, because the property under
// test IS the process exit status and nothing in-process can observe it.
namespace {

// Directory of this test binary, captured in main(). lc0 sits beside it in
// the build tree.
std::string& TestBinaryDir() {
  static std::string dir;
  return dir;
}

// A private directory for one test's fixtures and captured output, removed
// when the test ends. Fixed names in the shared TEMP would let concurrent
// test processes overwrite each other's inputs and logs, and the resulting
// verdicts would be about whichever process wrote last.
class ScopedTempDir {
 public:
  ScopedTempDir() {
    static std::atomic<unsigned> counter{0};
    const auto stamp = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    path_ = std::filesystem::temp_directory_path() /
            ("lc0_exit_test_" + std::to_string(stamp) + "_" +
             std::to_string(counter++));
    std::filesystem::create_directories(path_);
  }
  ~ScopedTempDir() {
    std::error_code ec;  // best effort; a leaked temp dir must not fail a test
    std::filesystem::remove_all(path_, ec);
  }
  ScopedTempDir(const ScopedTempDir&) = delete;
  ScopedTempDir& operator=(const ScopedTempDir&) = delete;

  std::string File(const char* name) const {
    return (path_ / name).string();
  }

 private:
  std::filesystem::path path_;
};

// Returns lc0's exit status, capturing its output so a caller can assert WHY
// it failed. Status alone is not enough: an early version of these fixtures
// omitted the weight magic, so the rejection cases "passed" by failing on
// "bad header" without ever reaching the validation they claim to test.
// Substring proving the binary actually started; without checking it, a
// missing or unfindable lc0.exe would make cmd return nonzero and every
// rejection test would pass without lc0 ever running.
constexpr const char* kLc0Started = "Loading weights file from";

int RunLc0(const ScopedTempDir& dir, const std::string& args,
           std::string* output = nullptr) {
  const std::string log = dir.File("output.txt");
  // cmd.exe strips the first and last quote of a command that begins with
  // one, which silently broke the redirect and left the captured output
  // empty. Wrapping the whole command in an additional pair is the documented
  // workaround.
  const std::string cmd = "\"\"" + TestBinaryDir() + "lc0.exe\" " + args +
                          " > \"" + log + "\" 2>&1\"";
  const int status = std::system(cmd.c_str());
  if (output) {
    std::ifstream in(log);
    *output = std::string((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  }
  return status;
}

// Writes a net where a serialized proto is all that is needed: zlib's gzread
// reads uncompressed files transparently, so the loader accepts this without
// the test having to compress anything.
std::string WriteNet(const ScopedTempDir& dir, const pblczero::Net& net,
                     const char* name) {
  const std::string path = dir.File(name);
  // The loader requires the weight magic (loader.cc:57, kWeightMagic 0x1c0).
  // Without it a net is rejected as "bad header" before anything else is
  // examined, which is how an early version of these tests passed for the
  // wrong reason.
  pblczero::Net loadable = net;
  loadable.set_magic(0x1c0);
  // loader.cc:184-187 rejects any encoding but LINEAR16 below version 0.33.0,
  // which is what FillLayer already writes (min/max plus quantised params).
  loadable.mutable_format()->set_weights_encoding(pblczero::Format::LINEAR16);
  WriteStringToFile(path, loadable.OutputAsString());
  return path;
}

}  // namespace

// A net this build rejects must fail the process, not just print. Uses the
// embedding-normalisation rejection because it is deterministic and needs no
// external file.
TEST(BenchmarkExitStatus, NonzeroOnRejectedNet) {
  if (TestBinaryDir().empty()) GTEST_SKIP() << "binary dir unknown";
  const ScopedTempDir dir;
  const std::string path =
      WriteNet(dir, MakeConsumingNet(kEmb, 0, kEmb, kEmb), "rejected.pb");
  std::string output;
  EXPECT_NE(RunLc0(dir, "benchmark --backend=blas --weights=\"" + path +
                       "\" --nodes=1 --num-positions=1",
                   &output),
            0)
      << "a rejected net must exit nonzero";
  // And for the right reason: it must be the validation that rejected it, not
  // a malformed fixture failing earlier.
  EXPECT_NE(output.find(kLc0Started), std::string::npos)
      << "lc0 did not run at all: " << output;
  EXPECT_NE(output.find("ip_emb_ln_betas"), std::string::npos)
      << "expected the embedding-normalisation rejection, got: " << output;
}

TEST(BenchmarkExitStatus, NonzeroOnMissingWeightsFile) {
  if (TestBinaryDir().empty()) GTEST_SKIP() << "binary dir unknown";
  const ScopedTempDir dir;
  // Guaranteed missing because the directory is fresh and this file is never
  // written into it.
  const std::string missing = dir.File("no_such_net.pb");
  std::string output;
  EXPECT_NE(RunLc0(dir,
                   "benchmark --backend=blas --weights=\"" + missing +
                       "\" --nodes=1 --num-positions=1",
                   &output),
            0)
      << "a missing weights file must exit nonzero";
  EXPECT_NE(output.find(kLc0Started), std::string::npos)
      << "lc0 did not run at all; nonzero here would prove nothing: " << output;
}

// The converse, so the fix cannot be "always fail": a net that loads and runs
// must still report success.
TEST(BenchmarkExitStatus, ZeroOnSuccessfulBenchmark) {
  if (TestBinaryDir().empty()) GTEST_SKIP() << "binary dir unknown";
  const ScopedTempDir dir;
  const std::string path =
      WriteNet(dir, MakeNetWithDims(NetDims(), 7801), "valid.pb");
  std::string output;
  EXPECT_EQ(RunLc0(dir, "benchmark --backend=blas --weights=\"" + path +
                       "\" --nodes=1 --num-positions=1",
                   &output),
            0)
      << "a benchmark that loads and runs must exit zero: " << output;
}


// The same three cases for backendbench. It is the tool an automated sweep
// drives, so a load failure that exits zero there is worse than in benchmark:
// the caller reads exit codes to decide which rows are real measurements, and
// a configuration that never loaded would be recorded as one that did.
TEST(BackendBenchExitStatus, NonzeroOnRejectedNet) {
  if (TestBinaryDir().empty()) GTEST_SKIP() << "binary dir unknown";
  const ScopedTempDir dir;
  const std::string path =
      WriteNet(dir, MakeConsumingNet(kEmb, 0, kEmb, kEmb), "rejected.pb");
  std::string output;
  EXPECT_NE(RunLc0(dir, "backendbench --backend=blas --weights=\"" + path +
                       "\" --batches=1 --start-batch-size=1 --max-batch-size=1",
                   &output),
            0)
      << "a rejected net must exit nonzero";
  EXPECT_NE(output.find(kLc0Started), std::string::npos)
      << "lc0 did not run at all: " << output;
  EXPECT_NE(output.find("ip_emb_ln_betas"), std::string::npos)
      << "expected the embedding-normalisation rejection, got: " << output;
}

TEST(BackendBenchExitStatus, NonzeroOnMissingWeightsFile) {
  if (TestBinaryDir().empty()) GTEST_SKIP() << "binary dir unknown";
  const ScopedTempDir dir;
  const std::string missing = dir.File("no_such_net.pb");
  std::string output;
  EXPECT_NE(RunLc0(dir, "backendbench --backend=blas --weights=\"" + missing +
                       "\" --batches=1 --start-batch-size=1 --max-batch-size=1",
                   &output),
            0)
      << "a missing weights file must exit nonzero";
  EXPECT_NE(output.find(kLc0Started), std::string::npos)
      << "lc0 did not run at all; nonzero here would prove nothing: " << output;
}

TEST(BackendBenchExitStatus, ZeroOnSuccessfulRun) {
  if (TestBinaryDir().empty()) GTEST_SKIP() << "binary dir unknown";
  const ScopedTempDir dir;
  const std::string path =
      WriteNet(dir, MakeNetWithDims(NetDims(), 7801), "valid.pb");
  std::string output;
  EXPECT_EQ(RunLc0(dir, "backendbench --backend=blas --weights=\"" + path +
                       "\" --batches=1 --start-batch-size=1 --max-batch-size=1",
                   &output),
            0)
      << "a backendbench run that loads must exit zero: " << output;
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // The subprocess tests need lc0, which sits beside this binary.
  if (argc > 0 && argv[0]) {
    const std::string self = argv[0];
    const size_t slash = self.find_last_of("/\\");
    lczero::TestBinaryDir() =
        slash == std::string::npos ? std::string() : self.substr(0, slash + 1);
  }
  return RUN_ALL_TESTS();
}
