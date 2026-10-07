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
#include <gtest/gtest-spi.h>  // EXPECT_FATAL_FAILURE, for RejectsSelfComparisonBackend

#include <cmath>
#include <cstring>
#include <random>
#include <set>
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
#include "neural/backends/directml/layers.h"
#include "neural/backends/directml/onnx_graph.h"
#include "neural/onnx/converter.h"
#include <thread>

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

// Agora thread 19 #699 Track 1 next knob: scales only the decay path's
// parameters (decay_a_w/decay_a_b/decay_b_w/decay_b_b + a_log/dt_bias),
// independent of and composable with the other *_SCALE knobs via the same
// RandomVec-composition mechanism. beta and every other KDA parameter are
// deliberately left alone -- "decay_a/b + a_log + dt_bias only" per #699's
// scope, since decay_scale = exp(a_log[head]) is the specific
// exponentiating path muse-spark's comment on LC0_TEST_RANDOMIZE=decay
// (below, MatchesBlasOnRealNetFromEnv) already singles out for the same
// reason. Diagnostic-only, test-file-only, no production code.
float DecayScale() {
  if (const char* s = getenv("LC0_TEST_DECAY_SCALE")) {
    return static_cast<float>(atof(s));
  }
  return 1.0f;
}

void FillKdaEncoder(pblczero::Net* net, std::mt19937& rng, const NetDims& d,
                    const KdaFeatures& feat = KdaFeatures()) {
  const int key_depth = d.heads * d.key_dim;
  const int value_depth = d.heads * d.value_dim;
  const float dscale = DecayScale();
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
            RandomVec(rng, d.embedding * d.gate_rank, 0.1f * dscale));
  FillLayer(kda->mutable_decay_a_b(),
            RandomVec(rng, d.gate_rank, 0.05f * dscale));
  FillLayer(kda->mutable_decay_b_w(),
            RandomVec(rng, d.gate_rank * key_depth, 0.1f * dscale));
  FillLayer(kda->mutable_decay_b_b(),
            RandomVec(rng, key_depth, 0.05f * dscale));
  FillLayer(kda->mutable_beta_w(), RandomVec(rng, d.embedding * d.heads, 0.1f));
  FillLayer(kda->mutable_beta_b(), RandomVec(rng, d.heads, 0.05f));
  FillLayer(kda->mutable_a_log(), RandomVec(rng, d.heads, 0.05f * dscale));
  FillLayer(kda->mutable_dt_bias(),
            RandomVec(rng, key_depth, 0.05f * dscale));
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

// Agora thread 19 #641: byte-for-byte MakeKdaMhaNet() above, except with 3
// KDA encoders ahead of the trailing MHA encoder instead of 1 -- same dims,
// same flags, same rng seed/sequence shape (just 2 more FillKdaEncoder
// calls). Depth is the only variable, matching the real trained nets'
// 3xKDA+1xMHA body exactly. Diagnostic-only fixture for the #637/#638/#639/
// #640 batch-contamination triage.
pblczero::Net MakeThreeKdaThenMhaNet() {
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
  FillKdaEncoder(&file, rng, d);
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

// LC0_TEST_BACKEND_OPTS mirrors the CLI's --backend-opts=... string (see
// factory.cc's AddSubdictFromString call) so a test can opt a backend into
// something normally gated off by default -- e.g. directml-fp16's
// allow_broken_fp16 exception gate at network_directml.cc, which an empty
// OptionsDict here would otherwise trip on every call.
void ApplyTestBackendOpts(OptionsDict* options) {
  if (const char* opts = getenv("LC0_TEST_BACKEND_OPTS")) {
    options->AddSubdictFromString(opts);
  }
}

Outputs RunNetwork(const std::string& backend, const WeightsFile& weights,
                   const InputPlanes& planes) {
  OptionsDict options;
  ApplyTestBackendOpts(&options);
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
  ApplyTestBackendOpts(&options);
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

// FP16-calibrated bar (agora thread 19 #501, memory-bank note 2538, user-
// ratified in chat 2026-09-07 "after a parity pass, and only if the nets
// pass parity"). The bar above was calibrated for FP32 accumulation noise
// (eps ~= 1.19e-7); IEEE 754 half has eps ~= 9.77e-4, so holding a genuinely
// fp16 backend to 5e-5 (< 0.05 ULP of its own representation) is not an
// achievable bar regardless of correctness. 5e-3 (0.5%) is the floor for
// the half-precision backends only (directml-fp16, and onnx-dml unless it
// runs with fp16=false, see TestBackendIsHalfPrecision) -- every other
// backend (directml, eigen, blas) keeps the original 5e-5 bar unchanged;
// callers opt in explicitly via the fp16 parameter, default false, so a call
// site that forgets to pass it silently keeps the tighter FP32 bar rather
// than silently loosening.
constexpr float kAbsTolFp16 = 5e-3f;
constexpr float kScaleTolFp16 = 3e-2f;

// q and d: bounded in [-1,1], so absolute only. Deliberately takes no
// reference value -- passing one would invite reintroducing relative scaling.
inline float BoundedOutputBound(bool fp16 = false) {
  return fp16 ? kAbsTolFp16 : kAbsTol;
}

// m and policy: unbounded, so the bar scales with the reference's own
// magnitude and never drops below the absolute floor.
inline float ScaledOutputBound(float reference_scale, bool fp16 = false) {
  const float abs_tol = fp16 ? kAbsTolFp16 : kAbsTol;
  const float scale_tol = fp16 ? kScaleTolFp16 : kScaleTol;
  return std::max(abs_tol, scale_tol * std::fabs(reference_scale));
}

// F5 (agora thread 19 #560, codex-sol's independent review): the worst-diff
// and argmax searches below both rely on `>` comparisons (`diff > worst`,
// `policy[i] > policy[best]`), and a NaN operand makes every such comparison
// false in both directions -- so a NaN in a non-winning policy element is
// silently skipped by the worst-diff search (it can never become the new
// worst), and a NaN at index 0 of either array pins argmax at index 0
// forever (nothing can ever beat it, and it can never lose to anything
// either). Both are real gaps in what "PASSED" has meant from this suite:
// a candidate array containing a stray NaN could still report a passing
// worst-diff and a coincidentally-matching argmax. Asserting finiteness of
// every value before either search runs closes this -- reproduced by
// codex-sol via source inspection (reference=[2,1,0] vs candidate=[2,NaN,0]
// gives worst=0 and equal argmax under the old logic), not by observing an
// actual NaN from this backend; this is a test-harness fix, not evidence
// the backend itself produces NaNs today.
inline void AssertFiniteOutputs(const Outputs& o, const char* who) {
  ASSERT_TRUE(std::isfinite(o.q)) << who << ": q is not finite (" << o.q << ")";
  ASSERT_TRUE(std::isfinite(o.d)) << who << ": d is not finite (" << o.d << ")";
  ASSERT_TRUE(std::isfinite(o.m)) << who << ": m is not finite (" << o.m << ")";
  for (size_t i = 0; i < o.policy.size(); ++i) {
    ASSERT_TRUE(std::isfinite(o.policy[i]))
        << who << ": policy[" << i << "] is not finite (" << o.policy[i]
        << ")";
  }
}

// Every sample of a multi-position batch, against BLAS.
//
// The single-position CompareBackends validates exactly one (batch, sample) =
// (1, 0), and every HLSL kernel here indexes by sample. A wrong per-row stride
// is invisible at batch 1, because row 0 starts at offset 0 whichever stride
// is used -- which is exactly how policy_finalize shipped ROW_STRIDE 4168
// against a 4288-wide reader, scrambling the policy of every sample after the
// first in every real search.
// agora thread 19 #620 package E3: LC0_TEST_BACKEND set to "blas" (by
// accident, or by a future copy-paste of an env-var line) would compare
// the blas reference against itself -- every parity assertion in this
// suite trivially passes, and the suite reports green while testing
// nothing. Explicit whitelist rather than a blacklist of just "blas":
// eigen is a deliberate, understood self-comparison-adjacent case (same
// network_blas.cc code, different GEMM, used to measure the accumulation-
// order noise floor -- see CompareBackends' own comment on that), so it's
// allowed; anything else not on this list fails loudly instead of quietly
// passing for the wrong reason.
void ValidateTestBackendChoice(const std::string& test_backend) {
  static const std::set<std::string> kAllowed = {
      "directml", "directml-fp16", "directml-onnx", "eigen", "onnx-dml"};
  if (kAllowed.find(test_backend) == kAllowed.end()) {
    FAIL() << "LC0_TEST_BACKEND=" << test_backend
           << " is not on the allowed list (directml, directml-fp16, "
              "directml-onnx, eigen, onnx-dml) -- comparing the blas "
              "reference against itself (or "
              "any other unrecognized backend) would make every parity "
              "assertion in this suite trivially pass without testing "
              "anything.";
  }
}

// The net as the backend under test receives it. The ONNX converter has no
// INPUT_EMBEDDING_NONE case for an attention body, while the blas reference
// computes that label exactly as PE_MAP (network_blas.cc only branches on
// PE_DENSE), so it is relabeled for the two backends built on the converter,
// onnx-dml and directml-onnx; the synthetic nets here nearly all say NONE and
// would otherwise never reach the converter.
bool UsesOnnxConverter(const std::string& test_backend) {
  return test_backend == "onnx-dml" || test_backend == "directml-onnx";
}

pblczero::Net NetForTestBackend(const std::string& test_backend,
                                const pblczero::Net& net) {
  using NF = pblczero::NetworkFormat;
  pblczero::Net out = net;
  auto* nf = out.mutable_format()->mutable_network_format();
  if (UsesOnnxConverter(test_backend) &&
      nf->input_embedding() == NF::INPUT_EMBEDDING_NONE) {
    nf->set_input_embedding(NF::INPUT_EMBEDDING_PE_MAP);
  }
  return out;
}

// Whether the backend under test computes in half precision, which selects
// the wider tolerance. onnx-dml is fp16 unless LC0_TEST_BACKEND_OPTS says
// fp16=false, and directml-onnx is fp16 only when it says fp16=true.
bool TestBackendIsHalfPrecision(const std::string& test_backend) {
  if (test_backend == "directml-fp16") return true;
  const bool is_half_by_default = test_backend == "onnx-dml";
  if (!is_half_by_default && test_backend != "directml-onnx") return false;
  OptionsDict options;
  ApplyTestBackendOpts(&options);
  return options.GetOrDefault<bool>("fp16", is_half_by_default);
}

void CompareBackendsBatch(const pblczero::Net& net, int batch) {
  const std::vector<InputPlanes> planes = EncodeDistinctPositions(batch);

  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  const std::vector<Outputs> reference =
      RunNetworkBatch("blas", net, planes);

  const char* backend_env = getenv("LC0_TEST_BACKEND");
  const std::string test_backend = backend_env ? backend_env : "directml";
  ValidateTestBackendChoice(test_backend);
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
  const std::vector<Outputs> dml = RunNetworkBatch(
      test_backend, NetForTestBackend(test_backend, net), planes);
  const bool fp16_bound = TestBackendIsHalfPrecision(test_backend);

  for (int n = 0; n < batch; ++n) {
    // F5: assert before searching, not after -- see AssertFiniteOutputs.
    AssertFiniteOutputs(dml[n], "directml");
    AssertFiniteOutputs(reference[n], "blas reference");

    EXPECT_NEAR(dml[n].q, reference[n].q, BoundedOutputBound(fp16_bound))
        << "sample " << n << ": WDL value Q diverges";
    EXPECT_NEAR(dml[n].d, reference[n].d, BoundedOutputBound(fp16_bound))
        << "sample " << n << ": WDL draw probability diverges";
    EXPECT_NEAR(dml[n].m, reference[n].m,
               ScaledOutputBound(reference[n].m, fp16_bound))
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
    EXPECT_LT(worst, ScaledOutputBound(ref_absmax, fp16_bound))
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
  ValidateTestBackendChoice(test_backend);
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
  const Outputs dml =
      RunNetwork(test_backend, NetForTestBackend(test_backend, net), planes);
  const bool fp16_bound = TestBackendIsHalfPrecision(test_backend);

  // F5: assert before searching, not after -- see AssertFiniteOutputs.
  AssertFiniteOutputs(dml, "directml");
  AssertFiniteOutputs(reference, "blas reference");

  // Bounds and their calibration live with the constants above.
  EXPECT_NEAR(dml.q, reference.q, BoundedOutputBound(fp16_bound))
      << "WDL value Q diverges between directml and blas";
  EXPECT_NEAR(dml.d, reference.d, BoundedOutputBound(fp16_bound))
      << "WDL draw probability diverges between directml and blas";
  EXPECT_NEAR(dml.m, reference.m, ScaledOutputBound(reference.m, fp16_bound))
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
  EXPECT_LT(worst, ScaledOutputBound(ref_absmax, fp16_bound))
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
// Agora thread 19 #685 Track 1 step (b) ("SCOPE TO CHAIN"): scales only the
// smolgen chain's own weight matrices (compress/dense1/dense2/the shared
// global table), independent of and composable with LC0_TEST_WEIGHT_SCALE
// (which scales every synthetic weight in the net). Biases and the LN1/LN2
// gammas/betas are deliberately left alone -- gammas are GAMMA_ONE's axis
// (queued behind this step), and biases were never implicated by the
// whole-net WEIGHT_SCALE sweep in the way the multiplicative weights were.
// Diagnostic-only, test-file-only, no production code.
float SmolgenChainScale() {
  if (const char* s = getenv("LC0_TEST_SMOLGEN_SCALE")) {
    return static_cast<float>(atof(s));
  }
  return 1.0f;
}

void FillSmolgen(pblczero::Net* net, std::mt19937& rng, const NetDims& d,
                 const SmolgenDims& sd) {
  const float sscale = SmolgenChainScale();
  auto* weights = net->mutable_weights();
  auto* enc = weights->mutable_encoder(weights->encoder_size() - 1);
  auto* smol = enc->mutable_mha()->mutable_smolgen();
  FillLayer(smol->mutable_compress(),
            RandomVec(rng, sd.hidden_channels * d.embedding, 0.1f * sscale));
  FillLayer(smol->mutable_dense1_w(),
            RandomVec(rng, sd.hidden_sz * 64 * sd.hidden_channels,
                      0.05f * sscale));
  FillLayer(smol->mutable_dense1_b(), RandomVec(rng, sd.hidden_sz, 0.05f));
  FillLayer(smol->mutable_ln1_gammas(), GammaVec(rng, sd.hidden_sz));
  FillLayer(smol->mutable_ln1_betas(), RandomVec(rng, sd.hidden_sz, 0.05f));
  FillLayer(smol->mutable_dense2_w(),
            RandomVec(rng, sd.gen_outputs * sd.hidden_sz, 0.05f * sscale));
  FillLayer(smol->mutable_dense2_b(), RandomVec(rng, sd.gen_outputs, 0.05f));
  FillLayer(smol->mutable_ln2_gammas(), GammaVec(rng, sd.gen_outputs));
  FillLayer(smol->mutable_ln2_betas(), RandomVec(rng, sd.gen_outputs, 0.05f));
  FillLayer(weights->mutable_smolgen_w(),
            RandomVec(rng, 64 * 64 * (sd.gen_outputs / d.heads),
                      0.05f * sscale));
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

// Batches that land on rungs the 8-rung ladder never had. Every size here is
// a rung in its own right, so each compiles a graph at load that no other
// test exercises, and each is a size the OLD ladder would have padded away
// to 16/32/64 -- meaning these paths ran in production under a padded graph
// and are only now evaluated at their true width.
//
// Distinct positions per slot, as always: a batch of identical inputs makes
// every row hold the same numbers, so any per-sample indexing error hides
// behind a green checkmark.
TEST(DirectMlKdaParity, MatchesBlasOnNewLadderRungTwelve) {
  CompareBackendsBatch(MakeKdaMlhNet(), 12);
}

TEST(DirectMlKdaParity, MatchesBlasOnNewLadderRungTwentyFour) {
  CompareBackendsBatch(MakeKdaMlhNet(), 24);
}

// One size BETWEEN two new rungs, so the padding path itself is covered: 20
// rounds up to 24, and the extra four rows must not disturb the twenty real
// ones.
TEST(DirectMlKdaParity, MatchesBlasOnBatchPaddedToNewRung) {
  CompareBackendsBatch(MakeKdaMlhNet(), 20);
}


TEST(DirectMlKdaParity, MatchesBlasOnGatedEmbeddingNet) {
  CompareBackends(MakeGatedNet(NetDims(), 6001));
}

TEST(DirectMlKdaParity, MatchesBlasOnGatedEmbeddingRealisticDims) {
  CompareBackends(MakeGatedNet(RealisticDims(), 6002));
}

// agora thread 19 D3: the gating path above was batch-1-only, and the
// per-sample batch_base stride its shader hand-computes needs batch>1 to
// mean anything. Same fixtures at batch 4 (fresh seeds per the file's seed
// diversity rule), mirroring MatchesBlasOnLocalConvBatch above.
TEST(DirectMlKdaParity, MatchesBlasOnGatedEmbeddingBatch) {
  CompareBackendsBatch(MakeGatedNet(NetDims(), 6003), 4);
}

TEST(DirectMlKdaParity, MatchesBlasOnGatedEmbeddingRealisticDimsBatch) {
  CompareBackendsBatch(MakeGatedNet(RealisticDims(), 6004), 4);
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

// Wide policy-encoder FFN. Every other fixture in this file leaves the policy
// encoder's dff at 64; this drives it to 384, a geometry nothing else here
// reaches. What it checks is that DirectML's graph creation and binding agree
// numerically with BLAS's ForwardEncoderLayer at that width, at batch 1 and
// batch 4.
//
// It does NOT guard a buffer-sizing invariant, and an earlier version of this
// comment wrongly claimed it did. ForwardEncoderLayer sizes its own scratch
// from dff before the FFN runs (network_blas.cc:513-514,
// `vec_adjust(encoder_buffer4, batch * kSquares * max(kSquares * heads,
// dff_size))`), so the FFN write is covered by construction and there is no
// gap for a test to protect.
//
// pol_emb must equal pol_dmodel, for the same reason MakeWidePolicyEncoderNet
// states: DirectML rejects the graph outright when they differ.
pblczero::Net MakeWideFfnPolicyEncoderNet(int dff) {
  NetDims d;
  d.pol_emb = 128;
  d.pol_dmodel = 128;
  std::mt19937 rng(9400 + dff);
  pblczero::Net file = MakeNetWithDims(d, 9400 + dff);
  FillPolicyEncoder(&file, rng, d.pol_emb, d.pol_dmodel, /*heads=*/8,
                    /*dff=*/dff);
  return file;
}

TEST(DirectMlKdaParity, MatchesBlasOnWideFfnPolicyEncoder) {
  CompareBackends(MakeWideFfnPolicyEncoderNet(384));
}

TEST(DirectMlKdaParity, MatchesBlasOnWideFfnPolicyEncoderBatch) {
  CompareBackendsBatch(MakeWideFfnPolicyEncoderNet(384), 4);
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

// agora thread 19 P4 (iv), muse-spark's #677/#668: MakeLocalConvNet above
// uses NetDims()'s small default dims -- real KDA nets run at
// RealisticDims() (embedding=128, heads=16). Same head-scale-amplifier
// reasoning as MakeLocalConvNet's own comment (the effect is otherwise too
// small to discriminate against the tolerance floor), at batch>1 so the
// per-sample batch_base stride this shader hand-computes is exercised at
// realistic scale too, not just the small synthetic dims above.
pblczero::Net MakeLocalConvRealisticNet(unsigned seed) {
  KdaFeatures feat;
  feat.local_conv = true;
  constexpr float kLocalConvHeadScale = 12.0f;
  return MakeNetWithDims(RealisticDims(), seed, feat, kLocalConvHeadScale);
}

TEST(DirectMlKdaParity, MatchesBlasOnLocalConvRealisticBatch) {
  CompareBackendsBatch(MakeLocalConvRealisticNet(9403), 4);
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

// Agora thread 19 #637-#642 (this iteration, user-directed): the real trained
// nets' KDA encoders all have qkv_silu=false (confirmed via
// DISABLED_TriageBatchContamination's diagnostic print on all 3 real nets --
// local_conv=0, qkv_silu=0, gate_rank=32, key_dim=32, value_dim=32,
// enc3.has_smolgen=1). Every synthetic fixture tested so far in this triage
// (MakeKdaMhaNet, MakeThreeKdaThenMhaNet, MakeFullRealisticNet) calls
// FillKdaEncoder with the DEFAULT KdaFeatures, whose qkv_silu defaults to
// true (line ~152) -- every batch>1 synthetic test run in this triage has
// therefore exercised the SWISH-activated q/k/v path, never the
// ACTIVATION_NONE path the real nets actually use. Byte-for-byte
// MakeFullRealisticNet(false, true, true) (the closest match that already
// passed clean at batch 2/4/8) except qkv_silu=false on all 3 KDA encoders --
// the one remaining known flag mismatch against the real nets' confirmed
// config. Diagnostic-only, no tolerance change.
pblczero::Net MakeFullRealisticNetNoQkvSilu() {
  const NetDims d = RealisticDims();
  const int input_size = kInputPlanes + 64;
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
  nf->set_input_embedding(NF::INPUT_EMBEDDING_NONE);
  nf->set_default_activation(NF::DEFAULT_ACTIVATION_MISH);
  nf->set_ffn_activation(NF::ACTIVATION_DEFAULT);
  nf->set_smolgen_activation(NF::ACTIVATION_SWISH);
  for (int dir : {9, 10, 11, 12}) {
    nf->add_kda_directions(static_cast<NF::KdaDirection>(dir));
  }

  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);
  KdaFeatures feat;
  feat.qkv_silu = false;
  FillKdaEncoder(&file, rng, d, feat);
  FillKdaEncoder(&file, rng, d, feat);
  FillKdaEncoder(&file, rng, d, feat);
  FillMhaEncoder(&file, rng, d);
  SmolgenDims sd;
  sd.hidden_channels = 32;
  sd.hidden_sz = 256;
  sd.gen_outputs = d.heads * 16;
  FillSmolgen(&file, rng, d, sd);
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

// Agora thread 19 #637-#642: MakeFullRealisticNet(false, true, true) /
// FullRealistic_MhaSmolgen above already matches the real trained nets'
// structure (3xKDA+1xMHA) AND dims (RealisticDims(): embedding=128,
// heads=16) AND has smolgen on the MHA layer -- the closest synthetic match
// to the real nets that exists, but until now only ever compared at batch=1.
// MakeKdaMhaNet()-based fixtures (small dims, no smolgen) passed clean at
// batch 2/4/8 regardless of KDA encoder depth (#640/#642) -- these three
// test whether real dims and/or smolgen are the missing factor. The
// NoSmolgen variant isolates dims alone; the Smolgen variant adds smolgen on
// top, so a pass/fail split between them separates which factor matters.
// Diagnostic-only, no tolerance change.
TEST(DirectMlKdaParity, FullRealistic_MhaNoSmolgen_BatchOfTwo) {
  CompareBackendsBatch(MakeFullRealisticNet(false, false, true), 2);
}
TEST(DirectMlKdaParity, FullRealistic_MhaNoSmolgen_BatchOfFour) {
  CompareBackendsBatch(MakeFullRealisticNet(false, false, true), 4);
}
TEST(DirectMlKdaParity, FullRealistic_MhaNoSmolgen_BatchOfEight) {
  CompareBackendsBatch(MakeFullRealisticNet(false, false, true), 8);
}
TEST(DirectMlKdaParity, FullRealistic_MhaSmolgen_BatchOfTwo) {
  CompareBackendsBatch(MakeFullRealisticNet(false, true, true), 2);
}
TEST(DirectMlKdaParity, FullRealistic_MhaSmolgen_BatchOfFour) {
  CompareBackendsBatch(MakeFullRealisticNet(false, true, true), 4);
}
TEST(DirectMlKdaParity, FullRealistic_MhaSmolgen_BatchOfEight) {
  CompareBackendsBatch(MakeFullRealisticNet(false, true, true), 8);
}

// The one remaining known flag mismatch against the real nets: qkv_silu=false
// (see MakeFullRealisticNetNoQkvSilu's comment above).
TEST(DirectMlKdaParity, FullRealistic_NoQkvSilu_BatchOfTwo) {
  CompareBackendsBatch(MakeFullRealisticNetNoQkvSilu(), 2);
}
TEST(DirectMlKdaParity, FullRealistic_NoQkvSilu_BatchOfFour) {
  CompareBackendsBatch(MakeFullRealisticNetNoQkvSilu(), 4);
}
TEST(DirectMlKdaParity, FullRealistic_NoQkvSilu_BatchOfEight) {
  CompareBackendsBatch(MakeFullRealisticNetNoQkvSilu(), 8);
}

// Agora thread 19 #648/#652 (muse-spark authorization item 3, user-directed
// this iteration): codex-sol's freshly-extracted real decay metadata found
// output_rms_norm=0 for all 3 real nets' KDA encoders -- KdaFeatures
// defaults output_rms_norm=true, and MakeFullRealisticNetNoQkvSilu above
// only overrode qkv_silu, leaving output_rms_norm at its true default. This
// is byte-for-byte MakeFullRealisticNetNoQkvSilu except output_rms_norm is
// ALSO false on all 3 KDA encoders -- now matching the real nets' confirmed
// config on every checked dimension: dims, depth, mixer types, smolgen,
// qkv_silu, output_rms_norm, gate_rank, key/value dim, local_conv.
// Diagnostic-only, no tolerance change.
pblczero::Net MakeFullRealisticNetNoQkvSiluNoRmsNorm() {
  const NetDims d = RealisticDims();
  const int input_size = kInputPlanes + 64;
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
  nf->set_input_embedding(NF::INPUT_EMBEDDING_NONE);
  nf->set_default_activation(NF::DEFAULT_ACTIVATION_MISH);
  nf->set_ffn_activation(NF::ACTIVATION_DEFAULT);
  nf->set_smolgen_activation(NF::ACTIVATION_SWISH);
  for (int dir : {9, 10, 11, 12}) {
    nf->add_kda_directions(static_cast<NF::KdaDirection>(dir));
  }

  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);
  KdaFeatures feat;
  feat.qkv_silu = false;
  feat.output_rms_norm = false;
  FillKdaEncoder(&file, rng, d, feat);
  FillKdaEncoder(&file, rng, d, feat);
  FillKdaEncoder(&file, rng, d, feat);
  FillMhaEncoder(&file, rng, d);
  SmolgenDims sd;
  sd.hidden_channels = 32;
  sd.hidden_sz = 256;
  sd.gen_outputs = d.heads * 16;
  FillSmolgen(&file, rng, d, sd);
  FillPolicyAndValueHeads(&file, rng, d, true, 32);
  return file;
}

TEST(DirectMlKdaParity, FullRealistic_NoQkvSiluNoRmsNorm_BatchOfTwo) {
  CompareBackendsBatch(MakeFullRealisticNetNoQkvSiluNoRmsNorm(), 2);
}
TEST(DirectMlKdaParity, FullRealistic_NoQkvSiluNoRmsNorm_BatchOfFour) {
  CompareBackendsBatch(MakeFullRealisticNetNoQkvSiluNoRmsNorm(), 4);
}
TEST(DirectMlKdaParity, FullRealistic_NoQkvSiluNoRmsNorm_BatchOfEight) {
  CompareBackendsBatch(MakeFullRealisticNetNoQkvSiluNoRmsNorm(), 8);
}

// Agora thread 19 #668 P1 step 1 (muse-spark's proposal, user-directed this
// iteration -- "you have the go ahead from me"): the smolgen-bias
// generation chain (m1 compress -> m2 dense1+LN -> m3 dense2+LN -> m4
// bias-GEMM) was found corrupted at batch>1 (#665), but every synthetic
// smolgen fixture tested so far (including MakeFullRealisticNetNoQkvSilu
// NoRmsNorm above) used SmolgenDims{32,256,heads*16=256} -- LARGER than the
// real nets' actual smolgen shape. Extracted directly from the real nets
// still on disk (resolving muse-spark's caveat that these were
// unverified): kda-native-935532/825532 both have hidden_channels=8,
// hidden_sz=32, gen_outputs=512 (kda-t1-55050: gen_outputs=256). This is
// byte-for-byte MakeFullRealisticNetNoQkvSiluNoRmsNorm except the smolgen
// dims match 935532/825532 exactly -- a FAIL here means shape/driver
// (matches muse-spark's small-M/small-N meta-command hypothesis); a PASS
// re-opens value-driven causes. Diagnostic-only, no tolerance change.
// Agora thread 19 #694 Track 1 next knob: scales only the input embedding
// (ip_emb_w), independent of and composable with LC0_TEST_WEIGHT_SCALE and
// LC0_TEST_SMOLGEN_SCALE (same RandomVec-composition mechanism). Bias left
// alone, mirroring SmolgenChainScale's precedent. Diagnostic-only,
// test-file-only, no production code.
float EmbScale() {
  if (const char* s = getenv("LC0_TEST_EMB_SCALE")) {
    return static_cast<float>(atof(s));
  }
  return 1.0f;
}

pblczero::Net MakeFullRealisticNetRealSmolgenDims() {
  const NetDims d = RealisticDims();
  const int input_size = kInputPlanes + 64;
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
  nf->set_input_embedding(NF::INPUT_EMBEDDING_NONE);
  nf->set_default_activation(NF::DEFAULT_ACTIVATION_MISH);
  nf->set_ffn_activation(NF::ACTIVATION_DEFAULT);
  nf->set_smolgen_activation(NF::ACTIVATION_SWISH);
  for (int dir : {9, 10, 11, 12}) {
    nf->add_kda_directions(static_cast<NF::KdaDirection>(dir));
  }

  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * input_size, 0.05f * EmbScale()));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(d.heads);
  KdaFeatures feat;
  feat.qkv_silu = false;
  feat.output_rms_norm = false;
  FillKdaEncoder(&file, rng, d, feat);
  FillKdaEncoder(&file, rng, d, feat);
  FillKdaEncoder(&file, rng, d, feat);
  FillMhaEncoder(&file, rng, d);
  SmolgenDims sd;
  sd.hidden_channels = 8;
  sd.hidden_sz = 32;
  sd.gen_outputs = 512;
  FillSmolgen(&file, rng, d, sd);
  FillPolicyAndValueHeads(&file, rng, d, true, 32);
  return file;
}

TEST(DirectMlKdaParity, FullRealistic_RealSmolgenDims_BatchOfTwo) {
  CompareBackendsBatch(MakeFullRealisticNetRealSmolgenDims(), 2);
}
TEST(DirectMlKdaParity, FullRealistic_RealSmolgenDims_BatchOfFour) {
  CompareBackendsBatch(MakeFullRealisticNetRealSmolgenDims(), 4);
}
TEST(DirectMlKdaParity, FullRealistic_RealSmolgenDims_BatchOfEight) {
  CompareBackendsBatch(MakeFullRealisticNetRealSmolgenDims(), 8);
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

// agora thread 19 P4 (v), muse-spark's #677/#668: both serpentine directions
// were single-position (batch=1) only -- the direction table is uploaded
// once and read identically by every sample in a batch (kda_recurrence.hlsl
// direction_order), so a batch>1 case exercises the same table lookup
// across multiple per-sample dispatch groups instead of just one.
TEST(DirectMlKdaParity, MatchesBlasOnKdaSerpentineNetBatch) {
  CompareBackendsBatch(MakeKdaSerpentineNet(), 4);
}

TEST(DirectMlKdaParity, MatchesBlasOnKdaSerpentineReverseNetBatch) {
  CompareBackendsBatch(MakeKdaSerpentineReverseNet(), 4);
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

// E1 (agora thread 19 #620/#628/#630/#631): implemented and verified working
// (correctly skips without LC0_TEST_REAL_NET, correctly caught a real
// batch-8 divergence on kda-t1-55050 with it set -- see #630). HELD out of
// the last commit (31c6bbb) per muse-spark's #631 directive c: landing it
// red without a root cause would muddy every subsequent suite run's
// signal. Restored here in the working tree (uncommitted, per the same
// directive) to run the #631 triage against it; will land once the
// triage localizes the finding (as a red-with-cause regression test, or
// green if the underlying issue is fixed first).
TEST(DirectMlKdaParity, MatchesBlasOnRealNetBatchOfEight) {
  const char* path = getenv("LC0_TEST_REAL_NET");
  if (!path) GTEST_SKIP() << "set LC0_TEST_REAL_NET to a .pb.gz to run";
  pblczero::Net net = LoadWeightsFromFile(path);
  CompareBackendsBatch(net, 8);
}

// Diagnostic dump, NOT a correctness test: writes the first KDA encoder's
// decay geometry (heads, key_dim, value_dim, directions) and its real
// trained a_log/dt_bias to a small binary file, for
// kda_recurrence_test_directml's MatchesCpuReferenceWithRealNetDecay to
// consume (agora thread 19 #462/#463's real-net decay ground-truth check).
// This binary already has full net-loading machinery that the standalone
// recurrence-test binary deliberately does not (its own comment: "no
// factory, no backend registration") -- dumping here and reading a plain
// file there avoids growing that binary's dependency footprint just to
// parse one net file. Skipped unless LC0_DUMP_KDA_DECAY names an output
// path.
TEST(DirectMlKdaParity, DumpRealNetKdaDecayForRecurrenceTest) {
  const char* net_path = getenv("LC0_TEST_REAL_NET");
  const char* out_path = getenv("LC0_DUMP_KDA_DECAY");
  if (!net_path || !out_path) {
    GTEST_SKIP() << "set LC0_TEST_REAL_NET and LC0_DUMP_KDA_DECAY to run";
  }
  pblczero::Net net = LoadWeightsFromFile(net_path);
  const MultiHeadWeights decoded{net.weights()};
  const MultiHeadWeights::KDA* kda = nullptr;
  for (const auto& enc : decoded.encoder) {
    if (enc.is_kda) {
      kda = &enc.kda;
      break;
    }
  }
  ASSERT_NE(kda, nullptr) << "net has no KDA encoder";
  const int32_t heads = static_cast<int32_t>(kda->a_log.size());
  ASSERT_EQ(kda->dt_bias.size(), static_cast<size_t>(heads) * kda->key_dim)
      << "a_log/dt_bias/key_dim disagree on head count";
  const auto& directions_pb = net.format().network_format().kda_directions();
  const std::vector<int32_t> directions(directions_pb.begin(),
                                        directions_pb.end());

  std::ofstream f(out_path, std::ios::binary);
  ASSERT_TRUE(f.good()) << "could not open " << out_path;
  auto write_i32 = [&](int32_t v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
  };
  write_i32(heads);
  write_i32(kda->key_dim);
  write_i32(kda->value_dim);
  write_i32(static_cast<int32_t>(directions.size()));
  for (int32_t d : directions) write_i32(d);
  f.write(reinterpret_cast<const char*>(kda->a_log.data()),
         kda->a_log.size() * sizeof(float));
  f.write(reinterpret_cast<const char*>(kda->dt_bias.data()),
         kda->dt_bias.size() * sizeof(float));
  ASSERT_TRUE(f.good()) << "write failed";
  CERR << "[decay-dump] heads=" << heads << " key_dim=" << kda->key_dim
       << " value_dim=" << kda->value_dim
       << " direction_count=" << directions.size() << " -> " << out_path;
  // agora #465's action item 2: which BuildKdaTails paths this net actually
  // exercises, before bisecting kda_tail1_compiled_ (mixed_in -> normed ->
  // dense). Not written to the dump file -- the recurrence test doesn't
  // need it, only this investigation does.
  CERR << "[decay-dump] output_rms_norm=" << kda->output_rms_norm
       << " output_gate=" << kda->output_gate
       << " local_conv=" << kda->local_conv
       << " qkv_silu=" << kda->qkv_silu;
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

// Batch case for the same net -- agora thread 19 #471's request, to run
// PE_DENSE embedding-only under FP16 at batch>1 as well as batch 1 before
// touching the real net's arena sizing again (see #468's PE_DENSE arena
// regression).
TEST(DirectMlKdaParity, MatchesBlasOnPeDenseBatch) {
  CompareBackendsBatch(MakePeDenseNet(), 4);
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

// agora thread 19 D3 rework (codex-sol #724): MatchesBlasOnGatedEmbedding*
// above covers has_gating_ on INPUT_EMBEDDING_NONE nets only (MakeGatedNet
// wraps MakeNetWithDims, which never sets PE_DENSE); MakePeDenseNet and
// MakePeDenseWithEncodersNet above never populate the gate tensors. That
// leaves the PE_DENSE embedding's own gating block (layers.cc's
// is_pe_dense_embedding_ branch, both in EnsureCompiled and in Eval's
// lazy-compile fallback -- structurally identical duplicated code, so one
// fixture covers both) with zero test coverage, exactly as codex-sol's
// review found: "MakePeDenseNet and FullRealistic(true,...) still leave
// gate tensors empty." Real trained nets ARE PE_DENSE with gating live, so
// this is the actual shape being shipped, not a synthetic corner case.
pblczero::Net MakePeDenseGatedNet(unsigned seed) {
  const NetDims d;
  pblczero::Net file = MakePeDenseWithEncodersNet();
  std::mt19937 rng(seed ^ 0x9e37u);
  auto* w = file.mutable_weights();
  // Same shape/scale convention as MakeGatedNet: values must vary across
  // squares (a per-channel broadcast would still agree with a per-square
  // read and prove nothing -- see the 39%-wrong bug this caught on the
  // non-PE_DENSE branch).
  FillLayer(w->mutable_ip_mult_gate(),
            RandomVec(rng, static_cast<size_t>(d.embedding) * 64, 0.3f));
  FillLayer(w->mutable_ip_add_gate(),
            RandomVec(rng, static_cast<size_t>(d.embedding) * 64, 0.1f));
  return file;
}

TEST(DirectMlKdaParity, MatchesBlasOnPeDenseGatedNet) {
  CompareBackends(MakePeDenseGatedNet(6005));
}

// Batch case, same reasoning as MatchesBlasOnGatedEmbeddingBatch: the
// per-sample batch_base stride the gating shader hand-computes needs
// batch>1 to mean anything.
TEST(DirectMlKdaParity, MatchesBlasOnPeDenseGatedBatch) {
  CompareBackendsBatch(MakePeDenseGatedNet(6006), 4);
}

TEST(DirectMlKdaParity, MatchesBlasOnKdaHybridNet) {
  CompareBackends(MakeKdaHybridNet());
}

TEST(DirectMlKdaParity, MatchesBlasOnKdaMhaNet) {
  CompareBackends(MakeKdaMhaNet());
}

// Agora thread 19 #639: MakeKdaMhaNet() (1 KDA encoder + 1 MHA encoder) was,
// until now, only ever compared at batch 1 -- the one synthetic fixture that
// mixes mixer types in a single body never exercised batch>1 on the KDA->MHA
// handoff. Real trained nets are 3xKDA+1xMHA and fail at every batch>=2; the
// all-KDA single-encoder MakeKdaMlhNet() batch tests above pass. These three
// close that exact coverage gap (agora #637/#638/#639) -- diagnostic-only,
// reporting numbers, no tolerance change.
TEST(DirectMlKdaParity, MatchesBlasOnKdaMhaNetBatchOfTwo) {
  CompareBackendsBatch(MakeKdaMhaNet(), 2);
}

TEST(DirectMlKdaParity, MatchesBlasOnKdaMhaNetBatchOfFour) {
  CompareBackendsBatch(MakeKdaMhaNet(), 4);
}

TEST(DirectMlKdaParity, MatchesBlasOnKdaMhaNetBatchOfEight) {
  CompareBackendsBatch(MakeKdaMhaNet(), 8);
}

// Agora thread 19 #641: candidate 1 from #640 -- does chaining 3 KDA
// encoders (matching the real nets' depth) ahead of the trailing MHA
// encoder reproduce the contamination that a single KDA+MHA pair (above)
// did not? Diagnostic-only, no tolerance change.
TEST(DirectMlKdaParity, MatchesBlasOnThreeKdaThenMhaNetBatchOfTwo) {
  CompareBackendsBatch(MakeThreeKdaThenMhaNet(), 2);
}

TEST(DirectMlKdaParity, MatchesBlasOnThreeKdaThenMhaNetBatchOfFour) {
  CompareBackendsBatch(MakeThreeKdaThenMhaNet(), 4);
}

TEST(DirectMlKdaParity, MatchesBlasOnThreeKdaThenMhaNetBatchOfEight) {
  CompareBackendsBatch(MakeThreeKdaThenMhaNet(), 8);
}

TEST(DirectMlKdaParity, MatchesBlasOnNoEncoderNet) {
  CompareBackends(MakeNoEncoderNet());
}


// ---------------------------------------------------------------------------
// Shader-blob cache. FXC dominates backend startup (the KDA recurrence shader
// is the expensive one and a net compiles it once per KDA encoder), so
// CompileHlsl memoises its results. These assert the cache's behaviour through
// its own counters rather than inferring it from timing, which would be flaky.
//
// A trivial shader is used deliberately: it compiles in milliseconds, so these
// test the cache and not FXC. Every test calls ResetForTesting() first, so no
// test depends on another's leftovers or on gtest's execution order.
constexpr char kCacheTestShader[] =
    "RWStructuredBuffer<float> Out : register(u0);\n"
    "[numthreads(1,1,1)]\n"
    "void CsMain(uint3 tid : SV_DispatchThreadID) { Out[0] = SCALE; }\n";

// Namespace scope so the concurrency test's lambdas can reach it.
constexpr char kInvalidShader[] = "still not valid HLSL @@@";

directml_backend::ComPtr<ID3DBlob> CompileCacheTestShader(
    const std::string& test_id, const char* scale) {
  return directml_backend::CompileHlsl(
      kCacheTestShader, sizeof(kCacheTestShader) - 1, "cache_test.hlsl",
      "CsMain", /*fp16=*/false, {{"SCALE", scale}, {"TESTID", test_id}});
}

// Waits for `count` callers to be blocked on someone else's flight, with a
// deadline. Returns false on timeout rather than spinning forever: if
// single-flight regresses, the waiters never arrive, and a test that hung
// would be strictly worse than one that fails -- a hang gives CI nothing to
// report and no stack to read.
bool WaitForFlightWaiters(size_t count) {
  namespace sc = directml_backend::shader_cache;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (sc::Waiting() < count) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

TEST(DirectMlShaderCache, IdenticalInputsHitWithoutRecompiling) {
  namespace sc = directml_backend::shader_cache;
  sc::ResetForTesting();

  auto first = CompileCacheTestShader("hit", "1.0");
  ASSERT_NE(first.Get(), nullptr);
  EXPECT_EQ(sc::Attempts(), 1) << "first call must actually compile";
  EXPECT_EQ(sc::Hits(), 0);

  auto second = CompileCacheTestShader("hit", "1.0");
  EXPECT_EQ(sc::Attempts(), 1) << "second call must not recompile";
  EXPECT_EQ(sc::Hits(), 1);
  // The same blob, not merely an equivalent one: callers create pipeline
  // states from it and sharing is the whole point.
  EXPECT_EQ(first.Get(), second.Get());
}

TEST(DirectMlShaderCache, DifferingDefinesDoNotCollide) {
  namespace sc = directml_backend::shader_cache;
  sc::ResetForTesting();

  // This is the failure that matters. The defines carry the KDA
  // specialisation (KDA_KEY_DIM/KDA_VALUE_DIM); if they were left out of the
  // key, two different specialisations would share one blob and the second
  // net would silently execute the first net's bytecode.
  auto a = CompileCacheTestShader("miss", "1.0");
  auto b = CompileCacheTestShader("miss", "2.0");
  EXPECT_EQ(sc::Attempts(), 2) << "differing defines must compile twice";
  EXPECT_EQ(sc::Hits(), 0);
  EXPECT_NE(a.Get(), b.Get()) << "differing defines must not share a blob";
}

TEST(DirectMlShaderCache, ConcurrentCallersCompileOnceAndShare) {
  namespace sc = directml_backend::shader_cache;
  sc::ResetForTesting();

  constexpr int kThreads = 8;
  std::vector<directml_backend::ComPtr<ID3DBlob>> results(kThreads);
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back(
        [&results, i] { results[i] = CompileCacheTestShader("conc", "1.0"); });
  }
  for (auto& t : threads) t.join();

  // Exactly one compile total: the losers of the race wait for the winner and
  // take its blob rather than each compiling their own copy.
  EXPECT_EQ(sc::Attempts(), 1);
  EXPECT_EQ(sc::Hits(), kThreads - 1);
  for (int i = 0; i < kThreads; ++i) {
    ASSERT_NE(results[i].Get(), nullptr);
    EXPECT_EQ(results[i].Get(), results[0].Get());
  }
}

TEST(DirectMlShaderCache, KeyCoversEveryCompilationInput) {
  using directml_backend::shader_cache::MakeKey;
  const std::vector<std::pair<std::string, std::string>> macros = {{"X", "1"},
                                                                  {"Y", "2"}};
  const std::vector<std::pair<std::string, std::string>> reordered = {
      {"Y", "2"}, {"X", "1"}};
  const char* src = "AA";
  const auto base = MakeKey(src, 2, "f.hlsl", "Main", "cs_5_1", 0, macros);

  EXPECT_EQ(base, MakeKey(src, 2, "f.hlsl", "Main", "cs_5_1", 0, macros))
      << "identical inputs must produce an identical key";
  EXPECT_NE(base, MakeKey("AB", 2, "f.hlsl", "Main", "cs_5_1", 0, macros));
  EXPECT_NE(base, MakeKey(src, 2, "g.hlsl", "Main", "cs_5_1", 0, macros));
  EXPECT_NE(base, MakeKey(src, 2, "f.hlsl", "Other", "cs_5_1", 0, macros));
  EXPECT_NE(base, MakeKey(src, 2, "f.hlsl", "Main", "cs_6_0", 0, macros));
  EXPECT_NE(base, MakeKey(src, 2, "f.hlsl", "Main", "cs_5_1", 1, macros));
  EXPECT_NE(base, MakeKey(src, 2, "f.hlsl", "Main", "cs_5_1", 0, reordered))
      << "macro order is part of what was compiled";

  // Length-prefixing exists so field boundaries cannot be forged by shifting
  // a character from one field into the next. Without it these two collide.
  EXPECT_NE(MakeKey("A", 1, "bc", "Main", "cs_5_1", 0, {}),
            MakeKey("A", 1, "b", "cMain", "cs_5_1", 0, {}));
}

TEST(DirectMlShaderCache, SaturationRespectsBoundAndStopsRetaining) {
  namespace sc = directml_backend::shader_cache;
  sc::ResetForTesting();
  const size_t capacity = sc::Capacity();
  ASSERT_GT(capacity, 0u);

  // Drive past the cap with distinct shaders. This is what makes the bound
  // testable: asserting Size() <= Capacity() on a near-empty cache holds
  // trivially and would not notice the insertion guard regressing.
  for (size_t i = 0; i < capacity + 8; ++i) {
    CompileCacheTestShader("sat" + std::to_string(i), "1.0");
  }
  EXPECT_EQ(sc::Size(), capacity) << "retained entries must stop at the bound";

  // Defined behaviour for a shader first seen after saturation: it compiles
  // and is returned, but is never retained, so repeat calls recompile rather
  // than evicting something else.
  const int attempts = sc::Attempts();
  const int hits = sc::Hits();
  auto a = CompileCacheTestShader("post_saturation", "1.0");
  auto b = CompileCacheTestShader("post_saturation", "1.0");
  ASSERT_NE(a.Get(), nullptr);
  ASSERT_NE(b.Get(), nullptr);
  EXPECT_EQ(sc::Attempts(), attempts + 2);
  EXPECT_EQ(sc::Hits(), hits);
  EXPECT_EQ(sc::Size(), capacity) << "the bound must still hold";
}

TEST(DirectMlShaderCache, FailedCompilationIsNotCachedAndRetries) {
  namespace sc = directml_backend::shader_cache;
  sc::ResetForTesting();
  constexpr char kInvalid[] = "this is not valid HLSL @@@";

  EXPECT_THROW(directml_backend::CompileHlsl(kInvalid, sizeof(kInvalid) - 1,
                                             "invalid.hlsl", "CsMain",
                                             /*fp16=*/false, {}),
               Exception);
  EXPECT_EQ(sc::Attempts(), 1) << "a failed compile is still an attempt";
  EXPECT_EQ(sc::Size(), 0u) << "failures must not be retained";

  // The second call must compile again and raise its own diagnostics, rather
  // than being served a cached failure.
  EXPECT_THROW(directml_backend::CompileHlsl(kInvalid, sizeof(kInvalid) - 1,
                                             "invalid.hlsl", "CsMain",
                                             /*fp16=*/false, {}),
               Exception);
  EXPECT_EQ(sc::Attempts(), 2);
  EXPECT_EQ(sc::Hits(), 0);
}

TEST(DirectMlShaderCache, ConcurrentDistinctKeysEachCompile) {
  namespace sc = directml_backend::shader_cache;
  sc::ResetForTesting();

  // The companion to the same-key case: single-flight must deduplicate
  // identical shaders WITHOUT making distinct shaders wait for each other.
  // Each thread compiles a different shader, so none may be served from the
  // cache and every one must produce its own blob.
  constexpr int kThreads = 8;
  std::vector<directml_backend::ComPtr<ID3DBlob>> results(kThreads);
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&results, i] {
      results[i] = CompileCacheTestShader("distinct" + std::to_string(i), "1.0");
    });
  }
  for (auto& t : threads) t.join();

  EXPECT_EQ(sc::Attempts(), kThreads);
  EXPECT_EQ(sc::Hits(), 0);
  EXPECT_EQ(sc::Size(), static_cast<size_t>(kThreads));
  for (int i = 0; i < kThreads; ++i) {
    ASSERT_NE(results[i].Get(), nullptr);
    for (int j = i + 1; j < kThreads; ++j) {
      EXPECT_NE(results[i].Get(), results[j].Get());
    }
  }
}

TEST(DirectMlShaderCache, ConcurrentFailuresShareOneAttemptThenRetryLater) {
  namespace sc = directml_backend::shader_cache;
  sc::ResetForTesting();

  // Single-flight must hold for FAILURES too, which is where the previous
  // implementation was wrong: it let every waiter start its own compile, so a
  // wave of N callers paid N times for the same failure. One attempt serves
  // the wave, every caller sees that attempt's failure, and nothing is
  // retained -- so a LATER call, arriving after the flight has retired,
  // legitimately starts a fresh attempt.
  // Deterministic, not opportunistic. The owning call is held inside the
  // flight hook until the other three have provably joined its flight, so
  // "one attempt for the wave" is forced rather than dependent on the threads
  // happening to overlap.
  constexpr int kThreads = 4;
  constexpr size_t kWaiters = kThreads - 1;
  std::atomic<bool> release{false};
  sc::SetFlightHookForTesting([&release] {
    while (!release.load()) std::this_thread::yield();
  });

  std::atomic<int> threw{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&threw] {
      try {
        directml_backend::CompileHlsl(kInvalidShader,
                                      sizeof(kInvalidShader) - 1,
                                      "invalid_concurrent.hlsl", "CsMain",
                                      /*fp16=*/false, {});
      } catch (const Exception&) {
        ++threw;
      }
    });
  }
  const bool joined = WaitForFlightWaiters(kWaiters);
  EXPECT_TRUE(joined) << "waiters never joined: single-flight has regressed";
  if (joined) {
    EXPECT_EQ(sc::InFlight(), 1u) << "the wave must be one flight, not four";
  }
  release.store(true);
  for (auto& t : threads) t.join();
  sc::SetFlightHookForTesting({});

  EXPECT_EQ(threw.load(), kThreads) << "every caller must observe the failure";
  EXPECT_EQ(sc::Attempts(), 1) << "one attempt serves the whole wave";
  EXPECT_EQ(sc::Size(), 0u) << "a failed key must leave nothing retained";
  EXPECT_EQ(sc::InFlight(), 0u) << "the failed flight must be retired";
  EXPECT_EQ(sc::Hits(), 0) << "joining a failed flight is not a hit";

  // A later call is a new wave and must compile again rather than inherit.
  EXPECT_THROW(directml_backend::CompileHlsl(
                   kInvalidShader, sizeof(kInvalidShader) - 1,
                   "invalid_concurrent.hlsl", "CsMain", /*fp16=*/false, {}),
               Exception);
  EXPECT_EQ(sc::Attempts(), 2);
}

TEST(DirectMlShaderCache, FailureLeavesRetainedBlobsIntact) {
  namespace sc = directml_backend::shader_cache;
  sc::ResetForTesting();

  // The retire-by-identity property, in the form that can be asserted
  // deterministically: a failing compile must not disturb what is already
  // retained under any other key, and must not leave the retained entry
  // reachable-but-broken. The previous implementation erased by key alone,
  // so a failure could delete an entry another call had installed.
  auto good = CompileCacheTestShader("survives_failure", "1.0");
  ASSERT_NE(good.Get(), nullptr);
  ASSERT_EQ(sc::Size(), 1u);

  EXPECT_THROW(directml_backend::CompileHlsl(
                   kInvalidShader, sizeof(kInvalidShader) - 1, "invalid.hlsl",
                   "CsMain", /*fp16=*/false, {}),
               Exception);

  EXPECT_EQ(sc::Size(), 1u) << "the failure must not evict a retained blob";
  const int hits = sc::Hits();
  auto again = CompileCacheTestShader("survives_failure", "1.0");
  EXPECT_EQ(sc::Hits(), hits + 1) << "the retained blob must still serve";
  EXPECT_EQ(again.Get(), good.Get());
}

TEST(DirectMlShaderCache, PostSaturationConcurrentCallersShareOneAttempt) {
  namespace sc = directml_backend::shader_cache;
  sc::ResetForTesting();
  const size_t capacity = sc::Capacity();

  for (size_t i = 0; i < capacity; ++i) {
    CompileCacheTestShader("presat" + std::to_string(i), "1.0");
  }
  ASSERT_EQ(sc::Size(), capacity) << "cache must be saturated for this test";

  // Past the retention bound a shader is not stored -- but deduplication of
  // work in flight is a separate concern from retention, so a wave of callers
  // for one post-capacity key must still cost exactly one compile. The earlier
  // implementation handed each caller a private entry here and compiled once
  // per caller.
  const int attempts = sc::Attempts();
  constexpr int kThreads = 8;
  constexpr size_t kWaiters = kThreads - 1;
  std::atomic<bool> release{false};
  sc::SetFlightHookForTesting([&release] {
    while (!release.load()) std::this_thread::yield();
  });

  std::vector<directml_backend::ComPtr<ID3DBlob>> results(kThreads);
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&results, i] {
      results[i] = CompileCacheTestShader("post_sat_wave", "1.0");
    });
  }
  // Hold the owner until every other caller has joined its flight, so this
  // asserts single-flight past the retention bound rather than a race won.
  const bool joined = WaitForFlightWaiters(kWaiters);
  EXPECT_TRUE(joined)
      << "waiters never joined past capacity: in-flight dedup has regressed";
  if (joined) EXPECT_EQ(sc::InFlight(), 1u);
  release.store(true);
  for (auto& t : threads) t.join();
  sc::SetFlightHookForTesting({});

  EXPECT_EQ(sc::Attempts(), attempts + 1) << "one attempt for the whole wave";
  EXPECT_EQ(sc::Size(), capacity) << "no retained growth past the bound";
  EXPECT_EQ(sc::InFlight(), 0u);
  for (int i = 0; i < kThreads; ++i) {
    ASSERT_NE(results[i].Get(), nullptr);
    EXPECT_EQ(results[i].Get(), results[0].Get()) << "the wave must share one blob";
  }

  // A later wave finds nothing retained and recompiles exactly once.
  const int after_wave = sc::Attempts();
  CompileCacheTestShader("post_sat_wave", "1.0");
  EXPECT_EQ(sc::Attempts(), after_wave + 1);
  EXPECT_EQ(sc::Size(), capacity);
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

// ---------------------------------------------------------------------------
// F8 & F6 regression coverage (agora thread 19 #578/#579): codex-sol's
// secondary review flagged that F8 (empty-batch guard) and F6 (fp16
// converter NaN/subnormal fix) landed in commit 1b32741 with only a
// standalone, out-of-tree bite-test as evidence -- durable in-tree coverage
// was a CHANGES_REQUIRED condition, endorsed by gemini-antigravity in #579.
// These two tests are that coverage.
// ---------------------------------------------------------------------------

// F8: ComputeBlocking() on a computation with zero AddInput calls used to
// underflow planes_.size() - 1 in the per-sample readback loop (size_t 0 - 1
// wraps to SIZE_MAX) after forwardEval() silently raised the batch size to
// min_batch_size_ while planes_ itself stayed empty. The fix is the early
// `if (planes_.empty()) return;` in network_directml.cc's ComputeBlocking().
// This reproduces the underflow's precondition directly rather than
// inferring it from the surrounding parity suite, which never constructs an
// empty-input computation.
TEST(DirectMlRegressionCoverage, EmptyBatchNoCrash) {
  if (!HasBackend("directml"))
    GTEST_SKIP() << "directml backend not compiled in";
  if (!DirectMlAvailability().available) {
    GTEST_SKIP() << "no usable directml device: "
                 << DirectMlAvailability().reason;
  }
  OptionsDict options;
  ApplyTestBackendOpts(&options);
  auto network =
      NetworkFactory::Get()->Create("directml", MakeKdaMlhNet(), options);
  auto computation = network->NewComputation();
  // Deliberately no AddInput() call: planes_ stays empty, batch size 0.
  ASSERT_NO_THROW(computation->ComputeBlocking())
      << "ComputeBlocking on an empty batch must return cleanly, not crash "
         "or throw (F8, agora thread 19 #560/#573/#578)";
}

// agora thread 19 P4 (vi), muse-spark's #677/#668: EmptyBatchNoCrash above
// only ever exercised directml (fp32). The fp16 path has its own readback
// conversion (host-side DmlHalf->float, network_directml.cc) and its own
// allow_broken_fp16 exception gate, both independent of F8's fix -- a
// separate empty-batch defect in either could exist without this test
// catching it. Same reproduction shape as F8, on directml-fp16 instead.
TEST(DirectMlRegressionCoverage, EmptyBatchNoCrashFp16) {
  if (!HasBackend("directml-fp16")) {
    GTEST_SKIP() << "directml-fp16 backend not compiled in";
  }
  OptionsDict options;
  options.Set<bool>("allow_broken_fp16", true);
  auto network =
      NetworkFactory::Get()->Create("directml-fp16", MakeKdaMlhNet(), options);
  auto computation = network->NewComputation();
  // Deliberately no AddInput() call: planes_ stays empty, batch size 0.
  ASSERT_NO_THROW(computation->ComputeBlocking())
      << "ComputeBlocking on an empty batch must return cleanly on the fp16 "
         "backend too, not crash or throw (P4 vi, agora thread 19 #677)";
}

// Agora thread 19 #696 Stream A4 ("transient-sizing probe only -- run max
// supported config to map the throw boundary; implementation [size-from-
// ladder] stays held as design"): originally aimed at transient_arena_'s
// FIXED 256MB size (network_directml.cc, DmlArena::Create call), but the
// actual result is a DIFFERENT, more urgent finding -- it never gets that
// far. With max_batch configured to its allowed ceiling (1024),
// EVERY request above 256 throws immediately (verified: batch 256 OK,
// batch 257 THREW, identical error at batch 320) -- a binary cliff, not a
// gradual boundary. Mechanism, traced to source: BatchLadder()
// (network_directml.cc) is {..., 192, 256, max_batch_size_}, so with
// max_batch_size_=1024 there is a single huge gap from 256 straight to
// 1024 -- forwardEval's round-up-to-ladder logic sends ANY batch in
// (256, 1024] straight to 1024, and CheckDispatchGroupCount's pre-existing
// package-B3 guard (layers.cc) then correctly rejects record_preprocess's
// resulting 1024*64 = 65536 thread groups, one over D3D12's 65535-per-
// dimension limit. So max_batch=1024 is not "works up to some point then
// runs out of transient scratch" -- it is unconditionally broken for any
// caller who ever requests a batch the ladder doesn't have a close rung
// for, well before arena sizing is ever relevant. No BLAS reference here
// (BLAS caps at 256, see "BLAS max batch size is 256" in every suite log)
// -- directml-only, and no correctness assertion, just a boundary map
// printed to stderr. DISABLED_ by default (matches
// DISABLED_TriageBatchContamination/DISABLED_DumpSingleRealNetEval's
// convention for manual-invocation-only diagnostics); run explicitly with
// --gtest_also_run_disabled_tests --gtest_filter=*TransientArenaMaxBatchProbe*.
TEST(DirectMlKdaParity, DISABLED_TransientArenaMaxBatchProbe) {
  OptionsDict options;
  options.Set<int>("max_batch", 1024);
  auto network = NetworkFactory::Get()->Create(
      "directml", MakeFullRealisticNetRealSmolgenDims(), options);
  for (int batch : {1, 32, 64, 128, 192, 256, 257, 320, 384, 448, 512, 576,
                    640, 704, 768, 832, 896, 960, 1024}) {
    auto computation = network->NewComputation();
    for (int i = 0; i < batch; ++i) {
      computation->AddInput(InputPlanes(EncodeStartPos()));
    }
    try {
      computation->ComputeBlocking();
      std::cerr << "A4 probe: batch " << batch << ": OK\n";
    } catch (const std::exception& e) {
      std::cerr << "A4 probe: batch " << batch << ": THREW: " << e.what()
                << "\n";
      break;
    }
  }
}

// agora thread 19 DML-4 (codex-sol #724): the two env-mutating tests below
// used to set process-wide vars at the top and clear them with a second
// hand-matched _putenv_s/setenv block just before their assertions -- any
// ASSERT_* failure or exception between those two points (which DOES
// happen: TwoNetworkInterleaveDoesNotCorruptResults's own ASSERT_EQ loop
// runs entirely inside that window) skips the restore and leaks the
// setting into every later test in the same process. A scoped RAII helper,
// same shape as ScopedTempDir above, makes that impossible: restores on
// every exit path, including an unset var back to genuinely unset (not "").
class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
#if defined(_WIN32)
    const char* prev = getenv(name);
    had_prev_ = prev != nullptr;
    if (had_prev_) prev_value_ = prev;
    _putenv_s(name, value);
#else
    const char* prev = getenv(name);
    had_prev_ = prev != nullptr;
    if (had_prev_) prev_value_ = prev;
    setenv(name, value, 1);
#endif
  }
  ~ScopedEnvVar() {
#if defined(_WIN32)
    _putenv_s(name_.c_str(), had_prev_ ? prev_value_.c_str() : "");
#else
    if (had_prev_) {
      setenv(name_.c_str(), prev_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
#endif
  }
  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

 private:
  std::string name_;
  bool had_prev_ = false;
  std::string prev_value_;
};

// agora thread 19 P4 (vii), muse-spark's #677/#668 (doubles as RR3's
// regression test): RR3 moved BodyDumps()/ProfileMarks() from process-wide
// statics into per-DmlExecScope storage specifically to remove a
// cross-network hazard that only existed when TWO DirectMlNetwork
// instances' evaluations interleaved. The original hazard needed genuine
// concurrent GPU work from two threads (network_check.cc's dual-backend
// comparison, or directml+directml-fp16 loaded together) to manifest --
// this test does not attempt that (it needs the "hook seam mirroring
// SetFlightHookForTesting" muse-spark's fuller Design 1 proposal named,
// which does not exist yet; filed as the P3-followup, not built here).
//
// agora thread 19 DML-4 (codex-sol #724): codex-sol's "A and B never
// coexist" was, on inspection, stronger than "sequential" -- the original
// eval_once lambda called NetworkFactory::Get()->Create(...) INSIDE itself
// on every invocation, so network A's C++ object was destroyed before B's
// was even constructed; they were never simultaneously alive at all, let
// alone overlapping in flight. Genuine overlapping GPU submission still
// needs the not-yet-built hook infrastructure noted above and is NOT added
// here. What this rework does add, safely and single-threaded: both
// DirectMlNetwork instances are constructed ONCE, up front, and stay alive
// for the whole test, so their per-instance state (weight uploads, arenas,
// the DmlDeviceContext they each own) genuinely coexists while calls
// alternate between them -- a real strengthening of "two networks coexist"
// even though true concurrent-submission overlap is still future work.
TEST(DirectMlRegressionCoverage, TwoNetworkInterleaveDoesNotCorruptResults) {
  if (!HasBackend("directml"))
    GTEST_SKIP() << "directml backend not compiled in";
  if (!DirectMlAvailability().available) {
    GTEST_SKIP() << "no usable directml device: "
                 << DirectMlAvailability().reason;
  }
  const InputPlanes planes = EncodeStartPos();

  // No new directory needed -- TestBinaryDir() (where the test binary
  // itself lives) already exists, and DumpBodyStage's writes create files,
  // not directories, at whatever prefix this names.
  const std::string dump_prefix =
      TestBinaryDir() + "interleave_dump_smoke";
  ScopedEnvVar dump_body_env("LC0_DUMP_BODY", dump_prefix.c_str());
  ScopedEnvVar profile_env("LC0_DML_PROFILE", "1");

  // Both networks constructed up front and held for the whole test -- see
  // the DML-4 comment above for why this matters.
  OptionsDict options_a, options_b;
  auto network_a = NetworkFactory::Get()->Create(
      "directml", MakeKdaMlhNet(), options_a);   // 1 KDA encoder.
  auto network_b = NetworkFactory::Get()->Create(
      "directml", MakeKdaMhaNet(), options_b);   // 1 KDA + 1 MHA encoder.

  auto eval_once = [&](Network* network) {
    auto computation = network->NewComputation();
    computation->AddInput(InputPlanes(planes));
    computation->ComputeBlocking();
    Outputs out;
    out.q = computation->GetQVal(0);
    out.d = computation->GetDVal(0);
    out.m = computation->GetMVal(0);
    out.policy.reserve(1858);
    for (int i = 0; i < 1858; ++i) out.policy.push_back(computation->GetPVal(0, i));
    return out;
  };

  const Outputs a1 = eval_once(network_a.get());
  const Outputs b = eval_once(network_b.get());
  const Outputs a2 = eval_once(network_a.get());

  (void)b;  // Only used to interleave a different network's eval between A's two.
  EXPECT_EQ(a1.q, a2.q) << "network A's Q changed after B's eval ran between "
                           "A's two calls -- possible cross-network state leak";
  EXPECT_EQ(a1.d, a2.d);
  EXPECT_EQ(a1.m, a2.m);
  for (int i = 0; i < 1858; ++i) {
    ASSERT_EQ(a1.policy[i], a2.policy[i])
        << "network A's policy[" << i << "] changed after B's interleaved eval";
  }
}

// agora thread 19 P4 (viii), muse-spark's #677/#668 (E15 -- RR2's own
// stated exit criterion, codex-sol's #650 audit): GPU-Based Validation
// messages only ever reached CERR as printed text -- nothing turned an
// ERROR/CORRUPTION severity into an actual test failure, so a regression
// reintroducing a resource-state bug would print a warning nobody's CI
// reads and every numeric test would still stay green. This closes that
// gate: enable the debug layer + GBV, reset GbvErrorCount(), run a
// representative net (the same one gbv.log's evidence used throughout
// this triage), and fail if anything ERROR/CORRUPTION-severity fired.
// Expected to PASS now that RR2/P2 fixed the one persistent violation this
// exact scenario used to report.
//
// agora thread 19 DML-4 (codex-sol #724): errors==0 alone cannot
// distinguish "validation ran clean" from "validation never actually
// turned on" -- D3D12GetDebugInterface/ID3D12Debug1/ID3D12InfoQueue1 all
// fail soft (network_directml.cc), so a machine missing the Windows SDK's
// Graphics Tools optional feature would pass this exact assertion for the
// wrong reason. Also assert GbvActuallyActive() (dml_common.h), set true
// only once SetEnableGPUBasedValidation AND RegisterMessageCallback have
// both actually succeeded -- this job must fail loudly on a machine where
// GBV silently didn't start, not report a false green.
TEST(DirectMlRegressionCoverage, GbvReportsNoErrorsOnRepresentativeNet) {
  if (!HasBackend("directml"))
    GTEST_SKIP() << "directml backend not compiled in";
  if (!DirectMlAvailability().available) {
    GTEST_SKIP() << "no usable directml device: "
                 << DirectMlAvailability().reason;
  }
  ScopedEnvVar debug_layer_env("LC0_DML_DEBUG_LAYER", "1");
  ScopedEnvVar gbv_env("LC0_DML_GBV", "1");
  using directml_backend::GbvActuallyActive;
  using directml_backend::GbvErrorCount;
  GbvErrorCount() = 0;

  {
    OptionsDict options;
    auto network =
        NetworkFactory::Get()->Create("directml", MakeKdaMlhNet(), options);
    auto computation = network->NewComputation();
    computation->AddInput(InputPlanes(EncodeStartPos()));
    computation->ComputeBlocking();
  }

  const int errors = GbvErrorCount();
  const bool gbv_active = GbvActuallyActive();
  ASSERT_TRUE(gbv_active)
      << "GPU-Based Validation never actually activated on this machine "
         "(D3D12GetDebugInterface/ID3D12Debug1/ID3D12InfoQueue1 failed "
         "soft somewhere -- see the CERR output above for which step) -- "
         "errors==0 in that case proves nothing; this job must fail "
         "loudly rather than report a false green (DML-4, agora #724)";
  EXPECT_EQ(errors, 0)
      << "GPU-Based Validation reported " << errors << " ERROR/CORRUPTION "
         "severity message(s) -- see the CERR output above for detail "
         "(P4 viii, agora thread 19 #677; this gate was green-lit by "
         "RR2/P2's fix, do not silently widen it back open)";
}

// F6: F32toF16Bits (reached only through DmlHalf's public float constructor
// -- the function itself is file-local) used to consume a pre-truncated
// 10-bit mantissa in its Inf/NaN and subnormal-rounding paths, discarding
// the sticky/payload bits those two paths specifically needed. Reproduces
// codex-sol's exact two counterexamples in-tree, matching the standalone
// bite-test run before commit 1b32741.
TEST(DirectMlRegressionCoverage, Fp16ConverterEdgeCases) {
  using directml_backend::DmlHalf;
  auto bits_to_float = [](uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  };

  // FP32 0x33000001: just above the midpoint between half zero and the
  // smallest positive subnormal -- round-to-nearest-even must round UP to
  // 0x0001. The pre-fix code lost the sticky bit and rounded down to 0x0000.
  {
    const DmlHalf h(bits_to_float(0x33000001u));
    EXPECT_EQ(h.bits, 0x0001u)
        << "subnormal round-to-nearest-even lost the sticky bit (F6)";
  }

  // FP32 0x7f800001: a NaN with its payload bit at position 0, below the old
  // >>13 truncation. Must stay a NaN (exponent field all-ones, mantissa
  // nonzero), not collapse to +Infinity (exponent all-ones, mantissa zero).
  {
    const DmlHalf h(bits_to_float(0x7f800001u));
    EXPECT_EQ(h.bits & 0x7c00u, 0x7c00u)
        << "expected the Inf/NaN exponent pattern";
    EXPECT_NE(h.bits, 0x7c00u)
        << "NaN payload was silently truncated to +Infinity (F6)";
  }

  // Ordinary values never reach the two paths touched by F6 (they land on
  // the ordinary-range rounding path, which was already correct) -- confirm
  // that by round-tripping float -> half -> float and checking the result
  // is finite and within fp16's representable precision of the original,
  // rather than merely "didn't crash".
  const float kOrdinary[] = {0.0f,      -0.0f,     1.0f,  -1.0f, 0.5f,
                             0.001f,    -3.14159f, 1e-5f, 65504.0f};
  for (float v : kOrdinary) {
    const float round_tripped = static_cast<float>(DmlHalf(v));
    ASSERT_TRUE(std::isfinite(round_tripped))
        << "ordinary value " << v << " round-tripped to a non-finite half";
    const float tolerance = std::max(std::fabs(v) * 1e-2f, 1e-6f);
    EXPECT_NEAR(round_tripped, v, tolerance)
        << "ordinary value " << v << " diverged beyond fp16 precision";
  }
}

// ---------------------------------------------------------------------------
// B1/R3 regression (agora thread 19 #620/#622): DmlDeviceContext::WaitForFence
// used to sample ID3D12Fence::GetCompletedValue() TWICE for what should be
// one logical checkpoint -- once for the device-removal sentinel check, a
// second, independent time for the `>= value` fast-path compare. A device
// removal landing between the two calls let the sentinel check see the
// pre-removal value (fine) while the second call's fresh UINT64_MAX
// satisfied `>= value` for any real value, returning success on a batch
// that never completed. Fixed to sample once and reuse that value for both
// decisions.
//
// This calls the REAL, unmodified DmlDeviceContext::WaitForFence (not a
// copy) via a minimal ID3D12Fence COM double whose GetCompletedValue()
// returns a scripted sequence, and whose SetEventOnCompletion immediately
// signals the real event handle it's given (simulating "the GPU completed
// right away") so the real WaitForSingleObject call in the slow path
// returns immediately instead of hanging on an event nothing would ever
// signal. Adapted from codex-sol's standalone harness shape
// (docs/review-evidence/directml-host-review-2026-09-07.cc), promoted to
// a real in-tree gtest exercising the production code directly.
namespace {
class ScriptedFence : public ID3D12Fence {
 public:
  explicit ScriptedFence(std::vector<UINT64> values)
      : values_(std::move(values)) {}

  // IUnknown -- not testing COM lifetime, stack-allocated test double.
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override {
    return E_NOINTERFACE;
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
  ULONG STDMETHODCALLTYPE Release() override { return 1; }

  // ID3D12Object
  HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override {
    return E_NOTIMPL;
  }
  HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT,
                                           const void*) override {
    return E_NOTIMPL;
  }
  HRESULT STDMETHODCALLTYPE
  SetPrivateDataInterface(REFGUID, const IUnknown*) override {
    return E_NOTIMPL;
  }
  HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return E_NOTIMPL; }

  // ID3D12DeviceChild
  HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void**) override {
    return E_NOTIMPL;
  }

  // ID3D12Fence -- the methods WaitForFence actually calls.
  UINT64 STDMETHODCALLTYPE GetCompletedValue() override {
    // .at() throws std::out_of_range (loudly) if WaitForFence samples more
    // times than the scripted scenario expects, rather than silently
    // repeating the last value -- a call-count regression fails visibly.
    return values_.at(call_count_++);
  }
  HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64,
                                                 HANDLE event) override {
    // Simulate the GPU completing immediately: signal the real event handle
    // WaitForFence passed in, so its real WaitForSingleObject(event,
    // INFINITE) call returns right away instead of blocking forever on an
    // event nothing else would ever signal in this test.
    if (event) ::SetEvent(event);
    return S_OK;
  }
  HRESULT STDMETHODCALLTYPE Signal(UINT64) override { return S_OK; }

  size_t CallCount() const { return call_count_; }

 private:
  std::vector<UINT64> values_;
  size_t call_count_ = 0;
};
}  // namespace

TEST(DirectMlRegressionCoverage, FenceRaceDoesNotMaskDeviceRemoval) {
  if (!DirectMlAvailability().available) {
    GTEST_SKIP() << "no usable directml device: "
                 << DirectMlAvailability().reason;
  }
  directml_backend::DmlDeviceContext ctx;
  ctx.Init(OptionsDict());

  // Scenario 1, the actual bug: removal lands between what used to be two
  // separate GetCompletedValue() calls. The fixed code samples once (call 0
  // returns 0, which is < the target of 1), falls through to the slow path,
  // and must throw on its second sample (call 1 returns UINT64_MAX).
  {
    ScriptedFence racing({0, UINT64_MAX});
    EXPECT_THROW(ctx.WaitForFence(&racing, 1), Exception)
        << "a device removal between fence samples must not be masked as "
           "success";
    EXPECT_EQ(racing.CallCount(), 2u)
        << "expected exactly one fast-path sample plus one post-wait sample";
  }

  // Scenario 2 (control): immediate removal must always throw, and must be
  // caught on the very first sample without ever reaching the slow path.
  {
    ScriptedFence removed({UINT64_MAX});
    EXPECT_THROW(ctx.WaitForFence(&removed, 1), Exception)
        << "immediate device removal must throw";
    EXPECT_EQ(removed.CallCount(), 1u)
        << "immediate removal must be caught on the first sample, before "
           "any wait";
  }

  // Scenario 3 (control): genuine completion must return normally via the
  // fast path, consuming exactly one sample.
  {
    ScriptedFence completed({1});
    EXPECT_NO_THROW(ctx.WaitForFence(&completed, 1))
        << "a fence that has already reached the target value must not "
           "throw";
    EXPECT_EQ(completed.CallCount(), 1u)
        << "a genuinely-completed fence must take the fast path (one "
           "sample), not fall through to the slow path";
  }
}

// B3 (agora thread 19 #620/#622): the dispatch-group-count guard's own
// threshold logic, tested directly (no GPU or real net needed -- it's pure
// host-side arithmetic) rather than only indirectly through an exotic wide
// real-net fixture. D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION is
// 65535; confirms the guard throws exactly above it and not at or below.
TEST(DirectMlRegressionCoverage, DispatchGroupCountGuardThreshold) {
  using directml_backend::CheckDispatchGroupCount;
  EXPECT_NO_THROW(CheckDispatchGroupCount(65535, "test"))
      << "exactly at the D3D12 limit must not throw";
  EXPECT_NO_THROW(CheckDispatchGroupCount(1, "test"))
      << "a trivially small group count must not throw";
  EXPECT_THROW(CheckDispatchGroupCount(65536, "test"), Exception)
      << "one over the D3D12 limit must throw";
  EXPECT_THROW(CheckDispatchGroupCount(256ull * 256ull, "test"), Exception)
      << "the concrete 256x256 wide-MHA-at-max-batch-256 case from #620 "
         "package B3 must throw (65536 groups)";
}

// RR4 (agora thread 19 #650/#652, codex-sol's release audit): network_
// directml.cc:789/874/1217/1236 (four Map() call sites) were unchecked --
// each site's dependent-use code (memset/memcpy/f.write through the mapped
// pointer) would run against a garbage pointer on a failed Map instead of
// failing cleanly at the call site. The fix at all 4 sites is
// ReportD3DErrors(...->Map(...), "..."), the exact same wrapper the
// InputsOutputs constructor's earlier A2 fix already established as the
// pattern. Real D3D12 Map() failure injection needs a mock/scripted
// resource (no such Map-failing double exists for a real allocated
// ID3D12Resource, unlike ScriptedFence's real COM interface for
// WaitForFence's B1 fix) -- so this failure-injection test targets the one
// shared mechanism every RR4 site (and every pre-existing ReportD3DErrors
// call in this backend) actually depends on: that a FAILED HRESULT reaches
// the dependent-use boundary as a clean throw, never a silent pass-through
// to code that dereferences whatever Map happened to write on failure.
TEST(DirectMlRegressionCoverage, ReportD3DErrorsThrowsOnFailedHresult) {
  using directml_backend::ReportD3DErrors;
  EXPECT_NO_THROW(ReportD3DErrors(S_OK, "ok case"))
      << "a successful HRESULT must not throw -- no behavior change on the "
         "success path is the explicit RR4 authorization condition";
  EXPECT_THROW(ReportD3DErrors(E_FAIL, "generic failure"), Exception)
      << "a failed HRESULT must throw before any caller can dereference an "
         "indeterminate pointer from the failed call";
  EXPECT_THROW(ReportD3DErrors(E_OUTOFMEMORY, "out of memory"), Exception)
      << "the specific HRESULT a real Map() failure is most likely to "
         "return (driver/allocator exhaustion) must also throw";
  EXPECT_THROW(ReportD3DErrors(DXGI_ERROR_DEVICE_REMOVED, "device removed"),
              Exception)
      << "a removed-device HRESULT, which a Map() call can legitimately "
         "return mid-eval, must also throw rather than hand back a garbage "
         "pointer";
}

// B4 (agora thread 19 #620/#622): kda_recurrence.hlsl computes
// `direction_index = head / (heads / direction_count)` on the GPU --
// direction_count 0 (empty kda_directions list) is a GPU-side divide by
// zero, and a count that does not evenly divide heads produces an
// out-of-range direction_index for the largest head indices. Both used to
// be uncaught (the per-element 1-16 range check in network_directml.cc is
// vacuously true on an empty list, and never checked the count against
// heads at all); EncoderBlock's KDA branch now throws at construction
// instead of letting either reach the GPU.
TEST(DirectMlKdaParity, KdaEmptyDirectionsRejected) {
  // ctx_.Init() (real device creation) runs before this validation in the
  // constructor, so on a machine with no usable DirectML device Create()
  // would throw for that unrelated reason first and this test would pass
  // for the wrong reason -- skip rather than give a false positive.
  if (!DirectMlAvailability().available) {
    GTEST_SKIP() << "no usable directml device: "
                 << DirectMlAvailability().reason;
  }
  pblczero::Net file = MakeKdaSerpentineNet();
  file.mutable_format()->mutable_network_format()->mutable_kda_directions()->clear();
  OptionsDict options;
  EXPECT_THROW(NetworkFactory::Get()->Create("directml", file, options),
               Exception)
      << "an empty kda_directions list (direction_count 0) must be "
         "rejected at load, not reach a GPU-side divide by zero";
}

TEST(DirectMlKdaParity, KdaDirectionCountMustDivideHeads) {
  if (!DirectMlAvailability().available) {
    GTEST_SKIP() << "no usable directml device: "
                 << DirectMlAvailability().reason;
  }
  // MakeKdaSerpentineNet's default NetDims has heads=8; 3 directions do not
  // evenly divide it (8 % 3 == 2), which used to silently mismap the
  // largest head indices to an out-of-range direction_index on the GPU.
  pblczero::Net file = MakeKdaSerpentineNet();
  auto* nf = file.mutable_format()->mutable_network_format();
  nf->mutable_kda_directions()->clear();
  using NF = pblczero::NetworkFormat;
  for (int dir : {1, 2, 3}) {
    nf->add_kda_directions(static_cast<NF::KdaDirection>(dir));
  }
  OptionsDict options;
  EXPECT_THROW(NetworkFactory::Get()->Create("directml", file, options),
               Exception)
      << "heads=8 is not evenly divisible by 3 scan directions; this must "
         "be rejected at load, not silently mismap head-to-direction "
         "indices on the GPU";
}

// E3 (agora thread 19 #620/#628): LC0_TEST_BACKEND=blas would compare the
// reference against itself, passing trivially without testing anything.
// Tests ValidateTestBackendChoice directly rather than via the env var
// (no GPU or real comparison needed -- it's pure host-side logic).
TEST(DirectMlRegressionCoverage, RejectsSelfComparisonBackend) {
  EXPECT_FATAL_FAILURE(ValidateTestBackendChoice("blas"), "blas");
  EXPECT_FATAL_FAILURE(ValidateTestBackendChoice("nonsense_backend"),
                       "nonsense_backend");
  // None of the allowed backends should produce any failure.
  ValidateTestBackendChoice("directml");
  ValidateTestBackendChoice("directml-fp16");
  ValidateTestBackendChoice("eigen");
}

// E2 (agora thread 19 #620/#628): directml-fp16 is gated behind
// allow_broken_fp16 precisely because its precision is known-incomplete
// (this session's own #620/#626 stash A/B on this exact net confirmed a
// policy-argmax mismatch that is bit-for-bit reproducible, not noise).
// "Gated but measured": Q/D/M must still sit inside the already-calibrated
// fp16 tolerance (kAbsTolFp16/kScaleTolFp16) -- if THOSE regress, something
// broke beyond the known gap. The policy argmax mismatch is asserted to
// CURRENTLY differ from blas, on purpose: if this ever starts passing, the
// underlying precision issue was fixed and this test must be updated (or
// removed) to say so explicitly, rather than a silent pass hiding that the
// gate is stricter than it needs to be. Constructs the network directly
// (not via RunNetwork's env-var-driven ApplyTestBackendOpts) so this test
// exercises directml-fp16 + allow_broken_fp16 regardless of what
// LC0_TEST_BACKEND/LC0_TEST_BACKEND_OPTS happen to be set to.
TEST(DirectMlKdaParity, Fp16ParityIsGatedButMeasured) {
  if (!HasBackend("directml-fp16")) {
    GTEST_SKIP() << "directml-fp16 backend not compiled in";
  }
  const pblczero::Net net = MakeKdaMlhNet();
  const InputPlanes planes = EncodeStartPos();

  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  const Outputs reference = RunNetwork("blas", net, planes);

  OptionsDict fp16_options;
  fp16_options.Set<bool>("allow_broken_fp16", true);
  auto fp16_network =
      NetworkFactory::Get()->Create("directml-fp16", net, fp16_options);
  auto fp16_computation = fp16_network->NewComputation();
  fp16_computation->AddInput(InputPlanes(planes));
  fp16_computation->ComputeBlocking();
  Outputs fp16_out;
  fp16_out.q = fp16_computation->GetQVal(0);
  fp16_out.d = fp16_computation->GetDVal(0);
  fp16_out.m = fp16_computation->GetMVal(0);
  fp16_out.policy.reserve(1858);
  for (int i = 0; i < 1858; ++i) {
    fp16_out.policy.push_back(fp16_computation->GetPVal(0, i));
  }

  AssertFiniteOutputs(fp16_out, "directml-fp16");
  AssertFiniteOutputs(reference, "blas reference");

  // The "fp16 bar" that must still hold.
  EXPECT_NEAR(fp16_out.q, reference.q, BoundedOutputBound(/*fp16=*/true))
      << "WDL value Q regressed beyond the already-calibrated fp16 bar";
  EXPECT_NEAR(fp16_out.d, reference.d, BoundedOutputBound(/*fp16=*/true))
      << "WDL draw probability regressed beyond the already-calibrated "
         "fp16 bar";
  EXPECT_NEAR(fp16_out.m, reference.m,
             ScaledOutputBound(reference.m, /*fp16=*/true))
      << "moves-left regressed beyond the already-calibrated fp16 bar";

  int fp16_best = 0, ref_best = 0;
  for (int i = 1; i < 1858; ++i) {
    if (fp16_out.policy[i] > fp16_out.policy[fp16_best]) fp16_best = i;
    if (reference.policy[i] > reference.policy[ref_best]) ref_best = i;
  }
  // Asserted as a KNOWN, currently-reproduced divergence -- see the test's
  // own comment above. Not a bug in the test; a tripwire on the bug it's
  // measuring.
  EXPECT_NE(fp16_best, ref_best)
      << "policy argmax now MATCHES blas (directml picks " << fp16_best
      << ", blas picks " << ref_best
      << ") -- the known fp16 precision gap this test tracks appears to be "
         "fixed; update or remove this test's asserted-broken expectation "
         "rather than leaving it silently passing for the wrong reason";
}

// TEMPORARY, uncommitted, agora thread 19 #620/#635 directive (a): the
// decisive contamination experiment. Runs the same 2 positions once as a
// batch of 2 (directml), and once each as an independent batch of 1
// (directml), then diffs batch-2's sample against the solo run of the
// SAME position. If they match, whatever's wrong is per-position numerics
// (not what the earlier findings look like); if they differ, sample 1's
// slot in the batch-2 run is reading contaminated/wrong data -- a
// stride/offset bug, not a precision one. Prints only; no EXPECT/ASSERT
// (diagnostic, not a permanent assertion), no fix, no tolerance change.
TEST(DirectMlKdaParity, DISABLED_TriageBatchContamination) {
  const char* path = getenv("LC0_TEST_REAL_NET");
  if (!path) GTEST_SKIP() << "set LC0_TEST_REAL_NET to a .pb.gz to run";
  pblczero::Net net = LoadWeightsFromFile(path);
  CERR << "[dims] encoders=" << net.weights().encoder_size()
       << " embedding=" << LayerAdapter(net.weights().ip_emb_b()).size()
       << " headcount=" << net.weights().headcount();
  for (int i = 0; i < net.weights().encoder_size(); ++i) {
    const auto& enc = net.weights().encoder(i);
    CERR << "[dims] enc" << i << " mixer="
         << (enc.mixer() == pblczero::Weights::EncoderLayer::MIXER_KDA
                 ? "KDA"
                 : "MHA");
    if (enc.mixer() == pblczero::Weights::EncoderLayer::MIXER_KDA) {
      CERR << "[dims] enc" << i << " local_conv=" << enc.kda().local_conv()
           << " qkv_silu=" << enc.kda().qkv_silu()
           << " gate_rank=" << enc.kda().gate_rank()
           << " key_dim=" << enc.kda().key_dim()
           << " value_dim=" << enc.kda().value_dim();
    } else {
      CERR << "[dims] enc" << i << " has_smolgen=" << enc.mha().has_smolgen();
      if (enc.mha().has_smolgen()) {
        const auto& sg = enc.mha().smolgen();
        const uint64_t emb = LayerAdapter(net.weights().ip_emb_b()).size();
        const uint64_t hidden_channels =
            emb ? LayerAdapter(sg.compress()).size() / emb : 0;
        // Agora #668 P1 step 1: resolving muse-spark's stated caveat ("exact
        // real C_c/D1/D2 unverified, .pb.gz no longer on disk") -- the net
        // is still on disk for this session, so print the REAL smolgen
        // dims directly instead of guessing/approximating them.
        CERR << "[dims] enc" << i
             << " smolgen hidden_channels=" << hidden_channels
             << " hidden_sz=" << LayerAdapter(sg.dense1_b()).size()
             << " gen_outputs=" << LayerAdapter(sg.dense2_b()).size();
      }
    }
  }
  // Agora #637 audit: mirror network_directml.cc's scratch_elems maximand
  // computation by hand (fp32, scale_rec=1) to check whether the MHA
  // encoder's 8*d_model term or a KDA encoder's term is the actual binding
  // constraint for these real nets -- see the reasoning posted to thread 19
  // about EvalMha's buffer1 needing 5*d_model*max_tokens*sizeof(float)
  // bytes out of a half-scratch region only guaranteed >= scratch_bytes_/2.
  {
    const MultiHeadWeights decoded{net.weights()};
    const uint64_t emb_size = decoded.ip_emb_b.size();
    uint64_t scratch_elems = 0;
    uint64_t mha_need = 0, kda_need = 0;
    for (const auto& enc : decoded.encoder) {
      if (enc.is_kda) {
        const uint64_t KD =
            (uint64_t)decoded.encoder_head_count * enc.kda.key_dim;
        const uint64_t VD =
            (uint64_t)decoded.encoder_head_count * enc.kda.value_dim;
        const uint64_t need =
            (2 * KD + VD + std::max<uint64_t>(2 * KD, VD + 3 * enc.kda.key_dim)) +
            enc.kda.gate_rank + emb_size +
            (enc.kda.local_conv ? emb_size : 0);
        kda_need = std::max(kda_need, need);
        scratch_elems = std::max(scratch_elems, need);
      } else {
        const uint64_t d_model =
            !enc.mha.q_w.empty() ? enc.mha.q_w.size() / emb_size : emb_size;
        const uint64_t need = 8 * d_model;
        mha_need = std::max(mha_need, need);
        scratch_elems = std::max(scratch_elems, need);
        CERR << "[arena] MHA encoder d_model=" << d_model
             << " buffer1_needs(5*d_model)=" << 5 * d_model
             << " half_of(8*d_model)=" << (8 * d_model) / 2;
      }
    }
    CERR << "[arena] scratch_elems(fp32 approx)=" << scratch_elems
         << " kda_term_max=" << kda_need << " mha_term(8*d_model)=" << mha_need
         << " -- buffer1 gets AlignUp(scratch_elems/2) elems, "
         << "MHA needs 5*d_model contiguous: "
         << ((scratch_elems / 2) >= 5 * (mha_need / 8) ? "OK (fits)"
                                                        : "SHORTFALL");
  }
  const std::vector<InputPlanes> planes = EncodeDistinctPositions(2);

  const std::vector<Outputs> batch2 = RunNetworkBatch("directml", net, planes);
  const Outputs solo0 = RunNetwork("directml", net, planes[0]);
  const Outputs solo1 = RunNetwork("directml", net, planes[1]);

  auto worst_policy_diff = [](const std::vector<float>& a,
                              const std::vector<float>& b, int* move) {
    float worst = 0.0f;
    for (int i = 0; i < 1858; ++i) {
      const float d = std::fabs(a[i] - b[i]);
      if (d > worst) {
        worst = d;
        *move = i;
      }
    }
    return worst;
  };

  int move0 = -1, move1 = -1;
  const float p0_diff =
      worst_policy_diff(batch2[0].policy, solo0.policy, &move0);
  const float p1_diff =
      worst_policy_diff(batch2[1].policy, solo1.policy, &move1);

  CERR << "[contamination] sample 0: batch2 vs solo -- Q diff="
       << std::fabs(batch2[0].q - solo0.q)
       << " D diff=" << std::fabs(batch2[0].d - solo0.d)
       << " policy worst diff=" << p0_diff << " at move " << move0;
  CERR << "[contamination] sample 1: batch2 vs solo -- Q diff="
       << std::fabs(batch2[1].q - solo1.q)
       << " D diff=" << std::fabs(batch2[1].d - solo1.d)
       << " policy worst diff=" << p1_diff << " at move " << move1;
  CERR << "[contamination] solo0 vs solo1 policy worst diff (sanity: these "
          "are genuinely different positions) ="
       << worst_policy_diff(solo0.policy, solo1.policy, &move0);
}

// Agora thread 19 #652 item 4 / #656 (muse-spark, user-directed): the RR1
// real-net LC0_DUMP_BODY bisection. LC0_DUMP_BODY's file prefix is fixed
// for the whole process and each forwardEval call's drain truncates and
// overwrites its stage files -- DISABLED_TriageBatchContamination above
// does 3 evaluations (batch2, solo0, solo1) in ONE process, so its dumps
// would only ever reflect the LAST call. This harness does exactly ONE
// evaluation per process invocation, chosen by LC0_TEST_TRIAGE_BATCH (1 =
// solo RunNetwork on position 0, N>1 = RunNetworkBatch on N distinct
// positions) -- run it twice with different LC0_DUMP_BODY prefixes (once
// at batch 1, once at the batch under investigation) to get two clean,
// non-overwritten dump sets to diff stage-by-stage. No EXPECT/ASSERT, no
// tolerance, no fix -- dumps only, exactly as directed. E1 untouched.
// Agora thread 19 #725/DML-1 residual localization (claude-opus, post-D1-fix):
// LC0_TEST_TRIAGE_BACKEND (default "directml") added so this same harness can
// capture a BLAS ".blas.<stage>.bin" dump set for the identical position,
// letting the stage-by-stage diff that originally localized RR1 to enc3 be
// rerun against the current source to see whether the much smaller residual
// (2pass/4fail per codex-sol's #723/#724) still enters at the same stage.
// Still dumps only -- no EXPECT/ASSERT, no tolerance, no fix.
TEST(DirectMlKdaParity, DISABLED_DumpSingleRealNetEval) {
  const char* path = getenv("LC0_TEST_REAL_NET");
  if (!path) GTEST_SKIP() << "set LC0_TEST_REAL_NET to a .pb.gz to run";
  const char* backend_env = getenv("LC0_TEST_TRIAGE_BACKEND");
  const std::string backend = backend_env ? backend_env : "directml";
  const char* batch_env = getenv("LC0_TEST_TRIAGE_BATCH");
  const int batch = batch_env ? std::atoi(batch_env) : 1;
  pblczero::Net net = LoadWeightsFromFile(path);
  if (batch <= 1) {
    // LC0_TEST_TRIAGE_POS (default 0): which EncodeDistinctPositions index
    // to run solo -- needed to get a genuine standalone-batch1 reference
    // for sample N>0 of a larger batch (position i's FEN only depends on
    // i, not on the requested count, so EncodeDistinctPositions(pos+1)[pos]
    // is byte-identical to that same slot in a bigger batch's plane list).
    const char* pos_env = getenv("LC0_TEST_TRIAGE_POS");
    const int pos = pos_env ? std::atoi(pos_env) : 0;
    const std::vector<InputPlanes> planes = EncodeDistinctPositions(pos + 1);
    RunNetwork(backend, net, planes[pos]);
  } else {
    const std::vector<InputPlanes> planes = EncodeDistinctPositions(batch);
    RunNetworkBatch(backend, net, planes);
  }
}

}  // namespace lczero

// ---------------------------------------------------------------------------
// OnnxGraph: the converter's output as the DirectML translator loads it. The
// loader refuses any operator outside its vocabulary, so these tests are also
// the contract between the converter and the translator: a converter change
// that emits a new operator fails here until the translator knows it.
namespace lczero {
namespace {

pblczero::ModelProto ConvertForTranslator(
    const pblczero::Net& net, WeightsToOnnxConverterOptions::DataType type) {
  WeightsToOnnxConverterOptions options;
  options.data_type = type;
  const pblczero::Net converted =
      ConvertWeightsToOnnx(NetForTestBackend("onnx-dml", net), options);
  pblczero::ModelProto model;
  model.ParseFromString(converted.onnx_model().model());
  return model;
}

// Every value a node produces must have a type and a positive static shape,
// and the model must end in a policy of 1858 moves per position.
void ExpectFullyInferred(const directml_backend::OnnxGraph& graph, int batch,
                         bool is_main_graph) {
  for (const auto& node : graph.nodes()) {
    for (const int output : node.outputs) {
      const auto& value = graph.values()[output];
      // LayerNormalization's optional outputs are never used.
      if (value.dims.empty() && node.op_type == "LayerNormalization") continue;
      EXPECT_NE(value.data_type, pblczero::TensorProto::UNDEFINED)
          << node.op_type << " " << node.name;
      for (const int64_t dim : value.dims) {
        EXPECT_GT(dim, 0) << node.op_type << " " << node.name;
      }
    }
    if (node.body) ExpectFullyInferred(*node.body, batch, false);
  }
  if (!is_main_graph) return;
  bool has_policy = false;
  for (const int output : graph.outputs()) {
    const auto& dims = graph.values()[output].dims;
    ASSERT_FALSE(dims.empty());
    EXPECT_EQ(dims[0], batch);
    has_policy = has_policy || dims == std::vector<int64_t>{batch, 1858};
  }
  EXPECT_TRUE(has_policy);
}

TEST(DirectMlOnnxGraph, LoadsWhatTheConverterEmits) {
  const std::vector<pblczero::Net> nets = {
      MakeKdaHybridNet(),
      MakeKdaMhaNet(),
      MakeThreeKdaThenMhaNet(),
      MakeKdaMlhNet(),
      MakeLocalConvNet(7),
      MakeFullRealisticNet(true, true, true),
      MakeFullRealisticNet(false, false, false),
      MakeKdaSerpentineNet(),
      MakeMhaMlhNet(),
      MakeSmolgenMhaNet(),
      MakePeDenseWithEncodersNet(),
      MakePeDenseGatedNet(11)};
  using DataType = WeightsToOnnxConverterOptions::DataType;
  for (size_t i = 0; i < nets.size(); ++i) {
    for (const DataType type : {DataType::kFloat32, DataType::kFloat16}) {
      const pblczero::ModelProto model = ConvertForTranslator(nets[i], type);
      for (const int batch : {1, 8}) {
        SCOPED_TRACE("net " + std::to_string(i) + ", batch " +
                     std::to_string(batch));
        const directml_backend::OnnxGraph graph(model, batch);
        ExpectFullyInferred(graph, batch, true);
      }
    }
  }
}

TEST(DirectMlOnnxGraph, InfersTheRecurrenceShapes) {
  const NetDims d;
  const int batch = 4;
  const pblczero::ModelProto model = ConvertForTranslator(
      MakeKdaMlhNet(), WeightsToOnnxConverterOptions::DataType::kFloat32);
  const directml_backend::OnnxGraph graph(model, batch);
  int scans = 0;
  for (const auto& node : graph.nodes()) {
    if (node.op_type != "Scan") continue;
    ++scans;
    ASSERT_TRUE(node.body);
    // One state and five scanned inputs; the body sees the inputs without
    // the axis of the 64 squares.
    ASSERT_EQ(node.inputs.size(), 6u);
    ASSERT_EQ(node.body->inputs().size(), 6u);
    const auto& query = graph.values()[node.inputs[1]].dims;
    const auto& body_query = node.body->values()[node.body->inputs()[1]].dims;
    ASSERT_EQ(query.size(), 4u);
    EXPECT_EQ(query[0], batch);
    EXPECT_EQ(query[1], 64);
    EXPECT_EQ(query[2], d.heads);
    EXPECT_EQ(body_query, (std::vector<int64_t>{query[0], query[2], query[3]}));
    // The stacked output puts the squares back on axis 1.
    ASSERT_EQ(node.outputs.size(), 2u);
    const auto& stacked = graph.values()[node.outputs[1]].dims;
    ASSERT_EQ(stacked.size(), 4u);
    EXPECT_EQ(stacked[0], batch);
    EXPECT_EQ(stacked[1], 64);
    EXPECT_EQ(stacked[2], d.heads);
  }
  EXPECT_GT(scans, 0);
}

TEST(DirectMlOnnxGraph, RefusesAnOperatorOutsideTheVocabulary) {
  pblczero::ModelProto model = ConvertForTranslator(
      MakeKdaMlhNet(), WeightsToOnnxConverterOptions::DataType::kFloat32);
  ASSERT_FALSE(model.graph().node().empty());
  model.mutable_graph()->mutable_node()->back().set_op_type("Erf");
  EXPECT_THROW(directml_backend::OnnxGraph(model, 1), Exception);
}

// The same on a real net, which exercises real dimensions and the embedding
// and head variants no synthetic net here has.
TEST(DirectMlOnnxGraph, LoadsARealNet) {
  const char* path = getenv("LC0_TEST_REAL_NET");
  if (!path) GTEST_SKIP() << "set LC0_TEST_REAL_NET to a .pb.gz to run";
  const pblczero::Net net = LoadWeightsFromFile(path);
  using DataType = WeightsToOnnxConverterOptions::DataType;
  for (const DataType type : {DataType::kFloat32, DataType::kFloat16}) {
    const pblczero::ModelProto model = ConvertForTranslator(net, type);
    const directml_backend::OnnxGraph graph(model, 16);
    ExpectFullyInferred(graph, 16, true);
  }
}

}  // namespace
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
