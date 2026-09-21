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
#include <cstdlib>
#include <latch>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "chess/board.h"
#include "chess/position.h"
#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/loader.h"
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

// Fixture knobs for the coverage the review asked for (S1 bias presence,
// S7 direction families / local conv). Defaults reproduce the original
// one-layer, all-biases, directions-1..4 fixture.
struct KdaNetOptions {
  int heads = 8;
  std::vector<int> directions = {1, 2, 3, 4};
  bool q_b = true;
  bool k_b = true;
  bool v_b = true;
  bool decay_a_b = true;
  bool gate_a_b = true;
  bool local_conv = false;
  bool qkv_silu = true;
  bool output_gate = true;
  bool output_rms_norm = true;
  // MHA mixer instead of the KDA mixer (item 4: both encoder stacks).
  bool mha_mixer = false;
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
pblczero::Net MakeKdaHybridNet(const KdaNetOptions& opt = {}) {
  const NetDims d;
  const int heads = opt.heads;
  const int key_depth = heads * d.key_dim;
  const int value_depth = heads * d.value_dim;
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
  // Directions from the fixture; heads % count == 0 is the loader's contract.
  for (int dir : opt.directions) {
    nf->add_kda_directions(static_cast<NF::KdaDirection>(dir));
  }

  // Input embedding.
  FillLayer(weights->mutable_ip_emb_w(),
            RandomVec(rng, d.embedding * input_size, 0.05f));
  FillLayer(weights->mutable_ip_emb_b(), RandomVec(rng, d.embedding, 0.05f));
  weights->set_headcount(heads);

  // One encoder layer: MHA or KDA mixer (item 4 covers both stacks).
  auto* enc = weights->add_encoder();
  if (opt.mha_mixer) {
    enc->set_mixer(pblczero::Weights::EncoderLayer::MIXER_MHA);
    auto* mha = enc->mutable_mha();
    // d_model = embedding (q_w.size() / embedding), dense back to embedding.
    FillLayer(mha->mutable_q_w(),
              RandomVec(rng, d.embedding * d.embedding, 0.1f));
    FillLayer(mha->mutable_q_b(), RandomVec(rng, d.embedding, 0.05f));
    FillLayer(mha->mutable_k_w(),
              RandomVec(rng, d.embedding * d.embedding, 0.1f));
    FillLayer(mha->mutable_k_b(), RandomVec(rng, d.embedding, 0.05f));
    FillLayer(mha->mutable_v_w(),
              RandomVec(rng, d.embedding * d.embedding, 0.1f));
    FillLayer(mha->mutable_v_b(), RandomVec(rng, d.embedding, 0.05f));
    FillLayer(mha->mutable_dense_w(),
              RandomVec(rng, d.embedding * d.embedding, 0.1f));
    FillLayer(mha->mutable_dense_b(), RandomVec(rng, d.embedding, 0.05f));
  } else {
  enc->set_mixer(pblczero::Weights::EncoderLayer::MIXER_KDA);
  auto* kda = enc->mutable_kda();
  FillLayer(kda->mutable_q_w(),
            RandomVec(rng, d.embedding * key_depth, 0.1f));
  if (opt.q_b) {
    FillLayer(kda->mutable_q_b(), RandomVec(rng, key_depth, 0.05f));
  }
  FillLayer(kda->mutable_k_w(),
            RandomVec(rng, d.embedding * key_depth, 0.1f));
  if (opt.k_b) {
    FillLayer(kda->mutable_k_b(), RandomVec(rng, key_depth, 0.05f));
  }
  FillLayer(kda->mutable_v_w(),
            RandomVec(rng, d.embedding * value_depth, 0.1f));
  if (opt.v_b) {
    FillLayer(kda->mutable_v_b(), RandomVec(rng, value_depth, 0.05f));
  }
  FillLayer(kda->mutable_decay_a_w(),
            RandomVec(rng, d.embedding * d.gate_rank, 0.1f));
  if (opt.decay_a_b) {
    FillLayer(kda->mutable_decay_a_b(), RandomVec(rng, d.gate_rank, 0.05f));
  }
  FillLayer(kda->mutable_decay_b_w(),
            RandomVec(rng, d.gate_rank * key_depth, 0.1f));
  FillLayer(kda->mutable_decay_b_b(), RandomVec(rng, key_depth, 0.05f));
  FillLayer(kda->mutable_beta_w(), RandomVec(rng, d.embedding * heads, 0.1f));
  FillLayer(kda->mutable_beta_b(), RandomVec(rng, heads, 0.05f));
  FillLayer(kda->mutable_a_log(), RandomVec(rng, heads, 0.05f));
  FillLayer(kda->mutable_dt_bias(), RandomVec(rng, key_depth, 0.05f));
  FillLayer(kda->mutable_gate_a_w(),
            RandomVec(rng, d.embedding * d.gate_rank, 0.1f));
  if (opt.gate_a_b) {
    FillLayer(kda->mutable_gate_a_b(), RandomVec(rng, d.gate_rank, 0.05f));
  }
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
  kda->set_output_gate(opt.output_gate);
  kda->set_output_rms_norm(opt.output_rms_norm);
  if (opt.local_conv) {
    // 3x3 depthwise: embedding * 9 weights, one bias per channel.
    FillLayer(kda->mutable_local_conv_w(),
              RandomVec(rng, d.embedding * 9, 0.05f));
    FillLayer(kda->mutable_local_conv_b(), RandomVec(rng, d.embedding, 0.05f));
  }
  kda->set_local_conv(opt.local_conv);
  kda->set_qkv_silu(opt.qkv_silu);
  }  // if (opt.mha_mixer) ... else (KDA mixer)
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

InputPlanes EncodeFen(const char* fen) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(fen);
  history.Reset(board, 0, 1);
  return EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE, history, 8,
      FillEmptyHistory::NO, nullptr);
}

InputPlanes EncodeStartPos() { return EncodeFen(ChessBoard::kStartposFen); }

struct Outputs {
  float q = 0.0f;
  float d = 0.0f;
  std::vector<float> policy;
};

Outputs ReadOutput(NetworkComputation* computation, size_t b) {
  Outputs out;
  out.q = computation->GetQVal(b);
  out.d = computation->GetDVal(b);
  out.policy.reserve(1858);
  for (int i = 0; i < 1858; ++i) {
    out.policy.push_back(computation->GetPVal(b, i));
  }
  return out;
}

std::vector<Outputs> RunNetwork(const std::string& backend,
                                const WeightsFile& weights,
                                const std::vector<InputPlanes>& batch,
                                const OptionsDict& options = OptionsDict()) {
  auto network = NetworkFactory::Get()->Create(backend, weights, options);
  auto computation = network->NewComputation();
  for (const InputPlanes& planes : batch) {
    computation->AddInput(InputPlanes(planes));
  }
  computation->ComputeBlocking();
  std::vector<Outputs> outs;
  outs.reserve(batch.size());
  for (size_t b = 0; b < batch.size(); ++b) {
    outs.push_back(ReadOutput(computation.get(), b));
  }
  return outs;
}

// Compares one output against its reference; returns an empty string on
// success or a one-line description of the first failure. No gtest macros,
// so it can run inside the concurrency helper below (and under a debugger on
// the crashing path).
std::string CompareOutputs(const Outputs& actual, const Outputs& reference,
                           float tol) {
  if (!std::isfinite(actual.q)) return "non-finite Q";
  if (!std::isfinite(actual.d)) return "non-finite D";
  if (std::fabs(actual.q - reference.q) > tol) return "Q diverges";
  if (std::fabs(actual.d - reference.d) > tol) return "D diverges";
  for (int i = 0; i < 1858; ++i) {
    if (!std::isfinite(actual.policy[i])) {
      return "non-finite policy at move " + std::to_string(i);
    }
    if (std::fabs(actual.policy[i] - reference.policy[i]) > tol) {
      return "policy diverges at move " + std::to_string(i);
    }
  }
  return "";
}

// Runs @rounds barrier-aligned concurrent computation pairs on @network,
// alternating which thread carries the two-input batch (different tensor
// sizes overlapping), and compares every result to its serial reference.
// Returns an empty string on success or a one-line failure description. No
// gtest macros, so the same helper drives the qualifier test and a debugger
// session on the experimental multi_stream path.
std::string RunConcurrentRounds(Network* network, const InputPlanes& pos_a,
                                const InputPlanes& pos_b,
                                const InputPlanes& pos_c,
                                const std::vector<Outputs>& ref_a1,
                                const std::vector<Outputs>& ref_b1,
                                const std::vector<Outputs>& ref_a2,
                                const std::vector<Outputs>& ref_b2, int rounds) {
  for (int round = 0; round < rounds; ++round) {
    const bool a_is_double = (round % 2) == 0;
    auto ca = network->NewComputation();
    auto cb = network->NewComputation();
    ca->AddInput(InputPlanes(pos_a));
    if (a_is_double) ca->AddInput(InputPlanes(pos_c));
    cb->AddInput(InputPlanes(pos_b));
    if (!a_is_double) cb->AddInput(InputPlanes(pos_c));

    std::latch start(3);
    std::exception_ptr ea, eb;
    std::vector<Outputs> oa, ob;
    std::thread ta([&] {
      start.arrive_and_wait();
      try {
        ca->ComputeBlocking();
        for (int b = 0; b < ca->GetBatchSize(); ++b) {
          oa.push_back(ReadOutput(ca.get(), b));
        }
      } catch (...) {
        ea = std::current_exception();
      }
    });
    std::thread tb([&] {
      start.arrive_and_wait();
      try {
        cb->ComputeBlocking();
        for (int b = 0; b < cb->GetBatchSize(); ++b) {
          ob.push_back(ReadOutput(cb.get(), b));
        }
      } catch (...) {
        eb = std::current_exception();
      }
    });
    start.arrive_and_wait();
    ta.join();
    tb.join();

    auto describe = [](const std::exception_ptr& e, const char* what) {
      if (!e) return std::string();
      try {
        std::rethrow_exception(e);
      } catch (const std::exception& ex) {
        return std::string(what) + " threw: " + ex.what();
      } catch (...) {
        return std::string(what) + " threw an unknown exception";
      }
    };
    if (auto msg = describe(ea, "computation A"); !msg.empty()) return msg;
    if (auto msg = describe(eb, "computation B"); !msg.empty()) return msg;

    const std::vector<Outputs>& ref_a = a_is_double ? ref_a2 : ref_a1;
    const std::vector<Outputs>& ref_b = a_is_double ? ref_b1 : ref_b2;
    if (oa.size() != ref_a.size()) {
      return "round " + std::to_string(round) + " A returned batch size " +
             std::to_string(oa.size()) + ", expected " +
             std::to_string(ref_a.size());
    }
    if (ob.size() != ref_b.size()) {
      return "round " + std::to_string(round) + " B returned batch size " +
             std::to_string(ob.size()) + ", expected " +
             std::to_string(ref_b.size());
    }
    for (size_t b = 0; b < oa.size(); ++b) {
      if (auto msg = CompareOutputs(oa[b], ref_a[b], 2e-4f); !msg.empty()) {
        return "round " + std::to_string(round) + " A: " + msg;
      }
    }
    for (size_t b = 0; b < ob.size(); ++b) {
      if (auto msg = CompareOutputs(ob[b], ref_b[b], 2e-4f); !msg.empty()) {
        return "round " + std::to_string(round) + " B: " + msg;
      }
    }
  }
  return "";
}

// Comparison with explicit finiteness: the old worst-diff loop ignored NaN
// (`NaN > worst` is false), so a NaN output could pass as "no divergence"
// (review S7).
void ExpectOutputsClose(const Outputs& actual, const Outputs& reference,
                        float tol, const std::string& context) {
  ASSERT_TRUE(std::isfinite(actual.q)) << context << ": non-finite Q";
  ASSERT_TRUE(std::isfinite(actual.d)) << context << ": non-finite D";
  EXPECT_NEAR(actual.q, reference.q, tol) << context;
  EXPECT_NEAR(actual.d, reference.d, tol) << context;
  float worst = 0.0f;
  int worst_move = -1;
  for (int i = 0; i < 1858; ++i) {
    ASSERT_TRUE(std::isfinite(actual.policy[i]))
        << context << ": non-finite policy at move " << i;
    const float diff = std::fabs(actual.policy[i] - reference.policy[i]);
    if (diff > worst) {
      worst = diff;
      worst_move = i;
    }
  }
  EXPECT_LT(worst, tol) << context << ": policy diverges (worst move "
                        << worst_move << ", diff " << worst << ")";
}

bool HasBackend(const std::string& name) {
  const auto& backends = NetworkFactory::Get()->GetBackendsList();
  return std::find(backends.begin(), backends.end(), name) != backends.end();
}

// Test-only environment variable scoping for the async-error injection seam
// (S2). Restores the unset state on destruction.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
  }
  ~ScopedEnv() {
#ifdef _WIN32
    _putenv_s(name_.c_str(), "");
#else
    unsetenv(name_.c_str());
#endif
  }

 private:
  std::string name_;
};

TEST(KdaParity, SyclMatchesBlasOnKdaHybridNet) {
  const pblczero::Net net = MakeKdaHybridNet();
  const std::vector<InputPlanes> batch = {EncodeStartPos()};

  // BLAS is the CPU reference; it must be present in any build this test
  // links (it is on by default).
  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  const auto reference = RunNetwork("blas", net, batch);

  if (!HasBackend("sycl")) {
    GTEST_SKIP() << "sycl backend not compiled in; nothing to compare";
  }

  std::vector<Outputs> sycl;
  try {
    sycl = RunNetwork("sycl", net, batch);
  } catch (const Exception& e) {
    GTEST_SKIP() << "no usable SYCL device: " << e.what();
  }
  ASSERT_EQ(sycl.size(), reference.size());

  // The two implementations accumulate in different orders (and the SYCL
  // kernels vectorize), so compare with a tolerance rather than exactly.
  // The recurrence itself is sequential-float in both, so the dominant
  // error source is the surrounding GEMMs; 2e-4 leaves two orders of
  // magnitude of headroom over what an actual math divergence produces
  // (a wrong traversal order or dropped gate produces errors ~1e-1+).
  ExpectOutputsClose(sycl[0], reference[0], 2e-4f, "sycl vs blas");
}

// S1: every present KDA projection bias must be applied even when its
// siblings are absent. Sweeps all 8 Q/K/V presence combinations against all
// 4 decay/gate-A combinations, in fp32 and (when compiled in) fp16, against
// the BLAS reference, which applies each bias independently.
TEST(KdaParity, BiasPresenceCombinationsMatchBlas) {
  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  const bool have_sycl = HasBackend("sycl");
  const bool have_sycl_fp16 = HasBackend("sycl-fp16");
  if (!have_sycl && !have_sycl_fp16) {
    GTEST_SKIP() << "sycl backend not compiled in; nothing to compare";
  }
  const std::vector<InputPlanes> batch = {EncodeStartPos()};
  for (int qkv = 0; qkv < 8; ++qkv) {
    for (int dg = 0; dg < 4; ++dg) {
      KdaNetOptions opt;
      opt.q_b = (qkv & 1) != 0;
      opt.k_b = (qkv & 2) != 0;
      opt.v_b = (qkv & 4) != 0;
      opt.decay_a_b = (dg & 1) != 0;
      opt.gate_a_b = (dg & 2) != 0;
      const pblczero::Net net = MakeKdaHybridNet(opt);
      const auto reference = RunNetwork("blas", net, batch);
      const std::string ctx =
          " (qkv=" + std::to_string(qkv) + ", dg=" + std::to_string(dg) + ")";
      if (have_sycl) {
        std::vector<Outputs> sycl;
        try {
          sycl = RunNetwork("sycl", net, batch);
        } catch (const Exception& e) {
          GTEST_SKIP() << "no usable SYCL device: " << e.what();
        }
        ExpectOutputsClose(sycl[0], reference[0], 2e-4f, "sycl" + ctx);
      }
      if (have_sycl_fp16) {
        std::vector<Outputs> fp16;
        try {
          fp16 = RunNetwork("sycl-fp16", net, batch);
        } catch (const Exception& e) {
          GTEST_SKIP() << "no usable SYCL fp16 device: " << e.what();
        }
        ExpectOutputsClose(fp16[0], reference[0], 5e-3f, "sycl-fp16" + ctx);
      }
    }
  }
}

// S7: direction families 1-16 (the 16 serpents need 16 heads), local conv
// enabled, and a real 2-item batch, against BLAS.
TEST(KdaParity, DirectionsLocalConvAndBatchMatchBlas) {
  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  if (!HasBackend("sycl")) {
    GTEST_SKIP() << "sycl backend not compiled in; nothing to compare";
  }
  const std::vector<InputPlanes> batch = {
      EncodeStartPos(),
      EncodeFen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")};
  std::vector<KdaNetOptions> configs;
  {
    KdaNetOptions full8;
    full8.directions = {1, 2, 3, 4, 5, 6, 7, 8};
    full8.local_conv = true;
    configs.push_back(full8);
    KdaNetOptions serpentine16;
    serpentine16.heads = 16;
    serpentine16.directions = {9, 10, 11, 12, 13, 14, 15, 16};
    serpentine16.local_conv = true;
    configs.push_back(serpentine16);
  }
  for (size_t c = 0; c < configs.size(); ++c) {
    const pblczero::Net net = MakeKdaHybridNet(configs[c]);
    const auto reference = RunNetwork("blas", net, batch);
    std::vector<Outputs> sycl;
    try {
      sycl = RunNetwork("sycl", net, batch);
    } catch (const Exception& e) {
      GTEST_SKIP() << "no usable SYCL device: " << e.what();
    }
    ASSERT_EQ(sycl.size(), reference.size());
    for (size_t b = 0; b < reference.size(); ++b) {
      ExpectOutputsClose(sycl[b], reference[b], 2e-4f,
                         "config " + std::to_string(c) + " batch " +
                             std::to_string(b));
    }
  }
}

// S3: max_batch is a hard capacity, not a hint. A batch of max_batch+1 must
// be rejected before it can write outside the host staging buffers, and the
// advertised/recommended sizes must respect the configured capacity.
TEST(KdaParity, SyclEnforcesMaxBatchCapacity) {
  ASSERT_TRUE(HasBackend("sycl"))
      << "sycl backend not compiled into the test binary";
  const pblczero::Net net = MakeKdaHybridNet();
  OptionsDict options;
  options.Set<int>("max_batch", 1);
  std::unique_ptr<Network> network;
  try {
    network = NetworkFactory::Get()->Create("sycl", WeightsFile(net), options);
  } catch (const Exception& e) {
    GTEST_SKIP() << "no usable SYCL device: " << e.what();
  }
  EXPECT_EQ(network->GetMaxBatchSize(), 1);
  EXPECT_LE(network->GetMiniBatchSize(), 1);
  auto computation = network->NewComputation();
  computation->AddInput(EncodeStartPos());
  EXPECT_THROW(computation->AddInput(EncodeStartPos()), Exception);
}

// S2: an async error delivered to the queue handler must fail the
// computation instead of being consumed, so its stale outputs can never be
// read as a successful result. Injected deterministically (a real device
// fault cannot be produced on demand safely); fail-closed, so the network
// stays failed for later computations too.
TEST(KdaParity, SyclAsyncErrorFailsComputation) {
  ASSERT_TRUE(HasBackend("sycl"))
      << "sycl backend not compiled into the test binary";
  const pblczero::Net net = MakeKdaHybridNet();

  // Control: without injection the same net computes successfully.
  {
    std::unique_ptr<Network> network;
    try {
      network = NetworkFactory::Get()->Create("sycl", WeightsFile(net),
                                              OptionsDict());
    } catch (const Exception& e) {
      GTEST_SKIP() << "no usable SYCL device: " << e.what();
    }
    auto computation = network->NewComputation();
    computation->AddInput(EncodeStartPos());
    ASSERT_NO_THROW(computation->ComputeBlocking());
  }

  ScopedEnv inject("LC0_TEST_SYCL_INJECT_ASYNC_ERROR", "1");
  std::unique_ptr<Network> network;
  try {
    network = NetworkFactory::Get()->Create("sycl", WeightsFile(net),
                                            OptionsDict());
  } catch (const Exception& e) {
    GTEST_SKIP() << "no usable SYCL device: " << e.what();
  }
  auto computation = network->NewComputation();
  computation->AddInput(EncodeStartPos());
  EXPECT_THROW(computation->ComputeBlocking(), std::exception);
  // Sticky: no later computation may read outputs from a failed device.
  auto second = network->NewComputation();
  second->AddInput(EncodeStartPos());
  EXPECT_THROW(second->ComputeBlocking(), std::exception);
}

// S4/S5: a constructor failure (injected USM allocation failure, standing
// in for real OOM or a late validation error) must release every buffer
// already allocated and must not poison later loads. Deterministic injection;
// each fail point re-arms because the env value changes.
TEST(KdaParity, SyclConstructorFailureUnwindsCleanly) {
  ASSERT_TRUE(HasBackend("sycl"))
      << "sycl backend not compiled into the test binary";
  const pblczero::Net net = MakeKdaHybridNet();
  const std::vector<int> fail_points = {1, 2, 5, 11, 23};
  for (int fail_at : fail_points) {
    ScopedEnv env("LC0_TEST_SYCL_ALLOC_FAIL_AT",
                  std::to_string(fail_at).c_str());
    try {
      auto network =
          NetworkFactory::Get()->Create("sycl", WeightsFile(net), OptionsDict());
      // This fixture has fewer than fail_at checked allocations; nothing to
      // exercise at this point.
      continue;
    } catch (const Exception& e) {
      if (std::string(e.what()).find("injected USM allocation failure") ==
          std::string::npos) {
        GTEST_SKIP() << "no usable SYCL device: " << e.what();
      }
    } catch (const std::exception& e) {
      GTEST_SKIP() << "SYCL setup failure: " << e.what();
    }
  }
  // Disarm the seam: the same net must still load and compute.
  ScopedEnv off("LC0_TEST_SYCL_ALLOC_FAIL_AT", "0");
  std::unique_ptr<Network> network;
  ASSERT_NO_THROW(network = NetworkFactory::Get()->Create(
                      "sycl", WeightsFile(net), OptionsDict()));
  auto computation = network->NewComputation();
  computation->AddInput(EncodeStartPos());
  ASSERT_NO_THROW(computation->ComputeBlocking());
}

// Item 4: output gate x output RMS norm x local conv combinations on a real
// 3-position (odd) batch, fp32 and fp16, against BLAS.
TEST(KdaParity, GateNormLocalConvMatrixMatchBlas) {
  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  const bool have_sycl = HasBackend("sycl");
  const bool have_sycl_fp16 = HasBackend("sycl-fp16");
  if (!have_sycl && !have_sycl_fp16) {
    GTEST_SKIP() << "sycl backend not compiled in; nothing to compare";
  }
  const std::vector<InputPlanes> batch = {
      EncodeStartPos(),
      EncodeFen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
      EncodeFen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")};
  for (bool gate : {false, true}) {
    for (bool norm : {false, true}) {
      for (bool local_conv : {false, true}) {
        KdaNetOptions opt;
        opt.output_gate = gate;
        opt.output_rms_norm = norm;
        opt.local_conv = local_conv;
        const pblczero::Net net = MakeKdaHybridNet(opt);
        const auto reference = RunNetwork("blas", net, batch);
        const std::string ctx = " (gate=" + std::to_string(gate) +
                                ", norm=" + std::to_string(norm) +
                                ", local_conv=" + std::to_string(local_conv) +
                                ")";
        if (have_sycl) {
          std::vector<Outputs> sycl;
          try {
            sycl = RunNetwork("sycl", net, batch);
          } catch (const Exception& e) {
            GTEST_SKIP() << "no usable SYCL device: " << e.what();
          }
          ASSERT_EQ(sycl.size(), reference.size());
          for (size_t b = 0; b < reference.size(); ++b) {
            ExpectOutputsClose(sycl[b], reference[b], 2e-4f,
                               "sycl" + ctx + " batch " + std::to_string(b));
          }
        }
        if (have_sycl_fp16) {
          std::vector<Outputs> fp16;
          try {
            fp16 = RunNetwork("sycl-fp16", net, batch);
          } catch (const Exception& e) {
            GTEST_SKIP() << "no usable SYCL fp16 device: " << e.what();
          }
          ASSERT_EQ(fp16.size(), reference.size());
          for (size_t b = 0; b < reference.size(); ++b) {
            ExpectOutputsClose(fp16[b], reference[b], 5e-3f,
                               "sycl-fp16" + ctx + " batch " +
                                   std::to_string(b));
          }
        }
      }
    }
  }
}

// Item 4: the MHA mixer stack (the non-KDA EncoderBlock branch) through both
// backends at batch 1 and 2, fp32 and fp16.
TEST(KdaParity, MhaStackMatchesBlas) {
  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  const bool have_sycl = HasBackend("sycl");
  const bool have_sycl_fp16 = HasBackend("sycl-fp16");
  if (!have_sycl && !have_sycl_fp16) {
    GTEST_SKIP() << "sycl backend not compiled in; nothing to compare";
  }
  KdaNetOptions opt;
  opt.mha_mixer = true;
  const pblczero::Net net = MakeKdaHybridNet(opt);
  const std::vector<InputPlanes> batch = {
      EncodeStartPos(),
      EncodeFen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")};
  const auto reference = RunNetwork("blas", net, batch);
  if (have_sycl) {
    std::vector<Outputs> sycl;
    try {
      sycl = RunNetwork("sycl", net, batch);
    } catch (const Exception& e) {
      GTEST_SKIP() << "no usable SYCL device: " << e.what();
    }
    ASSERT_EQ(sycl.size(), reference.size());
    for (size_t b = 0; b < reference.size(); ++b) {
      ExpectOutputsClose(sycl[b], reference[b], 2e-4f,
                         "mha sycl batch " + std::to_string(b));
    }
  }
  if (have_sycl_fp16) {
    std::vector<Outputs> fp16;
    try {
      fp16 = RunNetwork("sycl-fp16", net, batch);
    } catch (const Exception& e) {
      GTEST_SKIP() << "no usable SYCL fp16 device: " << e.what();
    }
    ASSERT_EQ(fp16.size(), reference.size());
    for (size_t b = 0; b < reference.size(); ++b) {
      ExpectOutputsClose(fp16[b], reference[b], 5e-3f,
                         "mha sycl-fp16 batch " + std::to_string(b));
    }
  }
}

// Item 4: real-net parity, opt-in via LC0_TEST_REAL_NET=<path to .pb.gz>.
// Explicit skip when the variable is unset; a construction failure is device
// unavailability (skip), an execution failure FAILS (never converted to a
// skip). The tolerance is a provisional diagnostic (2e-4 absolute or
// relative to the reference's own policy scale), not a release calibration.
TEST(KdaParity, SyclMatchesBlasOnRealNetFromEnv) {
  const char* path = std::getenv("LC0_TEST_REAL_NET");
  if (path == nullptr || *path == '\0') {
    GTEST_SKIP()
        << "LC0_TEST_REAL_NET not set: explicit skip of real-net parity";
  }
  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  if (!HasBackend("sycl")) {
    GTEST_SKIP() << "sycl backend not compiled in; nothing to compare";
  }
  const auto weights = LoadWeights(path);
  ASSERT_TRUE(weights.has_value()) << "cannot load net: " << path;

  const std::vector<InputPlanes> batch = {EncodeStartPos()};
  const auto reference = RunNetwork("blas", *weights, batch);

  std::unique_ptr<Network> network;
  try {
    network = NetworkFactory::Get()->Create("sycl", *weights, OptionsDict());
  } catch (const Exception& e) {
    GTEST_SKIP() << "no usable SYCL device: " << e.what();
  }
  auto computation = network->NewComputation();
  computation->AddInput(EncodeStartPos());
  try {
    computation->ComputeBlocking();
  } catch (const Exception& e) {
    FAIL() << "real-net SYCL execution failed: " << e.what();
  }
  const Outputs got = ReadOutput(computation.get(), 0);
  ASSERT_FALSE(reference.empty());
  float ref_absmax = 0.0f;
  for (float p : reference[0].policy) {
    ref_absmax = std::max(ref_absmax, std::fabs(p));
  }
  const float tol = std::max(2e-4f, 2e-4f * ref_absmax);
  ExpectOutputsClose(got, reference[0], tol, "real net sycl vs blas");
}

// Item 4: concurrent computations on one network. Repeated, barrier-aligned
// eval pairs (alternating which thread carries the two-input batch) must
// match the serial BLAS reference in the default serialized mode.
// multi_stream=true removes that serialization and is currently unsafe; it
// is rejected at construction unless the test-only
// LC0_TEST_SYCL_ALLOW_MULTI_STREAM hook is set, in which case the same
// rounds qualify the experimental path (and reproduce its access violation
// under a debugger with --gtest_catch_exceptions=0).
TEST(KdaParity, SyclSimultaneousComputationsMatchSerial) {
  ASSERT_TRUE(HasBackend("blas"))
      << "blas backend not compiled into the test binary";
  if (!HasBackend("sycl")) {
    GTEST_SKIP() << "sycl backend not compiled in; nothing to compare";
  }
  const pblczero::Net net = MakeKdaHybridNet();
  const InputPlanes pos_a = EncodeStartPos();
  const InputPlanes pos_b =
      EncodeFen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
  const InputPlanes pos_c =
      EncodeFen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
  const auto ref_a1 = RunNetwork("blas", net, {pos_a});
  const auto ref_b1 = RunNetwork("blas", net, {pos_b});
  const auto ref_a2 = RunNetwork("blas", net, {pos_a, pos_c});
  const auto ref_b2 = RunNetwork("blas", net, {pos_b, pos_c});

  constexpr int kRounds = 25;
  std::unique_ptr<Network> serial_network;
  try {
    serial_network = NetworkFactory::Get()->Create("sycl", WeightsFile(net),
                                                   OptionsDict());
  } catch (const Exception& e) {
    GTEST_SKIP() << "no usable SYCL device: " << e.what();
  }
  SCOPED_TRACE("multi_stream=false (serialized)");
  const std::string serial_failure =
      RunConcurrentRounds(serial_network.get(), pos_a, pos_b, pos_c, ref_a1,
                          ref_b1, ref_a2, ref_b2, kRounds);
  EXPECT_TRUE(serial_failure.empty()) << serial_failure;

  if (std::getenv("LC0_TEST_SYCL_ALLOW_MULTI_STREAM") == nullptr) {
    SCOPED_TRACE("multi_stream=true (rejected by default)");
    OptionsDict options;
    options.Set<bool>("multi_stream", true);
    EXPECT_THROW(
        NetworkFactory::Get()->Create("sycl", WeightsFile(net), options),
        Exception);
  } else {
    SCOPED_TRACE("multi_stream=true (test hook set)");
    std::unique_ptr<Network> multi_network;
    try {
      OptionsDict options;
      options.Set<bool>("multi_stream", true);
      multi_network =
          NetworkFactory::Get()->Create("sycl", WeightsFile(net), options);
    } catch (const Exception& e) {
      GTEST_SKIP() << "no usable SYCL device: " << e.what();
    }
    // Experimental path, only run when explicitly requested; it is currently
    // broken (the first computation creation fails on the test iGPU inside
    // InputsOutputs construction -- NewComputation -> GetInputsOutputs ->
    // make_unique<InputsOutputs> -> queue.memset). Catch the throwing form so
    // the harness reports it instead of aborting; a fail-fast exit of the
    // process on this path is still possible.
    try {
      const std::string multi_failure = RunConcurrentRounds(
          multi_network.get(), pos_a, pos_b, pos_c, ref_a1, ref_b1, ref_a2,
          ref_b2, kRounds);
      EXPECT_TRUE(multi_failure.empty()) << multi_failure;
    } catch (const std::exception& e) {
      ADD_FAILURE() << "multi_stream path threw: " << e.what();
    }
  }
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  if (rc == 0) {
    auto* unit = ::testing::UnitTest::GetInstance();
    if (unit->test_to_run_count() > 0 &&
        unit->skipped_test_count() == unit->test_to_run_count()) {
      // CI must be able to tell "all skipped" from "passed" (review item 5).
      std::cerr << "ALL TESTS SKIPPED - no usable SYCL backend/device; "
                   "distinct exit code 3" << std::endl;
      return 3;
    }
  }
  return rc;
}
