/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

// dag_classic mirror of classic/backend_failure_test.cc
// a real Backend::CreateComputation() failure between two
// iterations used to hit a stale-state bug -- InitializeIteration()
// cleared minibatch_ AFTER the throwing CreateComputation() call, so a
// throw on iteration N skipped the clear and left iteration (N-1)'s
// already-backed-up entries sitting in minibatch_. The worker's
// catch(...) -> CancelPendingMinibatchVisits() then walked those stale
// entries and cancelled reservations DoBackupUpdate had already
// legitimately released, double-cancelling/underflowing N-in-flight.
//
// This runs one iteration to real completion (so minibatch_ actually holds
// backed-up entries afterward) and then makes the NEXT CreateComputation()
// call throw, asserting the search still converges to exactly one bestmove
// and every node's N-in-flight returns to zero.

#include "search/dag_classic/search.h"

#include <atomic>
#include <stdexcept>

#include "chess/board.h"
#include "chess/callbacks.h"
#include "gtest/gtest.h"
#include "neural/backend.h"
#include "neural/shared_params.h"
#include "search/classic/stoppers/stoppers.h"
#include "search/dag_classic/params.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace dag_classic {
namespace {

// Synchronous, always-cache-hit computation: AddInput fills the result
// directly and reports FETCHED_IMMEDIATELY, so ComputeBlocking() is a
// no-op -- good enough to let a real gather/backup cycle complete.
class FakeSuccessComputation : public BackendComputation {
 public:
  size_t UsedBatchSize() const override { return used_; }
  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    if (result.q) *result.q = 0.0f;
    if (result.d) *result.d = 1.0f;
    if (result.m) *result.m = 0.0f;
    if (!result.p.empty()) {
      const float p = pos.legal_moves.empty()
                          ? 0.0f
                          : 1.0f / pos.legal_moves.size();
      for (auto& v : result.p) v = p;
    }
    ++used_;
    return FETCHED_IMMEDIATELY;
  }
  void ComputeBlocking() override {}

 private:
  size_t used_ = 0;
};

// CreateComputation() succeeds `succeed_count` times, then throws on every
// subsequent call -- simulating a real backend device failure mid-search.
class ThrowAfterNBackend : public Backend {
 public:
  explicit ThrowAfterNBackend(int succeed_count)
      : succeed_count_(succeed_count) {}

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{.has_mlh = false,
                             .has_wdl = true,
                             .runs_on_cpu = true,
                             .suggested_num_search_threads = 1,
                             .recommended_batch_size = 1,
                             .maximum_batch_size = 256};
  }
  std::unique_ptr<BackendComputation> CreateComputation() override {
    if (calls_.fetch_add(1) >= succeed_count_) {
      throw std::runtime_error(
          "ThrowAfterNBackend: injected CreateComputation failure");
    }
    return std::make_unique<FakeSuccessComputation>();
  }
  std::vector<EvalResult> EvaluateBatch(
      std::span<const EvalPosition>) override {
    return {};
  }
  std::optional<EvalResult> GetCachedEvaluation(
      const EvalPosition&) override {
    return std::nullopt;
  }
  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict&) override {
    return UPDATE_OK;
  }

 private:
  const int succeed_count_;
  std::atomic<int> calls_{0};
};

// Queues real entries into a batch (no immediate fetch), then throws from
// ComputeBlocking() itself once the batch reaches a real size -- unlike
// FakeSuccessComputation above, this exercises the path where pooled tasks
// have actually added work to the same in-flight computation before it
// fails (: FETCHED_IMMEDIATELY-everywhere never reaches
// a real ComputeBlocking() failure).
class RealBatchThenThrowComputation : public BackendComputation {
 public:
  size_t UsedBatchSize() const override { return queued_; }
  AddInputResult AddInput(const EvalPosition&, EvalResultPtr) override {
    ++queued_;
    return ENQUEUED_FOR_EVAL;
  }
  void ComputeBlocking() override {
    throw std::runtime_error(
        "RealBatchThenThrowComputation: injected ComputeBlocking failure");
  }

 private:
  size_t queued_ = 0;
};

class RealBatchThenThrowBackend : public Backend {
 public:
  BackendAttributes GetAttributes() const override {
    return BackendAttributes{.has_mlh = false,
                             .has_wdl = true,
                             .runs_on_cpu = true,
                             .suggested_num_search_threads = 1,
                             .recommended_batch_size = 32,
                             .maximum_batch_size = 256};
  }
  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<RealBatchThenThrowComputation>();
  }
  std::vector<EvalResult> EvaluateBatch(
      std::span<const EvalPosition>) override {
    return {};
  }
  std::optional<EvalResult> GetCachedEvaluation(
      const EvalPosition&) override {
    return std::nullopt;
  }
  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict&) override {
    return UPDATE_OK;
  }
};

TEST(SearchBackendFailure, ComputeBlockingThrowWithPooledTasksIsSafe) {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  SearchParams::Populate(&options);
  options.GetMutableDefaultsOptions()->Set(SharedBackendParams::kNNCacheSizeId,
                                           200000);
  // Real pooled tasks, so the batch that ComputeBlocking() fails on
  // actually contains contributions gathered by the pool, not just the
  // main thread's own picks.
  options.GetMutableDefaultsOptions()->Set(
      SearchParams::kTaskWorkersPerSearchWorkerId, 4);
  const auto option_dict = options.GetOptionsDict();

  RealBatchThenThrowBackend backend;

  NodeTree tree;
  tree.ResetToPosition(ChessBoard::kStartposFen, {});
  TranspositionTable tt;

  auto stopper = std::make_unique<classic::ChainedSearchStopper>();
  stopper->AddStopper(
      std::make_unique<classic::VisitsStopper>(5000, false));

  std::atomic<int> bestmove_count{0};
  auto responder = std::make_unique<CallbackUciResponder>(
      [&](const BestMoveInfo&) { ++bestmove_count; },
      [](const std::vector<ThinkingInfo>&) {});

  auto search = std::make_unique<Search>(
      tree, &backend, std::move(responder), MoveList(),
      std::chrono::steady_clock::now(), std::move(stopper),
      /*infinite=*/false, /*ponder=*/false, option_dict, &tt, nullptr);

  search->StartThreads(4);
  search->Wait();

  EXPECT_EQ(bestmove_count.load(), 1);
  EXPECT_EQ(tree.GetCurrentHead()->GetNInFlight(), 0u);
  EXPECT_LT(tree.GetCurrentHead()->GetNInFlight(), 1000u);
}

TEST(SearchBackendFailure, CreateComputationThrowDoesNotUnderflowOrHang) {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  SearchParams::Populate(&options);
  options.GetMutableDefaultsOptions()->Set(SharedBackendParams::kNNCacheSizeId,
                                           200000);
  const auto option_dict = options.GetOptionsDict();

  // 1st CreateComputation() call succeeds (real gather+backup completes),
  // every call after that throws.
  ThrowAfterNBackend backend(/*succeed_count=*/1);

  NodeTree tree;
  tree.ResetToPosition(ChessBoard::kStartposFen, {});
  TranspositionTable tt;

  auto stopper = std::make_unique<classic::ChainedSearchStopper>();
  stopper->AddStopper(
      std::make_unique<classic::VisitsStopper>(100000, false));

  std::atomic<int> bestmove_count{0};
  auto responder = std::make_unique<CallbackUciResponder>(
      [&](const BestMoveInfo&) { ++bestmove_count; },
      [](const std::vector<ThinkingInfo>&) {});

  auto search = std::make_unique<Search>(
      tree, &backend, std::move(responder), MoveList(),
      std::chrono::steady_clock::now(), std::move(stopper),
      /*infinite=*/false, /*ponder=*/false, option_dict, &tt, nullptr);

  // Single search thread: deterministic ordering between the one
  // successful iteration and the throwing one that follows it.
  search->StartThreads(1);
  search->Wait();

  // Bounded stop, exactly one bestmove -- not zero (hung/no response) and
  // not more than one (a double-send from confused state).
  EXPECT_EQ(bestmove_count.load(), 1);

  // No leaked/underflowed N-in-flight on the root: a double-cancel of
  // already-released reservations would drive this uint32_t negative
  // (wrapping to a huge value), not just leave it nonzero.
  EXPECT_EQ(tree.GetCurrentHead()->GetNInFlight(), 0u);
  EXPECT_LT(tree.GetCurrentHead()->GetNInFlight(), 1000u);
}

}  // namespace
}  // namespace dag_classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // GenerateLegalMoves() depends on the magic-bitboard sliding-piece attack
  // tables, normally built once by main() before any engine code runs.
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
