/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

// Fault-injection coverage for review #872 P1: n_in_flight_ reservations
// made mid-traversal inside PickNodesToExtendTask (a fresh
// TryStartScoreUpdate(), or a successful hand-off to the task pool) were
// invisible to CancelPendingMinibatch until they landed in a receiver
// entry -- an exception in the gap between reservation and emission would
// otherwise leak them, distinct from (and not covered by) the earlier
// #868 fix, which only made sure the pool itself was drained/rethrown
// correctly around an exception already in a receiver entry.
//
// This uses the g_testonly_throw_after_reservations seam
// (utils/testonly_throw_hook.h) to throw partway through a real gather,
// after some reservations exist only in scratch state (not yet a Visit/
// Collision entry), and asserts the search still converges to exactly one
// bestmove with N-in-flight back to zero on every node in the tree, not
// just the root -- a leak on some interior node would pass a root-only
// check while still corrupting that node's future UCT selection.

#include "search/classic/search.h"

#include <atomic>

#include "chess/board.h"
#include "chess/callbacks.h"
#include "gtest/gtest.h"
#include "neural/backend.h"
#include "neural/shared_params.h"
#include "search/classic/params.h"
#include "search/classic/stoppers/stoppers.h"
#include "utils/optionsparser.h"
#include "utils/testonly_throw_hook.h"

namespace lczero {
namespace classic {
namespace {

// Same deterministic, always-cache-hit computation as
// backend_failure_test.cc's FakeSuccessComputation: the backend itself
// never fails in this test, only the reservation hook does.
class FakeComputation : public BackendComputation {
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

class FakeBackend : public Backend {
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
    return std::make_unique<FakeComputation>();
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

// Recurses the whole tree, not just the root -- a leak on some interior
// node would otherwise pass a root-only check while still corrupting that
// node's future UCT selection.
void ExpectZeroNInFlightEverywhere(Node* node) {
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->GetNInFlight(), 0u)
      << "leaked reservation on a non-root node (N=" << node->GetN() << ")";
  for (auto& edge : node->Edges()) {
    if (edge.node() != nullptr) ExpectZeroNInFlightEverywhere(edge.node());
  }
}

class ReservationRollbackTest : public ::testing::Test {
 protected:
  void TearDown() override {
    // The hook is a process-global countdown -- always leave it disabled
    // so a test that doesn't fully consume it can't bleed into the next
    // test in this binary.
    g_testonly_throw_after_reservations.store(-1, std::memory_order_relaxed);
  }
};

// task_workers=0: the throw happens on the main thread's own traversal,
// inside PickNodesToExtendTask's inner picking loop (exercises the
// "reservation_ledger + level_reservations, no pool involved" path).
TEST_F(ReservationRollbackTest, SerialThrowMidTraversalLeavesNoReservation) {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  SearchParams::Populate(&options);
  options.GetMutableDefaultsOptions()->Set(SharedBackendParams::kNNCacheSizeId,
                                           200000);
  options.GetMutableDefaultsOptions()->Set(
      SearchParams::kTaskWorkersPerSearchWorkerId, 0);
  const auto option_dict = options.GetOptionsDict();

  FakeBackend backend;
  NodeTree tree;
  tree.ResetToPosition(ChessBoard::kStartposFen, {});

  auto stopper = std::make_unique<ChainedSearchStopper>();
  stopper->AddStopper(std::make_unique<VisitsStopper>(3000, false));

  std::atomic<int> bestmove_count{0};
  auto responder = std::make_unique<CallbackUciResponder>(
      [&](const BestMoveInfo&) { ++bestmove_count; },
      [](const std::vector<ThinkingInfo>&) {});

  auto search = std::make_unique<Search>(
      tree, &backend, std::move(responder), MoveList(),
      std::chrono::steady_clock::now(), std::move(stopper),
      /*infinite=*/false, /*ponder=*/false, option_dict, nullptr);

  // Let a few reservations happen for real (root's own, plus at least one
  // child's TryStartScoreUpdate) before throwing -- nonempty pending local
  // work at the moment of the throw, not an empty-traversal edge case.
  g_testonly_throw_after_reservations.store(5, std::memory_order_relaxed);

  search->StartThreads(1);
  search->Wait();

  EXPECT_EQ(bestmove_count.load(), 1);
  ExpectZeroNInFlightEverywhere(tree.GetCurrentHead());
}

// task_workers>0: the throw happens inside a task submitted to the pool,
// after that task has already made some reservations of its own --
// exercises the same rollback running on a pool worker thread instead of
// the main search thread, and proves a submitted subtree's reservations
// don't depend on the parent call's cleanup.
TEST_F(ReservationRollbackTest, PooledThrowMidTraversalLeavesNoReservation) {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  SearchParams::Populate(&options);
  options.GetMutableDefaultsOptions()->Set(SharedBackendParams::kNNCacheSizeId,
                                           200000);
  options.GetMutableDefaultsOptions()->Set(
      SearchParams::kTaskWorkersPerSearchWorkerId, 4);
  const auto option_dict = options.GetOptionsDict();

  FakeBackend backend;
  NodeTree tree;
  tree.ResetToPosition(ChessBoard::kStartposFen, {});

  auto stopper = std::make_unique<ChainedSearchStopper>();
  stopper->AddStopper(std::make_unique<VisitsStopper>(3000, false));

  std::atomic<int> bestmove_count{0};
  auto responder = std::make_unique<CallbackUciResponder>(
      [&](const BestMoveInfo&) { ++bestmove_count; },
      [](const std::vector<ThinkingInfo>&) {});

  auto search = std::make_unique<Search>(
      tree, &backend, std::move(responder), MoveList(),
      std::chrono::steady_clock::now(), std::move(stopper),
      /*infinite=*/false, /*ponder=*/false, option_dict, nullptr);

  // A bigger budget than the serial case: with several pool workers racing
  // to consume the same countdown, more reservations land before whichever
  // one crosses zero actually throws.
  g_testonly_throw_after_reservations.store(30, std::memory_order_relaxed);

  search->StartThreads(4);
  search->Wait();

  EXPECT_EQ(bestmove_count.load(), 1);
  ExpectZeroNInFlightEverywhere(tree.GetCurrentHead());
}

}  // namespace
}  // namespace classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
