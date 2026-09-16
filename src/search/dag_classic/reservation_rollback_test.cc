/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

// dag_classic mirror of classic/reservation_rollback_test.cc (agora
// #41/#872 P1): n_in_flight_ reservations made mid-traversal inside
// PickNodesToExtendTask (a fresh TryStartScoreUpdate(), or a successful
// hand-off to the task pool) were invisible to CancelPendingMinibatchVisits
// until they landed in a receiver entry -- an exception in the gap between
// reservation and emission would otherwise leak them.
//
// This uses the g_testonly_throw_after_reservations seam
// (utils/testonly_throw_hook.h) to throw partway through a real gather and
// asserts the search still converges to exactly one bestmove with
// N-in-flight back to zero on every node in the tree, not just the root.

#include "search/dag_classic/search.h"

#include <atomic>

#include "chess/board.h"
#include "chess/callbacks.h"
#include "gtest/gtest.h"
#include "neural/backend.h"
#include "neural/shared_params.h"
#include "search/classic/stoppers/stoppers.h"
#include "search/dag_classic/params.h"
#include "utils/optionsparser.h"
#include "utils/testonly_throw_hook.h"

namespace lczero {
namespace dag_classic {
namespace {

// Same deterministic, always-cache-hit computation as
// pool_equivalence_test.cc's FakeComputation: the backend itself never
// fails in this test, only the reservation hook does.
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
    // The seams are process-global counters -- always disable every site and
    // zero the execution counters so a test that doesn't fully consume a
    // countdown can't bleed into the next test in this binary.
    TestOnlyResetSeams();
  }
};

// task_workers=0: the throw happens on the main thread's own traversal.
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
  TranspositionTable tt;

  auto stopper = std::make_unique<classic::ChainedSearchStopper>();
  stopper->AddStopper(
      std::make_unique<classic::VisitsStopper>(3000, false));

  std::atomic<int> bestmove_count{0};
  auto responder = std::make_unique<CallbackUciResponder>(
      [&](const BestMoveInfo&) { ++bestmove_count; },
      [](const std::vector<ThinkingInfo>&) {});

  auto search = std::make_unique<Search>(
      tree, &backend, std::move(responder), MoveList(),
      std::chrono::steady_clock::now(), std::move(stopper),
      /*infinite=*/false, /*ponder=*/false, option_dict, &tt, nullptr);

  // Let a few reservations happen for real (root's own, plus at least one
  // child's TryStartScoreUpdate) before throwing -- nonempty pending local
  // work at the moment of the throw, not an empty-traversal edge case.
  TestOnlySetThrowCount(TestOnlyThrowSite::kAfterReservation, 5);

  search->StartThreads(1);
  search->Wait();

  EXPECT_EQ(bestmove_count.load(), 1);
  EXPECT_TRUE(TestOnlyWasThrowFired(TestOnlyThrowSite::kAfterReservation));
  ExpectZeroNInFlightEverywhere(tree.GetCurrentHead());
}

// task_workers>0: the throw happens inside a task submitted to the pool.
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
  TranspositionTable tt;

  auto stopper = std::make_unique<classic::ChainedSearchStopper>();
  stopper->AddStopper(
      std::make_unique<classic::VisitsStopper>(3000, false));

  std::atomic<int> bestmove_count{0};
  auto responder = std::make_unique<CallbackUciResponder>(
      [&](const BestMoveInfo&) { ++bestmove_count; },
      [](const std::vector<ThinkingInfo>&) {});

  auto search = std::make_unique<Search>(
      tree, &backend, std::move(responder), MoveList(),
      std::chrono::steady_clock::now(), std::move(stopper),
      /*infinite=*/false, /*ponder=*/false, option_dict, &tt, nullptr);

  // A bigger budget than the serial case: with several pool workers racing
  // to consume the same countdown, more reservations land before whichever
  // one crosses zero actually throws.
  TestOnlySetThrowCount(TestOnlyThrowSite::kAfterReservation, 30);

  search->StartThreads(4);
  search->Wait();

  EXPECT_EQ(bestmove_count.load(), 1);
  EXPECT_TRUE(TestOnlyWasThrowFired(TestOnlyThrowSite::kAfterReservation));
  ExpectZeroNInFlightEverywhere(tree.GetCurrentHead());
}

// ---- review #876 finding 3 / #874: DAG submitted-task entry guard ---------
//
// The seam fires at the entry of a submitted task, before its path copy and
// before its root ledger entry exist. The submitting call already released
// its own coverage (subtracting the handed-off share and removing the child
// from level_reservations), so the entry guard is the only owner and must
// cancel the covered amount across the whole submitted path.
TEST_F(ReservationRollbackTest, SubmittedTaskEntrySetupFailureLeavesNoReservation) {
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
  TranspositionTable tt;

  auto stopper = std::make_unique<classic::ChainedSearchStopper>();
  stopper->AddStopper(
      std::make_unique<classic::VisitsStopper>(3000, false));

  std::atomic<int> bestmove_count{0};
  auto responder = std::make_unique<CallbackUciResponder>(
      [&](const BestMoveInfo&) { ++bestmove_count; },
      [](const std::vector<ThinkingInfo>&) {});

  auto search = std::make_unique<Search>(
      tree, &backend, std::move(responder), MoveList(),
      std::chrono::steady_clock::now(), std::move(stopper),
      /*infinite=*/false, /*ponder=*/false, option_dict, &tt, nullptr);

  TestOnlyResetSeams();
  TestOnlySetThrowCount(TestOnlyThrowSite::kBeforeTaskEntrySetup, 0);
  search->StartThreads(4);
  search->Wait();

  EXPECT_EQ(bestmove_count.load(), 1);
  EXPECT_TRUE(TestOnlyWasThrowFired(TestOnlyThrowSite::kBeforeTaskEntrySetup))
      << "the armed submitted-task entry seam was never reached";
  EXPECT_GT(g_testonly_gathering_tasks_executed.load(), 0)
      << "no worker task ever executed";
  ExpectZeroNInFlightEverywhere(tree.GetCurrentHead());
}

// ---- review #874/#876: DAG post-submit ownership transfer -----------------
//
// The seam fires immediately after a successful Submit, in the window where
// the level's current_path.back().visits_ has not yet been zeroed. Without
// the transferred-share subtraction, this call's cleanup would cancel the
// handed-off budget at the parent and ancestors while the running task (or
// its receiver entries later) cancels it again.
TEST_F(ReservationRollbackTest, PooledPostSubmitThrowLeavesNoReservation) {
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
  TranspositionTable tt;

  auto stopper = std::make_unique<classic::ChainedSearchStopper>();
  stopper->AddStopper(
      std::make_unique<classic::VisitsStopper>(3000, false));

  std::atomic<int> bestmove_count{0};
  auto responder = std::make_unique<CallbackUciResponder>(
      [&](const BestMoveInfo&) { ++bestmove_count; },
      [](const std::vector<ThinkingInfo>&) {});

  auto search = std::make_unique<Search>(
      tree, &backend, std::move(responder), MoveList(),
      std::chrono::steady_clock::now(), std::move(stopper),
      /*infinite=*/false, /*ponder=*/false, option_dict, &tt, nullptr);

  TestOnlyResetSeams();
  TestOnlySetThrowCount(TestOnlyThrowSite::kAfterSubmit, 0);
  search->StartThreads(4);
  search->Wait();

  EXPECT_EQ(bestmove_count.load(), 1);
  EXPECT_TRUE(TestOnlyWasThrowFired(TestOnlyThrowSite::kAfterSubmit))
      << "the armed post-submit seam was never reached";
  EXPECT_GT(g_testonly_gathering_tasks_executed.load(), 0)
      << "no worker task ever executed";
  ExpectZeroNInFlightEverywhere(tree.GetCurrentHead());
}

}  // namespace
}  // namespace dag_classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
