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
// fails in this test, only the reservation hook does. The peaked mode
// concentrates policy on the first legal move, which makes
// estimated_visits_to_change_best large for unexpanded children, so a pick
// hands them a multi-visit budget -- the precondition for the
// visit-then-collision emission that review #878 finding 2 is about (with a
// uniform policy that estimate is always clamped to 1 and the collision
// half of the pair never materialises in practice).
class FakeComputation : public BackendComputation {
 public:
  explicit FakeComputation(bool peaked_policy) : peaked_(peaked_policy) {}
  size_t UsedBatchSize() const override { return used_; }
  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    if (result.q) *result.q = 0.0f;
    if (result.d) *result.d = 1.0f;
    if (result.m) *result.m = 0.0f;
    if (!result.p.empty()) {
      if (peaked_) {
        const size_t n = result.p.size();
        result.p[0] = n > 1 ? 0.9f : 1.0f;
        const float rest = n > 1 ? 0.1f / static_cast<float>(n - 1) : 0.0f;
        for (size_t i = 1; i < n; ++i) result.p[i] = rest;
      } else {
        const float p = pos.legal_moves.empty()
                            ? 0.0f
                            : 1.0f / pos.legal_moves.size();
        for (auto& v : result.p) v = p;
      }
    }
    ++used_;
    return FETCHED_IMMEDIATELY;
  }
  void ComputeBlocking() override {}

 private:
  size_t used_ = 0;
  const bool peaked_;
};

class FakeBackend : public Backend {
 public:
  explicit FakeBackend(bool peaked_policy = false)
      : peaked_(peaked_policy) {}

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{.has_mlh = false,
                             .has_wdl = true,
                             .runs_on_cpu = true,
                             .suggested_num_search_threads = 1,
                             .recommended_batch_size = 32,
                             .maximum_batch_size = 256};
  }
  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<FakeComputation>(peaked_);
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
  const bool peaked_;
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
  EXPECT_GT(TestOnlyGatheringTasksExecuted(), 0)
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
  EXPECT_GT(TestOnlyGatheringTasksExecuted(), 0)
      << "no worker task ever executed";
  ExpectZeroNInFlightEverywhere(tree.GetCurrentHead());
}

// ---- review #878 finding 2: partial emission double-owns a Visit ----------
//
// The seam fires between a successful Visit insertion and the collision
// insertion that follows it in the same stop_picking entry, with a
// multi-visit budget so both exist. The visit's share must already be
// relinquished by the ledger when this throws; otherwise this call's catch
// would cancel it once more and the emitted entry's own cancellation would
// cancel it a second time. A peaked policy is required to reach this at all:
// with uniform policy estimated_visits_to_change_best clamps to 1, so stop
// entries always carry a single visit and the collision half never
// materialises. The count sits well into the run so real search (verified by
// the executor counters) happens before the throw lands.
TEST_F(ReservationRollbackTest, PooledCollisionEmissionFailureLeavesNoReservation) {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  SearchParams::Populate(&options);
  options.GetMutableDefaultsOptions()->Set(SharedBackendParams::kNNCacheSizeId,
                                           200000);
  options.GetMutableDefaultsOptions()->Set(
      SearchParams::kTaskWorkersPerSearchWorkerId, 4);
  const auto option_dict = options.GetOptionsDict();

  FakeBackend backend(/*peaked_policy=*/true);
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
  TestOnlySetThrowCount(TestOnlyThrowSite::kBeforeEmission, 1200);
  search->StartThreads(4);
  search->Wait();

  EXPECT_EQ(bestmove_count.load(), 1);
  EXPECT_TRUE(TestOnlyWasThrowFired(TestOnlyThrowSite::kBeforeEmission))
      << "the armed between-visit-and-collision seam was never reached";
  EXPECT_GT(TestOnlyGatheringTasksExecuted(), 0)
      << "no worker task ever executed";
  ExpectZeroNInFlightEverywhere(tree.GetCurrentHead());
}

// ---- review #878 finding 3: promotion must commit atomically --------------
//
// The seam fires at the promotion's current_path growth point, before either
// parallel vector or the level's ownership changes. A real search long enough
// to need growth (the vectors start at 30 entries) must roll the whole level
// back exactly once: the parent entry stays live, level_reservations stays
// live, and the catch cancels both without double-cancelling any announced
// child. Fired-asserted so a run that never needed growth fails loudly
// instead of passing vacuously.
TEST_F(ReservationRollbackTest, PromotionPathGrowthFailureLeavesNoReservation) {
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

  TestOnlyResetSeams();
  // Override the workspace's warm-start reserve so the very first wide
  // promotion genuinely needs to grow the parallel vectors; hits=1 lets the
  // first (multi-child) promotion succeed and throws at the next one, so
  // rollback is exercised both before and after an attempted promotion.
  TestOnlySetWorkspaceReserveOverride(4);
  TestOnlySetThrowCount(TestOnlyThrowSite::kBeforePromotionPathGrowth, 1);
  search->StartThreads(1);
  search->Wait();

  EXPECT_EQ(bestmove_count.load(), 1);
  EXPECT_TRUE(
      TestOnlyWasThrowFired(TestOnlyThrowSite::kBeforePromotionPathGrowth))
      << "promotion never needed to grow current_path: the search did not "
         "reach a wide/deep enough level to exercise the atomic commit";
  ExpectZeroNInFlightEverywhere(tree.GetCurrentHead());
}

// Same scenario, arming the second capacity-growth point: current_path is
// grown first (unarmed), then reservation_ledger's growth throws. Failure at
// either site must leave the parallel vectors consistent.
TEST_F(ReservationRollbackTest, PromotionLedgerGrowthFailureLeavesNoReservation) {
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

  TestOnlyResetSeams();
  // See the path-growth test: the first promotion succeeds (growing both
  // vectors), then reservation_ledger's growth attempt throws.
  TestOnlySetWorkspaceReserveOverride(4);
  TestOnlySetThrowCount(TestOnlyThrowSite::kBeforePromotionLedgerGrowth, 1);
  search->StartThreads(1);
  search->Wait();

  EXPECT_EQ(bestmove_count.load(), 1);
  EXPECT_TRUE(
      TestOnlyWasThrowFired(TestOnlyThrowSite::kBeforePromotionLedgerGrowth))
      << "promotion never needed to grow reservation_ledger";
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
