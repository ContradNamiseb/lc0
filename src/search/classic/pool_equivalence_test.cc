/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

// Serial-vs-pooled equivalence coverage for review #867/#868: TaskWorkers=0
// (all picking/processing done inline on the search thread, task_pool_ is
// never constructed) and TaskWorkers>0 (work split across a TaskStealingPool)
// are two different code paths through the same PickNodesToExtend/
// GatherMinibatch logic. This asserts both converge cleanly to the same
// visit budget with zero leaked N-in-flight -- codex-sol's re-review
// (agora #867) asked for this as validation before approving the
// TaskStealingPool migration.
//
// review #872 point 2: the original version of this test used
// recommended_batch_size=1 and a one-visit collision budget, so
// AddInput always returned FETCHED_IMMEDIATELY, classic's gather split
// never triggered (child_limit never exceeded MinimumWorkSizeForPicking),
// and the pooled/serial comparison was only a loose total-playout ratio --
// none of it proved either config's worker paths actually executed.
// FakeBackend now queues into a real batch (ComputeBlocking() gets called
// for real), and RunFixedVisitSearch additionally walks the resulting tree
// and reports its node count and max depth, so every test below can assert
// real multi-level, multi-node exploration happened -- collisions alone
// never grow the tree, so a shallow/sparse tree here would mean the pool
// path degenerated into pure collisions instead of real picking/processing.

#include "search/classic/search.h"

#include <algorithm>
#include <atomic>
#include <mutex>

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

// Deterministic, real-batching computation: uniform policy/value removes
// real-NN timing/precision noise so a visit-count comparison between
// serial and pooled runs is meaningful, and queuing into ComputeBlocking()
// instead of fetching immediately means processing tasks (ProcessPickedTask)
// have real work to do, not just gathering ones.
//
// AddInput() is called concurrently by every gathering task feeding this
// same per-round computation_ (that's the whole point of pooled gathering,
// and what a real backend has to support too), so queued_ needs a lock --
// unlike the FETCHED_IMMEDIATELY version this replaces, which never had
// shared mutable state across a call.
class FakeComputation : public BackendComputation {
 public:
  size_t UsedBatchSize() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return queued_.size();
  }
  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    std::lock_guard<std::mutex> lock(mutex_);
    queued_.push_back(result);
    legal_move_counts_.push_back(pos.legal_moves.size());
    return ENQUEUED_FOR_EVAL;
  }
  void ComputeBlocking() override {
    // No concurrent AddInput() calls can be in flight once ComputeBlocking
    // is reached (matches the real GatherMinibatch/RunNNComputation
    // ordering), so the lock here is a belt-and-suspenders visibility
    // fence, not contention prevention.
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < queued_.size(); ++i) {
      EvalResultPtr result = queued_[i];
      if (result.q) *result.q = 0.0f;
      if (result.d) *result.d = 1.0f;
      if (result.m) *result.m = 0.0f;
      if (!result.p.empty()) {
        const float p = legal_move_counts_[i] == 0
                            ? 0.0f
                            : 1.0f / legal_move_counts_[i];
        for (auto& v : result.p) v = p;
      }
    }
  }

 private:
  mutable std::mutex mutex_;
  std::vector<EvalResultPtr> queued_;
  std::vector<size_t> legal_move_counts_;
};

class FakeBackend : public Backend {
 public:
  // runs_on_cpu controls which branch of SearchWorker's TaskWorkers==-1
  // heuristic gets exercised (search.h ~221): true resolves straight to 0
  // workers (serial), matching a real CPU backend; false takes the
  // hardware_concurrency()-based sizing path, matching a real GPU backend
  // like sycl-fp16 -- which is what -1 (the actual UCI default) resolves to
  // in the overwhelming majority of real games, so it's the config worth
  // the most equivalence coverage.
  explicit FakeBackend(bool runs_on_cpu = true) : runs_on_cpu_(runs_on_cpu) {}

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{.has_mlh = false,
                             .has_wdl = true,
                             .runs_on_cpu = runs_on_cpu_,
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

 private:
  bool runs_on_cpu_;
};

// Walks the whole tree, reporting how many nodes were actually visited
// (GetN() > 0) and how deep the deepest one is. Collisions never expand a
// node, so a real split/processing pipeline should reach visibly more
// nodes and depth than picking alone could produce serially in the same
// budget -- this is the "did the pool actually do search work" proxy in
// place of instrumenting TaskStealingPool's own counters directly (it has
// no test-visible identity here -- pool_equivalence_test only ever sees
// the Search/SearchWorker surface).
struct TreeShape {
  int visited_nodes = 0;
  int max_depth = 0;
};

void WalkTreeShape(Node* node, int depth, TreeShape* shape) {
  if (node->GetN() == 0) return;
  ++shape->visited_nodes;
  shape->max_depth = std::max(shape->max_depth, depth);
  for (Node* child : node->VisitedNodes()) {
    WalkTreeShape(child, depth + 1, shape);
  }
}

// Runs a fixed-visit search with the given TaskWorkers setting and returns
// (bestmove is set, final root N-in-flight, total playouts, tree shape, and
// the test-only execution-counter deltas proving which task paths ran).
struct RunResult {
  bool got_bestmove = false;
  uint32_t root_n_in_flight = 0xffffffffu;  // poisoned until set
  int64_t total_playouts = 0;
  TreeShape shape;
  int64_t gathering_executed = 0;
  int64_t processing_executed = 0;
  int64_t processing_calls = 0;
};

RunResult RunFixedVisitSearch(int task_workers, int visits, int threads,
                              bool runs_on_cpu = true) {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  SearchParams::Populate(&options);
  options.GetMutableDefaultsOptions()->Set(SharedBackendParams::kNNCacheSizeId,
                                           200000);
  options.GetMutableDefaultsOptions()->Set(
      SearchParams::kTaskWorkersPerSearchWorkerId, task_workers);
  const auto option_dict = options.GetOptionsDict();

  FakeBackend backend(runs_on_cpu);
  NodeTree tree;
  tree.ResetToPosition(ChessBoard::kStartposFen, {});

  auto stopper = std::make_unique<ChainedSearchStopper>();
  stopper->AddStopper(std::make_unique<VisitsStopper>(visits, false));

  RunResult result;
  auto responder = std::make_unique<CallbackUciResponder>(
      [&](const BestMoveInfo&) { result.got_bestmove = true; },
      [](const std::vector<ThinkingInfo>&) {});

  auto search = std::make_unique<Search>(
      tree, &backend, std::move(responder), MoveList(),
      std::chrono::steady_clock::now(), std::move(stopper),
      /*infinite=*/false, /*ponder=*/false, option_dict, nullptr);

  const int64_t gathering_before =
      g_testonly_gathering_tasks_executed.load(std::memory_order_relaxed);
  const int64_t processing_tasks_before =
      g_testonly_processing_tasks_executed.load(std::memory_order_relaxed);
  const int64_t processing_calls_before =
      g_testonly_processing_calls.load(std::memory_order_relaxed);

  search->StartThreads(threads);
  search->Wait();

  result.gathering_executed =
      g_testonly_gathering_tasks_executed.load(std::memory_order_relaxed) -
      gathering_before;
  result.processing_executed =
      g_testonly_processing_tasks_executed.load(std::memory_order_relaxed) -
      processing_tasks_before;
  result.processing_calls =
      g_testonly_processing_calls.load(std::memory_order_relaxed) -
      processing_calls_before;
  result.root_n_in_flight = tree.GetCurrentHead()->GetNInFlight();
  result.total_playouts = search->GetTotalPlayouts();
  WalkTreeShape(tree.GetCurrentHead(), /*depth=*/0, &result.shape);
  return result;
}

TEST(SearchPoolEquivalence, SerialConvergesCleanly) {
  RunResult r = RunFixedVisitSearch(/*task_workers=*/0, /*visits=*/5000,
                                    /*threads=*/4);
  EXPECT_TRUE(r.got_bestmove);
  EXPECT_EQ(r.root_n_in_flight, 0u);
  EXPECT_GE(r.total_playouts, 5000);
  EXPECT_GE(r.shape.visited_nodes, 500);
  EXPECT_GE(r.shape.max_depth, 4);
  // task_workers=0 never constructs the pool, so nothing may reach the
  // executor; processing still has to run (inline on the search thread).
  EXPECT_EQ(r.gathering_executed, 0);
  EXPECT_EQ(r.processing_executed, 0);
  EXPECT_GT(r.processing_calls, 0);
}

TEST(SearchPoolEquivalence, PooledConvergesCleanly) {
  RunResult r = RunFixedVisitSearch(/*task_workers=*/4, /*visits=*/5000,
                                    /*threads=*/4);
  EXPECT_TRUE(r.got_bestmove);
  EXPECT_EQ(r.root_n_in_flight, 0u);
  EXPECT_GE(r.total_playouts, 5000);
  // The real proof the pool path did search work rather than degenerating
  // into pure collisions: with real batching and 4 task workers, a shallow
  // or sparse tree here means gathering/processing tasks never actually
  // ran real picks, only what codex-sol's #872 review flagged as an
  // unproven "constructing worker threads is not evidence they executed."
  EXPECT_GE(r.shape.visited_nodes, 500);
  EXPECT_GE(r.shape.max_depth, 4);
  // review #876 f5: the executor counters themselves, not the tree shape,
  // prove worker tasks actually ran -- gathering picks on worker threads, and
  // the batched processing tasks the pool submits for larger batches.
  EXPECT_GT(r.gathering_executed, 0);
  EXPECT_GT(r.processing_executed, 0);
  EXPECT_GT(r.processing_calls, 0);
}

// Not a strict node-for-node match (thread scheduling can pick a slightly
// different path near the visit-budget boundary under either config), but
// both must land in the same ballpark for the identical uniform-eval
// backend -- a real behavioral divergence between the two code paths would
// show up as a large, not a marginal, gap. Tightened from the original
// 0.5x-2.0x band now that real batching (not FETCHED_IMMEDIATELY-always)
// makes both configs' pacing more comparable.
TEST(SearchPoolEquivalence, SerialAndPooledConvergeToComparablePlayouts) {
  RunResult serial = RunFixedVisitSearch(/*task_workers=*/0, /*visits=*/5000,
                                         /*threads=*/4);
  RunResult pooled = RunFixedVisitSearch(/*task_workers=*/4, /*visits=*/5000,
                                         /*threads=*/4);
  ASSERT_TRUE(serial.got_bestmove);
  ASSERT_TRUE(pooled.got_bestmove);
  EXPECT_EQ(serial.root_n_in_flight, 0u);
  EXPECT_EQ(pooled.root_n_in_flight, 0u);
  EXPECT_GE(serial.shape.visited_nodes, 500);
  EXPECT_GE(pooled.shape.visited_nodes, 500);
  // Executor counters: the serial config must never reach the pool executor,
  // the pooled one must have run both task types.
  EXPECT_EQ(serial.gathering_executed, 0);
  EXPECT_GT(pooled.gathering_executed, 0);
  EXPECT_GT(serial.processing_calls, 0);
  EXPECT_GT(pooled.processing_calls, 0);

  const double ratio = static_cast<double>(pooled.total_playouts) /
                       static_cast<double>(serial.total_playouts);
  EXPECT_GT(ratio, 0.7);
  EXPECT_LT(ratio, 1.4);

  const double node_ratio = static_cast<double>(pooled.shape.visited_nodes) /
                            static_cast<double>(serial.shape.visited_nodes);
  EXPECT_GT(node_ratio, 0.6);
  EXPECT_LT(node_ratio, 1.6);
}

// TaskWorkers=-1 (the actual UCI default -- real games never set this
// explicitly) resolves via a runtime heuristic in SearchWorker's
// constructor (search.h ~221), not a fixed value: runs_on_cpu backends
// collapse straight to 0 (serial), everything else sizes workers off
// hardware_concurrency(). Both branches need their own coverage since
// they're different code paths, not just different inputs to the same one.
TEST(SearchPoolEquivalence, DefaultHeuristicOnCpuBackendResolvesToSerial) {
  RunResult r = RunFixedVisitSearch(/*task_workers=*/-1, /*visits=*/5000,
                                    /*threads=*/4, /*runs_on_cpu=*/true);
  EXPECT_TRUE(r.got_bestmove);
  EXPECT_EQ(r.root_n_in_flight, 0u);
  EXPECT_GE(r.total_playouts, 5000);
  EXPECT_GE(r.shape.visited_nodes, 500);
  EXPECT_GE(r.shape.max_depth, 4);
  // -1 on a CPU backend collapses to 0 workers (no pool), per the heuristic.
  EXPECT_EQ(r.gathering_executed, 0);
  EXPECT_EQ(r.processing_executed, 0);
  EXPECT_GT(r.processing_calls, 0);
}

TEST(SearchPoolEquivalence, DefaultHeuristicOnGpuBackendResolvesToPooled) {
  // runs_on_cpu=false takes the hardware_concurrency()-sized branch -- this
  // is what a real sycl-fp16/sycl-auto game actually runs with, since -1 is
  // the option's default and nothing in a normal UCI session overrides it.
  RunResult r = RunFixedVisitSearch(/*task_workers=*/-1, /*visits=*/5000,
                                    /*threads=*/4, /*runs_on_cpu=*/false);
  EXPECT_TRUE(r.got_bestmove);
  EXPECT_EQ(r.root_n_in_flight, 0u);
  EXPECT_GE(r.total_playouts, 5000);
  EXPECT_GE(r.shape.visited_nodes, 500);
  EXPECT_GE(r.shape.max_depth, 4);
  // A pool was actually constructed and its workers executed tasks.
  EXPECT_GT(r.gathering_executed, 0);
  EXPECT_GT(r.processing_calls, 0);
}

}  // namespace
}  // namespace classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
