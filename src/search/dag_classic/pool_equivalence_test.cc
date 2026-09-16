/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

// dag_classic mirror of classic/pool_equivalence_test.cc (agora #867/#868):
// TaskWorkers=0 (serial, task_pool_ never constructed) and TaskWorkers>0
// (pooled) are separate code paths through the same PickNodesToExtend/
// GatherMinibatch logic, ported onto dag_classic this session. Also covers
// TaskWorkers=-1 (the actual UCI default) on both branches of its
// runs_on_cpu heuristic -- real games run with -1, not an explicit count.

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

namespace lczero {
namespace dag_classic {
namespace {

// Same deterministic, always-cache-hit computation as classic's equivalence
// test: uniform policy/value removes real-NN timing/precision noise so a
// visit-count comparison between serial and pooled runs is meaningful.
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
  // See classic/pool_equivalence_test.cc's FakeBackend for why this knob
  // exists: it picks which branch of SearchWorker's TaskWorkers==-1
  // heuristic (search.h ~229) gets exercised.
  explicit FakeBackend(bool runs_on_cpu = true) : runs_on_cpu_(runs_on_cpu) {}

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{.has_mlh = false,
                             .has_wdl = true,
                             .runs_on_cpu = runs_on_cpu_,
                             .suggested_num_search_threads = 1,
                             .recommended_batch_size = 1,
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

struct RunResult {
  bool got_bestmove = false;
  uint32_t root_n_in_flight = 0xffffffffu;  // poisoned until set
  int64_t total_playouts = 0;
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
  TranspositionTable tt;

  auto stopper = std::make_unique<classic::ChainedSearchStopper>();
  stopper->AddStopper(
      std::make_unique<classic::VisitsStopper>(visits, false));

  RunResult result;
  auto responder = std::make_unique<CallbackUciResponder>(
      [&](const BestMoveInfo&) { result.got_bestmove = true; },
      [](const std::vector<ThinkingInfo>&) {});

  auto search = std::make_unique<Search>(
      tree, &backend, std::move(responder), MoveList(),
      std::chrono::steady_clock::now(), std::move(stopper),
      /*infinite=*/false, /*ponder=*/false, option_dict, &tt, nullptr);

  search->StartThreads(threads);
  search->Wait();

  result.root_n_in_flight = tree.GetCurrentHead()->GetNInFlight();
  result.total_playouts = search->GetTotalPlayouts();
  return result;
}

TEST(DagSearchPoolEquivalence, SerialConvergesCleanly) {
  RunResult r = RunFixedVisitSearch(/*task_workers=*/0, /*visits=*/2000,
                                    /*threads=*/4);
  EXPECT_TRUE(r.got_bestmove);
  EXPECT_EQ(r.root_n_in_flight, 0u);
  EXPECT_GE(r.total_playouts, 2000);
}

TEST(DagSearchPoolEquivalence, PooledConvergesCleanly) {
  RunResult r = RunFixedVisitSearch(/*task_workers=*/4, /*visits=*/2000,
                                    /*threads=*/4);
  EXPECT_TRUE(r.got_bestmove);
  EXPECT_EQ(r.root_n_in_flight, 0u);
  EXPECT_GE(r.total_playouts, 2000);
}

TEST(DagSearchPoolEquivalence, SerialAndPooledConvergeToComparablePlayouts) {
  RunResult serial = RunFixedVisitSearch(/*task_workers=*/0, /*visits=*/2000,
                                         /*threads=*/4);
  RunResult pooled = RunFixedVisitSearch(/*task_workers=*/4, /*visits=*/2000,
                                         /*threads=*/4);
  ASSERT_TRUE(serial.got_bestmove);
  ASSERT_TRUE(pooled.got_bestmove);
  EXPECT_EQ(serial.root_n_in_flight, 0u);
  EXPECT_EQ(pooled.root_n_in_flight, 0u);

  const double ratio = static_cast<double>(pooled.total_playouts) /
                       static_cast<double>(serial.total_playouts);
  EXPECT_GT(ratio, 0.5);
  EXPECT_LT(ratio, 2.0);
}

TEST(DagSearchPoolEquivalence, DefaultHeuristicOnCpuBackendResolvesToSerial) {
  RunResult r = RunFixedVisitSearch(/*task_workers=*/-1, /*visits=*/2000,
                                    /*threads=*/4, /*runs_on_cpu=*/true);
  EXPECT_TRUE(r.got_bestmove);
  EXPECT_EQ(r.root_n_in_flight, 0u);
  EXPECT_GE(r.total_playouts, 2000);
}

TEST(DagSearchPoolEquivalence, DefaultHeuristicOnGpuBackendResolvesToPooled) {
  RunResult r = RunFixedVisitSearch(/*task_workers=*/-1, /*visits=*/2000,
                                    /*threads=*/4, /*runs_on_cpu=*/false);
  EXPECT_TRUE(r.got_bestmove);
  EXPECT_EQ(r.root_n_in_flight, 0u);
  EXPECT_GE(r.total_playouts, 2000);
}

}  // namespace
}  // namespace dag_classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
