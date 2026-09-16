/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#pragma once

#include <atomic>
#include <cstdint>
#include <exception>

// Test-only instrumentation: fault-injection seams and execution counters.
//
// Everything here is compiled in ONLY when LC0_TEST_INSTRUMENTATION is
// defined, which meson.build does for the test executables whose sources
// include this header. Production targets (lc0.exe) compile the stub
// overloads below instead, so the search/pool hot paths carry no shared
// atomic RMW and no extra branches (review #878 finding 5) -- "test-only by
// naming" is not enough when the call site executes per pick and per task.
//
// The seams are per-site counters so a test arms exactly the location it
// means to exercise, instead of racing one process-wide countdown whose
// winning site is whichever thread/level gets there first (#876 finding 5).
// Each site starts disabled (-1, one relaxed load per call site); a
// non-negative value N makes that site throw on its (N+1)-th hit, i.e. after
// N earlier hits were consumed.

namespace lczero {

enum class TestOnlyThrowSite : int {
  // Immediately after an n_in_flight_ increment made mid-traversal (classic
  // and dag_classic). The original #872 P1 seam.
  kAfterReservation = 0,
  // Immediately before a receiver push_back, simulating the emission itself
  // failing while the rollback ledger still owns the coverage (#876 f2).
  kBeforeEmission,
  // Immediately after a successful pool Submit, while the submitting call
  // still holds its own mirror of the handed-off coverage (#874/#876 f4).
  kAfterSubmit,
  // Entry of a submitted task, after its entry guard exists but before any
  // allocating setup (#876 f3).
  kBeforeTaskEntrySetup,
  // A rollback-record/level-reservation capacity growth point, before the
  // growth allocates (#876 f1: a growth failure must not orphan an increment
  // that has already been applied).
  kBeforeRecordGrowth,
  // The classic task's warm ledger reserve at call entry, before the entry
  // guard would previously have been constructed (#878 f1).
  kBeforeInitialReserve,
  // DAG promotion's current_path/reservation_ledger growth points, before
  // either vector changes (#878 f3).
  kBeforePromotionPathGrowth,
  kBeforePromotionLedgerGrowth,
  // A queue insertion in TaskStealingPool::Submit, before the task is
  // published (#878 f4).
  kPoolInsert,
  // The completion-slot reserve in TaskStealingPool (review #881): the
  // acceptance-time allocation that keeps worker-side completion publication
  // nonallocating.
  kPoolCompletionReserve,
  // Immediately before a collision insertion that follows a successful Visit
  // insertion in the same stop_picking entry (review #881 P2). Distinct from
  // kBeforeEmission, which fires for collision-only entries too.
  kBeforeCollisionAfterVisit,
  // Immediately before the destination reserve that merges drained task
  // results into minibatch_ (review #883).
  kBeforeMergeReserve,
  // Top of the pool executor callback, before the task's own code runs.
  kWorkerExecutor,
  // Top of ProcessPickedTask, before it touches the batch.
  kProcessing,
  kCount
};

#if defined(LC0_TEST_INSTRUMENTATION)

struct TestOnlyInjectedReservationThrow : public std::exception {
  const char* what() const noexcept override {
    return "TestOnlyInjectedReservationThrow";
  }
};

struct TestOnlyThrowCounters {
  std::atomic<int64_t> count[static_cast<int>(TestOnlyThrowSite::kCount)];
  TestOnlyThrowCounters() {
    for (auto& c : count) c.store(-1, std::memory_order_relaxed);
  }
};

// Context of the most recent seam throw (review #881 P2): recorded so a
// targeted test can state WHERE its call failed instead of merely that some
// worker task ran at some point. Pool executor threads mark this at task
// entry; TestOnlyMaybeThrowAt snapshots it when it throws.
inline thread_local bool t_testonly_in_pool_task = false;
inline std::atomic<bool> g_testonly_last_throw_in_pool_task{false};

static inline void TestOnlyMarkPoolTaskContext() {
  t_testonly_in_pool_task = true;
}

static inline bool TestOnlyLastThrowInPoolTask() {
  return g_testonly_last_throw_in_pool_task.load(std::memory_order_relaxed);
}

// Remaining collision share at the post-visit seam (review #881 P2): the seam
// may only fire with a share still left after the Visit was emitted.
inline std::atomic<int64_t> g_testonly_post_visit_remaining_share{-1};

static inline void TestOnlyRecordPostVisitRemainingShare(int share) {
  g_testonly_post_visit_remaining_share.store(share, std::memory_order_relaxed);
}

static inline int64_t TestOnlyPostVisitRemainingShare() {
  return g_testonly_post_visit_remaining_share.load(std::memory_order_relaxed);
}

// The single counter array; address-stable for the process lifetime.
inline TestOnlyThrowCounters g_testonly_throw_counters;

// Set when a site actually throws; cleared by TestOnlyResetSeams().
inline std::atomic<bool> g_testonly_throw_fired[static_cast<int>(
    TestOnlyThrowSite::kCount)] = {};

// Execution counters: tree size/depth does not prove which task path actually
// ran, and constructing a pool does not prove its workers executed anything.
// The executor increments these per task type on every executed task;
// ProcessPickedTask increments the call counter at entry (main-thread slices
// included), so pool_equivalence tests can assert gathering and processing
// work really happened.
inline std::atomic<int64_t> g_testonly_gathering_tasks_executed{0};
inline std::atomic<int64_t> g_testonly_processing_tasks_executed{0};
inline std::atomic<int64_t> g_testonly_processing_calls{0};

static inline void TestOnlyMaybeThrowAt(TestOnlyThrowSite site) {
  auto& counter = g_testonly_throw_counters.count[static_cast<int>(site)];
  int64_t v = counter.load(std::memory_order_relaxed);
  if (v < 0) return;
  if (counter.fetch_sub(1, std::memory_order_relaxed) == 0) {
    g_testonly_throw_fired[static_cast<int>(site)].store(
        true, std::memory_order_relaxed);
    g_testonly_last_throw_in_pool_task.store(t_testonly_in_pool_task,
                                             std::memory_order_relaxed);
    throw TestOnlyInjectedReservationThrow{};
  }
}

static inline void TestOnlyMaybeThrowAfterReservation() {
  TestOnlyMaybeThrowAt(TestOnlyThrowSite::kAfterReservation);
}

static inline void TestOnlyMaybeThrowOnRecordGrowth() {
  TestOnlyMaybeThrowAt(TestOnlyThrowSite::kBeforeRecordGrowth);
}

static inline void TestOnlyRecordGatheringTaskExecuted() {
  g_testonly_gathering_tasks_executed.fetch_add(1, std::memory_order_relaxed);
}

static inline void TestOnlyRecordProcessingTaskExecuted() {
  g_testonly_processing_tasks_executed.fetch_add(1, std::memory_order_relaxed);
}

static inline void TestOnlyRecordProcessingCall() {
  g_testonly_processing_calls.fetch_add(1, std::memory_order_relaxed);
}

static inline int64_t TestOnlyGatheringTasksExecuted() {
  return g_testonly_gathering_tasks_executed.load(std::memory_order_relaxed);
}

static inline int64_t TestOnlyProcessingTasksExecuted() {
  return g_testonly_processing_tasks_executed.load(std::memory_order_relaxed);
}

static inline int64_t TestOnlyProcessingCalls() {
  return g_testonly_processing_calls.load(std::memory_order_relaxed);
}

// Test-only: overrides the DAG TaskWorkspace's initial reserve for the
// parallel current_path/reservation_ledger vectors, so promotion growth --
// and the failure seams at it -- can be reached deterministically in a
// small search instead of only after hundreds of thousands of visits. -1
// keeps the production value. Reset by TestOnlyResetSeams().
inline std::atomic<int> g_testonly_workspace_reserve_override{-1};

static inline int TestOnlyWorkspaceReserve() {
  return g_testonly_workspace_reserve_override.load(std::memory_order_relaxed);
}

static inline void TestOnlySetWorkspaceReserveOverride(int reserve) {
  g_testonly_workspace_reserve_override.store(reserve,
                                              std::memory_order_relaxed);
}

// True once the given site has actually thrown since the last
// TestOnlyResetSeams(). A targeted test asserts this so it cannot pass
// vacuously when its armed seam was never reached.
static inline bool TestOnlyWasThrowFired(TestOnlyThrowSite site) {
  return g_testonly_throw_fired[static_cast<int>(site)].load(
      std::memory_order_relaxed);
}

// Arm one site: the next `hits` calls pass, then the following one throws.
static inline void TestOnlySetThrowCount(TestOnlyThrowSite site, int64_t hits) {
  g_testonly_throw_counters.count[static_cast<int>(site)].store(
      hits, std::memory_order_relaxed);
}

// Disable every fault seam and zero the execution counters. Tests call this
// in TearDown so a test that doesn't fully consume a countdown cannot bleed
// into the next test in this binary.
static inline void TestOnlyResetSeams() {
  for (auto& c : g_testonly_throw_counters.count) {
    c.store(-1, std::memory_order_relaxed);
  }
  for (auto& f : g_testonly_throw_fired) {
    f.store(false, std::memory_order_relaxed);
  }
  g_testonly_gathering_tasks_executed.store(0, std::memory_order_relaxed);
  g_testonly_processing_tasks_executed.store(0, std::memory_order_relaxed);
  g_testonly_processing_calls.store(0, std::memory_order_relaxed);
  g_testonly_workspace_reserve_override.store(-1, std::memory_order_relaxed);
  g_testonly_last_throw_in_pool_task.store(false, std::memory_order_relaxed);
  g_testonly_post_visit_remaining_share.store(-1, std::memory_order_relaxed);
}

#else  // !LC0_TEST_INSTRUMENTATION

// Production stubs: empty inline functions, so every call site above
// compiles away. The enum stays (call sites name its values).
static inline void TestOnlyMaybeThrowAt(TestOnlyThrowSite) {}
static inline void TestOnlyMaybeThrowAfterReservation() {}
static inline void TestOnlyMaybeThrowOnRecordGrowth() {}
static inline void TestOnlyRecordGatheringTaskExecuted() {}
static inline void TestOnlyRecordProcessingTaskExecuted() {}
static inline void TestOnlyRecordProcessingCall() {}
static inline void TestOnlyMarkPoolTaskContext() {}
static inline void TestOnlyRecordPostVisitRemainingShare(int) {}
static inline int TestOnlyWorkspaceReserve() { return -1; }

#endif  // LC0_TEST_INSTRUMENTATION

}  // namespace lczero
