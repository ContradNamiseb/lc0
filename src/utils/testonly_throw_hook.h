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

namespace lczero {

// Test-only fault-injection seams (agora #41/#872, split into per-site
// counters per review #876 finding 5): a test arms exactly the location it
// means to exercise instead of racing one process-wide countdown whose
// winning site is whichever thread/level gets there first. Each site starts
// disabled (-1, one relaxed load per call site); a non-negative value N makes
// that site throw on its (N+1)-th hit, i.e. after N earlier hits were
// consumed. The throw is TestOnlyInjectedReservationThrow, which the pool
// and the search's own cleanup paths treat as any other exception.
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
  // Entry of a submitted task, before any allocating setup and before its
  // inherited-coverage ledger records exist (#876 f3).
  kBeforeTaskEntrySetup,
  // A rollback-record/level-reservation capacity growth point, before the
  // growth allocates (#876 f1: a growth failure must not orphan an increment
  // that has already been applied).
  kBeforeRecordGrowth,
  // Top of the pool executor callback, before the task's own code runs.
  kWorkerExecutor,
  // Top of ProcessPickedTask, before it touches the batch.
  kProcessing,
  kCount
};

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

// The single counter array; address-stable for the process lifetime.
inline TestOnlyThrowCounters g_testonly_throw_counters;

// Set when a site actually throws; cleared by TestOnlyResetSeams().
inline std::atomic<bool> g_testonly_throw_fired[static_cast<int>(
    TestOnlyThrowSite::kCount)] = {};

// Execution counters (agora #41/#876): tree size/depth does not prove which
// task path actually ran, and constructing a pool does not prove its workers
// executed anything. The executor increments these per task type on every
// executed task; ProcessPickedTask increments the call counter at entry (main
// thread slices included), so pool_equivalence tests can assert gathering
// and processing work really happened. Test-only, monotonic, relaxed.
inline std::atomic<int64_t> g_testonly_gathering_tasks_executed{0};
inline std::atomic<int64_t> g_testonly_processing_tasks_executed{0};
inline std::atomic<int64_t> g_testonly_processing_calls{0};

inline void TestOnlyMaybeThrowAt(TestOnlyThrowSite site) {
  auto& counter = g_testonly_throw_counters.count[static_cast<int>(site)];
  int64_t v = counter.load(std::memory_order_relaxed);
  if (v < 0) return;
  if (counter.fetch_sub(1, std::memory_order_relaxed) == 0) {
    g_testonly_throw_fired[static_cast<int>(site)].store(
        true, std::memory_order_relaxed);
    throw TestOnlyInjectedReservationThrow{};
  }
}

// True once the given site has actually thrown since the last
// TestOnlyResetSeams(). A targeted test asserts this so it cannot pass
// vacuously when its armed seam was never reached (e.g. no submission ever
// happened, or the traversal never needed to grow the ledger).
inline bool TestOnlyWasThrowFired(TestOnlyThrowSite site) {
  return g_testonly_throw_fired[static_cast<int>(site)].load(
      std::memory_order_relaxed);
}

inline void TestOnlyMaybeThrowAfterReservation() {
  TestOnlyMaybeThrowAt(TestOnlyThrowSite::kAfterReservation);
}

inline void TestOnlyMaybeThrowOnRecordGrowth() {
  TestOnlyMaybeThrowAt(TestOnlyThrowSite::kBeforeRecordGrowth);
}

// Arm one site: the next `hits` calls pass, then the following one throws.
inline void TestOnlySetThrowCount(TestOnlyThrowSite site, int64_t hits) {
  g_testonly_throw_counters.count[static_cast<int>(site)].store(
      hits, std::memory_order_relaxed);
}

// Disable every fault seam and zero the execution counters. Tests call this
// in TearDown so a test that doesn't fully consume a countdown cannot bleed
// into the next test in this binary.
inline void TestOnlyResetSeams() {
  for (auto& c : g_testonly_throw_counters.count) {
    c.store(-1, std::memory_order_relaxed);
  }
  for (auto& f : g_testonly_throw_fired) {
    f.store(false, std::memory_order_relaxed);
  }
  g_testonly_gathering_tasks_executed.store(0, std::memory_order_relaxed);
  g_testonly_processing_tasks_executed.store(0, std::memory_order_relaxed);
  g_testonly_processing_calls.store(0, std::memory_order_relaxed);
}

}  // namespace lczero
