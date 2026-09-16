/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

// Tests for TaskStealingPool, motivated by the classic-search refactor review
// (agora #858): the publish-before-count race in Submit() let WaitForAll()
// return while tasks were still mutating the tree under the caller's
// exclusive lock, and a throwing executor killed the worker thread and
// stranded the counter. Both are regression-tested here.

#include "utils/task_stealing_pool.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace lczero {
namespace {

TEST(TaskStealingPool, RunsAllSubmittedTasksExactlyOnce) {
  std::atomic<int> total{0};
  std::atomic<int> executions{0};
  TaskStealingPool<int> pool(4, [&total, &executions](int& t, int) {
    total.fetch_add(t);
    executions.fetch_add(1);
  });
  std::vector<int> v;
  for (int i = 1; i <= 100; ++i) v.push_back(i);
  pool.Submit(std::move(v));
  pool.WaitForAll();
  EXPECT_EQ(total.load(), 5050);
  EXPECT_EQ(executions.load(), 100);
  auto done = pool.DrainCompleted();
  EXPECT_EQ(done.size(), 100u);
}

TEST(TaskStealingPool, SingleSubmitsRoundTrip) {
  std::atomic<int> seen{0};
  TaskStealingPool<int> pool(3, [&seen](int& t, int) {
    seen.fetch_add(t);
  });
  for (int i = 0; i < 30; ++i) pool.Submit(2);
  pool.WaitForAll();
  EXPECT_EQ(seen.load(), 60);
  EXPECT_EQ(pool.CompletedCount(), 30);
}

// The #858 finding 1 detector. A parent task submits children from INSIDE
// the executor and then lingers; with the old protocol (push to queue, then
// count) a thief could finish a child and drive active_tasks_ transiently to
// zero before the parent's increment landed, releasing WaitForAll early
// while the parent still ran. With count-before-publish, WaitForAll may wait
// too long but never too little -- executed() must equal submitted+children
// at every round boundary.
TEST(TaskStealingPool, SubmitFromTaskNeverReleasesWaitEarly) {
  constexpr int kParents = 6;
  constexpr int kChildren = 6;
  constexpr int kRounds = 120;
  std::atomic<int> executed{0};
  TaskStealingPool<int>* pool = nullptr;
  std::function<void(int&, int)> exec = [&executed, &pool](int& tag, int) {
    if (tag > 0) {
      for (int i = 0; i < kChildren; ++i) pool->Submit(-tag);
      // Widen the race window the old code lost: children can finish here,
      // before this parent's own completion decrement, while the parent's
      // per-child increments were historically still pending.
      std::this_thread::sleep_for(std::chrono::microseconds(30));
    }
    executed.fetch_add(1);
  };
  TaskStealingPool<int> the_pool(3, std::move(exec));
  pool = &the_pool;
  for (int round = 0; round < kRounds; ++round) {
    the_pool.Reset();
    std::vector<int> parents;
    for (int i = 1; i <= kParents; ++i) parents.push_back(i);
    the_pool.Submit(std::move(parents));
    the_pool.WaitForAll();
    ASSERT_EQ(executed.load(), kParents * (1 + kChildren) * (round + 1))
        << "WaitForAll() returned before nested submissions completed "
           "(round "
        << round << ") -- publish-before-count regression";
  }
  EXPECT_EQ(the_pool.CompletedCount(), kParents * (1 + kChildren));
}

// The #858 finding 2 companion: an executor exception used to escape into
// the std::thread (terminate) and/or strand active_tasks_ (hang WaitForAll
// while the caller holds nodes_mutex_). Now the pool must absorb it, keep
// waiting correctness, and stay usable.
TEST(TaskStealingPool, ThrowingExecutorDoesNotStrandWaitOrKillPool) {
  std::atomic<int> ran{0};
  TaskStealingPool<int> pool(2, [&ran](int& tag, int) {
    ++ran;
    if (tag == 13) throw std::runtime_error("injected executor failure");
  });
  std::vector<int> v;
  for (int i = 0; i < 20; ++i) v.push_back(i);
  pool.Submit(std::move(v));
  // WaitForAll() now rethrows the executor's exception on the caller's own
  // stack (review #863 finding 4: silently marking a partially-mutated task
  // as a plain success was the bug, not just "does WaitForAll hang"), but
  // every task -- including ones queued/stolen after the throwing one --
  // must still have run and reached completed_tasks_ first.
  EXPECT_THROW(pool.WaitForAll(), std::runtime_error);
  EXPECT_EQ(ran.load(), 20);
  EXPECT_EQ(pool.DrainCompleted().size(), 20u);
  // Still usable afterwards.
  pool.Reset();
  pool.Submit(5);
  pool.WaitForAll();
  EXPECT_EQ(pool.CompletedCount(), 1);
}

TEST(TaskStealingPool, WaitForAllRethrowsOnlyOnce) {
  // A second WaitForAll() call (no new work submitted, no Reset()) must not
  // rethrow the same exception again -- it was already delivered.
  TaskStealingPool<int> pool(1, [](int& tag, int) {
    if (tag == 1) throw std::runtime_error("boom");
  });
  pool.Submit(1);
  EXPECT_THROW(pool.WaitForAll(), std::runtime_error);
  EXPECT_NO_THROW(pool.WaitForAll());
}

// After the worker sleep path (real condvar sleep since the notify-protocol
// fix), work must still arrive promptly -- guards both the sleep itself and
// the Submit wakes-a-sleeper protocol.
TEST(TaskStealingPool, WorkArrivesAfterRealSleep) {
  std::atomic<int> sum{0};
  TaskStealingPool<int> pool(2, [&sum](int& t, int) { sum.fetch_add(t); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  std::vector<int> v{6, 7};
  pool.Submit(std::move(v));
  pool.WaitForAll();
  EXPECT_EQ(sum.load(), 13);
}

// The #866 P1-2 detector, at the level this pool actually guarantees:
// completed_tasks_ retains EVERY task -- including ones that finished
// cleanly -- whether or not WaitForAll() ends up throwing. This documents
// (and pins) the calling convention search.cc's PickNodesToExtend() relies
// on: a caller MUST still call DrainCompleted() when WaitForAll() throws, or
// a sibling task's real results (here, its own `results` payload, mirroring
// PickTask::results) are silently dropped even though the pool itself never
// lost them.
TEST(TaskStealingPool, CompletedResultsSurviveAThrowingSiblingUntilDrained) {
  struct ResultTask {
    int tag = 0;
    std::vector<int> results;
  };
  TaskStealingPool<ResultTask> pool(4, [](ResultTask& t, int) {
    if (t.tag == 13) throw std::runtime_error("injected executor failure");
    t.results.push_back(t.tag * 10);
  });
  std::vector<ResultTask> tasks;
  for (int i = 0; i < 20; ++i) tasks.push_back(ResultTask{i, {}});
  pool.Submit(std::move(tasks));

  // The caller-side pattern this test pins: catch, drain unconditionally,
  // merge, THEN rethrow -- not WaitForAll() followed unconditionally by
  // DrainCompleted(), which skips the drain entirely on the throwing path.
  std::exception_ptr pending;
  try {
    pool.WaitForAll();
  } catch (...) {
    pending = std::current_exception();
  }
  auto completed = pool.DrainCompleted();
  EXPECT_EQ(completed.size(), 20u);
  int merged_results = 0;
  for (auto& t : completed) merged_results += static_cast<int>(t.results.size());
  // 19 successful tasks each produced one result; task 13 threw before
  // appending its own.
  EXPECT_EQ(merged_results, 19);
  ASSERT_TRUE(pending != nullptr);
  EXPECT_THROW(std::rethrow_exception(pending), std::runtime_error);
}

TEST(TaskStealingPool, StealUnderLoadPreservesCounts) {
  std::atomic<long long> sum{0};
  TaskStealingPool<int> pool(4, [&sum](int& t, int) {
    sum.fetch_add(t);
  });
  std::vector<int> v;
  for (int i = 1; i <= 2000; ++i) v.push_back(i);
  pool.Submit(std::move(v));
  pool.WaitForAll();
  EXPECT_EQ(sum.load(), 2000LL * 2001 / 2);
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
