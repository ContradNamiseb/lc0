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
#include "utils/testonly_throw_hook.h"

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

// Fault-injection fixture for the Submit publication rollback (#878 finding
// 4) and the completion-publication contract (#881): always leave every seam
// disarmed so an unexpectedly surviving countdown cannot bleed into the next
// test in this binary.
class PoolInsertFailureTest : public ::testing::Test {
 protected:
  void TearDown() override { TestOnlyResetSeams(); }
};

// Task with an owned payload, so "partial result ownership" is observable.
struct PoolPayloadTask {
  int tag = 0;
  std::vector<int> payload;
};

// review #881: completion publication must not allocate on worker threads.
// The completion slot is reserved at acceptance, on the submitting thread, so
// a failure there rejects the task cleanly -- no phantom counts -- while
// already-accepted peers keep their payloads and drain normally.
TEST_F(PoolInsertFailureTest, CompletionReserveFailureRollsBackAcceptance) {
  std::atomic<int> ran{0};
  std::atomic<int> payload_entries{0};
  TaskStealingPool<PoolPayloadTask> pool(3, [&](PoolPayloadTask& t, int) {
    ran.fetch_add(1);
    payload_entries.fetch_add(static_cast<int>(t.payload.size()));
  });

  // Accepted peers published BEFORE the failure...
  std::vector<PoolPayloadTask> first;
  for (int i = 0; i < 3; ++i) {
    first.push_back(PoolPayloadTask{i, std::vector<int>(2, i)});
  }
  pool.Submit(std::move(first));

  // ...then the completion-slot reserve fails and this task is rejected.
  TestOnlySetThrowCount(TestOnlyThrowSite::kPoolCompletionReserve, 0);
  EXPECT_THROW(pool.Submit(PoolPayloadTask{99, std::vector<int>{9}}),
               TestOnlyInjectedReservationThrow);

  // Bounded completion: the accepted prefix runs with payloads intact, and
  // the rejected task never ran and left no phantom count behind.
  pool.WaitForAll();
  EXPECT_EQ(ran.load(), 3);
  EXPECT_EQ(payload_entries.load(), 6);
  EXPECT_EQ(pool.DrainCompleted().size(), 3u);
  EXPECT_EQ(pool.CompletedCount(), 3);

  // Next round is fully usable.
  TestOnlyResetSeams();
  pool.Reset();
  std::vector<PoolPayloadTask> second;
  for (int i = 0; i < 4; ++i) {
    second.push_back(PoolPayloadTask{i, std::vector<int>(1, i)});
  }
  pool.Submit(std::move(second));
  pool.WaitForAll();
  EXPECT_EQ(pool.CompletedCount(), 4);
  EXPECT_EQ(pool.DrainCompleted().size(), 4u);
}

// Capacity survives drains (review #881): DrainCompleted() must not drop the
// member vector's capacity, or a later worker completion would allocate again.
// Pinned behaviorally across submit/wait/drain rounds with partial-result
// payloads.
TEST_F(PoolInsertFailureTest, CompletionCapacitySurvivesDrains) {
  std::atomic<int> ran{0};
  TaskStealingPool<PoolPayloadTask> pool(4, [&ran](PoolPayloadTask& t, int) {
    ran.fetch_add(1);
    t.payload.clear();  // consume, like task results merged by the caller
  });

  for (int round = 0; round < 5; ++round) {
    std::vector<PoolPayloadTask> tasks;
    for (int i = 0; i < 12; ++i) {
      tasks.push_back(
          PoolPayloadTask{i, std::vector<int>(static_cast<size_t>(i) + 1, i)});
    }
    pool.Submit(std::move(tasks));
    pool.WaitForAll();
    auto done = pool.DrainCompleted();
    ASSERT_EQ(done.size(), 12u) << "round " << round;
    pool.Reset();
  }
  EXPECT_EQ(ran.load(), 60);
}

// A single-insertion failure must not strand a phantom active task (WaitForAll
// would hang with nothing left to complete it), must reject the task, and must
// leave the pool usable. Test timeouts bound the wait.
TEST_F(PoolInsertFailureTest, SingleInsertionFailureRollsBackCounts) {
  std::atomic<int> ran{0};
  TaskStealingPool<int> pool(2, [&ran](int& t, int) { ran.fetch_add(t); });

  TestOnlySetThrowCount(TestOnlyThrowSite::kPoolInsert, 0);
  EXPECT_THROW(pool.Submit(7), TestOnlyInjectedReservationThrow);

  // Must return promptly: the failed task is not active and was never queued.
  pool.WaitForAll();
  EXPECT_EQ(ran.load(), 0);
  EXPECT_EQ(pool.CompletedCount(), 0);
  EXPECT_EQ(pool.DrainCompleted().size(), 0u);

  // The pool is still usable for a subsequent round.
  TestOnlyResetSeams();
  pool.Reset();
  pool.Submit(5);
  pool.WaitForAll();
  EXPECT_EQ(ran.load(), 5);
  EXPECT_EQ(pool.CompletedCount(), 1);
}

// Batch publication is prefix-transactional: on a kth-insertion failure the
// already-queued prefix is ACCEPTED (counted, executed, drained), while the
// failed element and the tail are REJECTED (not queued, counts rolled back,
// still owned by the caller's vector -- their payloads were never moved).
TEST_F(PoolInsertFailureTest, BatchInsertionFailurePublishesOnlyAcceptedPrefix) {
  struct PayloadTask {
    int tag = 0;
    std::vector<int> payload;
  };

  std::atomic<int> ran{0};
  std::atomic<long long> sum{0};
  TaskStealingPool<PayloadTask> pool(4, [&ran, &sum](PayloadTask& t, int) {
    ran.fetch_add(1);
    sum.fetch_add(t.tag);
  });

  std::vector<PayloadTask> tasks;
  for (int i = 0; i < 10; ++i) {
    tasks.push_back(PayloadTask{i, std::vector<int>{i}});
  }

  // hits=3: insertions 0..2 succeed, insertion 3 throws.
  TestOnlySetThrowCount(TestOnlyThrowSite::kPoolInsert, 3);
  EXPECT_THROW(pool.Submit(std::move(tasks)), TestOnlyInjectedReservationThrow);

  // Accepted prefix: queued, counted, runs to completion, drains.
  pool.WaitForAll();
  EXPECT_EQ(ran.load(), 3);
  EXPECT_EQ(sum.load(), 0 + 1 + 2);
  EXPECT_EQ(pool.DrainCompleted().size(), 3u);

  // Rejection boundary: accepted elements were moved into the pool (payload
  // emptied), rejected ones still own theirs.
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(tasks[i].payload.empty()) << "element " << i;
  }
  for (int i = 3; i < 10; ++i) {
    EXPECT_FALSE(tasks[i].payload.empty()) << "element " << i;
  }

  // A subsequent round works normally.
  TestOnlyResetSeams();
  pool.Reset();
  pool.Submit(PayloadTask{7, std::vector<int>{7}});
  pool.WaitForAll();
  EXPECT_EQ(pool.CompletedCount(), 1);
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
