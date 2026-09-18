/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

// Coverage for 's request: "exactly 256 successful claims
// and a 257th local fallback." TryClaimRoundTaskSlot is dag_classic's
// MAX_TASKS==256 round-wide budget (search.cc), pulled out here so its
// exact-cap behavior under real concurrent contention has direct test
// coverage without needing to drive a real search past 256 split
// opportunities.

#include "utils/round_task_slot.h"

#include <atomic>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace lczero {
namespace {

TEST(RoundTaskSlot, SingleThreadedExactlyCapSucceedsRestFallBack) {
  std::atomic<int> counter{0};
  constexpr int kMaxTasks = 256;
  int successes = 0;
  int failures = 0;
  // 300 attempts against a 256 cap: the first 256 must succeed, the
  // remaining 44 (the "257th local fallback" case and beyond) must not.
  for (int i = 0; i < 300; i++) {
    if (TryClaimRoundTaskSlot(counter, kMaxTasks)) {
      successes++;
    } else {
      failures++;
    }
  }
  EXPECT_EQ(successes, kMaxTasks);
  EXPECT_EQ(failures, 300 - kMaxTasks);
  EXPECT_EQ(counter.load(), kMaxTasks);
}

TEST(RoundTaskSlot, ConcurrentContentionNeverExceedsCap) {
  std::atomic<int> counter{0};
  constexpr int kMaxTasks = 256;
  constexpr int kThreads = 8;
  constexpr int kAttemptsPerThread = 200;  // 1600 total, far past the cap.
  std::atomic<int> total_successes{0};

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; t++) {
    threads.emplace_back([&]() {
      int local_successes = 0;
      for (int i = 0; i < kAttemptsPerThread; i++) {
        if (TryClaimRoundTaskSlot(counter, kMaxTasks)) {
          local_successes++;
        }
      }
      total_successes.fetch_add(local_successes, std::memory_order_relaxed);
    });
  }
  for (auto& t : threads) t.join();

  // The exact invariant this protects: under real contention (this is the
  // scenario multiple recursively-submitting task-pool workers actually
  // hit), the shared counter must land at exactly the cap -- not under it
  // (would mean claims were lost) and not over it (would mean the round's
  // task budget was exceeded, the defect this cap exists to prevent).
  EXPECT_EQ(total_successes.load(), kMaxTasks);
  EXPECT_EQ(counter.load(), kMaxTasks);
}

TEST(RoundTaskSlot, ZeroCapAlwaysFallsBack) {
  std::atomic<int> counter{0};
  EXPECT_FALSE(TryClaimRoundTaskSlot(counter, /*max_tasks=*/0));
  EXPECT_EQ(counter.load(), 0);
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
