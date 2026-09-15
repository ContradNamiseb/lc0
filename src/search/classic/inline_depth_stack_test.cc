/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

// Boundary tests for the classic-search picking cache structures, encoding
// the probes from the agora #857 release review:
//  * InlineDepthStack must survive descents past its inline capacity
//    (push 257 used to be an ASan-confirmed stack-buffer-overflow scribbling
//    into adjacent CachedNodeData memory).
//  * Node::CopyPolicy's strided (AoS) path must agree with the contiguous
//    path for every edge count and must not touch neighboring fields.
// Compiled with -fno-access-control (meson) so the private nested types are
// reachable, same convention as the review probes.

#include <array>
#include <vector>

#include "search/classic/search.h"
#include "gtest/gtest.h"

namespace lczero {
namespace classic {
namespace {

using CachedNodeData = SearchWorker::CachedNodeData;
using ChildCache = SearchWorker::ChildCache;
using InlineDepthStack = SearchWorker::InlineDepthStack;

TEST(InlineDepthStack, InlinePathGatesExactlyByCount) {
  InlineDepthStack s;
  EXPECT_EQ(s.count, 0);
  for (int i = 0; i < InlineDepthStack::kInlineCapacity; ++i) s.push_back(i);
  EXPECT_EQ(s.count, InlineDepthStack::kInlineCapacity);
  EXPECT_EQ(s.back(), InlineDepthStack::kInlineCapacity - 1);
  for (int i = 0; i < InlineDepthStack::kInlineCapacity; ++i) {
    ASSERT_EQ(s.back(), InlineDepthStack::kInlineCapacity - 1 - i);
    s.pop_back();
  }
  EXPECT_EQ(s.count, 0);
  s.pop_back();  // popping empty is a no-op, must not go negative
  EXPECT_EQ(s.count, 0);
}

TEST(InlineDepthStack, SpillAcrossTheOldOverflowBoundary) {
  InlineDepthStack s;
  for (int i = 0; i < 257; ++i) s.push_back(i);  // 257 was the ASan repro
  EXPECT_EQ(s.count, 257);
  EXPECT_EQ(s.back(), 256);
  s.back() = 12345;  // back() must be a writable lvalue (search.cc uses it
                     // as `cache.vtp_last_filled_cache.back() = best_idx;`)
  EXPECT_EQ(s.back(), 12345);
  s.pop_back();  // drops the spilled entry
  EXPECT_EQ(s.count, InlineDepthStack::kInlineCapacity);
  EXPECT_EQ(s.back(), 255);  // last inline slot
  s.pop_back();
  EXPECT_EQ(s.count, 255);
  EXPECT_EQ(s.back(), 254);
}

TEST(InlineDepthStack, DeepDescentAndReuse) {
  InlineDepthStack s;
  for (int i = 0; i < 4000; ++i) s.push_back(-1);
  for (int i = 0; i < 4000; ++i) s.pop_back();
  EXPECT_EQ(s.count, 0);
  // Reuse after clear() at depth: spill storage must not leak values.
  for (int i = 0; i < 600; ++i) s.push_back(i);
  s.clear();
  EXPECT_EQ(s.count, 0);
  s.push_back(7);
  s.push_back(8);
  EXPECT_EQ(s.back(), 8);
}

TEST(CopyPolicyStrided, MatchesContiguousAndKeepsNeighbours) {
  for (int count : {1, 32, 218, 255}) {
    Node node(nullptr, 0);
    node.CreateEdges(MoveList(count));
    {
      int idx = 0;
      for (auto& edge : node.Edges()) {
        edge.edge()->SetP(1.0f / (idx + 2));
        if (++idx >= count) break;
      }
    }
    CachedNodeData cache;
    for (auto& item : cache.children) item.utility = 123.0f;
    std::array<float, 256> plain{};
    node.CopyPolicy(count, plain.data());
    node.CopyPolicy(count, &cache.children[0].policy, sizeof(ChildCache));
    for (int i = 0; i < 256; ++i) {
      EXPECT_EQ(cache.children[i].policy, plain[i]) << "count=" << count << " i=" << i;
      EXPECT_EQ(cache.children[i].utility, 123.0f) << "neighbour clobbered at i=" << i;
    }
  }
}

// CachedNodeData is deliberately NOT zero-initialized; this guards the
// documentation of that contract: fresh scalars must have their declared
// defaults (those DO have NSDMIs), and the struct must remain cheap to
// construct by value (the old cache{} form memset ~7-9KB per gather task).
TEST(CachedNodeData, ScalarsDefaultStackStartsEmpty) {
  CachedNodeData cache;
  EXPECT_EQ(cache.cache_filled_idx, -1);
  EXPECT_EQ(cache.max_policy_entries_needed, 0);
  EXPECT_EQ(cache.puct_mult, 0.0f);
  EXPECT_EQ(cache.vtp_last_filled_cache.count, 0);
}

}  // namespace
}  // namespace classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
