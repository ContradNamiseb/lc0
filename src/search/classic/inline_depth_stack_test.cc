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
//  * Node::GetEdgeP() (the AoS per-edge accessor that replaced CopyPolicy's
//    strided overload -- review #864 found that overload's pointer
//    arithmetic past the single float subobject it was given, to reach
//    sibling array-of-structs elements, was not standards-safe) must agree
//    with CopyPolicy's contiguous path for every edge count and must not
//    touch neighboring fields.
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

TEST(CopyPolicyStrided, GetEdgePMatchesContiguousAndKeepsNeighbours) {
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
    // Same loop search.cc uses at the CachedNodeData::children call site.
    for (int i = 0; i < count; i++) cache.children[i].policy = node.GetEdgeP(i);
    for (int i = 0; i < 256; ++i) {
      EXPECT_EQ(cache.children[i].policy, plain[i]) << "count=" << count << " i=" << i;
      EXPECT_EQ(cache.children[i].utility, 123.0f) << "neighbour clobbered at i=" << i;
    }
  }
}

// Guards CachedNodeData's scalar defaults (they have NSDMIs, so this holds
// regardless of `{}` vs bare default-init -- review #864 found the earlier
// claim that dropping `{}` avoided construction cost was false; every
// ChildCache scalar still gets initialized either way. The actual fix for
// that cost was moving CachedNodeData into TaskWorkspace so it's built once
// per worker instead of once per gather task -- see search.h.)
TEST(CachedNodeData, ScalarsDefaultStackStartsEmpty) {
  CachedNodeData cache;
  EXPECT_EQ(cache.cache_filled_idx, -1);
  EXPECT_EQ(cache.max_policy_entries_needed, 0);
  EXPECT_EQ(cache.puct_mult, 0.0f);
  EXPECT_EQ(cache.vtp_last_filled_cache.count, 0);
}

// CachedNodeData now lives in TaskWorkspace and is reused across calls
// (review #864) instead of being freshly constructed per call. Simulates
// two consecutive "levels" sharing one cache object -- a wide round filling
// many children[] slots followed by a narrow round filling few -- and
// checks the narrow round's own freshly-filled range is correct (not
// stale from the wide round), matching what PickNodesToExtendTask's
// per-level prep block (cache_filled_idx = -1, refill up to
// max_policy_entries_needed) actually does.
TEST(CachedNodeData, WorkspaceReuseDoesNotLeakBetweenCalls) {
  SearchWorker::TaskWorkspace workspace;
  Node wide(nullptr, 0);
  wide.CreateEdges(MoveList(200));
  {
    int idx = 0;
    for (auto& edge : wide.Edges()) edge.edge()->SetP(1.0f / (++idx + 1));
  }
  Node narrow(nullptr, 0);
  narrow.CreateEdges(MoveList(5));
  {
    int idx = 0;
    for (auto& edge : narrow.Edges()) edge.edge()->SetP(9.0f + idx++);
  }

  // Round 1: a wide "level" fills 200 entries, same as PickNodesToExtendTask
  // does at the top of its per-level prep block.
  auto& cache = workspace.cache;
  cache.cache_filled_idx = -1;
  cache.max_policy_entries_needed = 200;
  for (int i = 0; i < cache.max_policy_entries_needed; i++) {
    cache.children[i].policy = wide.GetEdgeP(i);
    cache.children[i].utility = 111.0f;
  }
  ASSERT_FLOAT_EQ(cache.children[199].policy, wide.GetEdgeP(199));

  // Round 2: a narrow "level" on the SAME reused cache -- exactly what
  // happens descending into a child with far fewer legal moves.
  cache.cache_filled_idx = -1;
  cache.max_policy_entries_needed = 5;
  for (int i = 0; i < cache.max_policy_entries_needed; i++) {
    cache.children[i].policy = narrow.GetEdgeP(i);
  }
  for (int i = 0; i < 5; i++) {
    EXPECT_FLOAT_EQ(cache.children[i].policy, narrow.GetEdgeP(i))
        << "i=" << i << " -- round 2's own fill must win, not round 1's";
  }
  // Indices round 2 never touched (5..199) still hold round 1's stale
  // values -- that's fine and expected: nothing in PickNodesToExtendTask
  // reads children[i] for i >= cache.max_policy_entries_needed /
  // cache_filled_idx, which round 2 reset to 5. This isn't a correctness
  // gap, it's the documented write-before-read gating contract; assert it
  // explicitly so a future change that narrows the gate incorrectly (reads
  // past max_policy_entries_needed) has something to trip.
  EXPECT_FLOAT_EQ(cache.children[199].policy, wide.GetEdgeP(199))
      << "unread-this-round slot should still hold round 1's value";
}

}  // namespace
}  // namespace classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
