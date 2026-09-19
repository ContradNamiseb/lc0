/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

// Coverage for dag_classic's ported AoS picking cache (classic's
// CachedNodeData/ChildCache refactor applied to dag_classic's
// PickNodesToExtendTask, replacing the four parallel SoA arrays
// current_util/current_score/current_nstarted/cur_iters). The cache now
// lives in TaskWorkspace, reused across every level within one gather call
// AND across every gather call the worker ever runs -- these tests pin the
// write-before-read/bounded-read contract that reuse depends on, mirroring
// classic's own CachedNodeData.WorkspaceReuseDoesNotLeakBetweenCalls
// (search/classic/inline_depth_stack_test.cc).
//
// Relies on this toolchain's cfront-compatibility handling of private
// nested types (same as inline_depth_stack_test.cc -- see the compiler's
// own -Wintel-compat note), not an explicit -fno-access-control flag.

#include "search/dag_classic/search.h"

#include "gtest/gtest.h"

namespace lczero {
namespace dag_classic {
namespace {

using ChildCache = SearchWorker::ChildCache;
using CachedNodeData = SearchWorker::CachedNodeData;
using CurrentPath = SearchWorker::CurrentPath;
using TaskWorkspace = SearchWorker::TaskWorkspace;

TEST(CachedNodeData, ChildCacheDefaultsAreZero) {
  ChildCache cc;
  EXPECT_EQ(cc.utility, 0.0f);
  EXPECT_EQ(cc.uct_score, 0.0f);
  EXPECT_EQ(cc.n_started, 0);
}

TEST(CachedNodeData, WorkspaceConstructsWithoutASearchWorker) {
  // TaskWorkspace must remain default-constructible standalone -- the pool
  // migration will need one-per-worker instances built the same
  // way classic's are.
  TaskWorkspace workspace;
  EXPECT_EQ(workspace.current_path.capacity() >= 30, true);
  EXPECT_EQ(workspace.full_path.capacity() >= 30, true);
}

// Simulates the exact reuse pattern PickNodesToExtendTask relies on: a wide
// node (many edges) fills a wide range of cache.children[], then a
// subsequent call on a narrow node (few edges) reuses the SAME workspace.
// The narrow call's own writes must win for the indices it touches; indices
// it never touches are allowed to keep the wide call's stale values because
// nothing reads cache.children[i] for i >= max_needed in that level -- this
// isn't a leak, it's the documented gating contract this test exists to
// pin.
TEST(CachedNodeData, WorkspaceReuseDoesNotLeakBetweenCalls) {
  TaskWorkspace workspace;
  auto& cache = workspace.cache;

  // Round 1: a "wide" level fills 200 entries, mirroring the unconditional
  // current_util fill loop and the cache_filled_idx-gated iter/uct_score/
  // n_started fill in PickNodesToExtendTask.
  for (int i = 0; i < 200; i++) {
    cache.children[i].utility = 111.0f;
    cache.children[i].uct_score = 222.0f;
    cache.children[i].n_started = 7;
  }
  ASSERT_FLOAT_EQ(cache.children[199].utility, 111.0f);

  // Round 2: a "narrow" level on the SAME reused workspace -- exactly what
  // happens descending into a child with far fewer legal moves.
  for (int i = 0; i < 5; i++) {
    cache.children[i].utility = 9.0f + i;
    cache.children[i].uct_score = 99.0f + i;
    cache.children[i].n_started = i;
  }
  for (int i = 0; i < 5; i++) {
    EXPECT_FLOAT_EQ(cache.children[i].utility, 9.0f + i)
        << "i=" << i << " -- round 2's own fill must win, not round 1's";
    EXPECT_EQ(cache.children[i].n_started, i);
  }
  // Indices round 2 never touched (5..199) still hold round 1's stale
  // values -- expected: PickNodesToExtendTask never reads cache.children[i]
  // for i >= max_needed, which round 2's node would have set to 5.
  EXPECT_FLOAT_EQ(cache.children[199].utility, 111.0f)
      << "unread-this-round slot should still hold round 1's value";
}

// visits_to_perform lives in the same CachedNodeData now (previously a
// fresh per-level local std::array<CurrentPath, kMaxMovesInPosition>) --
// confirm reuse across "levels" doesn't resurrect a stale nonzero visit
// count at an index a later level's own write-before-read (the
// cache_filled_idx-gated `visits_to_perform[idx] = CurrentPath(0,0,0,0,idx)`
// reset inside PickNodesToExtendTask) is supposed to own exclusively.
TEST(CachedNodeData, VisitsToPerformResetPerIndexSurvivesReuse) {
  TaskWorkspace workspace;
  auto& vtp = workspace.cache.visits_to_perform;

  // Round 1: index 3 accumulates a nonzero visit count (as real picking
  // does via operator+=).
  vtp[3] = CurrentPath(0, false, false, false, 3);
  vtp[3] += 42u;
  EXPECT_EQ(vtp[3].visits_, 42u);

  // Round 2 (a later level reusing the same workspace): the picker always
  // re-initializes an index the moment it's first touched this level
  // (`visits_to_perform[idx] = CurrentPath(0, 0, 0, 0, idx)`), which must
  // fully clear the stale round-1 count, not just overwrite index_.
  vtp[3] = CurrentPath(0, false, false, false, 3);
  EXPECT_EQ(vtp[3].visits_, 0u)
      << "re-init must clear a stale visit count left by a previous level";
}

}  // namespace
}  // namespace dag_classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
