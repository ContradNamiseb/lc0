/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

// Agora thread 19 DML-5/D2 (claude-opus, per codex-sol #724's ask for
// descriptor-level positive/negative tests of the ReinterpretView guard).
//
// This file supplies only main() -- the TEST_F cases themselves live inside
// layers.cc, guarded by LC0_DIRECTML_TESTS (see that file's comment above
// its gtest include), because GraphFactory<DataType> is defined entirely in
// layers.cc's anonymous namespace and has no external linkage: nothing
// outside that one translation unit can name it, so the tests have to be
// compiled as part of it. Same "standalone kernel-level test: links
// layers.cc only, no factory, no backend registration" pattern as
// test_kda_recurrence.cc.

#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
