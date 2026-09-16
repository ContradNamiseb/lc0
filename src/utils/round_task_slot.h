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

namespace lczero {

// Claims one of at most `max_tasks` slots from a shared round-wide counter.
// Multiple threads can call this concurrently while recursively submitting
// sub-tasks -- CAS rather than a lock since the only shared state is one
// int. Returns false once `max_tasks` slots are already claimed; the
// caller should fall back to handling that unit of work locally instead of
// submitting it to a pool.
inline bool TryClaimRoundTaskSlot(std::atomic<int>& counter, int max_tasks) {
  int expected = counter.load(std::memory_order_relaxed);
  while (expected < max_tasks) {
    if (counter.compare_exchange_weak(expected, expected + 1,
                                      std::memory_order_acq_rel)) {
      return true;
    }
  }
  return false;
}

}  // namespace lczero
