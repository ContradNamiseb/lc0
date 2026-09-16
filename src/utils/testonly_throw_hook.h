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

// Test-only fault-injection seam (agora #41/#872): a global, zero-cost-when-
// disabled countdown checked at points inside PickNodesToExtendTask (both
// classic and dag_classic) where a node reservation has just been made but
// not yet turned into a receiver entry or handed to a submitted task. Tests
// set the countdown, run a real gather, and assert n_in_flight_ is back to
// zero everywhere afterward -- proving the exception-safety rollback
// actually covers reservations still only in scratch state, not just ones
// already in the output receiver.
//
// Disabled (the default, -1) costs one relaxed atomic load per call site.
inline std::atomic<int64_t> g_testonly_throw_after_reservations{-1};

struct TestOnlyInjectedReservationThrow : public std::exception {
  const char* what() const noexcept override {
    return "TestOnlyInjectedReservationThrow";
  }
};

inline void TestOnlyMaybeThrowAfterReservation() {
  int64_t v =
      g_testonly_throw_after_reservations.load(std::memory_order_relaxed);
  if (v < 0) return;
  if (g_testonly_throw_after_reservations.fetch_sub(
          1, std::memory_order_relaxed) == 0) {
    throw TestOnlyInjectedReservationThrow{};
  }
}

}  // namespace lczero
