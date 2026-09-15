/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024-2025 The LCZero Authors

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
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "utils/logging.h"
#include "utils/spinhelper.h"

namespace lczero {

// A work-stealing threadpool where each worker owns a local deque.
// Workers push new tasks to their own deque and steal from others when idle.
//
// Template parameter Task must be movable and default-constructible.
// The executor callback receives (Task&, int tid) where tid identifies
// the worker thread (for workspace selection).
template <typename Task>
class TaskStealingPool {
 public:
  using Executor = std::function<void(Task&, int)>;

  // Creates a pool with `num_workers` threads. The `executor` callback
  // is invoked for each task, receiving the task and the worker's thread id.
  TaskStealingPool(int num_workers, Executor executor)
      : num_workers_(num_workers),
        executor_(std::move(executor)),
        workers_(num_workers) {
    for (int i = 0; i < num_workers_; i++) {
      threads_.emplace_back([this, i]() { WorkerLoop(i); });
    }
  }

  ~TaskStealingPool() { Shutdown(); }

  // Non-copyable, non-movable.
  TaskStealingPool(const TaskStealingPool&) = delete;
  TaskStealingPool& operator=(const TaskStealingPool&) = delete;

  // Submit a single task. Distributes to the least-loaded worker.
  // Publish protocol (review #858/#859): active_tasks_ is incremented BEFORE
  // the task becomes visible in any queue, and under wait_mutex_ so a worker
  // evaluating its wait predicate can never miss the increment. Overcounting
  // for a few instructions is harmless; undercounting released WaitForAll
  // early while tree mutation continued outside nodes_mutex_.
  void Submit(Task&& task) {
    int target = LeastLoaded();
    {
      std::lock_guard<std::mutex> lock(wait_mutex_);
      active_tasks_.fetch_add(1, std::memory_order_release);
    }
    {
      std::lock_guard<std::mutex> lock(workers_[target].queue_mutex);
      workers_[target].local_queue.push_back(std::move(task));
    }
    workers_[target].queue_size.fetch_add(1, std::memory_order_release);
    work_available_.notify_one();
  }

  // Submit a batch of tasks, distributed round-robin across workers.
  void Submit(std::vector<Task>&& tasks) {
    if (tasks.empty()) return;
    int n = static_cast<int>(tasks.size());
    {
      std::lock_guard<std::mutex> lock(wait_mutex_);
      active_tasks_.fetch_add(n, std::memory_order_release);
    }
    for (int i = 0; i < n; i++) {
      int target = i % num_workers_;
      {
        std::lock_guard<std::mutex> lock(workers_[target].queue_mutex);
        workers_[target].local_queue.push_back(std::move(tasks[i]));
      }
      workers_[target].queue_size.fetch_add(1, std::memory_order_release);
    }
    work_available_.notify_all();  // batch may need several workers.
  }

  // Block until all submitted tasks complete.
  // Uses condition variable instead of busy-spin.
  void WaitForAll() {
    std::unique_lock<std::mutex> lock(done_mutex_);
    all_done_.wait(lock, [this]() {
      return active_tasks_.load(std::memory_order_acquire) == 0;
    });
  }

  // Drain all completed tasks (moves them out). Call after WaitForAll().
  std::vector<Task> DrainCompleted() {
    std::lock_guard<std::mutex> lock(completed_mutex_);
    std::vector<Task> result = std::move(completed_tasks_);
    completed_tasks_.clear();
    return result;
  }

  // Reset counters for a new round of work.
  void Reset() {
    // All tasks should be complete before calling Reset. A nonzero
    // active_tasks_ here means an earlier WaitForAll() returned while work
    // was in flight (that was the #858 publish race) -- restore the counter
    // rather than poison every later round with a phantom that hangs or
    // instant-releases forever, and say so loudly.
    const int leftover = active_tasks_.exchange(0, std::memory_order_acq_rel);
    if (leftover != 0) {
      CERR << "TaskStealingPool::Reset with " << leftover
           << " tasks still counted active: an earlier WaitForAll() returned"
              " early. Counter restored; investigate the submit protocol.";
    }
    completed_.store(0, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(completed_mutex_);
      completed_tasks_.clear();
    }
  }

  // Get the number of completed tasks since last Reset.
  int CompletedCount() const {
    return completed_.load(std::memory_order_acquire);
  }

  // Signal shutdown and join all threads.
  void Shutdown() {
    if (shutdown_.load(std::memory_order_acquire)) return;
    // Same publish protocol as Submit: state under wait_mutex_, then notify.
    {
      std::lock_guard<std::mutex> lock(wait_mutex_);
      shutdown_.store(true, std::memory_order_release);
    }
    work_available_.notify_all();
    for (auto& t : threads_) {
      if (t.joinable()) t.join();
    }
    threads_.clear();
  }

 private:
  // Per-worker state, cache-line aligned to prevent false sharing.
  struct alignas(64) WorkerState {
    std::deque<Task> local_queue;
    std::mutex queue_mutex;
    std::atomic<int> queue_size{0};
  };

  // Find the worker with the fewest queued tasks.
  int LeastLoaded() const {
    int best = 0;
    int best_size = workers_[0].queue_size.load(std::memory_order_relaxed);
    for (int i = 1; i < num_workers_; i++) {
      int s = workers_[i].queue_size.load(std::memory_order_relaxed);
      if (s < best_size) {
        best_size = s;
        best = i;
      }
    }
    return best;
  }

  // Try to pop a task from the worker's own deque (LIFO for locality).
  bool TryPopOwn(int tid, Task& out) {
    if (workers_[tid].queue_size.load(std::memory_order_acquire) == 0) {
      return false;
    }
    std::lock_guard<std::mutex> lock(workers_[tid].queue_mutex);
    if (workers_[tid].local_queue.empty()) return false;
    out = std::move(workers_[tid].local_queue.back());
    workers_[tid].local_queue.pop_back();
    workers_[tid].queue_size.fetch_sub(1, std::memory_order_release);
    return true;
  }

  // Try to steal a task from another worker's deque (FIFO for breadth).
  bool TrySteal(int thief_tid, Task& out) {
    // Start from a cheap per-thread pseudo-random offset to avoid a thundering
    // herd on worker 0. (mt19937 + a distribution object per attempt showed up
    // as real cost in the spin loops -- review #859.)
    thread_local uint32_t rng = 2654435761u * static_cast<uint32_t>(
                                     reinterpret_cast<uintptr_t>(&rng)) +
                                 1013904223u;
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    int start = static_cast<int>(rng % static_cast<unsigned>(num_workers_));

    for (int attempt = 0; attempt < num_workers_; attempt++) {
      int victim = (start + attempt) % num_workers_;
      if (victim == thief_tid) continue;
      if (workers_[victim].queue_size.load(std::memory_order_acquire) == 0) {
        continue;
      }
      std::lock_guard<std::mutex> lock(workers_[victim].queue_mutex);
      if (workers_[victim].local_queue.empty()) continue;
      // Steal from front (FIFO) to get broad/large subtrees.
      out = std::move(workers_[victim].local_queue.front());
      workers_[victim].local_queue.pop_front();
      workers_[victim].queue_size.fetch_sub(1, std::memory_order_release);
      return true;
    }
    return false;
  }

  void WorkerLoop(int tid) {
    while (true) {
      Task task;
      bool got_task = false;

      // 1. Try own deque first (LIFO).
      got_task = TryPopOwn(tid, task);

      // 2. If empty, try stealing from others.
      if (!got_task) {
        got_task = TrySteal(tid, task);
      }

      // 3. If still no work, back off.
      if (!got_task) {
        if (shutdown_.load(std::memory_order_acquire)) return;

        // Short spin phase (128 iterations).
        bool found_during_spin = false;
        for (int spin = 0; spin < 128; spin++) {
          SpinloopPause();
          if (TryPopOwn(tid, task) || TrySteal(tid, task)) {
            found_during_spin = true;
            got_task = true;
            break;
          }
          if (shutdown_.load(std::memory_order_acquire)) return;
        }

        if (!found_during_spin) {
          // Yield phase.
          std::this_thread::yield();
          if (TryPopOwn(tid, task) || TrySteal(tid, task)) {
            got_task = true;
          } else {
            // Sleep until new work arrives or shutdown. This is a REAL sleep
            // now, not a 100us poll: Submit/Shutdown publish state under
            // wait_mutex_ before notifying, so the predicate can never be
            // missed (the old timed wait existed only to recover from that
            // lost-wakeup race -- review #858 finding 8).
            std::unique_lock<std::mutex> lock(wait_mutex_);
            work_available_.wait(lock, [this]() {
              return shutdown_.load(std::memory_order_acquire) ||
                     active_tasks_.load(std::memory_order_acquire) > 0;
            });
            if (shutdown_.load(std::memory_order_acquire) &&
                active_tasks_.load(std::memory_order_acquire) == 0) {
              return;
            }
            continue;  // Restart the loop to try popping/stealing.
          }
        }
      }

      if (got_task) {
        // Execute the task. An executor exception must not strand the
        // active_tasks_ count (that would hang the next WaitForAll under the
        // caller's lock) nor kill the worker thread (std::terminate) --
        // bookkeeping runs either way; the task's partial results still reach
        // completed_tasks_ so the caller's undo/cancel logic sees them.
        try {
          executor_(task, tid);
        } catch (const std::exception& e) {
          CERR << "TaskStealingPool: executor threw: " << e.what();
        } catch (...) {
          CERR << "TaskStealingPool: executor threw a non-standard exception";
        }
        {
          std::lock_guard<std::mutex> lock(completed_mutex_);
          completed_tasks_.push_back(std::move(task));
        }
        completed_.fetch_add(1, std::memory_order_release);
        int remaining = active_tasks_.fetch_sub(1, std::memory_order_acq_rel);
        if (remaining == 1) {
          // This was the last active task.
          std::lock_guard<std::mutex> lock(done_mutex_);
          all_done_.notify_all();
        }
      }
    }
  }

  int num_workers_;
  Executor executor_;
  std::vector<WorkerState> workers_;
  std::vector<std::thread> threads_;

  std::atomic<int> active_tasks_{0};
  std::atomic<int> completed_{0};
  std::atomic<bool> shutdown_{false};

  // Condvar for workers waiting for new work.
  std::mutex wait_mutex_;
  std::condition_variable work_available_;

  // Condvar for main thread waiting for all tasks to complete.
  std::mutex done_mutex_;
  std::condition_variable all_done_;

  // Storage for completed tasks (for DrainCompleted()).
  std::mutex completed_mutex_;
  std::vector<Task> completed_tasks_;
};

}  // namespace lczero
