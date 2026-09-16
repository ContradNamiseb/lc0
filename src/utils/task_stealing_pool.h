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
#include <exception>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>

#include "utils/logging.h"
#include "utils/spinhelper.h"
#include "utils/testonly_throw_hook.h"

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

  // Worker-thread publication paths move tasks with no recovery path on the
  // worker side (an exception there escapes into std::thread and calls
  // terminate), so the lifecycle contract is enforced at compile time
  // (review #881).
  static_assert(std::is_nothrow_move_constructible_v<Task>,
                "TaskStealingPool requires Task to be nothrow move "
                "constructible: queue pops and completion publication must "
                "not throw on worker threads");
  static_assert(std::is_nothrow_move_assignable_v<Task>,
                "TaskStealingPool requires Task to be nothrow move "
                "assignable: TryPopOwn/TrySteal assign into a worker-local "
                "Task");

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
  //
  // Exception safety (review #878 finding 4): if the queue insertion itself
  // throws (allocation failure), the task was never published, so the counts
  // raised for it are rolled back before the exception propagates -- a
  // phantom active_tasks_ entry would otherwise hang WaitForAll forever with
  // no task left to decrement it. Count-before-visibility is preserved:
  // workers only gate on queue_size before locking the queue, and that is
  // still raised after the counts, so a count-only transient can at most
  // wake a worker that finds nothing.
  //
  // Completion storage (review #881): the slot the worker will publish this
  // task into is reserved BEFORE the task becomes visible, so the
  // worker-side completion push_back can never allocate. A failure at that
  // reservation is still on the submitting thread and rolls the acceptance
  // back like any other pre-publication failure.
  void Submit(Task&& task) {
    int target = LeastLoaded();
    {
      std::lock_guard<std::mutex> lock(wait_mutex_);
      active_tasks_.fetch_add(1, std::memory_order_release);
      // Published under the same lock/before-notify protocol as
      // active_tasks_ above (review #866 P2): total_queued_ is what the
      // sleep predicate in WorkerLoop actually waits on, so an idle worker's
      // predicate check can never miss this task becoming available.
      total_queued_.fetch_add(1, std::memory_order_release);
    }
    bool slot_reserved = false;
    try {
      {
        std::lock_guard<std::mutex> lock(completed_mutex_);
        ReserveCompletionSlotsLocked(1);
        slot_reserved = true;
      }
      std::lock_guard<std::mutex> lock(workers_[target].queue_mutex);
      TestOnlyMaybeThrowAt(TestOnlyThrowSite::kPoolInsert);
      workers_[target].local_queue.push_back(std::move(task));
    } catch (...) {
      if (slot_reserved) {
        std::lock_guard<std::mutex> lock(completed_mutex_);
        DropCompletionSlotsLocked(1);
      }
      RollBackUnpublished(1);
      throw;
    }
    workers_[target].queue_size.fetch_add(1, std::memory_order_release);
    work_available_.notify_one();
  }

  // Submit a batch of tasks, distributed round-robin across workers.
  //
  // Failure ownership (review #878 finding 4): elements [0, published) are
  // ACCEPTED -- queued, counted, and they will execute (their Task objects
  // are moved-from in the caller's vector). The element whose insertion
  // throws and every element after it are REJECTED -- not queued, their
  // counts rolled back, and their Task objects are untouched, still owned by
  // the caller's vector after the exception. A partially published batch is
  // safe to unwind through: the accepted prefix is real work the caller must
  // still drain via WaitForAll()/DrainCompleted(), like any single Submit.
  void Submit(std::vector<Task>&& tasks) {
    if (tasks.empty()) return;
    int n = static_cast<int>(tasks.size());
    {
      std::lock_guard<std::mutex> lock(wait_mutex_);
      active_tasks_.fetch_add(n, std::memory_order_release);
      total_queued_.fetch_add(n, std::memory_order_release);
    }
    int published = 0;
    bool slots_reserved = false;
    try {
      {
        std::lock_guard<std::mutex> lock(completed_mutex_);
        ReserveCompletionSlotsLocked(static_cast<size_t>(n));
        slots_reserved = true;
      }
      for (int i = 0; i < n; i++) {
        int target = i % num_workers_;
        {
          std::lock_guard<std::mutex> lock(workers_[target].queue_mutex);
          TestOnlyMaybeThrowAt(TestOnlyThrowSite::kPoolInsert);
          workers_[target].local_queue.push_back(std::move(tasks[i]));
        }
        workers_[target].queue_size.fetch_add(1, std::memory_order_release);
        ++published;
      }
    } catch (...) {
      if (slots_reserved) {
        std::lock_guard<std::mutex> lock(completed_mutex_);
        DropCompletionSlotsLocked(static_cast<size_t>(n - published));
      }
      RollBackUnpublished(n - published);
      // The end-of-batch notify below is skipped on this path; wake workers
      // for the accepted prefix so it still executes.
      if (published > 0) work_available_.notify_all();
      throw;
    }
    work_available_.notify_all();  // batch may need several workers.
  }

  // Block until all submitted tasks complete. If any executor threw, the
  // first such exception is rethrown here on the caller's (owning search
  // thread's) stack after every task has still finished and reached
  // completed_tasks_ -- review #863 finding 4: the old behavior logged and
  // swallowed the exception, reporting a task that may have partially
  // mutated the tree (or recursively submitted children) as an ordinary
  // success. Rethrowing here lets it land in the same
  // catch(std::exception&)-driven cleanup path (CancelPendingMinibatch +
  // Stop) as a synchronous failure on the main gather thread.
  void WaitForAll() {
    {
      std::unique_lock<std::mutex> lock(done_mutex_);
      all_done_.wait(lock, [this]() {
        return active_tasks_.load(std::memory_order_acquire) == 0;
      });
    }
    std::exception_ptr to_throw;
    {
      std::lock_guard<std::mutex> lock(exception_mutex_);
      to_throw = first_exception_;
      first_exception_ = nullptr;
    }
    if (to_throw) std::rethrow_exception(to_throw);
  }

  // Drain all completed tasks (moves them out). Call after WaitForAll().
  //
  // The allocation this needs happens HERE, on the caller's thread, where a
  // failure is recoverable: if reserve() throws, completed_tasks_ is
  // untouched, so a retry still sees every accepted task's payload. Moving
  // the member vector out (as this used to) would drop its capacity and let
  // the next worker completion allocate again (review #881).
  std::vector<Task> DrainCompleted() {
    std::lock_guard<std::mutex> lock(completed_mutex_);
    std::vector<Task> result;
    result.reserve(completed_tasks_.size());
    for (auto& task : completed_tasks_) result.push_back(std::move(task));
    completed_tasks_.clear();  // keeps capacity for outstanding completions
    return result;
  }

  // Reset counters for a new round of work.
  void Reset() {
    // All tasks should be complete before calling Reset. A nonzero
    // active_tasks_ here means a round is genuinely still in flight (an
    // earlier WaitForAll() returned early -- that was the #858 publish
    // race -- or a caller simply skipped WaitForAll()). Forcing it to 0
    // with exchange(), as this used to do, doesn't restore anything: it
    // just discards the count while real tasks are still running, so a
    // live worker's later fetch_sub(1) on completion (WorkerLoop) underflows
    // the counter negative, corrupting every subsequent round's exit
    // condition (review #866 P2) -- worse than the hang it was trying to
    // avoid. Wait for the round to actually finish instead of lying about
    // it; this reuses WaitForAll()'s own exit condition so it returns
    // immediately in the (overwhelmingly common) already-idle case.
    int leftover = active_tasks_.load(std::memory_order_acquire);
    if (leftover != 0) {
      CERR << "TaskStealingPool::Reset called with " << leftover
           << " tasks still counted active: waiting for the round to finish"
              " for real instead of zeroing a live counter; investigate the"
              " submit/wait protocol.";
      std::unique_lock<std::mutex> lock(done_mutex_);
      all_done_.wait(lock, [this]() {
        return active_tasks_.load(std::memory_order_acquire) == 0;
      });
    }
    completed_.store(0, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(completed_mutex_);
      completed_tasks_.clear();
    }
    {
      // Defensive: WaitForAll() normally drains and clears this, but a
      // caller that skips WaitForAll() after Submit() (or bails out via its
      // own exception first) must not carry a stale exception into the
      // next round.
      std::lock_guard<std::mutex> lock(exception_mutex_);
      first_exception_ = nullptr;
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

  // Undo the counts for work that was counted but never queued (an insertion
  // failed). Waiters on active_tasks_ == 0 must not be left sleeping on a
  // phantom count that no worker will ever complete: if this brings the
  // count to zero with no task outstanding, wake them. Lock order here is
  // wait_mutex_ -> done_mutex_; no other path takes the two in the opposite
  // order (worker completions notify all_done_ under done_mutex_ alone).
  void RollBackUnpublished(int count) {
    if (count <= 0) return;
    std::lock_guard<std::mutex> wait_lock(wait_mutex_);
    total_queued_.fetch_sub(count, std::memory_order_release);
    if (active_tasks_.fetch_sub(count, std::memory_order_acq_rel) == count) {
      std::lock_guard<std::mutex> done_lock(done_mutex_);
      all_done_.notify_all();
    }
  }

  // Reserve completion storage for `n` accepted tasks. May throw; callers
  // invoke it BEFORE the tasks become visible, so a failure here is still on
  // the submitting thread and rolls the acceptance back instead of killing a
  // worker later. completed_mutex_ must be held.
  void ReserveCompletionSlotsLocked(size_t n) {
    TestOnlyMaybeThrowAt(TestOnlyThrowSite::kPoolCompletionReserve);
    completed_tasks_.reserve(completed_tasks_.size() +
                             outstanding_completions_ + n);
    outstanding_completions_ += n;
  }

  // Release slots for work that will never publish a completion (a rejected
  // or rolled-back acceptance). completed_mutex_ must be held.
  void DropCompletionSlotsLocked(size_t n) { outstanding_completions_ -= n; }

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
    total_queued_.fetch_sub(1, std::memory_order_relaxed);
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
      total_queued_.fetch_sub(1, std::memory_order_relaxed);
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
            //
            // The predicate is total_queued_ > 0, not active_tasks_ > 0
            // (review #866 P2): active_tasks_ stays positive for the entire
            // round, including while the round's last task is actually
            // executing and every queue is empty -- with that as the
            // predicate, every idle worker's wait() returns immediately
            // (predicate still true), fails to pop/steal, and goes straight
            // back through the spin+yield phases into another immediate-
            // return wait, i.e. a CPU busy-spin for as long as that one task
            // runs. total_queued_ tracks tasks actually sitting in a queue
            // (incremented alongside active_tasks_ in Submit before
            // publish, decremented by TryPopOwn/TrySteal on removal), so it
            // is zero exactly when there is nothing left to wake up for.
            std::unique_lock<std::mutex> lock(wait_mutex_);
            work_available_.wait(lock, [this]() {
              return shutdown_.load(std::memory_order_acquire) ||
                     total_queued_.load(std::memory_order_acquire) > 0;
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
        // completed_tasks_ so the caller's undo/cancel logic sees them. The
        // first exception seen this round is also stashed and rethrown by
        // WaitForAll() on the owning thread (review #863 finding 4) -- this
        // is not just a log-and-continue: the executor may have already
        // mutated shared tree state or recursively submitted children
        // before throwing, so the caller must not treat this as success.
        try {
          executor_(task, tid);
        } catch (const std::exception& e) {
          CERR << "TaskStealingPool: executor threw: " << e.what();
          std::lock_guard<std::mutex> elock(exception_mutex_);
          if (!first_exception_) first_exception_ = std::current_exception();
        } catch (...) {
          CERR << "TaskStealingPool: executor threw a non-standard exception";
          std::lock_guard<std::mutex> elock(exception_mutex_);
          if (!first_exception_) first_exception_ = std::current_exception();
        }
        {
          std::lock_guard<std::mutex> lock(completed_mutex_);
          // Cannot allocate: every accepted task had a completion slot
          // reserved before it was published (review #881), and Task is
          // nothrow-move-constructible (static_assert above). The payload
          // therefore always reaches DrainCompleted on the owning thread,
          // even after an executor throw.
          --outstanding_completions_;
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
  // Tasks currently sitting in some worker's queue, as opposed to
  // active_tasks_ (submitted but not yet drained into completed_tasks_,
  // which stays positive while the last task is actually executing with
  // nothing queued). This is what WorkerLoop's sleep predicate waits on --
  // see there (review #866 P2).
  std::atomic<int> total_queued_{0};
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
  // Accepted tasks whose completion slot is reserved but not yet filled;
  // guarded by completed_mutex_. Invariant (review #881):
  // completed_tasks_.capacity() >= completed_tasks_.size() +
  // outstanding_completions_, so the worker-side completion push_back never
  // allocates. Slots are taken in Submit before the task is published and
  // released by the worker when it stores the completed task; DrainCompleted
  // clears without shrinking, so the capacity survives drains.
  size_t outstanding_completions_ = 0;

  // First executor exception this round, rethrown by WaitForAll().
  std::mutex exception_mutex_;
  std::exception_ptr first_exception_;
};

}  // namespace lczero
