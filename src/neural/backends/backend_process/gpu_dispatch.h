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

// How the gpu-dispatch backend shares each batch out over the GPUs, each
// running in a backend process of its own. The ideas come from CPU schedulers:
// every GPU's speed is tracked as the kernel tracks a CPU's load, and each part
// of a batch goes to the GPU that would finish it first.

#pragma once

#include <cstddef>
#include <span>

namespace lczero {
namespace backend_process {

// How long a GPU takes for a batch: an overhead plus a time per position.
// Fitted by least squares to the batches the GPU has run, each weighing
// kDecay times the one after it, as the kernel's per-entity load tracking
// decays old CPU use, so a GPU that slows down, from heat or a display on it,
// shows it within a few batches.
class CostModel {
 public:
  static constexpr double kDecay = 0.9;

  CostModel(double overhead, double per_position)
      : overhead_(overhead), per_position_(per_position) {}

  double Predict(size_t batch_size) const {
    return overhead_ + per_position_ * batch_size;
  }
  double Overhead() const { return overhead_; }
  double PerPosition() const { return per_position_; }

  // Adds a batch of `batch_size` positions that took `seconds`.
  void Record(size_t batch_size, double seconds);

 private:
  // Decayed sums over the batches recorded: their weight, sizes, squared
  // sizes, times and sizes times times.
  double weight_ = 0;
  double sum_size_ = 0;
  double sum_size_squared_ = 0;
  double sum_seconds_ = 0;
  double sum_size_seconds_ = 0;
  double overhead_;
  double per_position_;
};

// The work sent to one GPU and when it should be done with it, kept as a CPU
// scheduler keeps the load of a run queue. The GPU is taken to run its
// batches one after another.
class GpuQueue {
 public:
  explicit GpuQueue(const CostModel& model) : model_(model) {}

  const CostModel& Model() const { return model_; }

  // When the GPU can start new work, as seen at `now`. A GPU that is late
  // with its work, as while its backend process restarts or if it hangs,
  // counts as busy for as long again as it is late, so new work goes
  // elsewhere.
  double FreeAt(double now) const;

  // Records that `batch_size` positions went to the GPU at `now`. Returns
  // the time they are predicted to take, which Complete() wants back.
  double Dispatch(size_t batch_size, double now);

  // Records that the positions sent at `sent_at` came back at `now`. Only a
  // batch run in the normal way teaches the cost model, not one that waited
  // for its backend process to restart.
  void Complete(size_t batch_size, double predicted, double sent_at, double now,
                bool learn);

 private:
  CostModel model_;
  double free_at_ = 0;
  double pending_seconds_ = 0;
  size_t pending_batches_ = 0;
  double last_finish_ = 0;
};

// A GPU as PlanBatch() sees it.
struct GpuLoad {
  // When the GPU is done with the work it has.
  double free_at;
  size_t maximum_batch_size;
  const CostModel* model;
};

// Shares a batch of `batch_size` positions out over the GPUs, setting
// sizes[i] to how many GPU i gets. The batch goes in chunks of `step`
// positions, each to the GPU that would finish it first given the work it
// already has, as an earliest-finish-time scheduler places a task, so the
// parts of a batch finish about together. A GPU the batch does not use yet
// also has to save its overhead, as a CPU scheduler moves a task only when
// that pays for the move: a small batch stays on one GPU and leaves the
// others to other batches. Returns false if the GPUs' maximum batch sizes are
// too small for the batch.
bool PlanBatch(size_t batch_size, size_t step, double now,
               std::span<const GpuLoad> gpus, std::span<size_t> sizes);

}  // namespace backend_process
}  // namespace lczero
