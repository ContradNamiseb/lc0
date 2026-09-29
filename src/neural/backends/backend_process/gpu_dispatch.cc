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

#include "neural/backends/backend_process/gpu_dispatch.h"

#include <algorithm>

namespace lczero {
namespace backend_process {
namespace {

// Below this variance of the batch sizes, relative to their mean square, the
// sizes are too alike to tell the overhead from the time per position.
constexpr double kMinSpread = 0.01;
// Keeps a GPU's time growing with the batch size, so that batches still
// spread over GPUs whose time does not.
constexpr double kMinPerPosition = 1e-9;

}  // namespace

void CostModel::Record(size_t batch_size, double seconds) {
  if (batch_size == 0) return;
  const double size = batch_size;
  seconds = std::max(seconds, 0.0);
  weight_ = weight_ * kDecay + 1;
  sum_size_ = sum_size_ * kDecay + size;
  sum_size_squared_ = sum_size_squared_ * kDecay + size * size;
  sum_seconds_ = sum_seconds_ * kDecay + seconds;
  sum_size_seconds_ = sum_size_seconds_ * kDecay + size * seconds;

  // The weight times the variance of the sizes.
  const double spread = sum_size_squared_ - sum_size_ * sum_size_ / weight_;
  if (spread > kMinSpread * sum_size_squared_) {
    const double slope =
        (sum_size_seconds_ - sum_size_ * sum_seconds_ / weight_) / spread;
    const double intercept = (sum_seconds_ - slope * sum_size_) / weight_;
    if (slope > kMinPerPosition && intercept >= 0) {
      overhead_ = intercept;
      per_position_ = slope;
    } else if (slope > kMinPerPosition) {
      // No overhead: the line through zero.
      overhead_ = 0;
      per_position_ = sum_size_seconds_ / sum_size_squared_;
    } else {
      // The time does not grow with the batch size.
      overhead_ = sum_seconds_ / weight_;
      per_position_ = kMinPerPosition;
    }
    return;
  }
  // Keeps the overhead and fits the time per position, as the batches have
  // shown no more than the sum of the two.
  overhead_ = std::min(overhead_, sum_seconds_ / weight_);
  per_position_ = std::max((sum_seconds_ - overhead_ * weight_) / sum_size_,
                           kMinPerPosition);
}

double GpuQueue::FreeAt(double now) const {
  if (pending_batches_ == 0 || now <= free_at_) return std::max(free_at_, now);
  return now + (now - free_at_);
}

double GpuQueue::Dispatch(size_t batch_size, double now) {
  const double predicted = model_.Predict(batch_size);
  // From FreeAt(), so that a late GPU stays late for its next planners.
  free_at_ = FreeAt(now) + predicted;
  pending_seconds_ += predicted;
  ++pending_batches_;
  return predicted;
}

void GpuQueue::Complete(size_t batch_size, double predicted, double sent_at,
                        double now, bool learn) {
  // The batch started when it was sent or when the GPU finished the one
  // before it, whichever came later.
  const double started = std::max(sent_at, last_finish_);
  last_finish_ = std::max(last_finish_, now);
  if (learn) model_.Record(batch_size, now - started);
  if (pending_batches_ > 0) --pending_batches_;
  pending_seconds_ =
      pending_batches_ == 0 ? 0 : std::max(pending_seconds_ - predicted, 0.0);
  free_at_ = now + pending_seconds_;
}

bool PlanBatch(size_t batch_size, size_t step, double now,
               std::span<const GpuLoad> gpus, std::span<size_t> sizes) {
  std::fill(sizes.begin(), sizes.end(), 0);
  step = std::max<size_t>(step, 1);
  for (size_t placed = 0; placed < batch_size;) {
    size_t chunk = std::min(step, batch_size - placed);
    size_t best = gpus.size();
    double best_finish = 0;
    for (size_t i = 0; i < gpus.size(); ++i) {
      const size_t size = sizes[i] + chunk;
      if (size > gpus[i].maximum_batch_size) continue;
      // A GPU the batch does not use yet also has to save its overhead.
      const double finish = std::max(gpus[i].free_at, now) +
                            gpus[i].model->Predict(size) +
                            (sizes[i] == 0 ? gpus[i].model->Overhead() : 0);
      if (best == gpus.size() || finish < best_finish) {
        best = i;
        best_finish = finish;
      }
    }
    if (best == gpus.size()) {
      // No GPU takes a whole chunk: the one with the most room takes what
      // fits.
      size_t room = 0;
      for (size_t i = 0; i < gpus.size(); ++i) {
        const size_t spare = gpus[i].maximum_batch_size - sizes[i];
        if (spare > room) {
          room = spare;
          best = i;
        }
      }
      if (room == 0) return false;
      chunk = std::min(chunk, room);
    }
    sizes[best] += chunk;
    placed += chunk;
  }
  return true;
}

}  // namespace backend_process
}  // namespace lczero
