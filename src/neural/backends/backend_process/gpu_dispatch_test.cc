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

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace lczero {
namespace backend_process {
namespace {

TEST(CostModel, FitsOverheadAndTimePerPosition) {
  CostModel model(0, 1);
  for (size_t size : {8, 64, 256, 32, 512, 128}) {
    model.Record(size, 0.002 + 1e-5 * size);
  }
  EXPECT_NEAR(model.Overhead(), 0.002, 1e-9);
  EXPECT_NEAR(model.PerPosition(), 1e-5, 1e-12);
}

TEST(CostModel, FollowsAGpuThatSlowsDown) {
  CostModel model(0, 1);
  const std::vector<size_t> sizes = {8, 64, 256, 32, 512, 128};
  for (size_t size : sizes) model.Record(size, 0.002 + 1e-5 * size);
  for (int i = 0; i < 5; ++i) {
    for (size_t size : sizes) model.Record(size, 0.002 + 3e-5 * size);
  }
  // Within 2%, as the old times still weigh 0.9^30 each.
  const double time = 0.002 + 3e-5 * 256;
  EXPECT_NEAR(model.Predict(256), time, 0.02 * time);
}

TEST(CostModel, KeepsTheOverheadForBatchesOfOneSize) {
  CostModel model(0.001, 1e-4);
  for (int i = 0; i < 10; ++i) model.Record(64, 0.005);
  EXPECT_DOUBLE_EQ(model.Overhead(), 0.001);
  EXPECT_NEAR(model.Predict(64), 0.005, 1e-12);
}

TEST(CostModel, TimeThatDoesNotGrowIsAllOverhead) {
  CostModel model(0, 1e-4);
  for (size_t size : {8, 64, 256, 32}) model.Record(size, 0.02);
  EXPECT_NEAR(model.Overhead(), 0.02, 1e-9);
  EXPECT_LT(model.PerPosition(), 1e-8);
}

std::vector<size_t> Plan(size_t batch_size, size_t step,
                         const std::vector<GpuLoad>& gpus) {
  std::vector<size_t> sizes(gpus.size());
  EXPECT_TRUE(PlanBatch(batch_size, step, 0, gpus, sizes));
  return sizes;
}

TEST(PlanBatch, SplitsEvenlyOverIdleEqualGpus) {
  const CostModel model(0.001, 1e-4);
  const std::vector<GpuLoad> gpus(4, {0, 1024, &model});
  EXPECT_EQ(Plan(256, 32, gpus), std::vector<size_t>(4, 64));
}

TEST(PlanBatch, GivesABusyGpuWhatItHasSpare) {
  const CostModel model(0, 1e-5);
  // GPU 0 has 1 ms of work left: 100 positions' worth.
  const std::vector<GpuLoad> gpus = {{0.001, 1024, &model}, {0, 1024, &model}};
  EXPECT_EQ(Plan(256, 1, gpus), (std::vector<size_t>{78, 178}));
}

TEST(PlanBatch, GivesAFasterGpuMore) {
  const CostModel fast(0, 1e-5);
  const CostModel slow(0, 2e-5);
  const std::vector<GpuLoad> gpus = {{0, 1024, &slow}, {0, 1024, &fast}};
  EXPECT_EQ(Plan(300, 1, gpus), (std::vector<size_t>{100, 200}));
}

TEST(PlanBatch, KeepsASmallBatchOnOneGpu) {
  const CostModel model(0.005, 1e-6);
  const std::vector<GpuLoad> gpus(4, {0, 1024, &model});
  EXPECT_EQ(Plan(8, 1, gpus), (std::vector<size_t>{8, 0, 0, 0}));
}

TEST(PlanBatch, KeepsToEachGpusMaximumBatchSize) {
  const CostModel fast(0, 1e-6);
  const CostModel slow(0, 1e-3);
  const std::vector<GpuLoad> gpus = {{0, 100, &fast}, {0, 1024, &slow}};
  // In whole chunks of 32 while one fits.
  EXPECT_EQ(Plan(250, 32, gpus), (std::vector<size_t>{96, 154}));
  std::vector<size_t> sizes(2);
  EXPECT_FALSE(PlanBatch(1200, 32, 0, gpus, sizes));
}

TEST(GpuQueue, LateGpuLooksBusierTheLaterItIs) {
  GpuQueue queue(CostModel(0, 1e-4));
  EXPECT_NEAR(queue.FreeAt(1), 1, 1e-12);
  queue.Dispatch(100, 1);
  EXPECT_NEAR(queue.FreeAt(1), 1.01, 1e-12);
  // 40 ms late.
  EXPECT_NEAR(queue.FreeAt(1.05), 1.09, 1e-12);
  queue.Complete(100, 0.01, 1, 1.06, false);
  EXPECT_NEAR(queue.FreeAt(1.07), 1.07, 1e-12);
}

// GPUs that run their batches one after another, overhead + per_position *
// size each, fed one batch at a time by a single search thread.
class Simulation {
 public:
  struct Gpu {
    double overhead;
    double per_position;
  };

  explicit Simulation(std::vector<Gpu> gpus)
      : gpus_(std::move(gpus)),
        busy_until_(gpus_.size()),
        sizes_(gpus_.size()) {
    for (size_t i = 0; i < gpus_.size(); ++i) {
      queues_.emplace_back(CostModel(1e-3, 1e-4));
    }
  }

  // Runs one batch, returning how long it took.
  double Run(size_t batch_size) {
    std::vector<GpuLoad> loads;
    for (const GpuQueue& queue : queues_) {
      loads.push_back({queue.FreeAt(now_), 1024, &queue.Model()});
    }
    EXPECT_TRUE(PlanBatch(batch_size, 8, now_, loads, sizes_));
    double finished = now_;
    for (size_t i = 0; i < gpus_.size(); ++i) {
      if (sizes_[i] == 0) continue;
      const double predicted = queues_[i].Dispatch(sizes_[i], now_);
      busy_until_[i] = std::max(busy_until_[i], now_) + gpus_[i].overhead +
                       gpus_[i].per_position * sizes_[i];
      queues_[i].Complete(sizes_[i], predicted, now_, busy_until_[i], true);
      finished = std::max(finished, busy_until_[i]);
    }
    const double took = finished - now_;
    now_ = finished;
    return took;
  }

  void SetGpu(size_t index, Gpu gpu) { gpus_[index] = gpu; }
  const std::vector<size_t>& Sizes() const { return sizes_; }

 private:
  std::vector<Gpu> gpus_;
  std::vector<GpuQueue> queues_;
  std::vector<double> busy_until_;
  std::vector<size_t> sizes_;
  double now_ = 0;
};

TEST(Simulation, LearnsUnevenGpusAndBalancesTheirParts) {
  Simulation simulation({{0.001, 2e-5}, {0.001, 4e-5}, {0.002, 2e-5}});
  const std::vector<size_t> batch_sizes = {64, 512, 128, 256, 384};
  for (int i = 0; i < 20; ++i) {
    for (size_t size : batch_sizes) simulation.Run(size);
  }
  // Parts that finish together take (512 + sum overhead / per_position) /
  // sum(1 / per_position) = 5.5 ms, give or take a chunk; an even split
  // takes 7.8 ms.
  EXPECT_LT(simulation.Run(512), 0.0059);
}

TEST(Simulation, MovesWorkOffAGpuThatSlowsDown) {
  Simulation simulation({{0.001, 2e-5}, {0.001, 2e-5}});
  for (int i = 0; i < 20; ++i) {
    for (size_t size : {64, 512, 256}) simulation.Run(size);
  }
  simulation.Run(512);
  const size_t before = simulation.Sizes()[0];
  simulation.SetGpu(0, {0.001, 6e-5});
  for (int i = 0; i < 20; ++i) {
    for (size_t size : {64, 512, 256}) simulation.Run(size);
  }
  simulation.Run(512);
  EXPECT_NEAR(before, 256, 8);
  EXPECT_NEAR(simulation.Sizes()[0], 128, 8);
}

}  // namespace
}  // namespace backend_process
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
