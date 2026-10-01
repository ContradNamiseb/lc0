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

// The engine side of the backend process. lc0 runs every backend it creates in
// a child process, so a crash in the backend or in the GPU driver costs a
// restart of that process instead of the engine. Batches cross in shared
// memory, see protocol.h.

#include "neural/backends/backend_process/child_process_backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "neural/backends/backend_process/interprocess.h"
#include "neural/backends/backend_process/protocol.h"
#include "neural/shared_params.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace backend_process {
namespace {

// Batches in flight at once; more search threads than this wait for a slot.
constexpr uint32_t kNumSlots = 16;
// No network backend reports a larger maximum batch size.
constexpr uint32_t kMaxBatch = 1024;
constexpr int kPollMilliseconds = 100;
constexpr int kStopWaitMilliseconds = 2000;
// Restarts while one batch is pending before it fails.
constexpr uint32_t kMaxRestarts = 2;
// A restarted backend process is stopped when it takes this many times as
// long to get ready as the first one did, and no less than the minimum.
constexpr int kRestartTimeFactor = 5;
constexpr std::chrono::seconds kMinimumRestartTime{60};

std::string Flag(const OptionId& id, const std::string& value) {
  return std::string("--") + id.long_flag() + "=" + value;
}

std::string_view Trim(std::string_view text) {
  const size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) return {};
  return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

std::string_view Unquote(std::string_view text) {
  if (text.size() >= 2 && (text.front() == '"' || text.front() == '\'') &&
      text.back() == text.front()) {
    return text.substr(1, text.size() - 2);
  }
  return text;
}

// Splits a backend options string at its top-level commas, so that
// "backend=cuda-fp16,(gpu=0),(gpu=1)" gives "backend=cuda-fp16", "(gpu=0)"
// and "(gpu=1)".
std::vector<std::string_view> SplitOptions(std::string_view text) {
  std::vector<std::string_view> entries;
  int depth = 0;
  char quote = 0;
  size_t start = 0;
  for (size_t i = 0; i <= text.size(); ++i) {
    const char c = i < text.size() ? text[i] : ',';
    if (quote) {
      if (c == quote) quote = 0;
    } else if (c == '"' || c == '\'') {
      quote = c;
    } else if (c == '(') {
      ++depth;
    } else if (c == ')') {
      --depth;
    } else if (c == ',' && depth == 0) {
      const std::string_view entry = Trim(text.substr(start, i - start));
      if (!entry.empty()) entries.push_back(entry);
      start = i + 1;
    }
  }
  return entries;
}

// One entry of a backend options string, read the way
// OptionsDict::AddSubdictFromString reads it: "key=value", or a group,
// "name(...)", "(...)" or "name".
struct OptionsEntry {
  std::string_view text;
  // The key, or the group's name.
  std::string_view key;
  // The value, or what the group's parentheses hold.
  std::string_view value;
  bool is_group;
};

OptionsEntry ParseEntry(std::string_view text) {
  size_t key_end = 0;
  if (text.front() == '"' || text.front() == '\'') {
    key_end = std::min(text.find(text.front(), 1), text.size() - 1) + 1;
  }
  const size_t separator = text.find_first_of("=(", key_end);
  OptionsEntry entry{text, Unquote(Trim(text.substr(0, separator))), {}, true};
  if (separator == std::string_view::npos) return entry;
  if (text[separator] == '=') {
    entry.value = Unquote(Trim(text.substr(separator + 1)));
    entry.is_group = false;
    return entry;
  }
  const size_t close = text.rfind(')');
  entry.value = text.substr(
      separator + 1, close > separator ? close - separator - 1 : text.npos);
  return entry;
}

struct DemuxGroup {
  std::string backend;
  std::string backend_options;
};

// The backend and options of each network that a demux backend configured
// with `backend_options` creates, as DemuxingNetwork works them out: a group
// reads the top-level options that it does not set itself, and runs the
// backend named by its "backend" option or else by its name. Empty for a
// configuration that leaves demux a single network or a group no backend.
std::vector<DemuxGroup> DemuxGroups(std::string_view backend_options) {
  std::vector<OptionsEntry> top_level;
  std::vector<OptionsEntry> groups;
  for (std::string_view text : SplitOptions(backend_options)) {
    const OptionsEntry entry = ParseEntry(text);
    (entry.is_group ? groups : top_level).push_back(entry);
  }
  std::vector<DemuxGroup> result;
  for (const OptionsEntry& group : groups) {
    std::vector<OptionsEntry> own;
    for (std::string_view text : SplitOptions(group.value)) {
      own.push_back(ParseEntry(text));
    }
    DemuxGroup& part = result.emplace_back();
    part.backend = group.key;
    auto add = [&part](const OptionsEntry& entry) {
      if (!entry.is_group && entry.key == "backend") {
        part.backend = entry.value;
        return;
      }
      // The number of demux's own threads for the group; the backend
      // process has its own, one per slot.
      if (!entry.is_group && entry.key == "threads") return;
      if (!part.backend_options.empty()) part.backend_options += ',';
      part.backend_options += entry.text;
    };
    for (const OptionsEntry& entry : top_level) {
      if (std::none_of(own.begin(), own.end(), [&](const OptionsEntry& o) {
            return !o.is_group && o.key == entry.key;
          })) {
        add(entry);
      }
    }
    for (const OptionsEntry& entry : own) add(entry);
    // demux fails on it, so let it.
    if (part.backend.empty()) return {};
  }
  if (result.size() < 2) return {};
  return result;
}

// The command line of each backend process that `options` ask for: one per
// group of a demux backend, so that each GPU has a process of its own, and
// one for any other backend.
std::vector<std::vector<std::string>> ProcessFlags(const OptionsDict& options) {
  std::vector<std::string> shared_flags;
  for (const OptionId* id :
       {&SharedBackendParams::kWeightsId, &SharedBackendParams::kHistoryFill}) {
    shared_flags.push_back(Flag(*id, options.Get<std::string>(*id)));
  }
  std::ostringstream temperature;
  temperature << std::setprecision(9)
              << options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp);
  shared_flags.push_back(
      Flag(SharedBackendParams::kPolicySoftmaxTemp, temperature.str()));

  const std::string backend =
      options.Get<std::string>(SharedBackendParams::kBackendId);
  const std::string backend_options =
      options.Get<std::string>(SharedBackendParams::kBackendOptionsId);
  std::vector<DemuxGroup> groups;
  if (backend == "demux") groups = DemuxGroups(backend_options);
  if (groups.empty()) groups.push_back({backend, backend_options});

  std::vector<std::vector<std::string>> flags;
  for (const DemuxGroup& group : groups) {
    std::vector<std::string>& process = flags.emplace_back(shared_flags);
    process.push_back(Flag(SharedBackendParams::kBackendId, group.backend));
    process.push_back(
        Flag(SharedBackendParams::kBackendOptionsId, group.backend_options));
  }
  return flags;
}

// Unique among the backends of all running lc0 processes.
std::string NewName() {
  static std::atomic<uint32_t> counter{0};
  return std::to_string(CurrentProcessId()) + "-" + std::to_string(counter++);
}

class ChildProcessBackend : public Backend {
 public:
  explicit ChildProcessBackend(std::vector<std::string> flags)
      : flags_(std::move(flags)),
        name_(NewName()),
        shared_memory_(
            SharedMemory::Create(name_, RegionSize(kNumSlots, kMaxBatch))),
        header_(static_cast<RegionHeader*>(shared_memory_.data())),
        state_changed_(NamedSemaphore::Create(StateName(name_))),
        slot_results_(kNumSlots, std::vector<EvalResultPtr>(kMaxBatch)) {
    header_->version = kVersion;
    header_->position_size = sizeof(Position);
    header_->num_slots = kNumSlots;
    header_->max_batch = kMaxBatch;
    header_->slot_stride = SlotStride(kMaxBatch);
    header_->magic = kMagic;
    for (uint32_t i = 0; i < kNumSlots; ++i) {
      requests_.push_back(NamedSemaphore::Create(RequestName(name_, i)));
      responses_.push_back(NamedSemaphore::Create(ResponseName(name_, i)));
      free_slots_.push_back(i);
    }
    {
      std::lock_guard lock(process_mutex_);
      StartProcess();
    }
    attributes_ = header_->attributes;
    attributes_.maximum_batch_size =
        std::min<int>(attributes_.maximum_batch_size, kMaxBatch);
    LOGFILE << "Backend process " << name_ << " started.";
  }

  ~ChildProcessBackend() override {
    header_->stop.store(1, std::memory_order_release);
    for (NamedSemaphore& request : requests_) request.Post();
    process_.Stop(kStopWaitMilliseconds);
  }

  BackendAttributes GetAttributes() const override { return attributes_; }

  std::unique_ptr<BackendComputation> CreateComputation() override;

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    Backend::UpdateConfiguration(options);
    // The backend process got these on its command line, so any change,
    // even to the softmax temperature, takes a new process.
    return ProcessFlags(options) ==
                   std::vector<std::vector<std::string>>{flags_}
               ? UPDATE_OK
               : NEED_RESTART;
  }

  // Blocks until a slot is free.
  uint32_t AcquireSlot() {
    std::unique_lock lock(slots_mutex_);
    // More batches at once than slots, e.g. from more than kNumSlots search
    // threads, would otherwise cap the parallelism without a word.
    if (free_slots_.empty() && !warned_all_slots_busy_) {
      warned_all_slots_busy_ = true;
      CERR << "All " << kNumSlots << " slots of the backend process are "
           << "busy; further batches wait for a free one.";
    }
    slot_freed_.wait(lock, [&] { return !free_slots_.empty(); });
    const uint32_t index = free_slots_.back();
    free_slots_.pop_back();
    return index;
  }

  void ReleaseSlot(uint32_t index) {
    {
      std::lock_guard lock(slots_mutex_);
      free_slots_.push_back(index);
    }
    slot_freed_.notify_one();
  }

  struct Request {
    uint64_t sequence;
    // The process starts before the request was sent.
    uint32_t first_start;
  };

  // Posts the slot's batch to the backend process.
  Request Send(uint32_t index) {
    SlotHeader* slot = Slot(index).header;
    // Only the slot's owner writes request_sequence, and slots change owners
    // under slots_mutex_, so a relaxed load sees the last request.
    const Request request{
        slot->request_sequence.load(std::memory_order_relaxed) + 1,
        starts_.load(std::memory_order_relaxed)};
    slot->failed = 0;
    slot->request_sequence.store(request.sequence, std::memory_order_release);
    requests_[index].Post();
    return request;
  }

  // Waits for the backend process to answer the request, restarting the
  // process if it dies meanwhile.
  void Receive(uint32_t index, const Request& request) {
    SlotHeader* slot = Slot(index).header;
    const uint64_t sequence = request.sequence;
    while (slot->response_sequence.load(std::memory_order_acquire) !=
           sequence) {
      if (responses_[index].Wait(kPollMilliseconds)) continue;
      std::lock_guard lock(process_mutex_);
      if (process_.IsRunning() ||
          slot->response_sequence.load(std::memory_order_acquire) == sequence) {
        continue;
      }
      // A batch that crashes every process it meets must not loop forever.
      // Marking it answered keeps the next process away from it.
      if (starts_.load(std::memory_order_relaxed) - request.first_start >=
          kMaxRestarts) {
        slot->response_sequence.store(sequence, std::memory_order_relaxed);
        throw Exception("The backend process keeps crashing");
      }
      CERR << "The backend process died, restarting it.";
      try {
        // The new process answers the pending batches, as their sequences
        // differ.
        StartProcess();
      } catch (...) {
        slot->response_sequence.store(sequence, std::memory_order_relaxed);
        throw;
      }
    }
    if (slot->failed) {
      throw Exception("The backend process failed to evaluate a batch");
    }
  }

  SlotView Slot(uint32_t index) const {
    return GetSlot(shared_memory_.data(), index);
  }

  // Kept across batches so a batch allocates nothing once warmed up. Only
  // the computation that owns the slot touches its entry.
  std::vector<EvalResultPtr>& SlotResults(uint32_t index) {
    return slot_results_[index];
  }

 private:
  // Starts the backend process and waits until it has loaded the network.
  // The first start may take any time, as nothing tells a slow load from one
  // that hangs. It gives the time limit of the later ones.
  // Requires process_mutex_.
  void StartProcess() {
    const auto start_time = std::chrono::steady_clock::now();
    header_->state.store(ProcessState::kStarting, std::memory_order_relaxed);
    header_->error[0] = '\0';
    std::vector<std::string> arguments = {
        ExecutablePath(), "backendprocess", "--name=" + name_,
        "--parent-process-id=" + std::to_string(CurrentProcessId())};
    arguments.insert(arguments.end(), flags_.begin(), flags_.end());
    process_ = ChildProcess::Spawn(arguments);
    starts_.fetch_add(1, std::memory_order_relaxed);
    while (true) {
      const ProcessState state = header_->state.load(std::memory_order_acquire);
      if (state == ProcessState::kReady) {
        if (!restart_time_limit_) {
          restart_time_limit_ = std::max<std::chrono::steady_clock::duration>(
              kMinimumRestartTime,
              kRestartTimeFactor *
                  (std::chrono::steady_clock::now() - start_time));
        }
        return;
      }
      if (state == ProcessState::kFailed) {
        throw Exception(std::string(
            header_->error, strnlen(header_->error, sizeof(header_->error))));
      }
      if (!process_.IsRunning()) {
        // It may have set the state just before it exited.
        if (header_->state.load(std::memory_order_acquire) !=
            ProcessState::kStarting) {
          continue;
        }
        throw Exception("The backend process exited while loading the network");
      }
      if (restart_time_limit_ && std::chrono::steady_clock::now() - start_time >
                                     *restart_time_limit_) {
        process_.Stop(0);
        throw Exception(
            "The backend process took too long to load the network");
      }
      // Woken as soon as the state is set. A post left over from an earlier
      // process only costs one more pass.
      state_changed_.Wait(10);
    }
  }

  const std::vector<std::string> flags_;
  const std::string name_;
  SharedMemory shared_memory_;
  RegionHeader* const header_;
  std::vector<NamedSemaphore> requests_;
  std::vector<NamedSemaphore> responses_;
  NamedSemaphore state_changed_;
  BackendAttributes attributes_;

  std::mutex process_mutex_;
  ChildProcess process_;
  // Set by the first start. Guarded by process_mutex_.
  std::optional<std::chrono::steady_clock::duration> restart_time_limit_;
  std::atomic<uint32_t> starts_{0};

  std::mutex slots_mutex_;
  std::condition_variable slot_freed_;
  std::vector<uint32_t> free_slots_;
  bool warned_all_slots_busy_ = false;
  std::vector<std::vector<EvalResultPtr>> slot_results_;
};

class ChildProcessComputation : public BackendComputation {
 public:
  explicit ChildProcessComputation(ChildProcessBackend* backend)
      : backend_(backend) {}

  // Also after Receive() threw: the slot is answered or its process is dead.
  ~ChildProcessComputation() override {
    if (slot_ >= 0) backend_->ReleaseSlot(slot_);
  }

  size_t UsedBatchSize() const override {
    return std::min<size_t>(size_.load(), kMaxBatch);
  }

  // The search's task workers call this concurrently, as they do
  // NetworkAsBackendComputation::AddInput.
  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    Write(size_.fetch_add(1), pos, result);
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    Send(UsedBatchSize());
    Receive();
  }

  // Puts the position at `index` of the batch; safe to call concurrently for
  // different indices.
  void Write(size_t index, const EvalPosition& pos, EvalResultPtr result) {
    if (pos.legal_moves.size() > kMaxLegalMoves) {
      throw Exception("Too many legal moves for the backend process");
    }
    if (index >= kMaxBatch) {
      throw Exception("Batch is larger than the backend process allows");
    }
    AcquireSlot();
    // Only the positions the encoder reads cross to the backend process.
    std::array<int, kCompactHistory> history;
    const int history_size = CompactHistoryForNN(
        backend_->GetAttributes().input_format, pos.pos, history);
    PositionRecord& record = backend_->Slot(slot_).positions[index];
    record.history_size = history_size;
    record.num_moves = pos.legal_moves.size();
    record.want_policy = !result.p.empty();
    for (int i = 0; i < history_size; ++i) {
      std::memcpy(record.history + i * sizeof(Position), &pos.pos[history[i]],
                  sizeof(Position));
    }
    std::memcpy(record.moves, pos.legal_moves.data(),
                pos.legal_moves.size() * sizeof(Move));
    results_[index] = result;
  }

  // Takes a slot of the backend process, unless this computation has one.
  void AcquireSlot() {
    std::call_once(slot_acquired_, [&] {
      slot_ = backend_->AcquireSlot();
      results_ = backend_->SlotResults(slot_).data();
    });
  }

  // Posts the first `batch_size` positions, which Write() has put.
  void Send(size_t batch_size) {
    batch_size_ = batch_size;
    if (batch_size == 0) return;
    backend_->Slot(slot_).header->batch_size = batch_size;
    request_ = backend_->Send(slot_);
  }

  // Waits for the batch that Send() posted and hands out its results.
  void Receive() {
    if (batch_size_ == 0) return;
    const SlotView slot = backend_->Slot(slot_);
    backend_->Receive(slot_, request_);
    for (size_t i = 0; i < batch_size_; ++i) {
      const ResultRecord& source = slot.results[i];
      const EvalResultPtr& destination = results_[i];
      if (destination.q) *destination.q = source.q;
      if (destination.d) *destination.d = source.d;
      if (destination.m) *destination.m = source.m;
      std::copy_n(source.p,
                  std::min<size_t>(destination.p.size(), kMaxLegalMoves),
                  destination.p.begin());
    }
  }

 private:
  ChildProcessBackend* const backend_;
  std::once_flag slot_acquired_;
  int slot_ = -1;
  EvalResultPtr* results_ = nullptr;
  std::atomic<size_t> size_ = 0;
  size_t batch_size_ = 0;
  ChildProcessBackend::Request request_{};
};

std::unique_ptr<BackendComputation> ChildProcessBackend::CreateComputation() {
  return std::make_unique<ChildProcessComputation>(this);
}

// Splits every batch across one backend process per demux group, much as
// DemuxingNetwork splits it across its networks, so that a crash on one GPU
// restarts only that GPU's process.
class DemuxingBackend : public Backend {
 public:
  explicit DemuxingBackend(std::vector<std::vector<std::string>> flags)
      : flags_(std::move(flags)), parts_(flags_.size()) {
    // The processes load their networks at the same time.
    std::vector<std::exception_ptr> errors(parts_.size());
    std::vector<std::thread> threads;
    for (size_t i = 0; i < parts_.size(); ++i) {
      threads.emplace_back([this, &errors, i] {
        try {
          parts_[i] = std::make_unique<ChildProcessBackend>(flags_[i]);
        } catch (...) {
          errors[i] = std::current_exception();
        }
      });
    }
    for (std::thread& thread : threads) thread.join();
    for (const std::exception_ptr& error : errors) {
      if (error) std::rethrow_exception(error);
    }

    attributes_ = parts_[0]->GetAttributes();
    int maximum_batch_size = attributes_.maximum_batch_size;
    for (const auto& part : parts_) {
      const BackendAttributes attributes = part->GetAttributes();
      attributes_.runs_on_cpu &= attributes.runs_on_cpu;
      attributes_.recommended_batch_size =
          std::min(attributes_.recommended_batch_size,
                   attributes.recommended_batch_size);
      maximum_batch_size =
          std::min(maximum_batch_size, attributes.maximum_batch_size);
      attributes_.preferred_batch_step = std::max(
          attributes_.preferred_batch_step, attributes.preferred_batch_step);
    }
    const int step = attributes_.preferred_batch_step =
        std::max(attributes_.preferred_batch_step, 1);
    attributes_.recommended_batch_size *= parts_.size();
    // DemuxingNetwork leaves Network::GetThreads() at its default.
    attributes_.suggested_num_search_threads = 1;
    // No part gets more whole steps than a batch this large gives it.
    attributes_.maximum_batch_size =
        parts_.size() * std::max(maximum_batch_size / step, 1) * step;
  }

  BackendAttributes GetAttributes() const override { return attributes_; }

  std::unique_ptr<BackendComputation> CreateComputation() override;

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    Backend::UpdateConfiguration(options);
    return ProcessFlags(options) == flags_ ? UPDATE_OK : NEED_RESTART;
  }

  ChildProcessBackend* Part(size_t index) const { return parts_[index].get(); }
  size_t NumParts() const { return parts_.size(); }
  // Which part gets a batch's first chunk; it moves on with every batch, so
  // that the extra chunks of uneven batches spread over the parts.
  size_t NextStart() { return start_.fetch_add(1) % parts_.size(); }

 private:
  const std::vector<std::vector<std::string>> flags_;
  std::vector<std::unique_ptr<ChildProcessBackend>> parts_;
  BackendAttributes attributes_;
  std::atomic<size_t> start_ = 0;
};

// Deals the batch out in chunks of the preferred batch step, chunk c going to
// part (start + c) % parts. Positions go straight into the parts' slots as
// they arrive, so no part needs the batch size to place them.
class DemuxingComputation : public BackendComputation {
 public:
  explicit DemuxingComputation(DemuxingBackend* backend)
      : step_(backend->GetAttributes().preferred_batch_step),
        maximum_batch_size_(backend->GetAttributes().maximum_batch_size),
        start_(backend->NextStart()) {
    parts_.reserve(backend->NumParts());
    for (size_t i = 0; i < backend->NumParts(); ++i) {
      parts_.push_back(
          std::make_unique<ChildProcessComputation>(backend->Part(i)));
    }
  }

  size_t UsedBatchSize() const override {
    return std::min(size_.load(), maximum_batch_size_);
  }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    const size_t index = size_.fetch_add(1);
    if (index >= maximum_batch_size_) {
      throw Exception("Batch is larger than the backend processes allow");
    }
    // In part order, so that computations waiting for slots never wait on
    // each other in a cycle.
    std::call_once(slots_acquired_, [&] {
      for (const auto& part : parts_) part->AcquireSlot();
    });
    const size_t chunk = index / step_;
    parts_[(start_ + chunk) % parts_.size()]->Write(
        chunk / parts_.size() * step_ + index % step_, pos, result);
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    const size_t batch_size = UsedBatchSize();
    const size_t chunks = (batch_size + step_ - 1) / step_;
    for (size_t i = 0; i < parts_.size(); ++i) {
      // The part's chunks are offset, offset + parts, ... below chunks; the
      // batch's last chunk may be short.
      const size_t offset = (i + parts_.size() - start_) % parts_.size();
      size_t size = 0;
      if (offset < chunks) {
        const size_t part_chunks =
            (chunks - offset + parts_.size() - 1) / parts_.size();
        size = part_chunks * step_;
        if (offset + (part_chunks - 1) * parts_.size() == chunks - 1) {
          size -= chunks * step_ - batch_size;
        }
      }
      parts_[i]->Send(size);
    }
    // Every part is waited for even after one fails, as a slot is released
    // only once its process has answered or died.
    std::exception_ptr error;
    for (const auto& part : parts_) {
      try {
        part->Receive();
      } catch (...) {
        if (!error) error = std::current_exception();
      }
    }
    if (error) std::rethrow_exception(error);
  }

 private:
  const size_t step_;
  const size_t maximum_batch_size_;
  const size_t start_;
  std::vector<std::unique_ptr<ChildProcessComputation>> parts_;
  std::once_flag slots_acquired_;
  std::atomic<size_t> size_ = 0;
};

std::unique_ptr<BackendComputation> DemuxingBackend::CreateComputation() {
  return std::make_unique<DemuxingComputation>(this);
}

}  // namespace

std::unique_ptr<Backend> CreateChildProcessBackend(const OptionsDict& options) {
  std::vector<std::vector<std::string>> flags = ProcessFlags(options);
  std::unique_ptr<Backend> backend;
  if (flags.size() == 1) {
    backend = std::make_unique<ChildProcessBackend>(std::move(flags[0]));
  } else {
    backend = std::make_unique<DemuxingBackend>(std::move(flags));
  }
  // Records the configuration, for IsSameConfiguration().
  backend->UpdateConfiguration(options);
  return backend;
}

}  // namespace backend_process
}  // namespace lczero
