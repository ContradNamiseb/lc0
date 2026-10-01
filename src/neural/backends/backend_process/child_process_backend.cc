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

#include "neural/backends/backend_process/gpu_dispatch.h"
#include "neural/backends/backend_process/interprocess.h"
#include "neural/backends/backend_process/protocol.h"
#include "neural/register.h"
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

// The backend that shares batches out over one backend process per GPU.
constexpr std::string_view kGpuDispatchName = "gpu-dispatch";

struct ProcessGroup {
  std::string backend;
  std::string backend_options;
};

// The backend and options of each GPU that gpu-dispatch configured with
// `backend_options` runs, read the way DemuxingNetwork reads its groups: a
// group reads the top-level options that it does not set itself, and runs the
// backend named by its "backend" option or else by its name. Options with no
// groups make one group. Empty if a group names no backend.
std::vector<ProcessGroup> ProcessGroups(std::string_view backend_options) {
  std::vector<OptionsEntry> top_level;
  std::vector<OptionsEntry> groups;
  for (std::string_view text : SplitOptions(backend_options)) {
    const OptionsEntry entry = ParseEntry(text);
    (entry.is_group ? groups : top_level).push_back(entry);
  }
  if (groups.empty()) groups.push_back({"()", "", "", true});
  std::vector<ProcessGroup> result;
  for (const OptionsEntry& group : groups) {
    std::vector<OptionsEntry> own;
    for (std::string_view text : SplitOptions(group.value)) {
      own.push_back(ParseEntry(text));
    }
    ProcessGroup& part = result.emplace_back();
    part.backend = group.key;
    auto add = [&part](const OptionsEntry& entry) {
      if (!entry.is_group && entry.key == "backend") {
        part.backend = entry.value;
        return;
      }
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
    if (part.backend.empty() || part.backend == kGpuDispatchName) return {};
  }
  return result;
}

// The command line of each backend process that `options` ask for: one per
// GPU of gpu-dispatch, and one for any other backend.
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
  std::vector<ProcessGroup> groups;
  if (backend == kGpuDispatchName) {
    groups = ProcessGroups(backend_options);
    if (groups.empty()) {
      throw Exception(
          "The gpu-dispatch backend needs a backend for every GPU, as in "
          "backend=cuda-fp16,(gpu=0),(gpu=1)");
    }
  } else {
    groups.push_back({backend, backend_options});
  }

  std::vector<std::vector<std::string>> flags;
  for (const ProcessGroup& group : groups) {
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

// Puts `pos` into `record`.
void WriteRecord(pblczero::NetworkFormat::InputFormat input_format,
                 const EvalPosition& pos, bool want_policy,
                 PositionRecord& record) {
  if (pos.legal_moves.size() > kMaxLegalMoves) {
    throw Exception("Too many legal moves for the backend process");
  }
  // Only the positions the encoder reads cross to the backend process.
  std::array<int, kCompactHistory> history;
  const int history_size = CompactHistoryForNN(input_format, pos.pos, history);
  record.history_size = history_size;
  record.num_moves = pos.legal_moves.size();
  record.want_policy = want_policy;
  for (int i = 0; i < history_size; ++i) {
    std::memcpy(record.history + i * sizeof(Position), &pos.pos[history[i]],
                sizeof(Position));
  }
  std::memcpy(record.moves, pos.legal_moves.data(),
              pos.legal_moves.size() * sizeof(Move));
}

// Copies the part of `from` that is in use.
void CopyRecord(const PositionRecord& from, PositionRecord& to) {
  to.history_size = from.history_size;
  to.num_moves = from.num_moves;
  to.want_policy = from.want_policy;
  std::memcpy(to.history, from.history, from.history_size * sizeof(Position));
  std::memcpy(to.moves, from.moves, from.num_moves * sizeof(Move));
}

// Hands the first `batch_size` results out to where the search wants them.
void HandOutResults(const ResultRecord* source,
                    const EvalResultPtr* destination, size_t batch_size) {
  for (size_t i = 0; i < batch_size; ++i) {
    if (destination[i].q) *destination[i].q = source[i].q;
    if (destination[i].d) *destination[i].d = source[i].d;
    if (destination[i].m) *destination[i].m = source[i].m;
    std::copy_n(source[i].p,
                std::min<size_t>(destination[i].p.size(), kMaxLegalMoves),
                destination[i].p.begin());
  }
}

// Seconds of std::chrono::steady_clock, as SlotHeader::finished_at counts.
double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
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

  // How many times the backend process has started. A batch sent after
  // start n and answered with the count still n ran in one process.
  uint32_t NumStarts() const { return starts_.load(std::memory_order_relaxed); }

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
    if (index >= kMaxBatch) {
      throw Exception("Batch is larger than the backend process allows");
    }
    AcquireSlot();
    WriteRecord(backend_->GetAttributes().input_format, pos, !result.p.empty(),
                backend_->Slot(slot_).positions[index]);
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
    backend_->Receive(slot_, request_);
    HandOutResults(backend_->Slot(slot_).results, results_, batch_size_);
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

// Before a GPU has run any batch: a guess, the same for every GPU, that the
// first batches correct.
constexpr double kPriorOverhead = 1e-3;
constexpr double kPriorPerPosition = 1e-4;

// Shares every batch out over one backend process per GPU, each part going
// to the GPU that would finish it first; see gpu_dispatch.h. A crash on one
// GPU restarts only that GPU's process, and until it is back its late work
// makes it look busy, so the other GPUs take the batches.
class GpuDispatchBackend : public Backend {
 public:
  // The positions of a batch until it is planned, and what its parts need.
  // Kept for later batches, so that a batch allocates nothing once warmed
  // up.
  struct Batch {
    struct Part {
      size_t first = 0;
      int slot = -1;
      ChildProcessBackend::Request request{};
      double sent_at = 0;
    };
    std::vector<PositionRecord> records;
    std::vector<EvalResultPtr> results;
    // Per GPU.
    std::vector<size_t> sizes;
    std::vector<double> predicted;
    std::vector<Part> parts;
  };

  explicit GpuDispatchBackend(std::vector<std::vector<std::string>> flags)
      : flags_(std::move(flags)), gpus_(flags_.size()) {
    // The processes load their networks at the same time.
    std::vector<std::exception_ptr> errors(gpus_.size());
    std::vector<std::thread> threads;
    for (size_t i = 0; i < gpus_.size(); ++i) {
      threads.emplace_back([this, &errors, i] {
        try {
          gpus_[i] = std::make_unique<ChildProcessBackend>(flags_[i]);
        } catch (...) {
          errors[i] = std::current_exception();
        }
      });
    }
    for (std::thread& thread : threads) thread.join();
    for (const std::exception_ptr& error : errors) {
      if (error) std::rethrow_exception(error);
    }

    attributes_ = gpus_[0]->GetAttributes();
    attributes_.recommended_batch_size = 0;
    attributes_.maximum_batch_size = 0;
    for (const auto& gpu : gpus_) {
      const BackendAttributes attributes = gpu->GetAttributes();
      attributes_.runs_on_cpu &= attributes.runs_on_cpu;
      attributes_.suggested_num_search_threads =
          std::max(attributes_.suggested_num_search_threads,
                   attributes.suggested_num_search_threads);
      attributes_.recommended_batch_size += attributes.recommended_batch_size;
      attributes_.maximum_batch_size += attributes.maximum_batch_size;
      attributes_.preferred_batch_step = std::max(
          attributes_.preferred_batch_step, attributes.preferred_batch_step);
      queues_.emplace_back(CostModel(kPriorOverhead, kPriorPerPosition));
    }
    attributes_.preferred_batch_step =
        std::max(attributes_.preferred_batch_step, 1);
    loads_.resize(gpus_.size());
  }

  BackendAttributes GetAttributes() const override { return attributes_; }

  std::unique_ptr<BackendComputation> CreateComputation() override;

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    Backend::UpdateConfiguration(options);
    return ProcessFlags(options) == flags_ ? UPDATE_OK : NEED_RESTART;
  }

  ChildProcessBackend* Gpu(size_t index) const { return gpus_[index].get(); }

  // Shares a batch of `batch_size` positions out over the GPUs and counts
  // the parts as sent: sets batch.sizes and batch.predicted.
  void Plan(size_t batch_size, Batch& batch) {
    std::lock_guard lock(queues_mutex_);
    const double now = Now();
    for (size_t i = 0; i < gpus_.size(); ++i) {
      loads_[i] = {
          queues_[i].FreeAt(now),
          static_cast<size_t>(gpus_[i]->GetAttributes().maximum_batch_size),
          &queues_[i].Model()};
    }
    if (!PlanBatch(batch_size, attributes_.preferred_batch_step, now, loads_,
                   batch.sizes)) {
      throw Exception("Batch is larger than the backend processes allow");
    }
    for (size_t i = 0; i < gpus_.size(); ++i) {
      batch.predicted[i] =
          batch.sizes[i] ? queues_[i].Dispatch(batch.sizes[i], now) : 0;
    }
  }

  // Records that GPU `index` finished a part of `batch_size` positions at
  // `finished_at`.
  void Complete(size_t index, size_t batch_size, double predicted,
                double sent_at, double finished_at, bool learn) {
    std::lock_guard lock(queues_mutex_);
    queues_[index].Complete(batch_size, predicted, sent_at, finished_at, learn);
  }

  std::unique_ptr<Batch> TakeBatch() {
    {
      std::lock_guard lock(batches_mutex_);
      if (!free_batches_.empty()) {
        std::unique_ptr<Batch> batch = std::move(free_batches_.back());
        free_batches_.pop_back();
        return batch;
      }
    }
    auto batch = std::make_unique<Batch>();
    batch->records.resize(attributes_.maximum_batch_size);
    batch->results.resize(attributes_.maximum_batch_size);
    batch->sizes.resize(gpus_.size());
    batch->predicted.resize(gpus_.size());
    batch->parts.resize(gpus_.size());
    return batch;
  }

  void ReturnBatch(std::unique_ptr<Batch> batch) {
    std::lock_guard lock(batches_mutex_);
    free_batches_.push_back(std::move(batch));
  }

 private:
  const std::vector<std::vector<std::string>> flags_;
  std::vector<std::unique_ptr<ChildProcessBackend>> gpus_;
  BackendAttributes attributes_;

  std::mutex queues_mutex_;
  std::vector<GpuQueue> queues_;
  // Only for Plan(), kept to spare it an allocation.
  std::vector<GpuLoad> loads_;

  std::mutex batches_mutex_;
  std::vector<std::unique_ptr<Batch>> free_batches_;
};

class GpuDispatchComputation : public BackendComputation {
 public:
  explicit GpuDispatchComputation(GpuDispatchBackend* backend)
      : backend_(backend), batch_(backend->TakeBatch()) {}

  // Also after ComputeBlocking() threw: every part it sent is answered or
  // its process is dead.
  ~GpuDispatchComputation() override {
    for (size_t i = 0; i < batch_->parts.size(); ++i) {
      int& slot = batch_->parts[i].slot;
      if (slot >= 0) backend_->Gpu(i)->ReleaseSlot(slot);
      slot = -1;
    }
    backend_->ReturnBatch(std::move(batch_));
  }

  size_t UsedBatchSize() const override {
    return std::min(size_.load(), batch_->records.size());
  }

  // The search's task workers call this concurrently.
  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    const size_t index = size_.fetch_add(1);
    if (index >= batch_->records.size()) {
      throw Exception("Batch is larger than the backend processes allow");
    }
    WriteRecord(backend_->GetAttributes().input_format, pos, !result.p.empty(),
                batch_->records[index]);
    batch_->results[index] = result;
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    const size_t batch_size = UsedBatchSize();
    if (batch_size == 0) return;
    GpuDispatchBackend::Batch& batch = *batch_;
    backend_->Plan(batch_size, batch);
    // Slots are taken in GPU order, so that computations waiting for slots
    // never wait on each other in a cycle. Each part is sent as soon as it
    // is copied, so the first GPUs start while the others are filled.
    size_t first = 0;
    for (size_t i = 0; i < batch.parts.size(); ++i) {
      GpuDispatchBackend::Batch::Part& part = batch.parts[i];
      const size_t size = batch.sizes[i];
      part.first = first;
      first += size;
      if (size == 0) continue;
      ChildProcessBackend* gpu = backend_->Gpu(i);
      part.slot = gpu->AcquireSlot();
      const SlotView slot = gpu->Slot(part.slot);
      for (size_t j = 0; j < size; ++j) {
        CopyRecord(batch.records[part.first + j], slot.positions[j]);
      }
      slot.header->batch_size = size;
      part.sent_at = Now();
      part.request = gpu->Send(part.slot);
    }
    // Every part is waited for even after one fails, as a slot is released
    // only once its process has answered or died.
    std::exception_ptr error;
    for (size_t i = 0; i < batch.parts.size(); ++i) {
      GpuDispatchBackend::Batch::Part& part = batch.parts[i];
      const size_t size = batch.sizes[i];
      if (size == 0) continue;
      ChildProcessBackend* gpu = backend_->Gpu(i);
      const SlotView slot = gpu->Slot(part.slot);
      bool answered = false;
      try {
        gpu->Receive(part.slot, part.request);
        answered = true;
      } catch (...) {
        if (!error) error = std::current_exception();
      }
      // A part that waited for a restart says nothing of the GPU's speed.
      const bool learn =
          answered && gpu->NumStarts() == part.request.first_start;
      backend_->Complete(i, size, batch.predicted[i], part.sent_at,
                         learn ? slot.header->finished_at * 1e-9 : Now(),
                         learn);
      if (answered) {
        HandOutResults(slot.results, &batch.results[part.first], size);
      }
      gpu->ReleaseSlot(part.slot);
      part.slot = -1;
    }
    if (error) std::rethrow_exception(error);
  }

 private:
  GpuDispatchBackend* const backend_;
  std::unique_ptr<GpuDispatchBackend::Batch> batch_;
  std::atomic<size_t> size_ = 0;
};

std::unique_ptr<BackendComputation> GpuDispatchBackend::CreateComputation() {
  return std::make_unique<GpuDispatchComputation>(this);
}

// Lists gpu-dispatch among the backends, so that the Backend option takes
// it. Only lc0 itself runs it, never a backend process.
class GpuDispatchFactory : public BackendFactory {
 public:
  int GetPriority() const override { return -1100; }
  std::string_view GetName() const override { return kGpuDispatchName; }
  std::unique_ptr<Backend> Create(const OptionsDict&) override {
    throw Exception("The gpu-dispatch backend runs only in lc0 itself");
  }
};

BackendManager::Register register_gpu_dispatch(
    std::make_unique<GpuDispatchFactory>());

}  // namespace

std::unique_ptr<Backend> CreateChildProcessBackend(const OptionsDict& options) {
  std::vector<std::vector<std::string>> flags = ProcessFlags(options);
  std::unique_ptr<Backend> backend;
  if (flags.size() == 1) {
    backend = std::make_unique<ChildProcessBackend>(std::move(flags[0]));
  } else {
    backend = std::make_unique<GpuDispatchBackend>(std::move(flags));
  }
  // Records the configuration, for IsSameConfiguration().
  backend->UpdateConfiguration(options);
  return backend;
}

}  // namespace backend_process
}  // namespace lczero
