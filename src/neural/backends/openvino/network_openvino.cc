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

// Structured to mirror the SYCL backend (network_sycl.cc.dp.cpp): this file
// only orchestrates -- construction, the InputsOutputs pool, and the
// NetworkComputation call sequence -- and inputs_outputs.h owns every
// pre-allocated buffer. There is no sycl/layers.h equivalent here: OpenVINO
// compiles and fuses the whole graph itself from the converted ONNX model,
// so there is no per-layer C++ dispatch to split out the way SYCL's
// FCLayer/SELayer/EncoderBlock classes do -- with two exceptions, both
// hand-written fused ops replacing an ONNX-exported subgraph that profiled
// as a bottleneck: KdaScanOp (kda_scan_op.h/.cc) replaces the ONNX-Scan
// -derived TensorIterator for the KDA recurrence (>98% of inference time
// before the fix), and SEResidualOp (se_residual_op.h/.cc) replaces the
// Squeeze-and-Excitation sub-block plus residual tail -- the ~200 small
// FullyConnected/Gemm-shaped ops that turned out to dominate runtime once
// KdaScanOp fixed the recurrence itself (see the profiling comment on
// ReplaceKdaScan).

#include <algorithm>
#include <chrono>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <openvino/core/op_extension.hpp>
#include <openvino/core/version.hpp>
#include <openvino/openvino.hpp>
#include <openvino/op/util/precision_sensitive_attribute.hpp>
#include <openvino/pass/manager.hpp>

#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "neural/onnx/converter.h"
#include "utils/bititer.h"
#include "utils/exception.h"
#include "utils/logging.h"

#include "neural/backends/openvino/inputs_outputs.h"
#include "neural/backends/openvino/kda_scan_kernel_source.h"
#include "neural/backends/openvino/kda_scan_op.h"
#include "neural/kda_directions.h"
#include "neural/backends/openvino/kda_scan_pass.h"
#include "neural/backends/openvino/se_residual_kernel_source.h"
#include "neural/backends/openvino/se_residual_op.h"
#include "neural/backends/openvino/se_residual_pass.h"

namespace lczero {
namespace openvino_backend {

static constexpr int kMaxBatchSize = 1024;

class OpenVinoNetwork;

class OpenVinoNetworkComputation : public NetworkComputation {
 public:
  OpenVinoNetworkComputation(OpenVinoNetwork* network, bool has_wdl,
                             bool has_mlh);
  ~OpenVinoNetworkComputation() override;

  void AddInput(InputPlanes&& input) override;
  void ComputeBlocking() override;
  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override;
  float GetDVal(int sample) const override;
  float GetPVal(int sample, int move_id) const override;
  float GetMVal(int sample) const override;

 private:
  OpenVinoNetwork* const network_;
  std::unique_ptr<InputsOutputs> io_;
  int batch_size_ = 0;
  bool has_wdl_;
  bool has_mlh_;
};

// Cached output ports, resolved once at network initialization so
// ComputeBlocking() never has to look a tensor up by name.
struct CachedOutputPorts {
  ov::Output<const ov::Node> policy_port;
  ov::Output<const ov::Node> value_port;
  ov::Output<const ov::Node> mlh_port;
};

class OpenVinoNetwork : public Network {
 public:
  OpenVinoNetwork(const WeightsFile& weights, const OptionsDict& options);
  ~OpenVinoNetwork() override { if (profile_) DumpProfile(); }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<OpenVinoNetworkComputation>(
        this, capabilities_.has_wdl(), capabilities_.has_mlh());
  }

  int GetMiniBatchSize() const override {
    // The batch the search gathers before evaluating when the user leaves
    // MinibatchSize at 0 (wrapper.cc surfaces this as recommended_batch_size).
    // 256 is the Network base-class default; returning kMaxBatchSize (1024)
    // here padded most default-config inferences up to 1024 rows of real GPU
    // work and quadrupled max_out_of_order on a backend whose per-inference
    // latency is already its weak spot. Custom-layer JIT shapes are handled
    // by BucketBatch()/batch_buckets_, not by a large suggested minibatch.
    return 256;
  }
  bool IsCpu() const override { return is_cpu_; }

  const CachedOutputPorts& ports() const { return ports_; }
  bool profile() const { return profile_; }

  // Folds one inference's per-op counters into the running totals. Called
  // with the InferRequest still intact, straight after infer().
  void AccumulateProfile(const ov::InferRequest& request) {
    // Not every primitive produces a profiling event -- the GPU plugin
    // throws CL_PROFILING_INFO_NOT_AVAILABLE for at least the custom-layer
    // KdaScanOp kernel, and it takes the whole request's counters down with
    // it. Losing some samples is fine for a diagnostic; killing the search
    // thread is not, so swallow it and report the drop count at the end.
    std::vector<ov::ProfilingInfo> infos;
    try {
      infos = request.get_profiling_info();
    } catch (const std::exception&) {
      std::lock_guard<std::mutex> lock(profile_mutex_);
      profile_failures_++;
      return;
    }

    std::lock_guard<std::mutex> lock(profile_mutex_);
    for (const auto& info : infos) {
      if (info.status != ov::ProfilingInfo::Status::EXECUTED) continue;
      auto& entry = profile_by_type_[info.exec_type];
      entry.us += info.real_time.count();
      entry.count++;
      auto& node = profile_by_node_[info.node_name];
      node.us += info.real_time.count();
      node.count++;
      node.exec_type = info.exec_type;
    }
    profile_infers_++;
  }
  int min_batch() const { return min_batch_; }
  bool is_ir_model() const { return is_ir_model_; }

  // Rounds an inference batch up to the next warmed bucket.
  //
  // SimpleGPU custom layers are JIT'd per (kernel source x concrete tensor
  // shape), so a dynamic batch dimension means every new batch size the
  // search happens to produce triggers a multi-second Intel graphics
  // compiler run *inside* the search. Measured on kda-native-455835 with an
  // empty driver cache: 32 nodes in 20s (1 nps), seven stalls of up to
  // 4.4s, 90 kernel binaries built. Restricting the batch to a handful of
  // shapes bounds that to a handful of compiles, which the constructor then
  // pays up front (see the warmup block).
  //
  // Only worth doing when a custom layer is actually present: OpenVINO's
  // own GPU kernels are effectively shape-agnostic here (the same cold-cache
  // run on native-only 791556 built 37 binaries and held 1016 nps), so for
  // those bucketing would buy nothing and cost padded rows of real compute.
  int BucketBatch(int n) const {
    if (batch_buckets_.empty()) return n;
    auto it = std::lower_bound(batch_buckets_.begin(), batch_buckets_.end(), n);
    return it == batch_buckets_.end() ? kMaxBatchSize : *it;
  }

  // Mirrors sycl/network_sycl.cc.dp.cpp's GetInputsOutputs/
  // ReleaseInputsOutputs free-list: an InferRequest plus its bound tensors
  // is too expensive to build fresh on every move, so completed ones are
  // recycled instead of destroyed.
  std::unique_ptr<InputsOutputs> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (free_inputs_outputs_.empty()) {
      return std::make_unique<InputsOutputs>(
          kMaxBatchSize, compiled_model_.create_infer_request());
    }
    auto io = std::move(free_inputs_outputs_.front());
    free_inputs_outputs_.pop_front();
    return io;
  }

  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputs> io) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    free_inputs_outputs_.push_back(std::move(io));
  }

 private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  CachedOutputPorts ports_;
  NetworkCapabilities capabilities_;
  bool is_cpu_ = false;
  // True when this model came from a pre-converted IR file (ir_path) rather
  // than ConvertWeightsToOnnx. An IR-loaded graph normally contains no
  // KdaScanOp (nothing matches ReplaceKdaScan's TensorIterator pattern --
  // and the constructor now rejects one that does, since the GPU
  // custom-layer config is converter-path-only), so it skips the GPU
  // custom-layer kernel setup -- but it still needs the forced-f32 precision
  // hint, for its own separate reason. See the device_config block in the
  // constructor.
  bool is_ir_model_ = false;
  // Mirrors the cuda backend's min_batch: it never evaluates fewer
  // than this many positions, padding the batch out instead. There
  // it is about output variance on tiny batches; here it also keeps
  // the GPU plugin off the degenerate small-batch path.
  int min_batch_ = 4;
  // Ascending batch shapes the search is allowed to use, and which warmup
  // pre-compiles. Empty means "no bucketing" -- see BucketBatch().
  std::vector<int> batch_buckets_;

  std::mutex io_mutex_;
  std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;

  // Per-op profiling, off unless the `profile` backend option is set.
  // Aggregated across every inference and dumped once at teardown -- a
  // per-call dump would be unreadable at hundreds of batches a second, and
  // the whole point is to find which kernels dominate in aggregate.
  struct ProfileEntry {
    uint64_t us = 0;
    uint64_t count = 0;
    std::string exec_type;
  };
  bool profile_ = false;
  std::mutex profile_mutex_;
  std::map<std::string, ProfileEntry> profile_by_type_;
  std::map<std::string, ProfileEntry> profile_by_node_;
  uint64_t profile_infers_ = 0;
  uint64_t profile_failures_ = 0;
  void DumpProfile();
};

OpenVinoNetworkComputation::OpenVinoNetworkComputation(
    OpenVinoNetwork* network, bool has_wdl, bool has_mlh)
    : network_(network),
      io_(network->GetInputsOutputs()),
      has_wdl_(has_wdl),
      has_mlh_(has_mlh) {}

OpenVinoNetworkComputation::~OpenVinoNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(io_));
}

void OpenVinoNetworkComputation::AddInput(InputPlanes&& input) {
  if (batch_size_ >= kMaxBatchSize) {
    throw Exception("OpenVINO batch size exceeded maximum of " +
                    std::to_string(kMaxBatchSize));
  }

  float* const batch_ptr =
      io_->input_val_mem_ + batch_size_ * (kInputPlanes * 64);
  std::memset(batch_ptr, 0, kInputPlanes * 64 * sizeof(float));

  int plane_idx = 0;
  for (const auto& plane : input) {
    float* const plane_ptr = batch_ptr + plane_idx * 64;
    for (auto sq : IterateBits(plane.mask)) {
      plane_ptr[sq] = plane.value;
    }
    plane_idx++;
  }
  batch_size_++;
}

void OpenVinoNetworkComputation::ComputeBlocking() {
  if (batch_size_ == 0) return;

  // Evaluate at least min_batch positions. Samples are independent,
  // so the padding rows cannot affect the real ones; they are zeroed
  // rather than left stale so the padding is at least well-defined,
  // and only batch_size_ results are ever read back.
  const int infer_batch =
      network_->BucketBatch(std::max(batch_size_, network_->min_batch()));
  if (infer_batch > batch_size_) {
    std::memset(io_->input_val_mem_ + batch_size_ * (kInputPlanes * 64),
                0,
                static_cast<size_t>(infer_batch - batch_size_) *
                    kInputPlanes * 64 * sizeof(float));
  }

  if (network_->is_ir_model()) {
    // The zero-copy ROI view below hands the graph a tensor whose strides
    // still describe the full kMaxBatchSize backing buffer. The ONNX path
    // tolerates that; the TF-imported graph does not -- with the view, a
    // batch of 1 matches the reference but every larger batch is wrong,
    // the classic stride signature. Copying into an exactly-sized tensor
    // costs one memcpy of at most a few hundred KB per call and is also
    // what the Python verification of this IR actually exercised.
    ov::Tensor exact(io_->input_tensor_.get_element_type(),
                     {static_cast<size_t>(infer_batch), kInputPlanes, 8, 8});
    std::memcpy(exact.data<float>(), io_->input_val_mem_,
                static_cast<size_t>(infer_batch) * kInputPlanes * 64 *
                    sizeof(float));
    io_->infer_request_.set_input_tensor(exact);
  } else {
    // Zero-copy dynamic-batch view over the pre-allocated input tensor.
    ov::Tensor batch_view(
        io_->input_tensor_, {0, 0, 0, 0},
        {static_cast<size_t>(infer_batch), kInputPlanes, 8, 8});
    io_->infer_request_.set_input_tensor(batch_view);
  }

  // Bind our own host buffers as the output tensors BEFORE inferring, so
  // the plugin writes results straight into memory we own. Reading back
  // from whatever get_tensor() hands us is what crashed: on the GPU plugin
  // that memory is not dependably host-readable, and no amount of bounds
  // checking helps because the pointer itself is the problem.
  const auto& ports = network_->ports();
  const size_t nb = static_cast<size_t>(infer_batch);
  io_->infer_request_.set_tensor(
      ports.policy_port,
      ov::Tensor(ov::element::f32, {nb, InputsOutputs::kPolicyWidth},
                 io_->policy_.data()));
  io_->infer_request_.set_tensor(
      ports.value_port,
      ov::Tensor(ov::element::f32,
                 {nb, has_wdl_ ? InputsOutputs::kValueWidth : size_t{1}},
                 io_->value_.data()));
  if (has_mlh_) {
    io_->infer_request_.set_tensor(
        ports.mlh_port,
        ov::Tensor(ov::element::f32, {nb, 1}, io_->moves_left_.data()));
  }

  io_->infer_request_.infer();

  if (network_->profile()) network_->AccumulateProfile(io_->infer_request_);
}

float OpenVinoNetworkComputation::GetQVal(int sample) const {
  if (has_wdl_) {
    const float* const wdl = io_->op_value_mem_ + sample * 3;
    return wdl[0] - wdl[2];
  }
  return io_->op_value_mem_[sample];
}

float OpenVinoNetworkComputation::GetDVal(int sample) const {
  if (has_wdl_) {
    return io_->op_value_mem_[sample * 3 + 1];
  }
  return 0.0f;
}

float OpenVinoNetworkComputation::GetPVal(int sample, int move_id) const {
  assert(move_id >= 0 && move_id < 1858);
  return io_->op_policy_mem_[sample * 1858 + move_id];
}

float OpenVinoNetworkComputation::GetMVal(int sample) const {
  if (has_mlh_) {
    return io_->op_moves_left_mem_[sample];
  }
  return 0.0f;
}

namespace {

// Portable pid for the per-process config directory below.
std::uint32_t ProcessId() {
#if defined(_WIN32)
  return static_cast<std::uint32_t>(::_getpid());
#else
  return static_cast<std::uint32_t>(::getpid());
#endif
}

// The per-process temp directory both WriteKdaScanGpuConfig and
// WriteSEResidualGpuConfig write their .cl sources into, and where the
// single merged CONFIG_FILE ends up. OpenVINO's GPU plugin loads multiple
// custom ops from one CONFIG_FILE as sibling <CustomLayer> elements with no
// wrapper (confirmed against the plugin's own parser: it walks
// document_element()/next_sibling(), not a single expected root) -- so
// every <Source filename=.../> in that file must resolve from the same
// directory as the merged xml itself (see the comment on WriteKdaScanGpuConfig
// below), which is why all of this shares one directory rather than each op
// getting its own.
//
// The directory is keyed on the process id: the merged XML embeds this net's
// KDA/SE geometry as -D defines and is rewritten (truncating whatever is
// there) on every backend load, so a fixed shared path would let two
// concurrent lc0 processes converting different nets -- match play,
// selfplay workers, lc0 plus a benchmark -- race on one file, and the loser
// would compile the winner's geometry with nothing logged. The .cl sources
// are net-independent, so only the XML needs this protection.
std::filesystem::path CustomLayerConfigDir() {
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() /
                 ("lc0_openvino_" + std::to_string(ProcessId()));
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    throw Exception("OpenVINO: could not create temp dir for GPU "
                    "custom-layer kernel configs: " + ec.message());
  }
  return dir;
}

// Writes the merged CONFIG_FILE from however many <CustomLayer> fragments
// were generated (one KdaScanOp fragment, one SEResidualOp fragment, both,
// or neither). Must be called before compile_model().
std::filesystem::path WriteMergedGpuConfig(
    const std::vector<std::string>& fragments) {
  namespace fs = std::filesystem;
  fs::path xml_path = CustomLayerConfigDir() / "lc0_custom_layers.xml";
  std::ofstream xml(xml_path, std::ios::trunc);
  if (!xml) {
    throw Exception("OpenVINO: could not write custom-layer config to '" +
                    xml_path.string() + "'.");
  }
  for (const auto& fragment : fragments) xml << fragment;
  xml.close();
  return xml_path;
}

// Writes the GPU custom-layer kernel source for this model's specific KDA
// dims (heads/key_dim/value_dim are static per net but not portable across
// different net shapes, so the kernel is JIT-compiled per model rather than
// shipped pre-built), and returns its <CustomLayer> XML fragment for
// WriteMergedGpuConfig to concatenate alongside any other custom op's.
std::string WriteKdaScanGpuConfig(int heads, int key_dim, int value_dim,
                                  int direction_count,
                                  const std::vector<int>& directions,
                                  bool fp16) {
  namespace fs = std::filesystem;
  // The kernel launches with a work-group VALUE_DIM_ wide (WorkSizes
  // local="1,X,1" below), which must fit CL_KERNEL_WORK_GROUP_SIZE --
  // commonly 256 on Intel iGPUs. Exceeding it fails at first execution with
  // a raw OpenCL error deep inside the plugin, so reject a net that cannot
  // fit at config time, where a clear message is possible.
  constexpr int kMaxWorkgroupWidth = 256;
  if (value_dim > kMaxWorkgroupWidth) {
    throw Exception(
        "OpenVINO: KDA value_dim " + std::to_string(value_dim) +
        " exceeds the " + std::to_string(kMaxWorkgroupWidth) +
        "-wide work-group this GPU kernel launches with.");
  }
  fs::path dir = CustomLayerConfigDir();

  fs::path cl_path = dir / "kda_scan.cl";
  std::ofstream cl(cl_path, std::ios::trunc);
  if (!cl) {
    throw Exception("OpenVINO: could not write KDA kernel source to '" +
                    cl_path.string() + "'.");
  }
  // Generated from the single canonical definition rather than hand-copied
  // into the kernel string -- see kda_scan_kernel_source.h.
  // All 16 directions, not just the net's own: the kernel indexes this
  // by dir - 1 taken straight from DIRECTIONS_LIST_, so a serpentine net
  // (directions 9-16) reads rows 8-15. 1 KiB of __constant either way.
  cl << "__constant uchar kDirectionTable[16][64] = {\n";
  for (int dir = 1; dir <= 16; ++dir) {
    cl << "  {";
    for (int token = 0; token < 64; ++token) {
      if (token) cl << ",";
      cl << KdaSquareForToken(dir, token);
    }
    cl << "}" << (dir < 16 ? "," : "") << "\n";
  }
  cl << "};\n\n";
  cl << kKdaScanKernelSource;
  cl.close();

  std::string dir_list_str;
  for (size_t i = 0; i < directions.size(); ++i) {
    if (i > 0) dir_list_str += ",";
    dir_list_str += std::to_string(directions[i]);
  }
  if (directions.empty()) {
    dir_list_str = "1,2,3,4,5,6,7,8";
  }

  std::string dtype_str = fp16 ? "half" : "float";

  // Source filename is resolved relative to the CONFIG_FILE's own
  // directory (not the executable, despite what the docs say, and not
  // usable as an absolute path -- it gets concatenated onto that
  // directory regardless). Both files are written into the same dir
  // above, so a bare filename is enough.
  std::ostringstream xml;
  xml << "<CustomLayer name=\"KdaScan\" type=\"SimpleGPU\" version=\"1\">\n"
      << "  <Kernel entry=\"kda_scan_kernel\">\n"
      << "    <Source filename=\"" << cl_path.filename().string() << "\"/>\n"
      << "  </Kernel>\n"
      << "  <Buffers>\n"
      << "    <Tensor arg-index=\"0\" type=\"input\" port-index=\"0\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"1\" type=\"input\" port-index=\"1\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"2\" type=\"input\" port-index=\"2\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"3\" type=\"input\" port-index=\"3\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"4\" type=\"input\" port-index=\"4\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"5\" type=\"input\" port-index=\"5\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"6\" type=\"input\" port-index=\"6\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"7\" type=\"output\" port-index=\"0\" format=\"BFYX\"/>\n"
      << "  </Buffers>\n"
      << "  <CompilerOptions options=\"-D HEADS_=" << heads
      << " -D KEY_DIM_=" << key_dim << " -D VALUE_DIM_=" << value_dim
      << " -D DIRECTION_COUNT_=" << direction_count
      << " -D DIRECTIONS_LIST_=" << dir_list_str
      << " -D DTYPE=" << dtype_str
      << (fp16 ? " -D FP16_SUPPORTED=1" : "")
      << "\"/>\n"
      << "  <WorkSizes global=\"B*Y,X,1\" local=\"1,X,1\" dim=\"output\"/>\n"
      << "</CustomLayer>\n";
  return xml.str();
}

// Same pattern as WriteKdaScanGpuConfig, for SEResidualOp. channels/
// se_filters are static per net (uniform across every residual block in an
// lc0 resnet tower), so -- like KdaScanOp -- exactly one compiled kernel
// covers every SEResidualOp instance in the model.
std::string WriteSEResidualGpuConfig(int channels, int se_filters,
                                     SEResidualOp::Activation activation,
                                     bool fp16) {
  namespace fs = std::filesystem;
  // Same work-group-width limit as the KDA kernel (local="1,F,1" with
  // F = channels).
  constexpr int kMaxWorkgroupWidth = 256;
  if (channels > kMaxWorkgroupWidth) {
    throw Exception(
        "OpenVINO: SE residual channel count " + std::to_string(channels) +
        " exceeds the " + std::to_string(kMaxWorkgroupWidth) +
        "-wide work-group this GPU kernel launches with.");
  }
  fs::path dir = CustomLayerConfigDir();

  fs::path cl_path = dir / "se_residual.cl";
  std::ofstream cl(cl_path, std::ios::trunc);
  if (!cl) {
    throw Exception("OpenVINO: could not write SE kernel source to '" +
                    cl_path.string() + "'.");
  }
  cl << kSEResidualKernelSource;
  cl.close();

  std::string dtype_str = fp16 ? "half" : "float";

  std::ostringstream xml;
  xml << "<CustomLayer name=\"SEResidual\" type=\"SimpleGPU\" version=\"1\">\n"
      << "  <Kernel entry=\"se_residual_kernel\">\n"
      << "    <Source filename=\"" << cl_path.filename().string() << "\"/>\n"
      << "  </Kernel>\n"
      << "  <Buffers>\n"
      << "    <Tensor arg-index=\"0\" type=\"input\" port-index=\"0\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"1\" type=\"input\" port-index=\"1\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"2\" type=\"input\" port-index=\"2\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"3\" type=\"input\" port-index=\"3\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"4\" type=\"input\" port-index=\"4\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"5\" type=\"input\" port-index=\"5\" format=\"BFYX\"/>\n"
      << "    <Tensor arg-index=\"6\" type=\"output\" port-index=\"0\" format=\"BFYX\"/>\n"
      << "  </Buffers>\n"
      << "  <CompilerOptions options=\"-D CHANNELS_=" << channels
      << " -D SE_FILTERS_=" << se_filters << " -D DTYPE=" << dtype_str
      << (fp16 ? " -D FP16_SUPPORTED=1" : "")
      << (activation == SEResidualOp::Activation::kMish
              ? " -D MISH_ACTIVATION=1"
              : "")
      << "\"/>\n"
      << "  <WorkSizes global=\"B*1,F,1\" local=\"1,F,1\" dim=\"output\"/>\n"
      << "</CustomLayer>\n";
  return xml.str();
}

}  // namespace

OpenVinoNetwork::OpenVinoNetwork(const WeightsFile& weights,
                                 const OptionsDict& options) {
  std::string device = options.GetOrDefault<std::string>("device", "GPU");
  // Accept the whole OpenVINO device-name grammar for GPU and CPU ("GPU",
  // "GPU.1" on multi-GPU, "CPU"), not exact equality: every GPU-specific
  // setting below (LATENCY hints, inference_precision, the custom-layer
  // CONFIG_FILE) was originally keyed on device == "GPU", so device=GPU.1
  // silently dropped all of it and failed to compile KDA nets with an
  // opaque plugin error. Composite devices (AUTO, MULTI:..., HETERO:...)
  // remain unsupported for custom-layer nets -- see the check after the
  // graph passes -- because the config mechanism targets one concrete
  // plugin.
  const bool is_gpu_device = device.rfind("GPU", 0) == 0;
  const bool is_cpu_device = device.rfind("CPU", 0) == 0;
  const int requested_min_batch = options.GetOrDefault<int>("min_batch", 4);
  if (requested_min_batch < 1 || requested_min_batch > kMaxBatchSize) {
    CERR << "OpenVINO: clamping min_batch=" << requested_min_batch
         << " to [1, " << kMaxBatchSize << "].";
  }
  // Both bounds matter: above kMaxBatchSize overflows the buffers, and <= 0
  // used to hang the constructor -- the bucket ladder below steps batches
  // up by doubling until they reach kMaxBatchSize, which 0 never does.
  min_batch_ = std::clamp(requested_min_batch, 1, kMaxBatchSize);
  is_cpu_ = is_cpu_device;

  const auto& format = weights.format().network_format();
  capabilities_.input_format = format.input();
  capabilities_.output_format = format.output();
  capabilities_.moves_left = format.moves_left();

  CERR << "Initializing OpenVINO backend on device [" << device
       << "], runtime " << ov::get_openvino_version().buildNumber << "...";

  // ir_path opts into loading a pre-converted OpenVINO IR (.xml/.bin) file
  // directly, bypassing ConvertWeightsToOnnx entirely -- see
  // lc0-training/tf/net_to_openvino_ir.py.
  const std::string ir_path = options.GetOrDefault<std::string>("ir_path", "");
  is_ir_model_ = !ir_path.empty();

  // Head selection, same option names and defaults as network_cuda.cc:381 and
  // network_sycl.cc.dp.cpp:427, so a multi-head net evaluates the same head
  // whichever backend runs it. Read unconditionally (not just on the
  // converter path) because OptionsDict rejects any option nothing read --
  // hardcoding these was why `openvino-auto.policy_head` came back as
  // "Unknown string option".
  const std::string policy_head =
      options.GetOrDefault<std::string>("policy_head", "vanilla");
  const std::string value_head =
      options.GetOrDefault<std::string>("value_head", "winner");

  std::shared_ptr<ov::Model> model;
  if (is_ir_model_) {
    CERR << "Loading pre-converted OpenVINO IR from: " << ir_path;
    // The IR was baked with whichever heads net_to_openvino_ir.py exported;
    // there is no conversion here to steer, so say so rather than silently
    // evaluating a different head than asked for.
    if (policy_head != "vanilla" || value_head != "winner") {
      CERR << "Warning: policy_head/value_head are ignored with ir_path -- the "
              "IR already has its heads baked in.";
    }
    model = core_.read_model(ir_path);
  } else {
    WeightsToOnnxConverterOptions converter_options;
    converter_options.opset = 17;
    converter_options.data_type =
        WeightsToOnnxConverterOptions::DataType::kFloat32;
    converter_options.policy_head = policy_head;
    converter_options.value_head = value_head;

    pblczero::Net onnx_net = ConvertWeightsToOnnx(weights, converter_options);
    std::string_view model_bytes = onnx_net.onnx_model().model();

    model = core_.read_model(
        std::string(model_bytes),
        ov::Tensor(ov::element::u8, {model_bytes.size()},
                   const_cast<char*>(model_bytes.data())));
  }

  // Dynamic batch shape: -1 lets OpenVINO recompile internal shape-dependent
  // constants once here rather than reshaping (and paying that cost) inside
  // ComputeBlocking() on every call.
  model->reshape({{-1, kInputPlanes, 8, 8}});

  for (const auto& output : model->outputs()) {
    const std::string name = output.get_any_name();
    if (name.find("policy") != std::string::npos) {
      ports_.policy_port = output;
    } else if (name.find("wdl") != std::string::npos ||
               name.find("value") != std::string::npos) {
      ports_.value_port = output;
    } else if (name.find("mlh") != std::string::npos) {
      ports_.mlh_port = output;
    }
  }
  if (!ports_.policy_port.get_node() || !ports_.value_port.get_node()) {
    throw Exception(
        "OpenVINO: model is missing a policy or value output -- " +
        std::string(is_ir_model_
                        ? "the IR file's output names don't match what "
                          "this backend expects (policy/wdl/value/mlh "
                          "substrings)."
                        : "ConvertWeightsToOnnx's naming must have "
                          "changed."));
  }

  core_.add_extension(ov::OpExtension<KdaScanOp>());
  core_.add_extension(ov::OpExtension<SEResidualOp>());

  // The fused KdaScanOp applies the direction permutation inside the kernel
  // (the exported graph used to carry it as explicit ops and no longer
  // does), so it has to be told the net's actual direction set. Read it from
  // the same place converter.cc:682, network_blas.cc and
  // network_sycl.cc.dp.cpp:552 read it; defaulting to {1..8} would silently
  // evaluate a net declaring any other set or count with the wrong traversal
  // order. Empty is not a valid KDA net -- converter.cc throws on it too --
  // but a non-KDA net legitimately has none, and there the pass simply never
  // matches, so only validate when the field is actually populated.
  const auto& net_format = weights.format().network_format();
  const std::vector<int> kda_directions(net_format.kda_directions().begin(),
                                        net_format.kda_directions().end());
  const int kda_direction_count = static_cast<int>(kda_directions.size());
  if (kda_direction_count > 0) {
    const int heads = static_cast<int>(weights.weights().headcount());
    if (heads > 0 && heads % kda_direction_count != 0) {
      throw Exception(
          "KDA directions must evenly divide the encoder heads (" +
          std::to_string(heads) + " heads, " +
          std::to_string(kda_direction_count) + " directions).");
    }
    // Same check BLAS (network_blas.cc) and SYCL (layers.cc.dp.cpp) do at
    // load. Here it also protects the GPU kernel: it indexes
    // kDirectionTable[dir - 1] straight from DIRECTIONS_LIST_, and that
    // table has exactly 16 rows -- an out-of-range direction from a
    // malformed net would be an out-of-bounds __constant read rather than
    // an error.
    for (int dir : kda_directions) {
      if (dir < 1 || dir > 16) {
        throw Exception("KDA direction " + std::to_string(dir) +
                        " is out of range [1, 16].");
      }
    }
  }

  {
    ov::pass::Manager manager;
    manager.register_pass<ReplaceKdaScan>(kda_direction_count, kda_directions);
    // ReplaceSqueezeExcite is deliberately NOT registered: measured slower
    // than the native ops it replaces. On 791556.pb.gz (15 SE residual
    // blocks), device=GPU, MinibatchSize=32 with min_batch=32 so every
    // inference uses one batch shape, over 60s searches: unfused 873/828
    // nps vs fused 719/708 nps, i.e. ~16% slower. The premise (SE is ~14%
    // of op time as many small dispatch-bound ops, so fusing it should
    // win) capped the possible gain at ~14% to begin with, and this
    // kernel -- one work-item per (batch, channel), 64 squares serial, two
    // barriers, and only SE_FILTERS_ of CHANNELS_ lanes active during FC1
    // -- does not beat OpenVINO's own kernels even for the ops it removes.
    //
    // Also worth knowing before re-enabling: custom SimpleGPU layers are
    // JIT-compiled lazily per (kernel source x tensor shape) and a cold
    // compile costs seconds. The graph uses a dynamic batch dim, so a real
    // search hits many shapes and stalls repeatedly until the driver's
    // on-disk cache warms. Pinning to one shape (as measured above) hides
    // it; anything re-enabling this op needs batch-shape bucketing, not
    // just a faster kernel.
    //
    // Left reachable rather than deleted so the measurement can be redone
    // without resurrecting the code, but off by default and GPU-only.
    // GPU-only because the fallback path cannot carry it: SEResidualOp,
    // like KdaScanOp, only implements evaluate() for f32
    // (se_residual_op.cc:64), so on CPU -- or anywhere else the custom
    // kernel is not what runs -- an f16 model would fail to compile rather
    // than fall back. KdaScanOp needs no equivalent guard: it is only ever
    // matched out of an ONNX-imported TensorIterator, which an f16 IR does
    // not contain (see the is_ir_model_ note above).
    if (options.GetOrDefault<bool>("se_fusion", false)) {
      if (is_gpu_device) {
        manager.register_pass<ReplaceSqueezeExcite>();
        CERR << "OpenVINO: SE residual fusion enabled -- measured ~16% "
                "SLOWER than the native ops on 791556; see the comment in "
                "network_openvino.cc before trusting a result from this.";
      } else {
        CERR << "OpenVINO: ignoring se_fusion -- it requires device=GPU.";
      }
    }
    manager.run_passes(model);
  }

  // Default fp16 to true on GPU for maximum throughput (2.3x-2.4x speedup)
  const bool want_fp16 = options.GetOrDefault<bool>("fp16", is_gpu_device);
  profile_ = options.GetOrDefault<bool>("profile", false);

  // Does this net actually have KDA layers? (ReplaceKdaScan above has
  // already turned any KDA TensorIterator into KdaScanOp nodes, so this is
  // a direct check, not a guess from the net's metadata.) Used below to
  // default kda_safe on automatically -- a non-KDA net has nothing for it
  // to protect and pays no cost either way, since the marking loop is a
  // no-op without a KdaScanOp to find.
  bool has_kda = false;
  for (const auto& node : model->get_ops()) {
    if (ov::as_type_ptr<KdaScanOp>(node)) {
      has_kda = true;
      break;
    }
  }

  // A graph that still carries a KdaScanOp needs either the GPU custom-layer
  // config (generated only for converter-path models, on one concrete GPU
  // plugin) or the CPU evaluate() fallback. Two load configurations can
  // reach here with neither available -- say so plainly instead of letting
  // compile_model() fail with an opaque plugin error:
  //   - ir_path + KDA on GPU: the pass does run on IR-loaded models (IR
  //     preserves layer names, so a KDA IR converted from the same ONNX can
  //     still match), but the CONFIG_FILE generation below is deliberately
  //     converter-path-only.
  //   - a composite device (AUTO, MULTI:..., HETERO:...): the custom-layer
  //     config mechanism targets one concrete plugin, and which plugin
  //     executes the op is decided at runtime.
  if (has_kda) {
    if (is_ir_model_ && is_gpu_device) {
      throw Exception(
          "OpenVINO: ir_path models containing KDA layers are not supported "
          "on GPU -- the custom KdaScan kernel config is only generated on "
          "the net-to-ONNX conversion path. Load the net without ir_path, or "
          "use device=CPU (the KdaScanOp CPU path handles it).");
    }
    if (!is_gpu_device && !is_cpu_device) {
      throw Exception(
          "OpenVINO: device='" + device +
          "' cannot run this net's fused KDA layers -- the custom-layer "
          "config targets one concrete plugin. Use an explicit GPU (e.g. "
          "'GPU' or 'GPU.1') or CPU.");
    }
  }

  // kda_safe: mixed precision. Keeps the KDA recurrence's tensors and the
  // value head in fp32 inside an otherwise-fp16 graph, since those are where
  // half rounding costs the most (measured ~1.6e-2 absolute value error at
  // plain fp16), while the bulk of the FC/FFN work keeps fp16 speed. Comes on
  // automatically for any net with KDA layers -- a KDA net should not need a
  // manual flag to avoid its own worst fp16 error -- but can be forced off
  // (raw fp16 throughput, ~6x faster, at the larger error) or on.
  const bool kda_safe = options.GetOrDefault<bool>("kda_safe", has_kda) &&
                        want_fp16 && is_gpu_device;
  if (kda_safe) {
    int marked = 0;
    for (const auto& node : model->get_ops()) {
      if (!ov::as_type_ptr<KdaScanOp>(node)) continue;
      // All 7 inputs, not just the 5 data tensors: the custom kernel is
      // compiled with one DTYPE, so a mixed f16-constant/f32-data op would
      // misread whichever side disagrees.
      for (size_t i = 0; i < node->get_input_size(); ++i) {
        ov::mark_as_precision_sensitive(node->input(i));
        ++marked;
      }
    }
    for (const auto& result : model->get_results()) {
      const auto& names = result->input_value(0).get_names();
      const bool is_value =
          std::any_of(names.begin(), names.end(), [](const std::string& n) {
            return n.find("value") != std::string::npos ||
                   n.find("wdl") != std::string::npos;
          });
      if (is_value) {
        ov::mark_as_precision_sensitive(result->input(0));
        ++marked;
      }
    }
    CERR << "OpenVINO: kda_safe marked " << marked
         << " inputs precision-sensitive (KDA scan + value head stay fp32).";
  }

  ov::AnyMap device_config;
  if (!is_gpu_device && want_fp16) {
    CERR << "OpenVINO: ignoring fp16 -- it only applies to GPU devices.";
  }
  if (is_gpu_device) {
    device_config[ov::hint::performance_mode.name()] =
        ov::hint::PerformanceMode::LATENCY;
    device_config[ov::hint::execution_mode.name()] =
        ov::hint::ExecutionMode::PERFORMANCE;
    // lc0 evaluates one position (or a small minibatch) at a time, so a
    // single inference stream matches the LATENCY intent: parallel streams
    // only add scheduling overhead and per-stream memory with no throughput
    // to gain at batch~1.
    device_config[ov::num_streams.name()] = ov::streams::Num(1);
    // Let the GPU plugin compile kernels across all host threads. The default
    // is conservative and shows up here as multi-minute first-run compiles on
    // a KDA net's large custom-op graph; this only affects compile time, not
    // inference correctness or the LATENCY hint above.
    device_config[ov::compilation_num_threads.name()] = 0;

    if (!is_ir_model_) {
      // One config fragment per custom-op TYPE found in the model, not per
      // node instance -- a compiled model has exactly one kernel binary per
      // <CustomLayer name="..."> type, shared by every instance of that op
      // (channels/se_filters, like KDA's heads/key_dim/value_dim, are
      // uniform across a net's whole tower, so the first instance's dims
      // speak for all of them). KdaScanOp and SEResidualOp are different
      // type names, so a kda-hybrid net needs -- and gets -- both fragments
      // concatenated into one CONFIG_FILE (OpenVINO's GPU plugin loads
      // sibling <CustomLayer> elements with no wrapper needed; confirmed
      // against the plugin's own parser, which walks
      // document_element()/next_sibling()).
      // One <CustomLayer> per op TYPE, generated from the dimensions of the
      // first instance found -- the kernel is compiled with those baked in
      // as -D defines, so every other instance of that type silently runs
      // the first one's geometry. That is true of the nets converter.cc
      // emits (one uniform encoder stack, one uniform residual tower), but
      // it is an assumption about the net rather than something the config
      // format enforces, so check it instead of trusting it: a net mixing
      // block sizes would otherwise produce wrong math with nothing logged.
      std::vector<std::string> config_fragments;
      std::shared_ptr<KdaScanOp> first_kda;
      for (const auto& node : model->get_ops()) {
        auto kda = ov::as_type_ptr<KdaScanOp>(node);
        if (!kda) continue;
        if (!first_kda) {
          first_kda = kda;
          continue;
        }
        if (kda->heads() != first_kda->heads() ||
            kda->key_dim() != first_kda->key_dim() ||
            kda->value_dim() != first_kda->value_dim() ||
            kda->direction_count() != first_kda->direction_count() ||
            kda->directions() != first_kda->directions()) {
          throw Exception(
              "OpenVINO: this net's KDA layers do not all share the same "
              "geometry ('" + kda->get_friendly_name() + "' differs from '" +
              first_kda->get_friendly_name() +
              "'), but the GPU custom-layer config can only describe one.");
        }
      }
      if (first_kda) {
        // With kda_safe the scan's tensors stay f32, so the kernel must be
        // compiled to read f32 -- half would misinterpret every buffer.
        config_fragments.push_back(WriteKdaScanGpuConfig(
            first_kda->heads(), first_kda->key_dim(), first_kda->value_dim(),
            first_kda->direction_count(), first_kda->directions(),
            want_fp16 && !kda_safe));
      }

      std::shared_ptr<SEResidualOp> first_se;
      for (const auto& node : model->get_ops()) {
        auto se = ov::as_type_ptr<SEResidualOp>(node);
        if (!se) continue;
        if (!first_se) {
          first_se = se;
          continue;
        }
        if (se->channels() != first_se->channels() ||
            se->se_filters() != first_se->se_filters() ||
            se->activation() != first_se->activation()) {
          throw Exception(
              "OpenVINO: this net's SE residual blocks do not all share the "
              "same geometry ('" + se->get_friendly_name() +
              "' differs from '" + first_se->get_friendly_name() +
              "'), but the GPU custom-layer config can only describe one.");
        }
      }
      if (first_se) {
        config_fragments.push_back(WriteSEResidualGpuConfig(
            first_se->channels(), first_se->se_filters(),
            first_se->activation(), want_fp16));
      }
      if (!config_fragments.empty()) {
        device_config["CONFIG_FILE"] =
            WriteMergedGpuConfig(config_fragments).string();
        // This model carries a per-shape-JIT'd custom layer, so confine the
        // search to a fixed ladder of batch shapes. Powers of two above
        // min_batch: few enough to warm in one startup pass, and since lc0's
        // batches cluster near MinibatchSize rather than spreading uniformly,
        // the padding they cost in practice is much less than the worst-case
        // 2x. Overridable for measurement.
        if (options.GetOrDefault<bool>("bucket_batches", true)) {
          for (int b = min_batch_; b < kMaxBatchSize; b *= 2) {
            batch_buckets_.push_back(b);
          }
          batch_buckets_.push_back(kMaxBatchSize);
        }
      }
    }

    device_config[ov::hint::inference_precision.name()] =
        want_fp16 ? ov::element::f16 : ov::element::f32;
  }

  if (profile_) device_config[ov::enable_profiling.name()] = true;

  // Model caching skips the multi-second kernel JIT on every run after the
  // first. Off by default for two reasons, both learned the hard way:
  //
  //   - It is *unsafe* with a custom GPU layer (KdaScanOp, SEResidualOp, or
  //     both). The cache key is computed from the model and the plugin
  //     config; it does not cover the CONFIG_FILE's OpenCL source. So the
  //     plugin happily restores a blob that does not match the custom
  //     kernel, and dereferences null inside the plugin on the first infer.
  //     Cold cache ran fine at 233 nps; the very next run crashed with
  //     0xC0000005. Every run after the first, on every net using a custom
  //     layer.
  //   - A relative default path silently creates a cache directory in
  //     whatever directory lc0 happens to be started from.
  //
  // So: opt in explicitly, and never for a model carrying a custom layer.
  const std::string cache_dir =
      options.GetOrDefault<std::string>("cache_dir", "");
  if (!cache_dir.empty()) {
    if (device_config.count("CONFIG_FILE")) {
      CERR << "OpenVINO: ignoring cache_dir -- this net uses a custom GPU "
              "kernel (KdaScanOp and/or SEResidualOp), and the model cache "
              "does not key on the custom-layer source, so a restored blob "
              "crashes the plugin.";
    } else {
      try {
        std::filesystem::create_directories(cache_dir);
        core_.set_property(ov::cache_dir(cache_dir));
      } catch (const std::exception& e) {
        CERR << "OpenVINO: could not use cache_dir '" << cache_dir
             << "': " << e.what();
      }
    }
  }

  compiled_model_ = core_.compile_model(model, device, device_config);
  CERR << "OpenVINO model compiled successfully on " << device
       << (want_fp16 && is_gpu_device ? " (FP16)" : " (FP32)") << ".";

  // compile_model() does not actually build the device kernels: the GPU
  // plugin JITs them lazily on first execution, through the Intel graphics
  // compiler. Measured on 791556.pb.gz with an empty driver cache, that is
  // ~9.8s before the first evaluation returns, versus ~1.3s once the
  // driver's on-disk cache (%LOCALAPPDATA%\NEO\neo_compiler_cache) has the
  // binaries. Left alone that cost lands inside the first search.
  //
  // Engine::SetPosition runs UpdateBackendConfig -- so this constructor --
  // on the `position` command, before `go`, and StrictUciTiming restarts
  // the clock at `go`, so paying it here is at worst the same total work
  // and at best entirely off the clock. It also gets it over with in one
  // deterministic block instead of stuttering the first search.
  //
  // One inference is enough for the native path: OpenVINO's own kernels
  // are shape-agnostic here (an empty cache filled with only ~25 entries
  // across a whole varying-batch search), so warming one shape warms them
  // all. That is NOT true of custom SimpleGPU layers, which are
  // specialized per concrete shape -- see the ReplaceSqueezeExcite note.
  if (options.GetOrDefault<bool>("warmup", is_gpu_device)) {
    try {
      auto io = GetInputsOutputs();
      // With a custom layer every bucket is a separate JIT, so warm them
      // all -- warming only one shape would leave the rest to stall the
      // search exactly as before. Without one, a single shape suffices.
      std::vector<int> warm_shapes = batch_buckets_;
      if (warm_shapes.empty()) warm_shapes.push_back(min_batch_);
      const auto warm_start = std::chrono::steady_clock::now();

      for (const int shape : warm_shapes) {
      const size_t nb = static_cast<size_t>(shape);
      std::memset(io->input_val_mem_, 0,
                  nb * kInputPlanes * 64 * sizeof(float));

      // Bind exactly what ComputeBlocking binds, so warmup compiles the
      // kernels the search will actually use rather than a variant of them.
      if (is_ir_model_) {
        ov::Tensor exact(io->input_tensor_.get_element_type(),
                         {nb, kInputPlanes, 8, 8});
        std::memcpy(exact.data<float>(), io->input_val_mem_,
                    nb * kInputPlanes * 64 * sizeof(float));
        io->infer_request_.set_input_tensor(exact);
      } else {
        io->infer_request_.set_input_tensor(ov::Tensor(
            io->input_tensor_, {0, 0, 0, 0}, {nb, kInputPlanes, 8, 8}));
      }
      io->infer_request_.set_tensor(
          ports_.policy_port,
          ov::Tensor(ov::element::f32, {nb, InputsOutputs::kPolicyWidth},
                     io->policy_.data()));
      io->infer_request_.set_tensor(
          ports_.value_port,
          ov::Tensor(ov::element::f32,
                     {nb, capabilities_.has_wdl() ? InputsOutputs::kValueWidth
                                                  : size_t{1}},
                     io->value_.data()));
      if (capabilities_.has_mlh()) {
        io->infer_request_.set_tensor(
            ports_.mlh_port,
            ov::Tensor(ov::element::f32, {nb, 1}, io->moves_left_.data()));
      }

      io->infer_request_.infer();
      }

      ReleaseInputsOutputs(std::move(io));
      const auto warm_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - warm_start)
              .count();
      CERR << "OpenVINO warmup: built kernels for " << warm_shapes.size()
           << " batch shape(s) in " << warm_ms << "ms.";
    } catch (const std::exception& e) {
      // Warmup is an optimization, never a reason to fail to start: if it
      // throws, the same work just happens lazily during the first search.
      CERR << "OpenVINO: warmup inference failed (" << e.what()
           << ") -- continuing; kernels will build on first use.";
    }
  }
}

void OpenVinoNetwork::DumpProfile() {
  std::lock_guard<std::mutex> lock(profile_mutex_);
  if (profile_infers_ == 0) {
    CERR << "OpenVINO profile: no usable counters were collected ("
         << profile_failures_ << " inferences had none available).";
    return;
  }

  uint64_t total_us = 0;
  for (const auto& [type, entry] : profile_by_type_) {
    (void)type;
    total_us += entry.us;
  }
  if (total_us == 0) {
    CERR << "OpenVINO profile: counters came back all-zero -- the plugin "
            "may not support per-op profiling on this device.";
    return;
  }

  // Sorted views. std::map gives name order, which is useless for finding
  // the hot kernels.
  auto by_time = [](const auto& a, const auto& b) {
    return a.second.us > b.second.us;
  };
  std::vector<std::pair<std::string, ProfileEntry>> types(
      profile_by_type_.begin(), profile_by_type_.end());
  std::sort(types.begin(), types.end(), by_time);
  std::vector<std::pair<std::string, ProfileEntry>> nodes(
      profile_by_node_.begin(), profile_by_node_.end());
  std::sort(nodes.begin(), nodes.end(), by_time);

  const double infers = static_cast<double>(profile_infers_);
  CERR << "";
  if (profile_failures_ > 0) {
    CERR << "OpenVINO profile: " << profile_failures_
         << " inferences reported no counters and were skipped.";
  }
  CERR << "OpenVINO profile over " << profile_infers_ << " inferences ("
       << (total_us / infers / 1000.0) << " ms of op time per inference):";
  CERR << "";
  CERR << "  by kernel implementation:";
  for (size_t i = 0; i < types.size() && i < 15; ++i) {
    const auto& [type, entry] = types[i];
    CERR << "    " << (100.0 * entry.us / total_us) << "%  "
         << (entry.us / infers / 1000.0) << " ms  " << entry.count / infers
         << " ops  " << type;
  }

  // Reference kernels are the plugin's unoptimized fallbacks. They still
  // run on the device -- this is not CPU fallback -- but a graph that
  // spends most of its time here is being executed by the slow path and is
  // usually better fixed by fusing ops away than by tuning anything.
  uint64_t ref_us = 0;
  for (const auto& [type, entry] : profile_by_type_) {
    if (type.find("_ref") != std::string::npos) ref_us += entry.us;
  }
  CERR << "";
  CERR << "  reference (unoptimized) kernels: " << (100.0 * ref_us / total_us)
       << "% of op time";

  CERR << "";
  CERR << "  hottest individual ops:";
  for (size_t i = 0; i < nodes.size() && i < 12; ++i) {
    const auto& [name, entry] = nodes[i];
    CERR << "    " << (entry.us / infers / 1000.0) << " ms  "
         << entry.exec_type << "  " << name;
  }
  CERR << "";
}

std::unique_ptr<Network> MakeOpenVinoNetwork(
    const std::optional<WeightsFile>& w, const OptionsDict& options) {
  if (!w) {
    throw Exception("The openvino backend requires a network file.");
  }
  return std::make_unique<OpenVinoNetwork>(*w, options);
}

std::unique_ptr<Network> MakeOpenVinoNetworkAuto(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  ov::Core core;
  auto devices = core.get_available_devices();
  OptionsDict opts = options;
  if (std::find(devices.begin(), devices.end(), "GPU") != devices.end()) {
    opts.Set("device", std::string("GPU"));
  } else {
    opts.Set("device", std::string("CPU"));
  }
  return MakeOpenVinoNetwork(weights, opts);
}

// Priority well below sycl (130-132), cudnn/dx12 (120), and cuda (102-104):
// measured GPU throughput here is currently ~5-8x slower than SYCL on this
// KDA-hybrid net (see docs/inference-backends-handoff.md), likely from
// OpenVINO serializing the KDA recurrence's ONNX Scan op into per-token
// dispatch, where SYCL runs it as one fused kernel. Still above blas/eigen
// (49-50): GPU execution should beat a CPU BLAS fallback when nothing
// faster is available. Do not raise this without re-measuring -- a wrong
// auto-selection would silently make search slower on any machine with
// both this and sycl/cuda/dx12 available.
REGISTER_NETWORK("openvino", MakeOpenVinoNetwork, 45)
REGISTER_NETWORK("openvino-auto", MakeOpenVinoNetworkAuto, 46)

}  // namespace openvino_backend
}  // namespace lczero
