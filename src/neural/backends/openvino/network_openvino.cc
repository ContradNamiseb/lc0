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
// FCLayer/SELayer/EncoderBlock classes do -- with one exception, KdaScanOp
// (kda_scan_op.h/.cc), a hand-written fused op that replaces the ONNX-Scan
// -derived TensorIterator OpenVINO would otherwise use for the KDA
// recurrence, which profiling showed was >98% of total inference time.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <openvino/core/op_extension.hpp>
#include <openvino/openvino.hpp>
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

  int GetMiniBatchSize() const override { return kMaxBatchSize; }
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
  // than ConvertWeightsToOnnx. An IR-loaded graph never contains KdaScanOp
  // (nothing matches ReplaceKdaScan's TensorIterator pattern in the first
  // place), so it skips the GPU custom-layer kernel setup -- but it still
  // needs the forced-f32 precision hint, for its own separate reason. See
  // the device_config block in the constructor.
  bool is_ir_model_ = false;
  // Mirrors the cuda backend's min_batch: it never evaluates fewer
  // than this many positions, padding the batch out instead. There
  // it is about output variance on tiny batches; here it also keeps
  // the GPU plugin off the degenerate small-batch path.
  int min_batch_ = 4;

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
  const int infer_batch = std::max(batch_size_, network_->min_batch());
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

// Writes the GPU custom-layer kernel source and its CONFIG_FILE XML for
// this model's specific KDA dims (heads/key_dim/value_dim are static per
// net but not portable across different net shapes, so the kernel is
// JIT-compiled per model rather than shipped pre-built). Returns the XML
// path to hand to ov::Core::set_property("GPU", {{"CONFIG_FILE", ...}}).
// Must be called before compile_model().
std::filesystem::path WriteKdaScanGpuConfig(int heads, int key_dim,
                                            int value_dim,
                                            int direction_count,
                                            const std::vector<int>& directions,
                                            bool fp16) {
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "lc0_openvino_kda_scan";
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    throw Exception("OpenVINO: could not create temp dir for the KdaScan "
                    "GPU kernel config: " + ec.message());
  }

  fs::path cl_path = dir / "kda_scan.cl";
  std::ofstream cl(cl_path, std::ios::trunc);
  // Generated from the single canonical definition rather than hand-copied
  // into the kernel string -- see kda_scan_kernel_source.h.
  cl << "__constant uchar kDirectionTable[8][64] = {\n";
  for (int dir = 1; dir <= 8; ++dir) {
    cl << "  {";
    for (int token = 0; token < 64; ++token) {
      if (token) cl << ",";
      cl << KdaSquareForToken(dir, token);
    }
    cl << "}" << (dir < 8 ? "," : "") << "\n";
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

  fs::path xml_path = dir / ("kda_scan_" + std::to_string(heads) + "x" +
                             std::to_string(key_dim) + "x" +
                             std::to_string(value_dim) + "_" +
                             (fp16 ? "fp16" : "fp32") + ".xml");
  std::ofstream xml(xml_path, std::ios::trunc);
  // Source filename is resolved relative to the CONFIG_FILE's own
  // directory (not the executable, despite what the docs say, and not
  // usable as an absolute path -- it gets concatenated onto that
  // directory regardless). Both files are written into the same dir
  // above, so a bare filename is enough.
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
  xml.close();
  return xml_path;
}

}  // namespace

OpenVinoNetwork::OpenVinoNetwork(const WeightsFile& weights,
                                 const OptionsDict& options) {
  std::string device = options.GetOrDefault<std::string>("device", "GPU");
  min_batch_ = std::min(options.GetOrDefault<int>("min_batch", 4),
                        kMaxBatchSize);
  is_cpu_ = (device == "CPU");

  const auto& format = weights.format().network_format();
  capabilities_.input_format = format.input();
  capabilities_.output_format = format.output();
  capabilities_.moves_left = format.moves_left();

  CERR << "Initializing OpenVINO backend on device [" << device << "]...";

  // ir_path opts into loading a pre-converted OpenVINO IR (.xml/.bin) file
  // directly, bypassing ConvertWeightsToOnnx entirely -- see
  // lc0-training/tf/net_to_openvino_ir.py.
  const std::string ir_path = options.GetOrDefault<std::string>("ir_path", "");
  is_ir_model_ = !ir_path.empty();

  std::shared_ptr<ov::Model> model;
  if (is_ir_model_) {
    CERR << "Loading pre-converted OpenVINO IR from: " << ir_path;
    model = core_.read_model(ir_path);
  } else {
    WeightsToOnnxConverterOptions converter_options;
    converter_options.opset = 17;
    converter_options.data_type =
        WeightsToOnnxConverterOptions::DataType::kFloat32;
    converter_options.policy_head = "vanilla";
    converter_options.value_head = "winner";

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
  {
    ov::pass::Manager manager;
    manager.register_pass<ReplaceKdaScan>();
    manager.run_passes(model);
  }

  // Default fp16 to true on GPU for maximum throughput (2.3x-2.4x speedup)
  const bool want_fp16 = options.GetOrDefault<bool>("fp16", device == "GPU");
  profile_ = options.GetOrDefault<bool>("profile", false);

  ov::AnyMap device_config;
  if (device != "GPU" && want_fp16) {
    CERR << "OpenVINO: ignoring fp16 -- it only applies to device=GPU.";
  }
  if (device == "GPU") {
    device_config[ov::hint::performance_mode.name()] =
        ov::hint::PerformanceMode::LATENCY;
    device_config[ov::hint::execution_mode.name()] =
        ov::hint::ExecutionMode::PERFORMANCE;

    if (!is_ir_model_) {
      for (const auto& node : model->get_ops()) {
        auto kda = ov::as_type_ptr<KdaScanOp>(node);
        if (!kda) continue;
        auto xml_path = WriteKdaScanGpuConfig(
            kda->heads(), kda->key_dim(), kda->value_dim(),
            kda->direction_count(), kda->directions(), want_fp16);
        device_config["CONFIG_FILE"] = xml_path.string();
        break;
      }
    }

    device_config[ov::hint::inference_precision.name()] =
        want_fp16 ? ov::element::f16 : ov::element::f32;
  }

  if (profile_) device_config[ov::enable_profiling.name()] = true;

  // Model caching skips the multi-second kernel JIT on every run after the
  // first. Off by default for two reasons, both learned the hard way:
  //
  //   - It is *unsafe* with the KdaScanOp custom layer. The cache key is
  //     computed from the model and the plugin config; it does not cover
  //     the CONFIG_FILE's OpenCL source. So the plugin happily restores a
  //     blob that does not match the custom kernel, and dereferences null
  //     inside the plugin on the first infer. Cold cache ran fine at 233
  //     nps; the very next run crashed with 0xC0000005. Every run after
  //     the first, on every KDA net.
  //   - A relative default path silently creates a cache directory in
  //     whatever directory lc0 happens to be started from.
  //
  // So: opt in explicitly, and never for a model carrying a custom layer.
  const std::string cache_dir =
      options.GetOrDefault<std::string>("cache_dir", "");
  if (!cache_dir.empty()) {
    if (device_config.count("CONFIG_FILE")) {
      CERR << "OpenVINO: ignoring cache_dir -- this net uses the KdaScanOp "
              "custom GPU kernel, and the model cache does not key on the "
              "custom-layer source, so a restored blob crashes the plugin.";
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
       << (want_fp16 && device == "GPU" ? " (FP16)" : " (FP32)") << ".";
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
