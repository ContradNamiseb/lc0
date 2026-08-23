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
#pragma once

#include <vector>

#include <openvino/openvino.hpp>

#include "neural/network.h"

namespace lczero {
namespace openvino_backend {

// Mirrors sycl/inputs_outputs.h: one InputsOutputs owns everything a single
// in-flight computation touches, allocated once at construction for the
// backend's fixed max batch size and never again per-call. Non-copyable and
// non-movable for the same reason the SYCL one is -- an ov::InferRequest can
// hold a live view over input_tensor_'s memory, so nothing here is safe to
// duplicate or leave in a moved-from state mid-inference.
struct InputsOutputs {
  InputsOutputs(const InputsOutputs&) = delete;
  InputsOutputs& operator=(const InputsOutputs&) = delete;
  InputsOutputs(InputsOutputs&&) = delete;
  InputsOutputs& operator=(InputsOutputs&&) = delete;

  InputsOutputs(int max_batch_size, ov::InferRequest infer_request)
      : policy_(static_cast<size_t>(max_batch_size) * kPolicyWidth),
        value_(static_cast<size_t>(max_batch_size) * kValueWidth),
        moves_left_(static_cast<size_t>(max_batch_size)),
        infer_request_(std::move(infer_request)) {
    // Pre-allocate the fixed-capacity input tensor once. AddInput() writes
    // into it directly; ComputeBlocking() sets a dynamic-batch *view* over
    // the same memory rather than allocating a new tensor per call.
    input_tensor_ = ov::Tensor(
        ov::element::f32,
        {static_cast<size_t>(max_batch_size), kInputPlanes, 8, 8});
    input_val_mem_ = input_tensor_.data<float>();

    op_policy_mem_ = policy_.data();
    op_value_mem_ = value_.data();
    op_moves_left_mem_ = moves_left_.data();
  }

  ~InputsOutputs() = default;

  // Declaration order matters here, and it is the opposite of what reads
  // naturally. ComputeBlocking() hands the request a view over
  // input_tensor_'s memory via set_input_tensor(), so the request must be
  // torn down while that memory is still alive. Members are destroyed in
  // reverse declaration order, so input_tensor_ is declared *before*
  // infer_request_ to make the request go first.
  ov::Tensor input_tensor_;
  float* input_val_mem_ = nullptr;

  // Output storage this struct owns, matching what sycl/inputs_outputs.h
  // does. ComputeBlocking() binds these as the request's output tensors via
  // set_tensor() before inferring, so the plugin writes results straight in
  // here. The obvious alternative -- reading back whatever get_tensor()
  // returns after infer() -- is not safe on the GPU plugin, where that
  // memory is not dependably host-readable: the reported size is correct
  // but the pointer itself can fault, which crashed the search.
  static constexpr size_t kPolicyWidth = 1858;
  static constexpr size_t kValueWidth = 3;  // WDL; scalar value uses [0].

  std::vector<float> policy_;
  std::vector<float> value_;
  std::vector<float> moves_left_;

  ov::InferRequest infer_request_;

  // Point into the owned vectors above, set once at construction and never
  // reseated, so they stay valid for this object's whole lifetime.
  const float* op_policy_mem_ = nullptr;
  const float* op_value_mem_ = nullptr;
  const float* op_moves_left_mem_ = nullptr;
};

}  // namespace openvino_backend
}  // namespace lczero
