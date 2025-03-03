// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/TensorOptions.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/types.h"

#include "glog/logging.h"

namespace torch {

class IntArrayRef {
 public:
  IntArrayRef() { LOG(FATAL) << "IntArrayRef() is not allowed!"; }
  IntArrayRef(const std::initializer_list<int64_t>& Vec) {
    LOG(FATAL) << "IntArrayRef() is not allowed!";
  }
};

class MemoryFormat {};

Tensor empty(IntArrayRef, c10::TensorOptions) {
  LOG(FATAL) << "Tensor::empty() is not allowed!";
  return *(Tensor*)nullptr;  // NOLINT
}
}  // namespace torch
