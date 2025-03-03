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

#include <glog/logging.h>
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/DeviceType.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/core/stream.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/fake_torch/c10/cuda/CUDAStream.h"

namespace torch {

class Event {
 public:
  Event() { cudaEventCreate(&cuda_event_); }
  ~Event() { cudaEventDestroy(cuda_event_); }
  explicit Event(const DeviceType _device_type) {
    LOG(FATAL) << "Not implemented";
  }
  void record(const c10::Stream& stream) { LOG(FATAL) << "Not implemented"; }
  // void record(const c10::cuda::CUDAStream& stream) {
  //   LOG(FATAL) << "Not implemented";
  // }
  void record(const cudaStream_t& stream) {
    cudaEventRecord(cuda_event_, stream);
  }

  cudaEvent_t cuda_event() const { return cuda_event_; }

 private:
  cudaEvent_t cuda_event_;
};

}  // namespace torch
