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

namespace torch {

enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1,          // CUDA.
  MKLDNN = 2,        // Reserved for explicit MKLDNN
  OPENGL = 3,        // OpenGL
  OPENCL = 4,        // OpenCL
  IDEEP = 5,         // IDEEP.
  HIP = 6,           // AMD HIP
  FPGA = 7,          // FPGA
  MAIA = 8,          // ONNX Runtime / Microsoft
  XLA = 9,           // XLA / TPU
  Vulkan = 10,       // Vulkan
  Metal = 11,        // Metal
  XPU = 12,          // XPU
  MPS = 13,          // MPS
  Meta = 14,         // Meta (tensors with no data)
  HPU = 15,          // HPU / HABANA
  VE = 16,           // SX-Aurora / NEC
  Lazy = 17,         // Lazy Tensors
  IPU = 18,          // Graphcore IPU
  MTIA = 19,         // Meta training and inference devices
  PrivateUse1 = 20,  // PrivateUse1 device
  // NB: If you add more devices:
  //  - Change the implementations of DeviceTypeName and isValidDeviceType
  //    in DeviceType.cpp
  //  - Change the number below
  COMPILE_TIME_MAX_DEVICE_TYPES = 21,
};

constexpr DeviceType kCUDA = DeviceType::CUDA;

}  // namespace torch
