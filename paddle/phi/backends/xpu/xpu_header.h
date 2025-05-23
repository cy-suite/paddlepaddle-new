/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_XPU
#include <map>
#include <string>
#include <unordered_map>

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#ifdef PADDLE_WITH_XPU_BKCL
#include "xpu/bkcl.h"
#endif
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"
#ifdef PADDLE_WITH_XPU_PLUGIN
#include "xpu/plugin.h"
#endif

namespace xpu = baidu::xpu::api;

template <typename T>
class XPUTypeTrait {
 public:
  using Type = T;
};

template <>
class XPUTypeTrait<phi::dtype::float16> {
 public:
  using Type = float16;
};

template <>
class XPUTypeTrait<phi::dtype::bfloat16> {
 public:
  using Type = bfloat16;
};

template <typename T>
class XPUTypeToPhiType {
 public:
  using Type = T;
};

template <>
class XPUTypeToPhiType<float16> {
 public:
  using Type = phi::dtype::float16;
};

template <>
class XPUTypeToPhiType<bfloat16> {
 public:
  using Type = phi::dtype::bfloat16;
};

// XPUCopyTypeTrait is the same as XPUTypeTrait except for double, int16_t, and
// uint8_t. Used for ops that simply copy data and do not need to calculate
template <typename T>
class XPUCopyTypeTrait {
 public:
  using Type = T;
};

template <>
class XPUCopyTypeTrait<phi::dtype::float16> {
 public:
  using Type = float16;
};

template <>
class XPUCopyTypeTrait<phi::dtype::bfloat16> {
 public:
  using Type = bfloat16;
};

template <>
class XPUCopyTypeTrait<double> {
 public:
  using Type = int64_t;
};

template <>
class XPUCopyTypeTrait<int16_t> {
 public:
  using Type = float16;
};

template <>
class XPUCopyTypeTrait<uint8_t> {
 public:
  using Type = int8_t;
};

#endif
