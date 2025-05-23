// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/core/xpu_cuda_stream.h"
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#else
namespace phi {
class XPUCUDAStream {};
}  // namespace phi
#endif

namespace py = pybind11;

namespace paddle {
namespace platform {
#ifdef PADDLE_WITH_XPU
XPUStream get_current_stream(int device_id = -1);
#endif
}  // namespace platform
namespace pybind {

void BindXpuStream(py::module* m);

}  // namespace pybind
}  // namespace paddle
