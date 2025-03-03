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

#include "paddle/phi/common/data_type.h"

namespace c10 {

using ScalarType = phi::DataType;

}

namespace torch {
constexpr auto kInt32 = c10::ScalarType::INT32;
constexpr auto kBool = c10::ScalarType::BOOL;
constexpr auto kFloat32 = c10::ScalarType::FLOAT32;
constexpr auto kByte = c10::ScalarType::INT8;
}  // namespace torch
