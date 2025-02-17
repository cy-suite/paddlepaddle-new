// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

// Manual jvp rules for dygraph forward_autograd
// there are three kind of jvp rules
// 1. Elementwise functions, e.g. tanh, we can reuse their vjp rules and just
// replace the out_grad with input_tangent
// 2. Linear functions with single input, e.g. scale, we can reuse their forward
// functions and just replace the input with input_tangent
// 3. Other case, e.g. concat/stack/batch_norm, we need to implement jvp rules
// manually
//
#pragma once

#include <vector>
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"

using Scalar = paddle::Scalar;
using Tensor = paddle::Tensor;

paddle::Tensor concat_jvp(const std::vector<Tensor>& x_ts, Scalar axis);
