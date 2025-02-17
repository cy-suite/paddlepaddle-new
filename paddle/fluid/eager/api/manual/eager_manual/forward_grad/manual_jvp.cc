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
// 1. elementwise functions, e.g. tanh, we can reuse their vjp rules and just
// replace the out_grad with input_tangent
// 2. linear functions with single input, e.g. scale, we can reuse their forward
// functions and just replace the input with input_tangent
// 3. other case, e.g. concat/stack/batch_norm, we need to implement jvp rules
// manually
//
#include "paddle/fluid/eager/api/manual/eager_manual/forward_grad/manual_jvp.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/optional.h"

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/phi/common/int_array.h"

Tensor concat_jvp(const std::vector<Tensor>& x_ts, Scalar axis) {
  std::vector<Tensor> fw_grads;
  for (const Tensor& t : x_ts) {
    if (egr::EagerUtils::nullable_autograd_meta(t)) {
      fw_grads.push_back(t._fw_grad(/*level*/ 0));
    } else {
      fw_grads.push_back(
          paddle::experimental::zeros(t.shape(), t.dtype(), t.place()));
    }
  }
  return concat_ad_func(fw_grads, axis);
}
