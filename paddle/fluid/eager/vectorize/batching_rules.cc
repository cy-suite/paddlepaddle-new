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

#include "paddle/fluid/eager/vectorize/batching_rules.h"

#include "paddle/fluid/eager/vectorize/vmap_transforms.cc"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/core/batched_tensor.h"

using Tensor = paddle::Tensor;

namespace paddle {
namespace vmap {

Tensor dot_batching_rule(const Tensor& x, const Tensor& y) {
  auto x_batched = phi::isBatchedTensor(x);
  auto y_batched = phi::isBatchedTensor(y);
  PD_CHECK(/*logical*/ x.dims().size() == 1 && /*logical*/ y.dims().size() == 1,
           "dot(x, y): Shape mismatch: vector "
           "(got `x` of size ",
           x.dims().size(),
           ") ",
           "and vector (got `y` of size ",
           y.dims().size(),
           ")");

  // See Note [Batching rules for matmul-like operators] for why we have cases
  if (x_batched && !y_batched) {
    // x_physical: [..., K], y_physical: [K]
    // View the tensors as [..., 1, K] and [K], perform matmul, and unsqueeze.
    auto x_physical = MultiBatchVmapTransform::logicalToPhysical(x);
    auto result = paddle::experimental::matmul(
        paddle::experimental::unsqueeze(x_physical.tensor(), {-2}), y);
    return x_physical.getPhysicalToLogicalMap().apply(
        paddle::experimental::squeeze(result, {-1}));
  } else if (!x_batched && y_batched) {
    // x_physical: [K], y_physical: [..., K]
    // View the tensors as [K] and [..., K, 1], perform matmul, and unsqueeze.
    auto y_physical = MultiBatchVmapTransform::logicalToPhysical(y);
    auto result = paddle::experimental::matmul(
        x, paddle::experimental::unsqueeze(y_physical.tensor(), {-1}));
    return y_physical.getPhysicalToLogicalMap().apply(
        paddle::experimental::squeeze(result, {-1}));
  } else if (x_batched && y_batched) {
    // x_physical: [..., K], y_physical: [..., K]
    // View the tensors as [..., 1, K] and [..., K, 1], perform matmul, and
    // unsqueeze.
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({x, y});
    auto result = paddle::experimental::matmul(
        paddle::experimental::unsqueeze(physical_args[0].tensor(), {-2}),
        paddle::experimental::unsqueeze(physical_args[1].tensor(), {-1}));
    return physical_args[0].getPhysicalToLogicalMap().apply(
        paddle::experimental::squeeze(
            paddle::experimental::squeeze(result, {-1}), {-1}));
  } else {
    PD_CHECK(false, "either x or y must be a BatchedTensor");
  }
}

}  // namespace vmap
}  // namespace paddle
