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

#include "paddle/fluid/eager/vectorize/batching_rules.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_grad_functions.h"
#include "paddle/fluid/eager/vectorize/vmap_transforms.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/core/batched_tensor.h"

using Tensor = paddle::Tensor;

paddle::Tensor matmul_ad_func(const paddle::Tensor& x,
                              const paddle::Tensor& y,
                              bool transpose_x,
                              bool transpose_y);
paddle::Tensor unsqueeze_ad_func(const paddle::Tensor& x,
                                 paddle::experimental::IntArray axis);
paddle::Tensor squeeze_ad_func(const paddle::Tensor& x,
                               paddle::experimental::IntArray axis);

paddle::Tensor tanh_grad_ad_func(const paddle::Tensor& out,
                                 const paddle::Tensor& grad_out);

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
    auto result =
        ::matmul_ad_func(::unsqueeze_ad_func(x_physical.tensor(), {-2}), y);
    return x_physical.getPhysicalToLogicalMap().apply(
        ::squeeze_ad_func(result, {-1}));
  } else if (!x_batched && y_batched) {
    // x_physical: [K], y_physical: [..., K]
    // View the tensors as [K] and [..., K, 1], perform matmul, and unsqueeze.
    auto y_physical = MultiBatchVmapTransform::logicalToPhysical(y);
    auto result =
        ::matmul_ad_func(x, ::unsqueeze_ad_func(y_physical.tensor(), {-1}));
    return y_physical.getPhysicalToLogicalMap().apply(
        ::squeeze_ad_func(result, {-1}));
  } else if (x_batched && y_batched) {
    // x_physical: [..., K], y_physical: [..., K]
    // View the tensors as [..., 1, K] and [..., K, 1], perform matmul, and
    // unsqueeze.
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({x, y});
    auto result =
        ::matmul_ad_func(::unsqueeze_ad_func(physical_args[0].tensor(), {-2}),
                         ::unsqueeze_ad_func(physical_args[1].tensor(), {-1}));
    return physical_args[0].getPhysicalToLogicalMap().apply(
        ::squeeze_ad_func(::squeeze_ad_func(result, {-1}), {-1}));
  } else {
    PD_CHECK(false, "either x or y must be a BatchedTensor");
  }
}

Tensor matmul_batching_rule(const Tensor& x,
                            const Tensor& y,
                            bool transpose_x,
                            bool transpose_y) {
  auto x_batched = phi::isBatchedTensor(x);
  auto y_batched = phi::isBatchedTensor(y);

  PD_CHECK(/*logical*/ x.dims().size() == 2 && /*logical*/ y.dims().size() == 2,
           "mm(x, y): Shape mismatch: expected matrix "
           "(got `x` of size [%s]) ",
           "and matrix (got `y` of size [%s])",
           x.dims(),
           y.dims());

  // See Note [Batching rules for matmul-like operators] for why we have cases
  if (x_batched && !y_batched) {
    auto x_physical = MultiBatchVmapTransform::logicalToPhysical(x);
    auto result =
        ::matmul_ad_func(x_physical.tensor(), y, transpose_x, transpose_y);
    return x_physical.getPhysicalToLogicalMap().apply(result);
  } else if (!x_batched && y_batched) {
    auto y_physical = MultiBatchVmapTransform::logicalToPhysical(y);
    auto result =
        ::matmul_ad_func(x, y_physical.tensor(), transpose_x, transpose_y);
    return y_physical.getPhysicalToLogicalMap().apply(result);
  } else if (x_batched && y_batched) {
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({x, y});
    auto result = ::matmul_ad_func(physical_args[0].tensor(),
                                   physical_args[1].tensor(),
                                   transpose_x,
                                   transpose_y);
    return physical_args[0].getPhysicalToLogicalMap().apply(
        ::squeeze_ad_func(::squeeze_ad_func(result, {-1}), {-1}));
  }
  PD_CHECK(false, "either x or y must be a BatchedTensor");
}

Tensor tanh_batching_rule(const Tensor& x) {
  auto* x_batched = phi::unsafeGetBatchedImpl(x);
  auto output_physical = ::tanh_ad_func(x_batched->value());
  auto old_bdims = x_batched->bdims();
  return phi::makeBatched(output_physical,
                          phi::BatchDims(old_bdims.begin(), old_bdims.end()));
}

Tensor tanh_grad_batching_rule(const Tensor& out, const Tensor& grad_out) {
  auto physical_args =
      BroadcastingVmapTransform::logicalToPhysical({out, grad_out});
  const auto& out_physical = physical_args[0].tensor();
  const auto& grad_out_physical = physical_args[1].tensor();
  auto result = (1 - out_physical * out_physical) * grad_out_physical;
  return physical_args[0].getPhysicalToLogicalMap().apply(result);
}

}  // namespace vmap
}  // namespace paddle
