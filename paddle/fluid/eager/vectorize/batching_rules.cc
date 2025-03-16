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

/*traceable interface*/
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_grad_functions.h"

/*non-traceable interface*/
#include "paddle/phi/api/backward/backward_api_base.h"
#include "paddle/phi/api/include/api.h"

#include "paddle/fluid/eager/vectorize/vmap_transforms.h"
#include "paddle/phi/core/batched_tensor.h"

namespace paddle {
namespace vmap {

/***********************************
Utility functions for batching rules
************************************/
paddle::Tensor mT(const paddle::Tensor& x) {
  // Alias of matrix_transpose_ad_func
  int ndim = x.dims().size();
  if (ndim <= 1) {
    return x;
  }
  std::vector<int> perm(ndim, 0);
  std::iota(perm.begin(), perm.end(), 0);
  std::swap(perm[ndim - 1], perm[ndim - 2]);
  return ::transpose_ad_func(x, perm);
}

/*****************************************************
Batching rule for traceable function(xx_ad_func) below
******************************************************/
paddle::Tensor dot_batching_rule(const paddle::Tensor& x,
                                 const paddle::Tensor& y) {
  auto x_batched = phi::isBatchedTensor(x);
  auto y_batched = phi::isBatchedTensor(y);
  PD_CHECK(/*logical*/ x.dims().size() <= 2 && /*logical*/ y.dims().size() <= 2,
           "dot(x, y): Shape mismatch: vector "
           "(got `x` of size %s) and vector (got `y` of size %s)",
           x.dims(),
           y.dims());

  // See Note [Batching rules for matmul-like operators] for why we have cases
  if (x_batched && !y_batched) {
    // x_physical: [..., K], y_physical: [K]
    // View the tensors as [..., 1, K] and [K], perform matmul, and unsqueeze.
    auto x_physical = MultiBatchVmapTransform::logicalToPhysical(x);
    auto result =
        matmul_ad_func(::unsqueeze_ad_func(x_physical.tensor(), {-2}), y);
    return x_physical.getPhysicalToLogicalMap().apply(
        squeeze_ad_func(result, {-1}));
  } else if (!x_batched && y_batched) {
    // x_physical: [K], y_physical: [..., K]
    // View the tensors as [K] and [..., K, 1], perform matmul, and unsqueeze.
    auto y_physical = MultiBatchVmapTransform::logicalToPhysical(y);
    auto result =
        matmul_ad_func(x, ::unsqueeze_ad_func(y_physical.tensor(), {-1}));
    return y_physical.getPhysicalToLogicalMap().apply(
        squeeze_ad_func(result, {-1}));
  } else if (x_batched && y_batched) {
    // x_physical: [..., K], y_physical: [..., K]
    // View the tensors as [..., 1, K] and [..., K, 1], perform matmul, and
    // unsqueeze.
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({x, y});
    auto result =
        matmul_ad_func(::unsqueeze_ad_func(physical_args[0].tensor(), {-2}),
                       ::unsqueeze_ad_func(physical_args[1].tensor(), {-1}));
    return physical_args[0].getPhysicalToLogicalMap().apply(
        squeeze_ad_func(result, {-1, -1}));
  } else {
    PD_CHECK(false, "either x or y must be a BatchedTensor");
  }
}

paddle::Tensor matmul_batching_rule(const paddle::Tensor& x,
                                    const paddle::Tensor& y,
                                    bool transpose_x,
                                    bool transpose_y) {
  auto x_batched = phi::isBatchedTensor(x);
  auto y_batched = phi::isBatchedTensor(y);

  // See Note [Batching rules for matmul-like operators] for why we have cases
  if (x_batched && !y_batched) {
    auto x_physical = MultiBatchVmapTransform::logicalToPhysical(x);
    auto result =
        matmul_ad_func(x_physical.tensor(), y, transpose_x, transpose_y);
    return x_physical.getPhysicalToLogicalMap().apply(result);
  } else if (!x_batched && y_batched) {
    auto y_physical = MultiBatchVmapTransform::logicalToPhysical(y);
    auto result =
        matmul_ad_func(x, y_physical.tensor(), transpose_x, transpose_y);
    return y_physical.getPhysicalToLogicalMap().apply(result);
  } else if (x_batched && y_batched) {
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({x, y});
    auto result = matmul_ad_func(physical_args[0].tensor(),
                                 physical_args[1].tensor(),
                                 transpose_x,
                                 transpose_y);
    return physical_args[0].getPhysicalToLogicalMap().apply(
        squeeze_ad_func(result, {-1, -1}));
  }
  PD_CHECK(false, "either x or y must be a BatchedTensor");
}

void matmul_grad_batching_rule(const paddle::Tensor& x,
                               const paddle::Tensor& y,
                               const paddle::Tensor& grad_out,
                               bool transpose_x,
                               bool transpose_y,
                               paddle::Tensor* grad_x,
                               paddle::Tensor* grad_y) {
  auto x_batched = phi::isBatchedTensor(x);
  auto y_batched = phi::isBatchedTensor(y);
  auto grad_out_batched = phi::isBatchedTensor(grad_out);

  // See Note [Batching rules for matmul-like operators] for why we have cases
  if (grad_x) {
    if (grad_out_batched && !y_batched) {
      auto grad_out_physical =
          MultiBatchVmapTransform::logicalToPhysical(grad_out);
      *grad_x =
          matmul_ad_func(grad_out_physical.tensor(), y, false, !transpose_y);
      *grad_x = grad_out_physical.getPhysicalToLogicalMap().apply(*grad_x);
    } else if (!grad_out_batched && y_batched) {
      auto y_physical = MultiBatchVmapTransform::logicalToPhysical(y);
      *grad_x =
          matmul_ad_func(grad_out, y_physical.tensor(), false, !transpose_y);
      *grad_x = y_physical.getPhysicalToLogicalMap().apply(*grad_x);
    } else if (x_batched && y_batched) {
      auto physical_args =
          MultiBatchVmapTransform::logicalToPhysical({grad_out, y});
      *grad_x = matmul_ad_func(physical_args[0].tensor(),
                               physical_args[1].tensor(),
                               false,
                               !transpose_y);
      *grad_x = physical_args[0].getPhysicalToLogicalMap().apply(
          squeeze_ad_func(*grad_x, {-1, -1}));
    }
    if (transpose_x) {
      *grad_x = mT(*grad_x);
    }
  }
  if (grad_y) {
    if (x_batched && !grad_out_batched) {
      auto x_physical = MultiBatchVmapTransform::logicalToPhysical(x);
      *grad_y =
          matmul_ad_func(x_physical.tensor(), grad_out, !transpose_x, false);
      *grad_y = x_physical.getPhysicalToLogicalMap().apply(*grad_y);
    } else if (!x_batched && grad_out_batched) {
      auto grad_out_physical =
          MultiBatchVmapTransform::logicalToPhysical(grad_out);
      *grad_y =
          matmul_ad_func(x, grad_out_physical.tensor(), !transpose_x, false);
      *grad_y = grad_out_physical.getPhysicalToLogicalMap().apply(*grad_y);
    } else if (x_batched && grad_out_batched) {
      auto physical_args =
          MultiBatchVmapTransform::logicalToPhysical({x, grad_out});
      *grad_y = matmul_ad_func(physical_args[0].tensor(),
                               physical_args[1].tensor(),
                               !transpose_x,
                               false);
      *grad_y = physical_args[0].getPhysicalToLogicalMap().apply(
          squeeze_ad_func(*grad_y, {-1, -1}));
    }
    if (transpose_y) {
      *grad_y = mT(*grad_y);
    }
  }
}

paddle::Tensor scale_batching_rule(const paddle::Tensor& x,
                                   paddle::experimental::Scalar scale,
                                   paddle::experimental::Scalar bias,
                                   bool bias_after_scale) {
  auto* x_batched = phi::unsafeGetBatchedImpl(x);
  auto output_physical =
      ::scale_ad_func(x_batched->value(), scale, bias, bias_after_scale);
  auto old_bdims = x_batched->bdims();
  return phi::makeBatched(output_physical,
                          phi::BatchDims(old_bdims.begin(), old_bdims.end()));
}

paddle::Tensor full_like_batching_rule(const paddle::Tensor& x,
                                       paddle::experimental::Scalar value,
                                       phi::DataType dtype,
                                       paddle::Place place) {
  auto* x_batched = phi::unsafeGetBatchedImpl(x);
  auto output_physical =
      ::full_like_ad_func(x_batched->value(), value, dtype, place);
  auto old_bdims = x_batched->bdims();
  return phi::makeBatched(output_physical,
                          phi::BatchDims(old_bdims.begin(), old_bdims.end()));
}

paddle::Tensor ones_like_batching_rule(const paddle::Tensor& x,
                                       phi::DataType dtype,
                                       paddle::Place place) {
  auto* x_batched = phi::unsafeGetBatchedImpl(x);
  auto output_physical = ::ones_like_ad_func(x_batched->value(), dtype, place);
  auto old_bdims = x_batched->bdims();
  return phi::makeBatched(output_physical,
                          phi::BatchDims(old_bdims.begin(), old_bdims.end()));
}

paddle::Tensor zeros_like_batching_rule(const paddle::Tensor& x,
                                        phi::DataType dtype,
                                        paddle::Place place) {
  auto* x_batched = phi::unsafeGetBatchedImpl(x);
  auto output_physical = ::zeros_like_ad_func(x_batched->value(), dtype, place);
  auto old_bdims = x_batched->bdims();
  return phi::makeBatched(output_physical,
                          phi::BatchDims(old_bdims.begin(), old_bdims.end()));
}

void tanh_grad_batching_rule(const paddle::Tensor& out,
                             const paddle::Tensor& grad_out,
                             paddle::Tensor* grad_x) {
  auto physical_args =
      BroadcastingVmapTransform::logicalToPhysical({out, grad_out});
  const auto& out_physical = physical_args[0].tensor();
  const auto& grad_out_physical = physical_args[1].tensor();
  *grad_x = (1 - out_physical * out_physical) * grad_out_physical;
  *grad_x = physical_args[0].getPhysicalToLogicalMap().apply(*grad_x);
}

paddle::Tensor cast_batching_rule(const paddle::Tensor& x,
                                  phi::DataType dtype) {
  auto* x_batched = phi::unsafeGetBatchedImpl(x);
  auto output_physical = ::cast_ad_func(x_batched->value(), dtype);
  auto old_bdims = x_batched->bdims();
  return phi::makeBatched(output_physical,
                          phi::BatchDims(old_bdims.begin(), old_bdims.end()));
}

/*Simple unary batching rule*/
#define REGISTER_UNARY_AD_FUNC_BATCHING_RULE(func_name, ad_func)              \
  paddle::Tensor func_name##_batching_rule(const paddle::Tensor& x) {         \
    auto* x_batched = phi::unsafeGetBatchedImpl(x);                           \
    auto output_physical = ad_func(x_batched->value());                       \
    auto old_bdims = x_batched->bdims();                                      \
    return phi::makeBatched(                                                  \
        output_physical, phi::BatchDims(old_bdims.begin(), old_bdims.end())); \
  }

REGISTER_UNARY_AD_FUNC_BATCHING_RULE(abs, ::abs_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(acos, ::acos_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(acosh, ::acosh_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(angle, ::angle_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(as_complex, ::as_complex_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(as_real, ::as_real_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(asin, ::asin_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(asinh, ::asinh_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(atan, ::atan_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(atanh, ::atanh_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(assign, ::assign_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(hardswish, ::hardswish_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(bernoulli, ::bernoulli_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(bitwise_not, ::bitwise_not_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(ceil, ::ceil_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(conj, ::conj_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(cos, ::cos_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(cosh, ::cosh_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(digamma, ::digamma_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(erf, ::erf_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(erfinv, ::erfinv_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(exp, ::exp_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(expm1, ::expm1_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(floor, ::floor_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(gammaln, ::gammaln_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(i0, ::i0_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(i0e, ::i0e_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(i1, ::i1_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(i1e, ::i1e_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(imag, ::imag_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(isfinite, ::isfinite_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(isinf, ::isinf_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(isnan, ::isnan_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(lgamma, ::lgamma_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(log, ::log_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(log10, ::log10_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(log1p, ::log1p_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(log2, ::log2_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(logical_not, ::logical_not_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(logsigmoid, ::logsigmoid_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(poisson, ::poisson_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(real, ::real_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(reciprocal, ::reciprocal_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(relu, ::relu_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(relu6, ::relu6_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(round, ::round_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(rsqrt, ::rsqrt_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(sigmoid, ::sigmoid_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(sign, ::sign_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(softsign, ::softsign_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(sin, ::sin_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(sinh, ::sinh_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(sqrt, ::sqrt_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(square, ::square_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(tan, ::tan_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(tanh, ::tanh_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(tanh_shrink, ::tanh_shrink_ad_func)
REGISTER_UNARY_AD_FUNC_BATCHING_RULE(trunc, ::trunc_ad_func)
#undef REGISTER_UNARY_AD_FUNC_BATCHING_RULE

};  // namespace vmap
};  // namespace paddle
