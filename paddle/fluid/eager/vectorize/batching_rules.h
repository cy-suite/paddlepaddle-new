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

#include "paddle/fluid/eager/vectorize/vmap_transforms.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/core/batched_tensor.h"

/*****************************************************
Batching rule for traceable function(xx_ad_func) below
******************************************************/
namespace paddle {
namespace vmap {

#define DECLARE_UNARY_AD_FUNC_BATCHING_RULE(func_name) \
  paddle::Tensor func_name##_batching_rule(const paddle::Tensor& x);

DECLARE_UNARY_AD_FUNC_BATCHING_RULE(abs)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(acos)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(acosh)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(angle)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(as_complex)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(as_real)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(asin)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(asinh)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(assign)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(hardswish)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(atan)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(atanh)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(bernoulli)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(bitwise_not)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(ceil)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(conj)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(cos)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(cosh)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(digamma)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(erf)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(erfinv)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(exp)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(expm1)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(floor)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(gammaln)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(i0)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(i0e)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(i1)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(i1e)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(imag)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(isfinite)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(isinf)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(isnan)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(lgamma)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(log)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(log10)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(log1p)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(log2)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(logical_not)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(logsigmoid)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(poisson)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(real)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(reciprocal)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(relu)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(relu6)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(round)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(rsqrt)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(sigmoid)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(sign)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(softsign)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(sin)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(sinh)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(sqrt)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(square)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(tan)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(tanh)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(tanh_shrink)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(trunc)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(full_like)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(ones_like)
DECLARE_UNARY_AD_FUNC_BATCHING_RULE(zeros_like)
#undef DECLARE_UNARY_AD_FUNC_BATCHING_RULE

// forward function, Tensor as return type
paddle::Tensor dot_batching_rule(const paddle::Tensor& x,
                                 const paddle::Tensor& y);
paddle::Tensor matmul_batching_rule(const paddle::Tensor& x,
                                    const paddle::Tensor& y,
                                    bool transpose_x,
                                    bool transpose_y);
paddle::Tensor scale_batching_rule(const paddle::Tensor& x,
                                   paddle::experimental::Scalar scale,
                                   paddle::experimental::Scalar bias,
                                   bool bias_after_scale);
paddle::Tensor full_like_batching_rule(
    const paddle::Tensor& x,
    paddle::experimental::Scalar value,
    phi::DataType dtype = phi::DataType::UNDEFINED,
    paddle::Place place = {});
paddle::Tensor zeros_like_batching_rule(
    const paddle::Tensor& x,
    phi::DataType dtype = phi::DataType::UNDEFINED,
    paddle::Place place = {});
paddle::Tensor ones_like_batching_rule(
    const paddle::Tensor& x,
    phi::DataType dtype = phi::DataType::UNDEFINED,
    paddle::Place place = {});
paddle::Tensor cast_batching_rule(const paddle::Tensor& x, phi::DataType dtype);

// backward function, void ass return type
void tanh_grad_batching_rule(const paddle::Tensor& out,
                             const paddle::Tensor& grad_out,
                             paddle::Tensor* grad_x);
void matmul_grad_batching_rule(const paddle::Tensor& x,
                               const paddle::Tensor& y,
                               const paddle::Tensor& grad_out,
                               bool transpose_x,
                               bool transpose_y,
                               paddle::Tensor* grad_x,
                               paddle::Tensor* grad_y);

};  // namespace vmap
};  // namespace paddle
