// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/common/ddim.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/unsqueeze_kernel.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedAdalnScaleResidualXpuKernel(
    const Context& ctx,
    const DenseTensor& input1,
    const DenseTensor& input2,
    const DenseTensor& unsqueezed1,
    const DenseTensor& unsqueezed2,
    const DenseTensor& unsqueezed3,
    const paddle::optional<DenseTensor>& ln_weight,
    const paddle::optional<DenseTensor>& ln_bias,
    const int begin_norm_axis,
    const float epsilon,
    const float scale_op_weight,
    const float scale_op_bias,
    const bool bias_after_scale,
    DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto* input1_data = reinterpret_cast<const XPUType*>(input1.data<T>());
  auto* input2_data = reinterpret_cast<const XPUType*>(input2.data<T>());
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));

  int r = xpu::SUCCESS;
  auto xpu_ctx = static_cast<const phi::XPUContext*>(&ctx);
  auto input1_shape = input1.dims();
  int m = 1;
  int n = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    m *= input1_shape[i];
  }
  for (int i = begin_norm_axis; i < input1_shape.size(); i++) {
    n *= input1_shape[i];
  }

  // broadcast_mul
  r = baidu::xpu::api::broadcast_mul(
      xpu_ctx->x_context(),
      input1_data,
      reinterpret_cast<const XPUType*>(unsqueezed1.data<T>()),
      out_data,
      common::vectorize<int>(input1.dims()),
      common::vectorize<int>(unsqueezed1.dims()));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

  // broadcast_add
  r = baidu::xpu::api::broadcast_add(xpu_ctx->x_context(),
                                     out_data,
                                     input2_data,
                                     out_data,
                                     common::vectorize<int>(input1.dims()),
                                     common::vectorize<int>(input2.dims()));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

  // LayerNorm
  const float* ln_scale_data =
      ln_weight.get_ptr() ? ln_weight.get_ptr()->data<float>() : NULL;
  const float* ln_bias_data =
      ln_bias.get_ptr() ? ln_bias.get_ptr()->data<float>() : NULL;
  r = baidu::xpu::api::layer_norm(xpu_ctx->x_context(),
                                  out_data,
                                  out_data,
                                  m,
                                  n,
                                  epsilon,
                                  ln_scale_data,
                                  ln_bias_data,
                                  nullptr,
                                  nullptr);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");

  // Scale
  DenseTensor scaled_out;
  scaled_out.Resize(unsqueezed2.dims());
  ctx.template Alloc<T>(&scaled_out);
  r = baidu::xpu::api::scale(
      xpu_ctx->x_context(),
      reinterpret_cast<const XPUType*>(unsqueezed2.data<T>()),
      reinterpret_cast<XPUType*>(scaled_out.data<T>()),
      unsqueezed2.numel(),
      bias_after_scale,
      scale_op_weight,
      scale_op_bias);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");

  // broadcast_mul
  r = baidu::xpu::api::broadcast_mul(
      xpu_ctx->x_context(),
      out_data,
      reinterpret_cast<XPUType*>(scaled_out.data<T>()),
      out_data,
      common::vectorize<int>(out->dims()),
      common::vectorize<int>(scaled_out.dims()));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

  // broadcast_add
  r = baidu::xpu::api::broadcast_add(
      xpu_ctx->x_context(),
      out_data,
      reinterpret_cast<const XPUType*>(unsqueezed3.data<T>()),
      out_data,
      common::vectorize<int>(out->dims()),
      common::vectorize<int>(unsqueezed3.dims()));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_adaLN_scale_residual_xpu_kernel,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedAdalnScaleResidualXpuKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
