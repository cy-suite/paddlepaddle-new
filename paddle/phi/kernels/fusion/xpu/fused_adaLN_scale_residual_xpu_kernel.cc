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
// #include "paddle/common/ddim.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/adaptive_layernorm.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedAdalnScaleResidualXpuKernel(const Context& ctx,
                                      const DenseTensor& input1,
                                      const DenseTensor& input2,
                                      const DenseTensor& tensor1,
                                      const DenseTensor& tensor2,
                                      const DenseTensor& tensor3,
                                      const DenseTensor& ln_weight,
                                      const DenseTensor& ln_bias,
                                      int begin_norm_axis,
                                      float epsilon,
                                      float scale_op_weight,
                                      float scale_op_bias,
                                      bool bias_after_scale,
                                      DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto* input1_data = reinterpret_cast<const XPUType*>(input1.data<T>());
  auto* input2_data = reinterpret_cast<const XPUType*>(input2.data<T>());
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));

  int r = xpu::SUCCESS;
  auto xpu_ctx = static_cast<const phi::XPUContext*>(&ctx);

  // broadcast_mul
  r = baidu::xpu::api::broadcast_mul(
      xpu_ctx->x_context(),
      input1_data,
      reinterpret_cast<const XPUType*>(tensor1.data<T>()),
      out_data,
      common::vectorize<int>(input1.dims()),
      common::vectorize<int>(tensor1.dims()));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

  // broadcast_add
  r = baidu::xpu::api::broadcast_add(xpu_ctx->x_context(),
                                     out_data,
                                     input2_data,
                                     out_data,
                                     common::vectorize<int>(input1.dims()),
                                     common::vectorize<int>(input2.dims()));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

  // adaLN
  AdaptiveLayerNormXPUKernel<T, Context>(ctx,
                                         *out,
                                         ln_weight,
                                         ln_bias,
                                         tensor2,
                                         tensor3,
                                         begin_norm_axis,
                                         epsilon,
                                         scale_op_weight,
                                         scale_op_bias,
                                         bias_after_scale,
                                         out);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_adaLN_scale_residual_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedAdalnScaleResidualXpuKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
