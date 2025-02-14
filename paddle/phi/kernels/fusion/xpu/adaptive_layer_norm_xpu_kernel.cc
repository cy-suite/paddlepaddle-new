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

#include <vector>
#include "paddle/common/ddim.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void AdaptiveLayerNormXPUKernel(const Context& ctx,
                                const DenseTensor& x,
                                const DenseTensor& scale,
                                const DenseTensor& bias,
                                const DenseTensor& tensor1,
                                const DenseTensor& tensor2,
                                int begin_norm_axis,
                                const float epsilon,
                                const float factor,
                                const float scale_bias,
                                bool bias_after_scale,
                                const int64_t unsqueeze_axis1,
                                const int64_t unsqueeze_axis2,
                                DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto* in_data = reinterpret_cast<const XPUType*>(x.data<T>());
  auto* scale_data = reinterpret_cast<const float*>(scale.data<float>());
  auto* bias_data = reinterpret_cast<const float*>(bias.data<float>());
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  auto* tensor1_data = reinterpret_cast<const XPUType*>(tensor1.data<T>());
  auto* tensor2_data = reinterpret_cast<const XPUType*>(tensor2.data<T>());

  int r = xpu::SUCCESS;
  auto xpu_ctx = static_cast<const phi::XPUContext*>(&ctx);
  auto x_shape = x.dims();
  int m = 1;
  int n = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    m *= x_shape[i];
  }
  for (int i = begin_norm_axis; i < x_shape.size(); i++) {
    n *= x_shape[i];
  }
  std::vector<int> x_shape_vec = common::vectorize<int>(x_shape);
  r = baidu::xpu::api::layer_norm(xpu_ctx->x_context(),
                                  in_data,
                                  out_data,
                                  m,
                                  n,
                                  epsilon,
                                  scale_data,
                                  bias_data,
                                  nullptr,   // mean
                                  nullptr);  // variance
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");

  if (tensor1_data) {
    auto tensor1_unsqueezed = phi::funcs::Unsqueeze(tensor1, unsqueeze_axis1);
    std::vector<int> tensor1_shape_vec =
        common::vectorize<int>(tensor1_unsqueezed.dims());

    DenseTensor* scale_out = new phi::DenseTensor();
    auto scale_out_ptr =
        reinterpret_cast<XPUType*>(ctx.template Alloc<T>(scale_out));
    r = baidu::xpu::api::scale(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType*>(tensor1_unsqueezed.data<T>()),
        scale_out_ptr,
        tensor1_unsqueezed.numel(),
        bias_after_scale,
        factor,
        scale_bias);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    r = baidu::xpu::api::broadcast_mul(xpu_ctx->x_context(),
                                       out_data,
                                       scale_out_ptr,
                                       out_data,     // 原址输出
                                       x_shape_vec,  // layer_norm输出的维度
                                       tensor1_shape_vec);  // scale输出的维度
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");
    delete scale_out;
  }
  if (tensor2_data) {
    auto tensor2_unsqueezed = phi::funcs::Unsqueeze(tensor2, unsqueeze_axis2);
    std::vector<int> tensor2_shape_vec =
        common::vectorize<int>(tensor2_unsqueezed.dims());
    r = baidu::xpu::api::broadcast_add(
        xpu_ctx->x_context(),
        out_data,
        reinterpret_cast<const XPUType*>(tensor2_unsqueezed.data<T>()),
        out_data,            // 原址输出
        x_shape_vec,         // broadcast_mul输出的维度
        tensor2_shape_vec);  // unsqueeze输出的维度
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
  }
  return;
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(adaptive_layer_norm_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::AdaptiveLayerNormXPUKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
