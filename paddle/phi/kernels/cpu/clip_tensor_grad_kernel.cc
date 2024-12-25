// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/clip_tensor_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"

namespace phi {

template <typename T, typename Context>
void ClipTensorGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& min,
                          const DenseTensor& max,
                          const DenseTensor& out_grad,
                          DenseTensor* x_grad) {
  DenseTensor ex_min;
  MetaTensor meta_min(&ex_min);
  CastInferMeta(min, x.dtype(), &meta_min);
  DenseTensor ex_max;
  MetaTensor meta_max(&ex_max);
  CastInferMeta(max, x.dtype(), &meta_max);
  phi::CastKernel<T, Context>(dev_ctx, min, x.dtype(), &ex_min);
  phi::CastKernel<T, Context>(dev_ctx, max, x.dtype(), &ex_max);

  const T* x_data = x.data<T>();
  const T* min_data = ex_min.data<T>();
  const T* max_data = ex_max.data<T>();
  auto numel = x.numel();
  auto* dout = out_grad.data<T>();

  auto* dx = dev_ctx.template Alloc<T>(x_grad);
  for (int i = 0; i < numel; i++) {
    dx[i] = (x_data[i] > min_data[i] && x_data[i] < max_data[i])
                ? dout[i]
                : static_cast<T>(0);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(clip_tensor_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ClipTensorGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
