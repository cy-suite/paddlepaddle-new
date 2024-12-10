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

#include "paddle/phi/kernels/clip_tensor_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {

template <typename T, typename Context>
void ClipTensorGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& min,
                          const DenseTensor& max,
                          const DenseTensor& out_grad,
                          DenseTensor* x_grad) {
  phi::DenseTensor ex_min;
  phi::DenseTensor ex_max;
  phi::DenseTensor ex_x;
  std::vector<int> real_target_shape = common::vectorize<int>(x_grad->dims());
  if (x.dims() != x_grad->dims()) {
    phi::ExpandKernel<T, Context>(
        dev_ctx, x, real_target_shape, &ex_x);
  } else {
    ex_x = x;
  }
  if (min.dims() != x_grad->dims()) {
    phi::ExpandKernel<T, Context>(
        dev_ctx, min, real_target_shape, &ex_min);
  } else {
    ex_min = min;
  }
  if (max.dims() != x_grad->dims()) {
    phi::ExpandKernel<T, Context>(
        dev_ctx, max, real_target_shape, &ex_max);
  } else {
    ex_max = max;
  }
  phi::CastKernel<T, Context>(dev_ctx, ex_min, ex_x.dtype(), &ex_min);
  phi::CastKernel<T, Context>(dev_ctx, ex_max, ex_x.dtype(), &ex_max);

  phi::DenseTensor x_ls_min;
  phi::LessThanKernel<T, Context>(dev_ctx, ex_min, ex_x, &x_ls_min);
  phi::DenseTensor x_ls_max;
  phi::LessThanKernel<T, Context>(dev_ctx, ex_x, ex_max, &x_ls_max);
  phi::DenseTensor out;
  EqualKernel<T, Context>(dev_ctx, x_ls_min, x_ls_max, &out);
  phi::DenseTensor zero_tensor(x_grad->dtype());
  FullKernel<T, Context>(dev_ctx,
                         common::vectorize(x_grad->dims()),
                         0.0f,
                         zero_tensor.dtype(),
                         &zero_tensor);
  phi::WhereKernel<T, Context>(dev_ctx, out, out_grad, zero_tensor, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(clip_tensor_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ClipTensorGradKernel,
                   float,
                   phi::dtype::float16,
                   int64_t,
                   int) {}
