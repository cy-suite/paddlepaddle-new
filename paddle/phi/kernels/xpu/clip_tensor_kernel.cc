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

#include "paddle/phi/kernels/clip_tensor_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {

template <typename T, typename Context>
void ClipTensorKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& min,
                      const DenseTensor& max,
                      DenseTensor* out) {
  phi::DenseTensor ex_min;
  phi::DenseTensor ex_max;
  phi::DenseTensor ex_x;
  std::vector<int> real_target_shape = common::vectorize<int>(out->dims());
  if (x.dims() != out->dims()) {
    phi::ExpandKernel<T, Context>(dev_ctx, x, real_target_shape, &ex_x);
  } else {
    ex_x = x;
  }
  if (min.dims() != out->dims()) {
    phi::ExpandKernel<T, Context>(dev_ctx, min, real_target_shape, &ex_min);
  } else {
    ex_min = min;
  }
  if (max.dims() != out->dims()) {
    phi::ExpandKernel<T, Context>(dev_ctx, max, real_target_shape, &ex_max);
  } else {
    ex_max = max;
  }
  phi::CastKernel<T, Context>(dev_ctx, ex_min, ex_x.dtype(), &ex_min);
  phi::CastKernel<T, Context>(dev_ctx, ex_max, ex_x.dtype(), &ex_max);

  phi::DenseTensor x_ls_min;
  phi::LessThanKernel<T, Context>(dev_ctx, ex_x, ex_min, &x_ls_min);
  phi::DenseTensor tem_out;
  phi::WhereKernel<T, Context>(dev_ctx, x_ls_min, ex_min, ex_x, &tem_out);

  phi::DenseTensor x_ls_max;
  phi::LessThanKernel<T, Context>(dev_ctx, ex_max, ex_x, &x_ls_max);
  phi::WhereKernel<T, Context>(dev_ctx, x_ls_max, ex_max, tem_out, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(clip_tensor,
                   XPU,
                   ALL_LAYOUT,
                   phi::ClipTensorKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int) {}
