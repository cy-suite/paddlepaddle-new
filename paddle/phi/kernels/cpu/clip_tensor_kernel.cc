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

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"

namespace phi {

template <typename T, typename Context>
void ClipTensorKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& min,
                      const DenseTensor& max,
                      DenseTensor* out) {
  DenseTensor ex_min;
  DenseTensor ex_max;
  DenseTensor ex_x;
  std::vector<int> real_target_shape = common::vectorize<int>(out->dims());
  if (x.dims() != out->dims()) {
    phi::ExpandKernel<T, Context>(
        dev_ctx, x, real_target_shape, &ex_x);
  } else {
    ex_x = x;
  }
  if (min.dims() != out->dims()) {
    phi::ExpandKernel<T, Context>(
        dev_ctx, min, real_target_shape, &ex_min);
  } else {
    ex_min = min;
  }
  if (max.dims() != out->dims()) {
    phi::ExpandKernel<T, Context>(
        dev_ctx, max, real_target_shape, &ex_max);
  } else {
    ex_max = max;
  }
  phi::CastKernel<T, Context>(dev_ctx, ex_min, ex_x.dtype(), &ex_min);
  phi::CastKernel<T, Context>(dev_ctx, ex_max, ex_x.dtype(), &ex_max);
  
  const T* x_data = ex_x.data<T>();
  const T* min_data = ex_min.data<T>();
  const T* max_data = ex_max.data<T>();

  auto x_numel = ex_x.numel();

  T* out_data = dev_ctx.template Alloc<T>(out);

  for (int i = 0; i < x_numel; i++) {
    out_data[i] = x_data[i] < min_data[i] ? min_data[i] : x_data[i];
    out_data[i] = out_data[i] > max_data[i] ? max_data[i] : out_data[i];
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(clip_tensor,
                   CPU,
                   ALL_LAYOUT,
                   phi::ClipTensorKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
