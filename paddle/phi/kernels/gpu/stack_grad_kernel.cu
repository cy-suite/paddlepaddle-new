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

#include "paddle/phi/kernels/stack_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/stack_and_unstack.h"

namespace phi {

template <typename T, typename Context>
void StackGradKernel(const Context& ctx,
                     const DenseTensor& out_grad,
                     int axis,
                     std::vector<DenseTensor*> x_grad) {
  if (axis < 0) axis += out_grad.dims().size();

  int64_t split_dim = out_grad.dims()[axis];
  PADDLE_ENFORCE_EQ(
      split_dim,
      x_grad.size(),
      common::errors::InvalidArgument(
          "Output x_grad's size should be equal to the split_dim, but"
          " received split_dim is:%d x_grad's size is:%d.",
          split_dim,
          x_grad.size()));

  funcs::UnStackRawKernel<T, Context>(ctx, out_grad, axis, &x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(stack_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::StackGradKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   uint8_t,
                   int16_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float8_e5m2,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
