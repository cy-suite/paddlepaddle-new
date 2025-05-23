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

#include "paddle/phi/kernels/stack_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/stack_and_unstack.h"

namespace phi {

template <typename T, typename Context>
void StackKernel(const Context& ctx,
                 const std::vector<const DenseTensor*>& x,
                 int axis,
                 DenseTensor* out) {
  funcs::StackRawKernel<T, Context>(ctx, x, axis, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(stack,
                   GPU,
                   ALL_LAYOUT,
                   phi::StackKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float8_e5m2,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
