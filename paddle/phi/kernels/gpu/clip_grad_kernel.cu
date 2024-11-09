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

#include "paddle/phi/kernels/clip_grad_kernel.h"

#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
class ClipGradGPUFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T min_, const T max_) const {
    return (y > min_ && y < max_) ? x : static_cast<T>(0);
  }
};

template <typename T, typename Context>
void ClipGradKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    const DenseTensor& min,
                    const DenseTensor& max,
                    DenseTensor* x_grad) {
  std::vector<const DenseTensor*> ins = {&out_grad, &x, &min, &max};
  std::vector<DenseTensor*> outs = {x_grad};
  auto functor = ClipGradGPUFunctor<T>();
  dev_ctx.template Alloc<T>(x_grad);
  phi::funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(clip_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ClipGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
