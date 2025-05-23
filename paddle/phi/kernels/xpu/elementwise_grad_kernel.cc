//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/elementwise_grad_kernel.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename T, typename Context>
void MaximumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       DenseTensor* dx,
                       DenseTensor* dy) {
  if (dout.numel() == 0) {
    if (dx) {
      if (dx->numel() == 0) {
        dev_ctx.template Alloc<T>(dx);
      } else {
        phi::Full<T, Context>(
            dev_ctx, phi::IntArray(common::vectorize(dx->dims())), 0, dx);
      }
    }
    if (dy) {
      if (dy->numel() == 0) {
        dev_ctx.template Alloc<T>(dy);
      } else {
        phi::Full<T, Context>(
            dev_ctx, phi::IntArray(common::vectorize(dy->dims())), 0, dy);
      }
    }
    return;
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  int axis = -1;
  auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              const XPUType* z,
              const XPUType* dz,
              XPUType* dy,
              XPUType* dx,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_max_grad<XPUType>(
        ctx, x, y, z, dz, dy, dx, xshape, yshape);
  };

  XPUElementwiseGrad<T, XPUType>(dev_ctx, x, y, dout, axis, dx, dy, f, true);
}

template <typename T, typename Context>
void MinimumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       DenseTensor* dx,
                       DenseTensor* dy) {
  if (dout.numel() == 0) {
    if (dx) {
      if (dx->numel() == 0) {
        dev_ctx.template Alloc<T>(dx);
      } else {
        phi::Full<T, Context>(
            dev_ctx, phi::IntArray(common::vectorize(dx->dims())), 0, dx);
      }
    }
    if (dy) {
      if (dy->numel() == 0) {
        dev_ctx.template Alloc<T>(dy);
      } else {
        phi::Full<T, Context>(
            dev_ctx, phi::IntArray(common::vectorize(dy->dims())), 0, dy);
      }
    }
    return;
  }
  using XPUType = typename XPUTypeTrait<T>::Type;
  int axis = -1;
  auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              const XPUType* z,
              const XPUType* dz,
              XPUType* dy,
              XPUType* dx,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_min_grad<XPUType>(
        ctx, x, y, z, dz, dy, dx, xshape, yshape);
  };

  XPUElementwiseGrad<T, XPUType>(dev_ctx, x, y, dout, axis, dx, dy, f, true);
}

}  // namespace phi

PD_REGISTER_KERNEL(maximum_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaximumGradKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(minimum_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MinimumGradKernel,
                   float,
                   phi::dtype::float16) {}
