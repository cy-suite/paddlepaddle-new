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

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

namespace phi {

template <typename T, typename Context>
void MaximumRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      int axis,
                      DenseTensor* out) {
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_max<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

template <typename T, typename Context>
void MinimumRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      int axis,
                      DenseTensor* out) {
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_min<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

template <typename T, typename Context>
void RemainderRawKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        int axis,
                        DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_mod<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

template <typename T, typename Context>
void FloorDivideRawKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          int axis,
                          DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_floordiv<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

template <typename T, typename Context>
void ElementwisePowRawKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             int axis,
                             DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_pow<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, axis, out, f);
}

}  // namespace phi

PD_REGISTER_KERNEL(floor_divide_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::FloorDivideRawKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(maximum_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaximumRawKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(minimum_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::MinimumRawKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(remainder_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::RemainderRawKernel,
                   float,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(elementwise_pow_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowRawKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
