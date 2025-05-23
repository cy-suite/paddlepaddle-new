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

#include "paddle/phi/kernels/mean_all_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename T, typename Context>
void MeanAllKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   DenseTensor* out) {
  if (x.numel() == 0) {
    phi::Full<T, Context>(
        dev_ctx, phi::IntArray(common::vectorize(out->dims())), NAN, out);
    return;
  }
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto* input = &x;
  auto* output = out;
  dev_ctx.template Alloc<T>(out);
  const T* x_data = input->data<T>();
  T* y_data = output->data<T>();
  std::vector<int64_t> x_shape;
  x_shape.push_back(1);
  x_shape.push_back(input->numel());
  std::vector<int64_t> rdims = {1};
  int r = xpu::reduce_mean(dev_ctx.x_context(),
                           reinterpret_cast<const XPUType*>(x_data),
                           reinterpret_cast<XPUType*>(y_data),
                           x_shape,
                           rdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mean_all");
}
}  // namespace phi

PD_REGISTER_KERNEL(
    mean_all, XPU, ALL_LAYOUT, phi::MeanAllKernel, float, phi::dtype::float16) {
}
