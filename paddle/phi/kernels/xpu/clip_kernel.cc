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

#include "paddle/phi/kernels/clip_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {

template <typename T, typename Context>
void ClipKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const Scalar& min,
                const Scalar& max,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  using XPUDataType = typename XPUTypeTrait<T>::Type;
  auto x_data = reinterpret_cast<const XPUDataType*>(x.data<T>());
  auto out_data = reinterpret_cast<XPUDataType*>(out->data<T>());
  int r = xpu::clip_v2(dev_ctx.x_context(),
                       x_data,
                       out_data,
                       x.numel(),
                       static_cast<XPUDataType>(min.to<T>()),
                       static_cast<XPUDataType>(max.to<T>()));

  PADDLE_ENFORCE_EQ(r,
                    XPU_SUCCESS,
                    common::errors::External("XPU API(clip_v2) return wrong "
                                             "value[%d %s]",
                                             r,
                                             XPUAPIErrorMsg[r]));
}

template <typename T, typename Context>
void ClipTensorKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& min,
                      const DenseTensor& max,
                      DenseTensor* out) {
  using XPUDataType = typename XPUTypeTrait<T>::Type;
  const XPUDataType* x_data = reinterpret_cast<const XPUDataType*>(x.data<T>());
  const XPUDataType* min_data =
      reinterpret_cast<const XPUDataType*>(min.data<T>());
  const XPUDataType* max_data =
      reinterpret_cast<const XPUDataType*>(max.data<T>());
  XPUDataType* out_data =
      reinterpret_cast<XPUDataType*>(dev_ctx.template Alloc<T>(out));

  auto min_dims = common::vectorize<int>(min.dims());
  if (min_dims.size() == 0) {
    min_dims = std::vector<int>({1});
  }
  auto max_dims = common::vectorize<int>(max.dims());
  if (max_dims.size() == 0) {
    max_dims = std::vector<int>({1});
  }

  DenseTensor min_tensor(phi::DataType::BOOL);
  LessThanKernel<T, Context>(dev_ctx, x, min, &min_tensor);

  auto min_tensor_dims = common::vectorize<int>(min_tensor.dims());
  if (min_tensor_dims.size() == 0) {
    min_tensor_dims = std::vector<int>({1});
  }

  const bool* min_tensor_data = min_tensor.data<bool>();
  int ret = xpu::select(dev_ctx.x_context(),
                        min_tensor_data,
                        min_data,
                        x_data,
                        out_data,
                        min_tensor_dims,
                        min_dims);

  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "xpu::select");

  DenseTensor max_tensor(phi::DataType::BOOL);
  LessThanKernel<T, Context>(dev_ctx, max, x, &max_tensor);

  auto max_tensor_dims = common::vectorize<int>(max_tensor.dims());
  if (max_tensor_dims.size() == 0) {
    max_tensor_dims = std::vector<int>({1});
  }

  const bool* max_tensor_data = max_tensor.data<bool>();
  int ret2 = xpu::select(dev_ctx.x_context(),
                         max_tensor_data,
                         max_data,
                         x_data,
                         out_data,
                         max_tensor_dims,
                         max_dims);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret2, "xpu::select");
}

}  // namespace phi

PD_REGISTER_KERNEL(clip,
                   XPU,
                   ALL_LAYOUT,
                   phi::ClipKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int) {}

PD_REGISTER_KERNEL(clip_tensor,
                   XPU,
                   ALL_LAYOUT,
                   phi::ClipTensorKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int) {}
