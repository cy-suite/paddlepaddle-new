// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/memcpy_kernel.h"
#include <vector>

namespace phi {
namespace fusion {

static std::vector<int64_t> get_zshape(const std::vector<int64_t>& wshape, const std::vector<int64_t>& xshape, const std::vector<int64_t>& yshape) {
    std::vector<int64_t> ret;
    int64_t shape_len_max = std::max<int64_t>(wshape.size(), std::max<int64_t>(xshape.size(), yshape.size()));
    for (int64_t i = 0; i < shape_len_max; i++) {
        int64_t wi = i + wshape.size() - shape_len_max;
        int64_t xi = i + xshape.size() - shape_len_max;
        int64_t yi = i + yshape.size() - shape_len_max;
        wi = ((wi < 0) ? 1 : wshape[wi]);
        xi = ((xi < 0) ? 1 : xshape[xi]);
        yi = ((yi < 0) ? 1 : yshape[yi]);
        ret.push_back(std::max<int64_t>(wi, std::max<int64_t>(xi, yi)));
    }
    return ret;
}

template <typename T, typename Context>
static T* expand_array(const Context& ctx, const T* input, T* output, int m, int n, int k) {
  int r = xpu::SUCCESS;
  #pragma unroll
    for (int64_t i = 0; i < m; ++i) {
        const T* input_row_start = input + i * k;
        T* output_row_start = output + i * n * k;
        #pragma unroll
        for (int64_t j = 0; j < n; ++j) {
           r = xpu_memcpy(output_row_start + j * k,
                 input_row_start,
                 k * sizeof(T),
                 XPUMemcpyKind::XPU_DEVICE_TO_DEVICE);
  PADDLE_ENFORCE_EQ(r, 0, common::errors::Fatal("xpu_memcpy failed."));
        }
    }
    return output;
}

template <typename T, typename Context>
void FusedMultiplyAddXpuKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& w,
                      DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto* xpu_ctx = static_cast<const phi::XPUContext*>(&ctx);
  int r = xpu::SUCCESS;

  const std::vector<int64_t> x_shape_vec = common::vectorize<int64_t>(x.dims());
  const std::vector<int64_t> y_shape_vec = common::vectorize<int64_t>(y.dims());
  const std::vector<int64_t> w_shape_vec = common::vectorize<int64_t>(w.dims());

  auto z_shape_vec = get_zshape(w_shape_vec, x_shape_vec, y_shape_vec);
  auto calculate_element_num =
          [](const std::vector<int64_t> &vec) -> int64_t {
        int64_t accu_mul = 1;
        for (auto num : vec) {
          accu_mul *= num;
        }
        return accu_mul;
      };
  int64_t out_element_num = calculate_element_num(z_shape_vec);
  DenseTensor x_reshaped(x);
  DenseTensor y_reshaped(y);
  DenseTensor w_reshaped(w);
  const auto& z_dims = common::make_ddim(z_shape_vec);
  x_reshaped.ResizeAndAllocate(z_dims);
  y_reshaped.ResizeAndAllocate(z_dims);
  w_reshaped.ResizeAndAllocate(z_dims);

  auto* x_data = x_reshaped.data<T>();
  auto* y_data = y_reshaped.data<T>();
  auto* w_data = w_reshaped.data<T>();
  auto* out_data = ctx.template Alloc<T>(out);

  if(w_shape_vec[1] == 1){
    r = xpu::broadcast(ctx.x_context(), reinterpret_cast<const XPUType*>(w.data<T>()), reinterpret_cast< XPUType*>(w_data), w_shape_vec, z_shape_vec);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
    // expand_array(ctx, w.data<T>(), w_data, z_shape_vec[0], z_shape_vec[1], z_shape_vec[2]);
  }
  if(y_shape_vec[1] == 1){
    r = xpu::broadcast(ctx.x_context(), reinterpret_cast<const XPUType*>(y.data<T>()), reinterpret_cast< XPUType*>(y_data), y_shape_vec, z_shape_vec);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
    // expand_array(ctx, y.data<T>(), y_data, z_shape_vec[0], z_shape_vec[1], z_shape_vec[2]);
  }
  if(x_shape_vec[1] == 1){
    r = xpu::broadcast(ctx.x_context(), reinterpret_cast<const XPUType*>(x.data<T>()), reinterpret_cast< XPUType*>(x_data), x_shape_vec, z_shape_vec);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
    // expand_array(ctx, x.data<T>(), x_data, z_shape_vec[0], z_shape_vec[1], z_shape_vec[2]);
  }
  
  r = xpu::addcmul(ctx.x_context(),
                       reinterpret_cast<const XPUType*>(w_data),
                       reinterpret_cast<const XPUType*>(x_data),
                       reinterpret_cast<const XPUType*>(y_data),
                       reinterpret_cast<XPUType*>(out_data),
                       1.0f,
                       out_element_num);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "addcmul");
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_multiply_add_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMultiplyAddXpuKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
