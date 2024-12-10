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

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/onednn/elementwise_kernel.cc"

namespace phi {
template <typename T, typename Context>
void ClipTensorKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const DenseTensor& min,
                const DenseTensor& max,
                DenseTensor* out) {
  phi::DenseTensor out_max;
  phi::ElementwiseKernel<T, dnnl::algorithm::binary_max>(dev_ctx, x, min, -1, &out_max);
  phi::ElementwiseKernel<T, dnnl::algorithm::binary_min>(dev_ctx, out_max, max, -1, out);

}
}  // namespace phi

PD_REGISTER_KERNEL(
    clip_tensor, OneDNN, ONEDNN, phi::ClipTensorKernel, float, phi::dtype::bfloat16) {}
