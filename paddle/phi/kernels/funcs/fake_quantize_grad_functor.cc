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

#include "paddle/phi/kernels/funcs/fake_quantize_grad_functor.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {
namespace funcs {

template <typename Context, typename T>
void FakeQuantizeDequantizeGradLSQFunctor<Context, T>::operator()(
    const Context &dev_ctx,
    const DenseTensor &x,
    const DenseTensor &scale,
    const DenseTensor &out_grad,
    const float lsq_factor,
    const int bin_cnt,
    const int round_type,
    DenseTensor *x_grad,
    DenseTensor *scale_grad) {
  phi::Transform<Context> trans;

  T scale_v = scale.data<T>()[0];
  T min_bound = -bin_cnt - static_cast<T>(1);
  T max_bound = bin_cnt;

  DenseTensor scale_grad_elem;
  scale_grad_elem.Resize({x.dims()});
  T *scale_grad_elem_data = dev_ctx.template Alloc<T>(
      &scale_grad_elem, scale_grad_elem.numel() * sizeof(T));

  trans(dev_ctx,
        out_grad.data<T>(),
        out_grad.data<T>() + x.numel(),
        dev_ctx.template Alloc<T>(x_grad),
        phi::funcs::LSQClipGradFunctor<T>(min_bound, max_bound));

  trans(dev_ctx,
        x.data<T>(),
        x.data<T>() + x.numel(),
        scale_grad_elem_data,
        phi::funcs::LSQScaleGradFunctor<T>(
            bin_cnt, lsq_factor, inverse(scale_v), round_type));

  DenseTensor scale_grad_elem_2;
  scale_grad_elem_2.Resize({x.dims()});
  dev_ctx.template Alloc<T>(&scale_grad_elem_2,
                            scale_grad_elem_2.numel() * sizeof(T));

  MultiplyKernel<T, Context>(
      dev_ctx, out_grad, scale_grad_elem, &scale_grad_elem_2);

  dev_ctx.template Alloc<T>(scale_grad);
  std::vector<int> v_dims(x.dims().size());
  std::iota(v_dims.begin(), v_dims.end(), 0);
  IntArray reduce_dims(v_dims);

  phi::SumKernel<T, Context>(
      dev_ctx, scale_grad_elem_2, reduce_dims, x.dtype(), 0, scale_grad);
  scale_grad->Resize(scale.dims());
}
template class FakeQuantizeDequantizeGradLSQFunctor<CPUContext, float>;
template class FakeQuantizeDequantizeGradLSQFunctor<CPUContext, double>;

}  // namespace funcs
}  // namespace phi
