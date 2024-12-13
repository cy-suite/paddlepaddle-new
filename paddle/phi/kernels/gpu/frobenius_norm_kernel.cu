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

#include "paddle/phi/kernels/frobenius_norm_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/gpu/reduce.h"

namespace phi {

template <typename T, typename Context>
void FrobeniusNormKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const IntArray& dims,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* out) {
  if (x.numel() == 0) {
    phi::DenseTensor cpu_out;

    if (reduce_all || static_cast<int64_t>(dims.size()) == x.dims().size()) {
      cpu_out.Resize({});
    } else if (keep_dim) {
      std::vector<int64_t> out_dims(x.dims().size());
      for (int i = 0; i < x.dims().size(); ++i) {
        out_dims[i] = x.dims()[i];
      }
      for (int64_t i : dims.GetData()) {
        out_dims[i] = 1;
      }
      cpu_out.Resize(phi::make_ddim(out_dims));
    } else {
      std::vector<int64_t> out_dims;
      for (int i = 0; i < x.dims().size(); ++i) {
        if (std::find(dims.GetData().begin(), dims.GetData().end(), i) ==
            dims.GetData().end()) {
          out_dims.push_back(x.dims()[i]);
        }
      }
      cpu_out.Resize(phi::make_ddim(out_dims));
    }

    cpu_out.mutable_data<T>(dev_ctx.GetPlace());
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, &cpu_out, static_cast<T>(0));
    *out = cpu_out;
    return;
  }

  reduce_all = recompute_reduce_all(x, dims.GetData(), reduce_all);
  auto out_dtype = x.dtype();
  phi::Reduce<T, kps::AddFunctor, kps::SquareFunctor>(
      dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);

  SqrtKernel<T, Context>(dev_ctx, *out, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    frobenius_norm, GPU, ALL_LAYOUT, phi::FrobeniusNormKernel, float, double) {}
