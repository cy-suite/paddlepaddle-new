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

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"

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
  const T* in_data = x.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);
  auto numel = x.numel();
  auto rank = x.dims().size();
  auto place = dev_ctx.GetPlace();
  auto stream = dev_ctx.stream();

  if (rank == 0) {  // scalar
    memory_utils::Copy(
        place, out_data, place, in_data, numel * sizeof(T), stream);
    return;
  }

  std::vector<int> reduce_dims;
  reduce_dims.reserve(rank);
  for (decltype(rank) i = 0; i < rank; ++i) {
    reduce_dims.push_back(i);
  }
  funcs::ReduceKernel<T,
                      T,
                      kps::AddFunctor,
                      kps::IdentityFunctor<T>,
                      /*is_mean*/ true>(
      dev_ctx, x, out, kps::IdentityFunctor<T>(), reduce_dims);
}

}  // namespace phi

PD_REGISTER_KERNEL(mean_all,
                   GPU,
                   ALL_LAYOUT,
                   phi::MeanAllKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
