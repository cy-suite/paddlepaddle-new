// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/kernels/impl/lu_kernel_impl.h"
#include "paddle/phi/kernels/lu_solve_kernel.h"

namespace phi {

template <typename T, typename Context>
class LuSolveFunctor {
 public:
  void operator()(const Context& dev_ctx,
                  char trans,
                  int n,
                  int nrhs,
                  T *a,
                  int lda,
                  int *ipiv,
                  T *b,
                  int ldb,
                  int *info);
};

template <typename T, typename Context>
void LuSolveKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& lu,
                   const DenseTensor& pivots,
                   const std::string& trans,
                   DenseTensor* out) {
  // Get lu matrix dimensions
  auto lu_dims = lu.dims();
  // Get x matrix dimensions
  auto x_dims = x.dims();

  // Allocate output tensor
  dev_ctx.template Alloc<T>(out);
  // Copy x to out since cusolverDn*getrs overwrites the input
  // lapack column major order
  *out = phi::Transpose2DTo6D<Context, T>(dev_ctx, x);
  DenseTensor tem_lu = phi::Transpose2DTo6D<Context, T>(dev_ctx, lu);

  // Prepare LAPACK parameters
  char trans_char = (trans == "N") ? 'N' : ((trans == "T") ? 'T' : 'C');
  int n_int = lu_dims[lu_dims.size() - 1];
  int nrhs_int = x_dims[x_dims.size() - 1];
  int lda = std::max(1, n_int);  // Leading dimension of A (LU matrix)
  int ldb = std::max(1, n_int);  // Leading dimension of B (RHS/solution matrix)
  int info = 0;

  int outdims = out->dims();
  int outrank = outdims.size();
  int batchsize = product(common::slice_ddim(outdims, 0, outrank - 2));
  auto out_data = out->data<T>();
  auto lu_data = reinterpret_cast<T*>(const_cast<T*>(tem_lu.data<T>()));
  auto pivots_data =
      reinterpret_cast<int*>(const_cast<int*>(pivots.data<int>()));

  LuSolveFunctor<T, Context> functor;
  for (int i = 0; i < batchsize; i++) {
    auto* out_data_item = &out_data[i * lda * nrhs_int];
    auto* lu_data_item = &lu_data[i * ldb * n_int];
    auto* pivots_data_item = &pivots_data[i * n_int];
    functor(trans_char,
            n_int,
            nrhs_int,
            lu_data_item,
            lda,
            pivots_data_item,
            out_data_item,
            ldb,
            &info);
  }
  *out = Transpose2DTo6D<Context, T>(dev_ctx, *out);
}

}  // namespace phi
