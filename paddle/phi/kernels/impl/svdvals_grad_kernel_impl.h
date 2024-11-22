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

#pragma once

#include "glog/logging.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/diag_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/phi/kernels/svd_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void SvdvalsGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& s_grad,
                       DenseTensor* x_grad) {
  if (!s_grad.get_ptr() || s_grad.get_ptr()->numel() == 0) {
    funcs::SetConstant<Context, T>()(dev_ctx, x_grad, T(0.0));
    x_grad->Resize(x.dims());
    return;
  }

  const DenseTensor& dS = *(s_grad.get_ptr());
  DenseTensor dX_term = Diag<T, Context>(dev_ctx, dS, 0, 0);
  VLOG(1) << "dX_term shape: " << dX_term.dims();
  int rows = x.dims()[x.dims().size() - 2];
  int cols = x.dims()[x.dims().size() - 1];
  int k = std::min(rows, cols);

  DenseTensor U, VH, S_recomputed;
  DDim u_dims = {rows, k};
  DDim s_dims = {k};
  DDim vh_dims = {k, cols};
  U.Resize(u_dims);
  VH.Resize(vh_dims);
  S_recomputed.Resize(s_dims);
  phi::SvdKernel<T, Context>(dev_ctx,
                             x,
                             true,
                             &U,
                             &S_recomputed,
                             &VH);  // Crucial: recomputing SVD
  VLOG(1) << "S_grad shape: " << s_grad.dims();
  *x_grad =
      Matmul<T, Context>(dev_ctx, Matmul<T, Context>(dev_ctx, U, dX_term), VH);
}
}  // namespace phi
