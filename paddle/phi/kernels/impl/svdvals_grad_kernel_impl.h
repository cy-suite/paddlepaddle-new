#pragma once

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
                       const DenseTensor& s,
                       const paddle::optional<DenseTensor>& s_grad,
                       DenseTensor* x_grad) {
  if (!s_grad.get_ptr() || s_grad.get_ptr()->numel() == 0) {
    funcs::SetConstant<Context, T>()(dev_ctx, x_grad, T(0.0));
    x_grad->Resize(x.dims());
    return;
  }

  const DenseTensor& dS = *(s_grad.get_ptr());
  DenseTensor dX_term = Diag<T, Context>(dev_ctx, dS, 0, 0);

  DenseTensor U, VH, S_recomputed;
  phi::SvdKernel<T, Context>(dev_ctx,
                             x,
                             false,
                             &U,
                             &S_recomputed,
                             &VH);  // Crucial: recomputing SVD

  *x_grad =
      Matmul<T, Context>(dev_ctx, Matmul<T, Context>(dev_ctx, U, dX_term), VH);
}
}  // namespace phi