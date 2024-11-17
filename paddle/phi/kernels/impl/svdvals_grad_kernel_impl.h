#pragma once
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void SvdvalsGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& s,
                       const paddle::optional<DenseTensor>& s_grad,
                       DenseTensor* x_grad) {
  const auto& dX = *x_grad;
  int m = dX.dims()[dX.dims().size() - 2];
  int n = dX.dims()[dX.dims().size() - 1];
  int k = s.dims()[s.dims().size() - 1];

  // Ensure s_grad is provided
  if (!s_grad.get_ptr()) {
    phi::errors::InvalidArgument("s_grad must be provided for svdvals_grad");
  }

  const auto& gS = *(s_grad.get_ptr());
  auto S_diag =
      Diag<T, Context>(dev_ctx, gS, 0, 0);  // Construct diagonal tensor from gS
  auto U = DenseTensor();                   // Placeholder for U
  auto VH = DenseTensor();                  // Placeholder for VH

  // Approximate gradient computation for X
  auto sigma_term =
      Matmul<T, Context>(dev_ctx, Matmul<T, Context>(dev_ctx, U, S_diag), VH);

  // Write the result back to x_grad
  *x_grad = sigma_term;
}

}  // namespace phi