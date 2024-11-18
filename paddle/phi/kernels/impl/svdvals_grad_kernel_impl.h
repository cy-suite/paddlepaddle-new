#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/diag_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"

namespace phi {

template <typename T, typename Context>
void SvdvalsGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& s,
                       const paddle::optional<DenseTensor>& s_grad,
                       DenseTensor* x_grad) {
  if (!s_grad.get_ptr()) {
    funcs::SetConstant<Context, T>()(dev_ctx, x_grad, T(0));
    return;
  }

  const DenseTensor& gS = *(s_grad.get_ptr());
  int m = x.dims()[0];
  int n = x.dims()[1];
  int k = s.dims()[0];  // k = min(m, n)

  // 1. Allocate memory for U, VT, and working space:
  DenseTensor u, vt;
  u.Resize({m, m});   // Full U (adjust if you want a "thin" U)
  vt.Resize({n, n});  // Full VT (adjust for "thin" VT)
  dev_ctx.template Alloc<T>(&u);
  dev_ctx.template Alloc<T>(&vt);

  DenseTensor work;
  int lwork = -1;                // Query optimal workspace size
  std::vector<T> work_query(1);  // Workspace for query
  int info;
  phi::funcs::lapackSvd<T>(
      'A',  // All singular vectors (U and VT)
      m,
      n,
      x.data<T>(),
      m,            // Leading dimension of x
      s.data<T>(),  // Singular values (already computed in forward)
      u.data<T>(),
      m,  // Leading dimension of U
      vt.data<T>(),
      n,  // Leading dimension of VT
      work_query.data(),
      lwork,
      nullptr,  // iwork (not needed for float/double)
      &info);

  lwork = static_cast<int>(work_query[0]);  // Get optimal size
  work.Resize({lwork});
  dev_ctx.template Alloc<T>(&work);

  // 2. Perform SVD:
  phi::funcs::lapackSvd<T>(
      'A',
      m,
      n,
      x.data<T>(),
      m,
      s.data<T>(),  // Overwrites singular values (not used here)
      u.data<T>(),
      m,
      vt.data<T>(),
      n,
      work.data<T>(),
      lwork,
      nullptr,
      &info);

  if (info != 0) {
    // Handle LAPACK error (e.g., throw an exception)
    PADDLE_THROW(
        phi::errors::External("Lapack SVD function failed. Info = %d", info));
  }

  DenseTensor v = Transpose<T, Context>(dev_ctx, vt);  // Get V

  // Now you have U and V as DenseTensors (VT has been transposed to V):
  DenseTensor sigma_term = Multiply<T, Context>(dev_ctx, Unsqueeze(gS, -2), u);

  DenseTensor vh =
      Slice<T, Context>(dev_ctx, v, {v.dims().size() - 2}, {0}, {k});

  sigma_term = Matmul<T, Context>(dev_ctx, sigma_term, vh);
  *x_grad = sigma_term;
}

}  // namespace phi