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

static DenseTensor Unsqueeze(const DenseTensor& x, int axis = 0) {
  // don't copy data, only change the dims
  DenseTensor out;
  out.ShareDataWith(x);
  std::vector<int> out_shape = common::vectorize<int>(x.dims());
  if (axis >= 0) {
    auto index = (out_shape.begin() + axis);
    out_shape.insert(index, 1);
  } else if (axis < 0) {
    auto index = (out_shape.end() + axis + 1);
    out_shape.insert(index, 1);
  }
  out.Resize(common::make_ddim(out_shape));
  return out;
}
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

  DenseTensor x_copy = x;  // Create a copy! Important to avoid const
                           // correctness issues with LAPACK
  dev_ctx.template Alloc<T>(&x_copy);
  Copy(dev_ctx, x, dev_ctx.GetPlace(), false, &x_copy);

  // 1. Allocate memory (using Thin SVD for potential memory optimization):
  DenseTensor u, vt;
  u.Resize({m, k});   // Thin U
  vt.Resize({k, n});  // Thin VT
  dev_ctx.template Alloc<T>(&u);
  dev_ctx.template Alloc<T>(&vt);

  DenseTensor work;
  int lwork = -1;
  std::vector<T> work_query(1);
  int info;

  DenseTensor temp_s;  // Avoid overwriting input 's'
  temp_s.Resize({k});
  dev_ctx.template Alloc<T>(&temp_s);

  phi::funcs::lapackSvd<T>('S',
                           m,
                           n,
                           x_copy.data<T>(),
                           m,
                           temp_s.data<T>(),
                           u.data<T>(),
                           m,
                           vt.data<T>(),
                           n,
                           work_query.data(),
                           lwork,
                           nullptr,
                           &info);

  lwork = static_cast<int>(work_query[0]);
  work.Resize({lwork});
  dev_ctx.template Alloc<T>(&work);

  phi::funcs::lapackSvd<T>('S',
                           m,
                           n,
                           x_copy.data<T>(),
                           m,
                           temp_s.data<T>(),
                           u.data<T>(),
                           m,
                           vt.data<T>(),
                           n,
                           work.data<T>(),
                           lwork,
                           nullptr,
                           &info);

  if (info != 0) {
    PADDLE_THROW(phi::errors::External("Lapack SVD failed. Info = %d", info));
  }

  DenseTensor v = Transpose<T, Context>(dev_ctx, vt);  // Get V (n x k)

  DenseTensor sigma_term = Multiply<T, Context>(dev_ctx, Unsqueeze(gS, -2), u);
  sigma_term =
      Matmul<T, Context>(dev_ctx, sigma_term, v);  // Use v directly (thin SVD)
  *x_grad = sigma_term;
}
}  // namespace phi