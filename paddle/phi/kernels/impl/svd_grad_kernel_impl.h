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
#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/diag_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <class T, class Context>
static DenseTensor Fill(const Context& ctx,
                        std::vector<int> shape,
                        T fill_value) {
  DenseTensor ret;
  ret.Resize(common::make_ddim(shape));
  ctx.template Alloc<T>(&ret);
  funcs::SetConstant<Context, T>()(ctx, &ret, fill_value);
  return ret;
}

template <class T, class Context>
static DenseTensor Eye(const Context& dev_ctx, int n) {
  auto output = Fill<T, Context>(dev_ctx, {n}, T(1));
  auto ret = Diag<T, Context>(dev_ctx, output, 0, 0);
  return ret;
}

template <class T, class Context>
static DenseTensor Infinits(const Context& ctx, std::vector<int> shape) {
  auto value = static_cast<T>(std::numeric_limits<double>::infinity());
  return Fill<T, Context>(ctx, shape, value);
}

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
DenseTensor Hermitian(const Context& dev_ctx, const DenseTensor& x) {
  return ::phi::TransposeLast2Dim<T>(dev_ctx, Conj<T, Context>(dev_ctx, x));
}

template <typename T, typename Context>
struct SvdGradFunctor {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& u,
                  const DenseTensor& vh,
                  const DenseTensor& s,
                  const paddle::optional<DenseTensor>& u_grad,
                  const paddle::optional<DenseTensor>& vh_grad,
                  const paddle::optional<DenseTensor>& s_grad,
                  bool full_matrices,
                  DenseTensor* x_grad) {
    const auto& dX = *x_grad;
    int m = dX.dims()[dX.dims().size() - 2];
    int n = dX.dims()[dX.dims().size() - 1];
    int k = s.dims()[s.dims().size() - 1];
    DenseTensor U, VH, dU, dV, dVH;
    if (full_matrices) {
      // if full_matrices is set, slice the U and VT to k columns
      U = Slice<T, Context>(dev_ctx, u, {u.dims().size() - 1}, {0}, {k});
      // If m < n for input matrices A, we partition A = [X|Y] and R = [U|V]

      VH = Slice<T, Context>(dev_ctx, vh, {vh.dims().size() - 2}, {0}, {k});
      if (u_grad.get_ptr() != nullptr) {
        dU = Slice<T, Context>(
            dev_ctx, *(u_grad.get_ptr()), {u.dims().size() - 1}, {0}, {k});
      }
      if (vh_grad.get_ptr() != nullptr) {
        dVH = Slice<T, Context>(
            dev_ctx, *(vh_grad.get_ptr()), {vh.dims().size() - 2}, {0}, {k});
      }
    } else {
      U = u;
      VH = vh;
      if (u_grad.get_ptr() != nullptr) {
        dU = *(u_grad.get_ptr());
      }
      if (vh_grad.get_ptr() != nullptr) {
        dVH = *(vh_grad.get_ptr());
      }
    }
    auto s_inverse = Pow<T, Context>(dev_ctx, s, -1);
    auto s_square = Pow<T, Context>(dev_ctx, s, 2);
    auto F = Subtract<T, Context>(
        dev_ctx, Unsqueeze(s_square, -2), Unsqueeze(s_square, -1));
    F = Add<T, Context>(
        dev_ctx,
        F,
        Diag<T, Context>(dev_ctx, Infinits<T, Context>(dev_ctx, {k}), 0, 0));
    F = Pow<T, Context>(dev_ctx, F, -1);
    DenseTensor sigma_term = Fill<T, Context>(dev_ctx, {1}, T(0.0));
    DenseTensor u_term = Fill<T, Context>(dev_ctx, {1}, T(0.0));
    DenseTensor v_term = Fill<T, Context>(dev_ctx, {1}, T(0.0));

    if (s_grad.get_ptr() != nullptr) {
      const DenseTensor& gS = *(s_grad.get_ptr());
      sigma_term = Multiply<T, Context>(dev_ctx, Unsqueeze(gS, -2), U);
      sigma_term = Matmul<T, Context>(dev_ctx, sigma_term, VH);
    }

    if (u_grad.get_ptr() != nullptr) {
      auto UTG = Matmul<T, Context>(dev_ctx, U, dU, true, false);
      auto GTU = Matmul<T, Context>(dev_ctx, dU, U, true, false);
      u_term = Multiply<T, Context>(
          dev_ctx,
          Multiply<T, Context>(
              dev_ctx, Subtract<T, Context>(dev_ctx, UTG, GTU), F),
          Unsqueeze(s, -2));
      u_term = Matmul<T, Context>(dev_ctx, U, u_term);
      if (m > k) {
        auto project = Subtract<T, Context>(
            dev_ctx,
            Eye<T, Context>(dev_ctx, m),
            Matmul<T, Context>(dev_ctx, U, U, false, true));
        u_term = Add<T, Context>(
            dev_ctx,
            u_term,
            Multiply<T, Context>(dev_ctx,
                                 Matmul<T, Context>(dev_ctx, project, dU),
                                 Unsqueeze(s_inverse, -2)));
      }
      u_term = Matmul<T, Context>(dev_ctx, u_term, VH);
    }
    if (vh_grad.get_ptr() != nullptr) {
      auto UTG = Matmul<T, Context>(dev_ctx, VH, dVH, false, true);
      auto GTU = Matmul<T, Context>(dev_ctx, dVH, VH, false, true);
      v_term = Multiply<T, Context>(
          dev_ctx,
          Matmul<T, Context>(
              dev_ctx,
              Multiply<T, Context>(
                  dev_ctx, Subtract<T, Context>(dev_ctx, UTG, GTU), F),
              VH),
          Unsqueeze(s, -1));
      if (n > k) {
        auto project = Subtract<T, Context>(
            dev_ctx,
            Eye<T, Context>(dev_ctx, n),
            Matmul<T, Context>(dev_ctx, VH, VH, true, false));
        v_term = Add<T, Context>(
            dev_ctx,
            v_term,
            Multiply<T, Context>(dev_ctx,
                                 Matmul<T, Context>(dev_ctx, dVH, project),
                                 Unsqueeze(s_inverse, -1)));
      }
      v_term = Matmul<T, Context>(dev_ctx, U, v_term);
    }

    *x_grad = Add<T, Context>(
        dev_ctx, Add<T, Context>(dev_ctx, u_term, sigma_term), v_term);
  }
};

template <typename T, typename Context>
struct SvdGradFunctor<phi::dtype::complex<T>, Context> {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& u,
                  const DenseTensor& vh,
                  const DenseTensor& s,
                  const paddle::optional<DenseTensor>& u_grad,
                  const paddle::optional<DenseTensor>& vh_grad,
                  const paddle::optional<DenseTensor>& s_grad,
                  bool full_matrices,
                  DenseTensor* x_grad) {
    using C = phi::dtype::complex<T>;
    // const auto& dX = *x_grad;
    // int m = dX.dims()[dX.dims().size() - 2];
    // int n = dX.dims()[dX.dims().size() - 1];
    int k = s.dims()[s.dims().size() - 1];
    DenseTensor U, VH, dU, dVH;
    DenseTensor J, K, L;
    DenseTensor S = Cast<T, Context>(dev_ctx, s, u.dtype());
    DenseTensor S_Matrix = Multiply<C, Context>(
        dev_ctx, Eye<C, Context>(dev_ctx, k), Unsqueeze(S, -2));
    const DenseTensor dS =
        Cast<T, Context>(dev_ctx, *(s_grad.get_ptr()), u.dtype());
    DenseTensor dS_Matrix = Multiply<C, Context>(
        dev_ctx, Eye<C, Context>(dev_ctx, k), Unsqueeze(dS, -2));
    if (full_matrices) {
      // if full_matrices is set, slice the U and VT to k columns
      U = Slice<C, Context>(dev_ctx, u, {u.dims().size() - 1}, {0}, {k});
      // If m < n for input matrices A, we partition A = [X|Y] and R = [U|V]

      VH = Slice<C, Context>(dev_ctx, vh, {vh.dims().size() - 2}, {0}, {k});
      if (u_grad.get_ptr() != nullptr) {
        dU = Slice<C, Context>(
            dev_ctx, *(u_grad.get_ptr()), {u.dims().size() - 1}, {0}, {k});
      } else {
        auto dU_dims = u.dims();
        dU_dims[dU_dims.size() - 1] = k;
        dU.Resize(dU_dims);
        dev_ctx.template Alloc<C>(&dU);
        dU = Fill<C, Context>(dev_ctx, common::vectorize<int>(dU_dims), C(0.0));
      }
      if (vh_grad.get_ptr() != nullptr) {
        dVH = Slice<C, Context>(
            dev_ctx, *(vh_grad.get_ptr()), {vh.dims().size() - 2}, {0}, {k});
      } else {
        auto dVH_dims = vh.dims();
        dVH_dims[dVH_dims.size() - 2] = k;
        dVH.Resize(dVH_dims);
        dev_ctx.template Alloc<C>(&dVH);
        dVH =
            Fill<C, Context>(dev_ctx, common::vectorize<int>(dVH_dims), C(0.0));
      }
    } else {
      U = u;
      VH = vh;
      if (u_grad.get_ptr() != nullptr) {
        dU = *(u_grad.get_ptr());
      } else {
        dU.Resize(u.dims());
        dev_ctx.template Alloc<C>(&dU);
        dU = Fill<C, Context>(
            dev_ctx, common::vectorize<int>(dU.dims()), C(0.0));
      }
      if (vh_grad.get_ptr() != nullptr) {
        dVH = *(vh_grad.get_ptr());
      } else {
        dVH.Resize(vh.dims());
        dev_ctx.template Alloc<C>(&dVH);
        dVH = Fill<C, Context>(
            dev_ctx, common::vectorize<int>(dVH.dims()), C(0.0));
      }
    }
    auto s_inverse = Pow<C, Context>(dev_ctx, S, -1);
    auto s_square = Pow<C, Context>(dev_ctx, S, 2);
    auto F = Subtract<C, Context>(
        dev_ctx, Unsqueeze(s_square, -2), Unsqueeze(s_square, -1));
    F = Add<C, Context>(
        dev_ctx,
        F,
        Diag<C, Context>(dev_ctx, Infinits<C, Context>(dev_ctx, {k}), 0, 0));
    F = Pow<C, Context>(dev_ctx, F, -1);
    J = Multiply<C, Context>(
        dev_ctx,
        F,
        Matmul<C, Context>(dev_ctx, Hermitian<C, Context>(dev_ctx, U), dU));
    K = Multiply<C, Context>(
        dev_ctx,
        F,
        Matmul<C, Context>(dev_ctx, VH, Hermitian<C, Context>(dev_ctx, dVH)));
    L = Multiply<C, Context>(
        dev_ctx,
        Eye<C, Context>(dev_ctx, k),
        Matmul<C, Context>(dev_ctx, VH, Hermitian<C, Context>(dev_ctx, dVH)));
    DenseTensor USVH = Fill<C, Context>(dev_ctx, {1}, C(0.0));
    DenseTensor u_term = Fill<C, Context>(dev_ctx, {1}, C(0.0));
    DenseTensor s_term = Fill<C, Context>(dev_ctx, {1}, C(0.0));
    DenseTensor v_term = Fill<C, Context>(dev_ctx, {1}, C(0.0));

    USVH = Matmul<C, Context>(
        dev_ctx, Matmul<C, Context>(dev_ctx, U, dS_Matrix), VH);
    u_term = Matmul<C, Context>(
        dev_ctx,
        Matmul<C, Context>(
            dev_ctx,
            Matmul<C, Context>(
                dev_ctx,
                U,
                Add<C, Context>(dev_ctx, J, Hermitian<C, Context>(dev_ctx, J))),
            S_Matrix),
        VH);
    s_term = Matmul<C, Context>(
        dev_ctx,
        Matmul<C, Context>(
            dev_ctx,
            Matmul<C, Context>(dev_ctx, U, S_Matrix),
            Add<C, Context>(dev_ctx, K, Hermitian<C, Context>(dev_ctx, K))),
        VH);

    v_term = Multiply<C, Context>(
        dev_ctx,
        Fill<C, Context>(dev_ctx, {1}, C(0.5)),
        Matmul<C, Context>(
            dev_ctx,
            Matmul<C, Context>(
                dev_ctx,
                Matmul<C, Context>(dev_ctx, U, Unsqueeze(s_inverse, -2)),
                Subtract<C, Context>(
                    dev_ctx, Hermitian<C, Context>(dev_ctx, L), L)),
            VH));
    *x_grad = Add<C, Context>(
        dev_ctx,
        Add<C, Context>(
            dev_ctx, Add<C, Context>(dev_ctx, USVH, u_term), s_term),
        v_term);
  }
};

template <typename T, typename Context>
void SvdGradKernel(const Context& dev_ctx,
                   const DenseTensor& x UNUSED,
                   const DenseTensor& u,
                   const DenseTensor& vh,
                   const DenseTensor& s,
                   const paddle::optional<DenseTensor>& u_grad,
                   const paddle::optional<DenseTensor>& vh_grad,
                   const paddle::optional<DenseTensor>& s_grad,
                   bool full_matrices,
                   DenseTensor* x_grad) {
  SvdGradFunctor<T, Context>()(
      dev_ctx, u, vh, s, u_grad, vh_grad, s_grad, full_matrices, x_grad);
}

}  // namespace phi
