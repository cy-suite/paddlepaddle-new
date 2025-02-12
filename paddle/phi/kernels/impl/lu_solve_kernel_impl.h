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

#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/lu_solve_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
class LUSolveFunctor {
 public:
  void operator()(const Context& dev_ctx,
                  bool trans,
                  int M,
                  int N,
                  T* Adata,
                  int lda,
                  const int* ipiv,
                  T* Bdata,
                  int* devInfo);
};

template <typename T, typename Context>
void LUSolveKernel(const Context& dev_ctx,
                   const DenseTensor& lu,
                   const DenseTensor& pivots,
                   const DenseTensor& b,
                   bool trans,
                   DenseTensor* out) {
  // Get shapes
  auto lu_dims = common::vectorize(lu.dims());
  auto b_dims = common::vectorize(b.dims());

  // Validate matrix dimensions
  const int lu_ndim = lu_dims.size();
  const int b_ndim = b_dims.size();
  PADDLE_ENFORCE_GE(lu_ndim, 2, "LU tensor must have at least 2 dimensions");
  PADDLE_ENFORCE_GE(b_ndim, 2, "b tensor must have at least 2 dimensions");

  const int matrix_size = lu_dims[lu_ndim - 1];
  PADDLE_ENFORCE_EQ(
      lu_dims[lu_ndim - 1],
      lu_dims[lu_ndim - 2],
      "LU tensor must be a square matrix in the last 2 dimensions, but got "
      "shape (%d, %d)",
      lu_dims[lu_ndim - 2],
      lu_dims[lu_ndim - 1]);
  PADDLE_ENFORCE_EQ(
      b_dims[b_ndim - 2],
      matrix_size,
      "The second-to-last dimension of b must match LU's dimension, but got "
      "b(%d) != lu(%d)",
      b_dims[b_ndim - 2],
      matrix_size);

  // Handle broadcasting for batch dimensions
  const int max_batch_dims = std::max(lu_ndim - 2, b_ndim - 2);
  std::vector<int64_t> output_batch_shape(max_batch_dims);

  // Right-align batch dimensions and apply broadcasting rules
  for (int i = 0; i < max_batch_dims; ++i) {
    const int lu_batch_idx = lu_ndim - 3 - i;
    const int b_batch_idx = b_ndim - 3 - i;
    const int64_t lu_dim = lu_batch_idx >= 0 ? lu_dims[lu_batch_idx] : 1;
    const int64_t b_dim = b_batch_idx >= 0 ? b_dims[b_batch_idx] : 1;

    PADDLE_ENFORCE_EQ(lu_dim == 1 || b_dim == 1 || lu_dim == b_dim,
                      true,
                      "Incompatible broadcast shape at batch dimension %d",
                      i);
    output_batch_shape[max_batch_dims - 1 - i] = std::max(lu_dim, b_dim);
  }

  // Construct the final broadcast shapes
  std::vector<int64_t> lu_bst_dims_vec(output_batch_shape);
  lu_bst_dims_vec.push_back(matrix_size);
  lu_bst_dims_vec.push_back(matrix_size);

  std::vector<int64_t> b_bst_dims_vec(output_batch_shape);
  b_bst_dims_vec.push_back(matrix_size);
  b_bst_dims_vec.push_back(b_dims[b_ndim - 1]);

  IntArray lu_bst_dims(lu_bst_dims_vec);
  IntArray b_bst_dims(b_bst_dims_vec);

  // Broadcast b to temporary tensor b_bst
  DenseTensor b_bst = phi::Empty<T, Context>(dev_ctx, b_bst_dims);
  ExpandKernel<T, Context>(dev_ctx, b, b_bst_dims, &b_bst);

  // Broadcast lu to temporary tensor lu_bst
  DenseTensor lu_bst = phi::Empty<T, Context>(dev_ctx, lu_bst_dims);
  ExpandKernel<T, Context>(dev_ctx, lu, lu_bst_dims, &lu_bst);

  // Calculate the conjugate of b_bst (for complex types)
  DenseTensor b_bst_conj = Conj<T, Context>(dev_ctx, b_bst);
  b_bst_conj = phi::TransposeLast2Dim<T>(dev_ctx, b_bst_conj);

  // Calculate the conjugate of lu_bst as well
  DenseTensor lu_bst_conj = Conj<T, Context>(dev_ctx, lu_bst);
  lu_bst_conj = phi::TransposeLast2Dim<T>(dev_ctx, lu_bst_conj);
  T* lu_bst_conj_data = lu_bst_conj.data<T>();

  // Copy the conjugate of b_bst to result tensor
  DenseTensor result;
  Copy<Context>(dev_ctx, b_bst_conj, dev_ctx.GetPlace(), false, &result);
  T* res_data = result.data<T>();

  int lu_bst_ndim = lu_bst_dims_vec.size();
  int matrix_m = static_cast<int>(lu_bst_dims_vec[lu_bst_ndim - 2]);
  int matrix_n = static_cast<int>(b_bst_dims_vec[lu_bst_ndim - 1]);
  int batchsize =
      product(common::slice_ddim(lu_bst.dims(), 0, lu_bst_ndim - 2));

  DenseTensor info = phi::Empty<int, Context>(dev_ctx, IntArray({batchsize}));
  int* info_data = info.data<int>();

  const int* ipiv_data = pivots.data<int>();

  LUSolveFunctor<T, Context> functor;
  for (int i = 0; i < batchsize; ++i) {
    functor(dev_ctx,
            trans,
            matrix_m,
            matrix_n,
            lu_bst_conj_data + i * matrix_m * matrix_m,
            std::max(1, matrix_m),
            ipiv_data + i * matrix_m,
            res_data + i * matrix_m * matrix_n,
            info_data + i);
  }

  // Transpose the result back and compute its conjugate to restore the proper
  // output order
  result = phi::TransposeLast2Dim<T>(dev_ctx, result);

  out->Resize(common::make_ddim(b_bst_dims_vec));
  ConjKernel<T, Context>(dev_ctx, result, out);
}

}  // namespace phi
