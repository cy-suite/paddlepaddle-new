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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/lu_solve_kernel_impl.h"

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/rocsolver.h"
#else
#include "paddle/phi/backends/dynload/cusolver.h"
#endif

namespace phi {

#ifdef PADDLE_WITH_HIP
template <typename T>
void rocsolver_getrs(const solverHandle_t& handle,
                     rocblas_operation trans,
                     int M,
                     int N,
                     T* Adata,
                     int lda,
                     const int* ipiv,
                     T* Bdata,
                     int ldb);

template <>
void rocsolver_getrs<float>(const solverHandle_t& handle,
                            rocblas_operation trans,
                            int M,
                            int N,
                            float* Adata,
                            int lda,
                            const int* ipiv,
                            float* Bdata,
                            int ldb) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::rocsolver_sgetrs(
      handle, trans, M, N, Adata, lda, ipiv, Bdata, ldb));
}

template <>
void rocsolver_getrs<double>(const solverHandle_t& handle,
                             rocblas_operation trans,
                             int M,
                             int N,
                             double* Adata,
                             int lda,
                             const int* ipiv,
                             double* Bdata,
                             int ldb) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::rocsolver_dgetrs(
      handle, trans, M, N, Adata, lda, ipiv, Bdata, ldb));
}
#endif

template <typename T>
class LUSolveFunctor<T, GPUContext> {
 public:
  void operator()(const GPUContext& dev_ctx,
                  bool trans,
                  int M,
                  int N,
                  T* Adata,
                  int lda,
                  const int* ipiv,
                  T* Bdata,
                  int* devInfo) {
    auto handle = dev_ctx.cusolver_dn_handle();

#ifdef PADDLE_WITH_HIP
    rocblas_operation trans_op =
        trans ? rocblas_operation_transpose : rocblas_operation_none;
    rocsolver_getrs<T>(handle, trans_op, M, N, Adata, lda, ipiv, Bdata, lda);
#else
    // Workspace sizes differ for different data types
    int workspace_size;
    if (std::is_same<T, float>::value) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cusolverDnSgetrs(handle,
                                         trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                         M,
                                         N,
                                         reinterpret_cast<const float*>(Adata),
                                         lda,
                                         ipiv,
                                         reinterpret_cast<float*>(Bdata),
                                         lda,
                                         devInfo));
    } else if (std::is_same<T, double>::value) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cusolverDnDgetrs(handle,
                                         trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                         M,
                                         N,
                                         reinterpret_cast<const double*>(Adata),
                                         lda,
                                         ipiv,
                                         reinterpret_cast<double*>(Bdata),
                                         lda,
                                         devInfo));
    }
#endif
  }
};

}  // namespace phi

PD_REGISTER_KERNEL(
    lu_solve, GPU, ALL_LAYOUT, phi::LUSolveKernel, float, double) {}
