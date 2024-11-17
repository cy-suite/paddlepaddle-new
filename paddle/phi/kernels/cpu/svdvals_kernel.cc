#include "paddle/phi/kernels/svdvals_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"

namespace phi {

template <typename T>
void LapackSvdvals(const T* X, T* S, int rows, int cols) {
  // Using N to neglect computing U、VH
  char jobz = 'N';
  int mx = std::max(rows, cols);
  int mn = std::min(rows, cols);
  T* a = const_cast<T*>(X);
  int lda = rows;
  int lwork = 4 * mn + mx;
  std::vector<T> work(lwork);
  int info = 0;
  phi::funcs::lapackSvd<T>(jobz,
                           rows,
                           cols,
                           a,
                           lda,
                           S,
                           nullptr,  // U is not needed
                           1,        // dummy dimension for U
                           nullptr,  // VH is not needed
                           1,        // dummy dimension for VH
                           work.data(),
                           lwork,
                           nullptr,  // iwork is not needed
                           &info);
  if (info < 0) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "This %s-th argument has an illegal value.", info));
  }
  if (info > 0) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "SVD computation did not converge. Input matrix may be invalid."));
  }
}

template <typename T>
void BatchSvdvals(const T* X, T* S, int rows, int cols, int batches) {
  int stride = rows * cols;
  int stride_s = std::min(rows, cols);
  for (int i = 0; i < batches; i++) {
    LapackSvdvals<T>(X + i * stride, S + i * stride_s, rows, cols);
  }
}

template <typename T, typename Context>
void SvdvalsKernel(const Context& dev_ctx,
                   const DenseTensor& X,
                   DenseTensor* S) {
  auto x_dims = X.dims();
  int rows = static_cast<int>(x_dims[x_dims.size() - 2]);
  int cols = static_cast<int>(x_dims[x_dims.size() - 1]);
  int batches = static_cast<int>(X.numel() / (rows * cols));

  // Validate dimensions
  PADDLE_ENFORCE_GT(
      rows,
      0,
      phi::errors::InvalidArgument("The row of Input(X) must be > 0."));
  PADDLE_ENFORCE_GT(
      cols,
      0,
      phi::errors::InvalidArgument("The column of Input(X) must be > 0."));

  // Allocate memory for output
  auto* S_out = dev_ctx.template Alloc<phi::dtype::Real<T>>(S);
  
  // Transpose the last two dimensions for LAPACK compatibility
  DenseTensor trans_x = ::phi::TransposeLast2Dim<T>(dev_ctx, X);
  auto* x_data = trans_x.data<T>();

  // Perform batch SVD computation for singular values
  BatchSvdvals<T>(x_data, S_out, rows, cols, batches);
}

}  // namespace phi

// Register the kernel for CPU
PD_REGISTER_KERNEL(
    svdvals, CPU, ALL_LAYOUT, phi::SvdvalsKernel, float, double) {}