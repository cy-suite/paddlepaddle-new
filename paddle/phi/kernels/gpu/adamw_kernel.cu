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

#include "paddle/phi/kernels/adamw_kernel.h"

#include <math.h>
#include <vector>
#include "glog/logging.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

// Template accessor design
template <typename MT, bool IsCpu>
struct BetaPowAccessor;

template <typename MT>
struct BetaPowAccessor<MT, true> {  // CPU accessor
  const MT beta1;
  const MT beta2;

  BetaPowAccessor(const MT* beta1_pow, const MT* beta2_pow)
      : beta1(*beta1_pow), beta2(*beta2_pow) {}

  __device__ __forceinline__ MT GetBeta1() const { return beta1; }
  __device__ __forceinline__ MT GetBeta2() const { return beta2; }
};

template <typename MT>
struct BetaPowAccessor<MT, false> {  // GPU pointer
  const MT* beta1_pow;
  const MT* beta2_pow;

  BetaPowAccessor(const MT* beta1, const MT* beta2)
      : beta1_pow(beta1), beta2_pow(beta2) {}

  __device__ __forceinline__ MT GetBeta1() const { return *beta1_pow; }
  __device__ __forceinline__ MT GetBeta2() const { return *beta2_pow; }
};

// Unified kernel template
template <typename T,   // Parameter type
          typename TG,  // Gradient type
          typename MT,  // Multi-precision type
          typename TM,  // Moment estimation type
          typename BetaAccessor>
__global__ void AdamWKernel(MT beta1,
                            MT beta2,
                            MT epsilon,
                            MT coeff,
                            MT lr_ratio,
                            const MT* lr,
                            const TG* grad,
                            const T* param,
                            T* param_out,
                            const MT* master_param,
                            MT* master_param_out,
                            const TM* moment1,
                            TM* moment1_out,
                            const TM* moment2,
                            TM* moment2_out,
                            const TM* moment2_max,
                            TM* moment2_max_out,
                            BetaAccessor beta_accessor,
                            int64_t numel,
                            bool amsgrad) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const MT lr_val = *lr * lr_ratio;

  for (int64_t i = idx; i < numel; i += gridDim.x * blockDim.x) {
    // Read data and convert precision
    MT p = master_param ? master_param[i] : static_cast<MT>(param[i]);
    const MT g = static_cast<MT>(grad[i]);
    MT m1 = static_cast<MT>(moment1[i]);  // Automatic type conversion
    MT m2 = static_cast<MT>(moment2[i]);

    // Get beta powers
    const MT beta1_pow = beta_accessor.GetBeta1();
    const MT beta2_pow = beta_accessor.GetBeta2();

    // Weight decay
    p *= (1.0f - lr_val * coeff);

    // Update first moment
    m1 = beta1 * m1 + (1.0f - beta1) * g;
    // Update second moment
    m2 = beta2 * m2 + (1.0f - beta2) * g * g;

    // Calculate denominator
    MT denom;
    if (amsgrad) {
      const MT m2_max = static_cast<MT>(moment2_max[i]);
      const MT m2_max_new = max(m2, m2_max);
      moment2_max_out[i] =
          static_cast<TM>(m2_max_new);  // Convert back to storage type
      denom = sqrt(m2_max_new) / sqrt(1.0f - beta2_pow) + epsilon;
    } else {
      denom = sqrt(m2) / sqrt(1.0f - beta2_pow) + epsilon;
    }

    // Update parameters
    p += (m1 / denom) * (-lr_val / (1.0f - beta1_pow));

    // Write back results
    moment1_out[i] = static_cast<TM>(m1);  // Convert back to storage type
    moment2_out[i] = static_cast<TM>(m2);
    param_out[i] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[i] = p;
    }
  }
}

// Beta power update kernel
template <typename MT>
__global__ void UpdateBetaPowKernel(MT beta1,
                                    MT beta2,
                                    const MT* beta1_pow,
                                    const MT* beta2_pow,
                                    MT* beta1_pow_out,
                                    MT* beta2_pow_out) {
  beta1_pow_out[0] = beta1 * beta1_pow[0];
  beta2_pow_out[0] = beta2 * beta2_pow[0];
}

template <typename T, typename Context>
void AdamwDenseKernel(const Context& dev_ctx,
                      const DenseTensor& param,
                      const DenseTensor& grad,
                      const DenseTensor& learning_rate,
                      const DenseTensor& moment1,
                      const DenseTensor& moment2,
                      const paddle::optional<DenseTensor>& moment2_max,
                      const DenseTensor& beta1_pow,
                      const DenseTensor& beta2_pow,
                      const paddle::optional<DenseTensor>& master_param,
                      const paddle::optional<DenseTensor>& skip_update,
                      const Scalar& beta1,
                      const Scalar& beta2,
                      const Scalar& epsilon,
                      float lr_ratio,
                      float coeff,
                      bool with_decay,
                      bool lazy_mode,
                      int64_t min_row_size_to_use_multithread,
                      bool multi_precision,
                      bool use_global_beta_pow,
                      bool amsgrad,
                      DenseTensor* param_out,
                      DenseTensor* moment1_out,
                      DenseTensor* moment2_out,
                      DenseTensor* moment2_max_out,
                      DenseTensor* beta1_pow_out,
                      DenseTensor* beta2_pow_out,
                      DenseTensor* master_param_outs) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  constexpr int kThreadsPerBlock = 512;
  const int64_t numel = param.numel();
  const int blocks = (numel + kThreadsPerBlock - 1) / kThreadsPerBlock;

  // Skip update logic
  if (skip_update.is_initialized() && skip_update->data<bool>()[0]) {
    phi::Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    phi::Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    phi::Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    if (amsgrad) {
      phi::Copy(dev_ctx,
                moment2_max.get(),
                dev_ctx.GetPlace(),
                false,
                moment2_max_out);
    }
    if (!use_global_beta_pow) {
      phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
      phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    }
    return;
  }

  // Prepare parameters
  const MPDType beta1_val = beta1.to<MPDType>();
  const MPDType beta2_val = beta2.to<MPDType>();
  const MPDType epsilon_val = epsilon.to<MPDType>();
  const MPDType coeff_val =
      with_decay ? static_cast<MPDType>(coeff) : MPDType(0.0);
  const MPDType lr_ratio_val = static_cast<MPDType>(lr_ratio);

  // Master parameter pointer
  const MPDType* master_in =
      multi_precision ? master_param->data<MPDType>() : nullptr;
  MPDType* master_out = multi_precision
                            ? dev_ctx.template Alloc<MPDType>(master_param_outs)
                            : nullptr;

  // Determine BetaPow location
  const bool beta_pow_on_cpu =
      beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace();

  // Determine gradient type
  const bool use_bfloat32_grad = grad.dtype() == phi::DataType::FLOAT32;
  // Determine moment type
  const bool use_bfloat16_moments =
      moment1.dtype() == phi::DataType::BFLOAT16 &&
      moment2.dtype() == phi::DataType::BFLOAT16;

#define LAUNCH_ADAMW_KERNEL(MOMENT_T)                                          \
  if (beta_pow_on_cpu) {                                                       \
    BetaPowAccessor<MPDType, true> accessor(beta1_pow.data<MPDType>(),         \
                                            beta2_pow.data<MPDType>());        \
    if (use_bfloat32_grad) {                                                   \
      AdamWKernel<T, float, MPDType, MOMENT_T, BetaPowAccessor<MPDType, true>> \
          <<<blocks, kThreadsPerBlock, 0, dev_ctx.stream()>>>(                 \
              beta1_val,                                                       \
              beta2_val,                                                       \
              epsilon_val,                                                     \
              coeff_val,                                                       \
              lr_ratio_val,                                                    \
              learning_rate.data<MPDType>(),                                   \
              grad.data<float>(),                                              \
              param.data<T>(),                                                 \
              dev_ctx.template Alloc<T>(param_out),                            \
              master_in,                                                       \
              master_out,                                                      \
              moment1.data<MOMENT_T>(),                                        \
              dev_ctx.template Alloc<MOMENT_T>(moment1_out),                   \
              moment2.data<MOMENT_T>(),                                        \
              dev_ctx.template Alloc<MOMENT_T>(moment2_out),                   \
              moment2_max ? moment2_max->data<MOMENT_T>() : nullptr,           \
              amsgrad ? dev_ctx.template Alloc<MOMENT_T>(moment2_max_out)      \
                      : nullptr,                                               \
              accessor,                                                        \
              numel,                                                           \
              amsgrad);                                                        \
    } else {                                                                   \
      AdamWKernel<T, T, MPDType, MOMENT_T, BetaPowAccessor<MPDType, true>>     \
          <<<blocks, kThreadsPerBlock, 0, dev_ctx.stream()>>>(                 \
              beta1_val,                                                       \
              beta2_val,                                                       \
              epsilon_val,                                                     \
              coeff_val,                                                       \
              lr_ratio_val,                                                    \
              learning_rate.data<MPDType>(),                                   \
              grad.data<T>(),                                                  \
              param.data<T>(),                                                 \
              dev_ctx.template Alloc<T>(param_out),                            \
              master_in,                                                       \
              master_out,                                                      \
              moment1.data<MOMENT_T>(),                                        \
              dev_ctx.template Alloc<MOMENT_T>(moment1_out),                   \
              moment2.data<MOMENT_T>(),                                        \
              dev_ctx.template Alloc<MOMENT_T>(moment2_out),                   \
              moment2_max ? moment2_max->data<MOMENT_T>() : nullptr,           \
              amsgrad ? dev_ctx.template Alloc<MOMENT_T>(moment2_max_out)      \
                      : nullptr,                                               \
              accessor,                                                        \
              numel,                                                           \
              amsgrad);                                                        \
    }                                                                          \
  } else {                                                                     \
    BetaPowAccessor<MPDType, false> accessor(beta1_pow.data<MPDType>(),        \
                                             beta2_pow.data<MPDType>());       \
    if (use_bfloat32_grad) {                                                   \
      AdamWKernel<T,                                                           \
                  float,                                                       \
                  MPDType,                                                     \
                  MOMENT_T,                                                    \
                  BetaPowAccessor<MPDType, false>>                             \
          <<<blocks, kThreadsPerBlock, 0, dev_ctx.stream()>>>(                 \
              beta1_val,                                                       \
              beta2_val,                                                       \
              epsilon_val,                                                     \
              coeff_val,                                                       \
              lr_ratio_val,                                                    \
              learning_rate.data<MPDType>(),                                   \
              grad.data<float>(),                                              \
              param.data<T>(),                                                 \
              dev_ctx.template Alloc<T>(param_out),                            \
              master_in,                                                       \
              master_out,                                                      \
              moment1.data<MOMENT_T>(),                                        \
              dev_ctx.template Alloc<MOMENT_T>(moment1_out),                   \
              moment2.data<MOMENT_T>(),                                        \
              dev_ctx.template Alloc<MOMENT_T>(moment2_out),                   \
              moment2_max ? moment2_max->data<MOMENT_T>() : nullptr,           \
              amsgrad ? dev_ctx.template Alloc<MOMENT_T>(moment2_max_out)      \
                      : nullptr,                                               \
              accessor,                                                        \
              numel,                                                           \
              amsgrad);                                                        \
    } else {                                                                   \
      AdamWKernel<T, T, MPDType, MOMENT_T, BetaPowAccessor<MPDType, false>>    \
          <<<blocks, kThreadsPerBlock, 0, dev_ctx.stream()>>>(                 \
              beta1_val,                                                       \
              beta2_val,                                                       \
              epsilon_val,                                                     \
              coeff_val,                                                       \
              lr_ratio_val,                                                    \
              learning_rate.data<MPDType>(),                                   \
              grad.data<T>(),                                                  \
              param.data<T>(),                                                 \
              dev_ctx.template Alloc<T>(param_out),                            \
              master_in,                                                       \
              master_out,                                                      \
              moment1.data<MOMENT_T>(),                                        \
              dev_ctx.template Alloc<MOMENT_T>(moment1_out),                   \
              moment2.data<MOMENT_T>(),                                        \
              dev_ctx.template Alloc<MOMENT_T>(moment2_out),                   \
              moment2_max ? moment2_max->data<MOMENT_T>() : nullptr,           \
              amsgrad ? dev_ctx.template Alloc<MOMENT_T>(moment2_max_out)      \
                      : nullptr,                                               \
              accessor,                                                        \
              numel,                                                           \
              amsgrad);                                                        \
    }                                                                          \
  }

  // Select template instantiation based on moment type
  if (use_bfloat16_moments) {
    LAUNCH_ADAMW_KERNEL(bfloat16)
  } else {
    LAUNCH_ADAMW_KERNEL(MPDType)
  }
#undef LAUNCH_ADAMW_KERNEL

  // Update beta_pow
  if (!use_global_beta_pow) {
    if (beta_pow_on_cpu) {
      auto* beta1_pow_out_data =
          dev_ctx.template HostAlloc<MPDType>(beta1_pow_out);
      auto* beta2_pow_out_data =
          dev_ctx.template HostAlloc<MPDType>(beta2_pow_out);
      beta1_pow_out_data[0] = beta1_val * beta1_pow.data<MPDType>()[0];
      beta2_pow_out_data[0] = beta2_val * beta2_pow.data<MPDType>()[0];
    } else {
      UpdateBetaPowKernel<MPDType><<<1, 1, 0, dev_ctx.stream()>>>(
          beta1_val,
          beta2_val,
          beta1_pow.data<MPDType>(),
          beta2_pow.data<MPDType>(),
          dev_ctx.template Alloc<MPDType>(beta1_pow_out),
          dev_ctx.template Alloc<MPDType>(beta2_pow_out));
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(adamw,
                   GPU,
                   ALL_LAYOUT,
                   phi::AdamwDenseKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(9).SetBackend(phi::Backend::ALL_BACKEND);

  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(5).SetBackend(phi::Backend::UNDEFINED);
}
