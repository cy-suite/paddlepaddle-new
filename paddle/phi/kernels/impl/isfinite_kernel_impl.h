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
#include <cmath>
#include <string>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/kernels/funcs/isfinite_functor.h"
#include "paddle/phi/kernels/isfinite_kernel.h"

namespace phi {
using Tensor = DenseTensor;

template <typename DeviceContext, typename T>
struct IsfiniteFunctor {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor& in,
                  DenseTensor* output);
};

template <typename DeviceContext, typename T>
struct IsnanFunctor {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor& in,
                  DenseTensor* output);
};

template <typename DeviceContext, typename T>
struct IsinfFunctor {
  void operator()(const DeviceContext& ctx,
                  const DenseTensor& in,
                  DenseTensor* output);
};

template <typename T>
struct IsfiniteFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* out_data = ctx.template Alloc<bool>(output);
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      const T a = in_a[i];
      out_data[i] = std::isfinite(a);
    }
  }
};

template <>
struct IsfiniteFunctor<phi::CPUContext, phi::dtype::float16> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<phi::dtype::float16>();
    auto* out_data = ctx.template Alloc<bool>(output);
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      const phi::dtype::float16& a = in_a[i];
      out_data[i] = phi::dtype::isfinite(a);
    }
  }
};

template <>
struct IsfiniteFunctor<phi::CPUContext, phi::dtype::bfloat16> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<phi::dtype::bfloat16>();
    auto* out_data = ctx.template Alloc<bool>(output);
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      const phi::dtype::bfloat16& a = in_a[i];
      out_data[i] = phi::dtype::isfinite(a);
    }
  }
};

template <typename T>
struct IsinfFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* out_data = ctx.template Alloc<bool>(output);
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      const T a = in_a[i];
      out_data[i] = std::isinf(a);
    }
  }
};

template <>
struct IsinfFunctor<phi::CPUContext, phi::dtype::float16> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<phi::dtype::float16>();
    auto* out_data = ctx.template Alloc<bool>(output);
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      const phi::dtype::float16& a = in_a[i];
      out_data[i] = phi::dtype::isinf(a);
    }
  }
};

template <>
struct IsinfFunctor<phi::CPUContext, phi::dtype::bfloat16> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<phi::dtype::bfloat16>();
    auto* out_data = ctx.template Alloc<bool>(output);
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      const phi::dtype::bfloat16& a = in_a[i];
      out_data[i] = phi::dtype::isinf(a);
    }
  }
};

template <typename T>
struct IsnanFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<T>();
    auto* out_data = ctx.template Alloc<bool>(output);
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      const T a = in_a[i];
      out_data[i] = std::isnan(a);
    }
  }
};

template <>
struct IsnanFunctor<phi::CPUContext, phi::dtype::float16> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<phi::dtype::float16>();
    auto* out_data = ctx.template Alloc<bool>(output);
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      const phi::dtype::float16& a = in_a[i];
      out_data[i] = phi::dtype::isnan(a);
    }
  }
};

template <>
struct IsnanFunctor<phi::CPUContext, phi::dtype::bfloat16> {
  void operator()(const phi::CPUContext& ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    auto* in_a = in.data<phi::dtype::bfloat16>();
    auto* out_data = ctx.template Alloc<bool>(output);
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      const phi::dtype::bfloat16& a = in_a[i];
      out_data[i] = phi::dtype::isnan(a);
    }
  }
};

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
__global__ void IsfiniteCUDAKernel(const T* in_data, int num, bool* out_data) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool val;
  for (int i = idx; i < num; i += blockDim.x * gridDim.x) {
    const T& a = static_cast<T>(in_data[i]);
    out_data[i] = isfinite(a);
  }
}

template <typename T>
__global__ void IsnanCUDAKernel(const T* in_data, int num, bool* out_data) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool val;
  for (int i = idx; i < num; i += blockDim.x * gridDim.x) {
    const T& a = static_cast<T>(in_data[i]);
    out_data[i] = isnan(a);
  }
}

template <typename T>
__global__ void IsinfCUDAKernel(const T* in_data, int num, bool* out_data) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool val;
  for (int i = idx; i < num; i += blockDim.x * gridDim.x) {
    const T& a = static_cast<T>(in_data[i]);
    out_data[i] = isinf(a);
  }
}

template <typename T>
struct IsfiniteFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    int num = in.numel();
    const T* in_data = in.data<T>();
    bool* out_data = dev_ctx.template Alloc<bool>(output);
    int block = 1024;
    int grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;
#ifdef PADDLE_WITH_HIP
    hipMemset(out_data, true, num * sizeof(bool));
#else
    cudaMemset(out_data, true, num * sizeof(bool));
#endif
    IsfiniteCUDAKernel<T>
        <<<grid, block, 0, dev_ctx.stream()>>>(in_data, num, out_data);
  }
};

template <typename T>
struct IsnanFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    int num = in.numel();
    const T* in_data = in.data<T>();
    bool* out_data = dev_ctx.template Alloc<bool>(output);
    int block = 1024;
    int grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;
#ifdef PADDLE_WITH_HIP
    hipMemset(out_data, true, num * sizeof(bool));
#else
    cudaMemset(out_data, true, num * sizeof(bool));
#endif
    IsnanCUDAKernel<T>
        <<<grid, block, 0, dev_ctx.stream()>>>(in_data, num, out_data);
  }
};

template <typename T>
struct IsinfFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* output) {
    int num = in.numel();
    const T* in_data = in.data<T>();
    bool* out_data = dev_ctx.template Alloc<bool>(output);
    int block = 1024;
    int grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;
#ifdef PADDLE_WITH_HIP
    hipMemset(out_data, true, num * sizeof(bool));
#else
    cudaMemset(out_data, true, num * sizeof(bool));
#endif
    IsinfCUDAKernel<T>
        <<<grid, block, 0, dev_ctx.stream()>>>(in_data, num, out_data);
  }
};
#endif

template <typename T, typename Context>
void IsfiniteKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    DenseTensor* out) {
  IsfiniteFunctor<Context, T>()(dev_ctx, x, out);
}
template <typename T, typename Context>
void IsinfKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  IsinfFunctor<Context, T>()(dev_ctx, x, out);
}
template <typename T, typename Context>
void IsnanKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  IsnanFunctor<Context, T>()(dev_ctx, x, out);
}
}  // namespace phi
