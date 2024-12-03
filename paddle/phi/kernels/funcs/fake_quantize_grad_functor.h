/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/impl/clip_kernel_impl.h"

namespace phi {
namespace funcs {

template <typename T>
inline HOSTDEVICE T inverse(T s) {
  T eps = static_cast<T>(1e-6);
  T one = static_cast<T>(1.0);
  return s <= static_cast<T>(1e-30) ? one / (s + eps) : one / s;
}

template <typename T>
struct Compare {
  bool operator()(const T a, const T b) { return (std::abs(a) < std::abs(b)); }
};

template <typename T>
inline HOSTDEVICE T roundWithTiesToEven(T x) {
  T xLower = floor(x);
  T xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  T dLower = x - xLower;
  T dUpper = xUpper - x;
  return static_cast<T>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}

template <typename T>
class LSQClipGradFunctor {
 public:
  explicit LSQClipGradFunctor(const T min_bound, const T max_bound)
      : min_bound_(min_bound), max_bound_(max_bound) {}
  HOSTDEVICE T operator()(const T x) const {
    T out = static_cast<T>(0);
    if (out >= min_bound_ && out <= max_bound_) {
      out = x;
    }
    return out;
  }

 private:
  T min_bound_;
  T max_bound_;
};

template <typename T>
class LSQScaleGradFunctor {
 public:
  explicit LSQScaleGradFunctor(const int bin_cnt,
                               const float lsq_factor,
                               const T inv_s,
                               const int round_type)
      : bin_cnt_(bin_cnt),
        lsq_factor_(lsq_factor),
        inv_s_(inv_s),
        round_type_(round_type) {}
  HOSTDEVICE T operator()(const T x) const {
    T x_quant_round = x;
    T x_quant = x * inv_s_;
    T min_bound_ = -bin_cnt_ - static_cast<T>(1);
    T max_bound_ = bin_cnt_;

    if (round_type_ == 0) {
      T x_0 = roundWithTiesToEven(x_quant);
      x_0 = x_0 > max_bound_ ? max_bound_ : x_0;
      x_0 = x_0 < min_bound_ ? min_bound_ : x_0;
      x_quant_round = x_0;
    } else {
      T x_1 = round(x_quant);
      x_1 = x_1 > max_bound_ ? max_bound_ : x_1;
      x_1 = x_1 < min_bound_ ? min_bound_ : x_1;
      x_quant_round = x_1;
    }
    T elem = x_quant_round - x_quant;
    if (x_quant < min_bound_) {
      elem = min_bound_;
    } else if (x_quant > max_bound_) {
      elem = max_bound_;
    }
    elem = elem * lsq_factor_;
    return static_cast<T>(elem);
  }

 private:
  int bin_cnt_;
  T lsq_factor_;
  T inv_s_;
  int round_type_;
};

template <typename Context, typename T>
class FakeQuantizeDequantizeGradLSQFunctor {
 public:
  void operator()(const Context &dev_ctx,
                  const DenseTensor &x,
                  const DenseTensor &scale,
                  const DenseTensor &out_grad,
                  const float lsq_factor,
                  const int bin_cnt,
                  const int round_type,
                  DenseTensor *x_grad,
                  DenseTensor *scale_grad);
};

}  // namespace funcs
}  // namespace phi
