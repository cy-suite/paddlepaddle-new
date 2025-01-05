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
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {
// do not used
template <typename XPUType, XPUFCCalcType calc_type>
struct XPUConvQuantTypeTrait {};

// Because xpudnn only supports tf32 quant for bf16, we use it for all
// fc_calc_type.
template <XPUFCCalcType calc_type>
struct XPUConvQuantTypeTrait<XPUTypeBF16, calc_type> {
  using Type = tfloat32;
};

// fp16 uses itself as the default quant type.
template <XPUFCCalcType calc_type>
struct XPUConvQuantTypeTrait<XPUTypeFP16, calc_type> {
  using Type = XPUTypeFP16;
};

// template <>
// struct XPUConvQuantTypeTrait<XPUTypeFP16, XPUFCCalcType::FC_FLOAT> {
//   using Type = float;
// };

template <>
struct XPUConvQuantTypeTrait<XPUTypeFP16, XPUFCCalcType::FC_INT16> {
  using Type = int16_t;
};

// float used tf32 as the default quant type.
template <XPUFCCalcType calc_type>
struct XPUConvQuantTypeTrait<float, calc_type> {
  using Type = tfloat32;
};

template <>
struct XPUConvQuantTypeTrait<float, XPUFCCalcType::FC_FLOAT> {
  using Type = float;
};

template <>
struct XPUConvQuantTypeTrait<float, XPUFCCalcType::FC_INT16> {
  using Type = int16_t;
};

template <>
struct XPUConvQuantTypeTrait<float, XPUFCCalcType::FC_INT32> {
  using Type = int;
};

template <>
struct XPUConvQuantTypeTrait<float, XPUFCCalcType::FC_INT32_WITH_LL> {
  using Type = int_with_ll_t;
};

#define PD_PRIVATE_XPU_CONV_CASE(TYPE, calc_type, ...)                   \
  case calc_type: {                                                      \
    using TGEMM = typename XPUConvQuantTypeTrait<TYPE, calc_type>::Type; \
    __VA_ARGS__();                                                       \
    break;                                                               \
  }

#define PD_VISIT_XPU_CONV_TYPES(TYPE, calc_type, func_name, ...)             \
  do {                                                                       \
    switch (calc_type) {                                                     \
      PD_PRIVATE_XPU_CONV_CASE(TYPE, XPUFCCalcType::FC_FLOAT, __VA_ARGS__)   \
      PD_PRIVATE_XPU_CONV_CASE(TYPE, XPUFCCalcType::FC_TF32, __VA_ARGS__)    \
      PD_PRIVATE_XPU_CONV_CASE(TYPE, XPUFCCalcType::FC_FLOAT16, __VA_ARGS__) \
      PD_PRIVATE_XPU_CONV_CASE(TYPE, XPUFCCalcType::FC_INT16, __VA_ARGS__)   \
      PD_PRIVATE_XPU_CONV_CASE(TYPE, XPUFCCalcType::FC_INT32, __VA_ARGS__)   \
      PD_PRIVATE_XPU_CONV_CASE(                                              \
          TYPE, XPUFCCalcType::FC_INT32_WITH_LL, __VA_ARGS__)                \
      default:                                                               \
        PADDLE_THROW(common::errors::InvalidArgument(                        \
            "Function " #func_name " got invalid fc calc type %d",           \
            static_cast<int>(calc_type)));                                   \
    }                                                                        \
  } while (0)
}  // namespace phi
