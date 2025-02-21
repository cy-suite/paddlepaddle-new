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

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "paddle/phi/kernels/funcs/aligned_vector.h"


namespace phi {
namespace fusion {


template <typename T>
T *SafeGetTensorPtr(const DenseTensor &t) {
  return const_cast<T*>(t.data<T>());
}

template <typename T>
T *SafeGetTensorPtr(const DenseTensor *t) {
  return t ? SafeGetTensorPtr<T>(*t) : nullptr;
}

template <typename T>
T *SafeGetTensorPtr(const paddle::optional<DenseTensor>& t) {
  return t ? SafeGetTensorPtr<T>(t.get()) : nullptr;
}

__forceinline__ __host__ __device__ int div_up(int a, int b) {
  return (a + b - 1) / b;
}

enum PosEncMode {
  kNonePos,
  kRoPE,
  kAliBi
};

enum AppendCacheType {
  CacheT,
  CacheInt8Hw,
  CacheInt4CwZp
};

template<typename T>
struct cascade_attn_type_traits {
  using type = T;
};

template<>
struct cascade_attn_type_traits<phi::dtype::bfloat16> {
  using type = __nv_bfloat16;
};

template<>
struct cascade_attn_type_traits<phi::dtype::float16> {
  using type = half;
};

template<typename T>
struct cascade_attn_nv_type2_traits {
  using type = T;
};

template<>
struct cascade_attn_nv_type2_traits<__nv_bfloat16> {
  using type = __nv_bfloat162;
};

template<>
struct cascade_attn_nv_type2_traits<half> {
  using type = half2;
};

template<AppendCacheType cache_type>
struct vec_traits {
  using type = b128_t;
};

template<>
struct vec_traits<AppendCacheType::CacheInt8Hw> {
  using type = b64_t;
};

template<>
struct vec_traits<AppendCacheType::CacheInt4CwZp> {
  using type = b32_t;
};

template<typename T, AppendCacheType cache_type>
struct cache_type_traits {
  using type = T;
};

template<typename T>
struct cache_type_traits<T, AppendCacheType::CacheInt8Hw> {
  using type = uint8_t;
};

template<typename T>
struct cache_type_traits<T, AppendCacheType::CacheInt4CwZp> {
  using type = uint8_t;
};

__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x, uint32_t y) {
  return (x > y) ? x - y : 0U;
}

/******************************FASTER CAST*********************************/
inline __device__ static void convert_int8(__nv_bfloat16* result, const uint32_t& source) { // 4 int8 each time
  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(result);
  uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

  static constexpr uint32_t fp32_base = 0x4B000000;
  float fp32_intermediates[4];

  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);
  fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7651);
  fp32_intermediates_casted[2] = __byte_perm(i8s, fp32_base, 0x7652);
  fp32_intermediates_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);

#pragma unroll
  for (int ii = 0; ii < 4; ++ii) {
    fp32_intermediates[ii] -= 8388736.f;// (8388608.f + 128.f);
  }

#pragma unroll
  for (int ii = 0; ii < 2; ++ii) {
    bf16_result_ptr[ii] = __byte_perm(fp32_intermediates_casted[2 * ii + 0],
                                      fp32_intermediates_casted[2 * ii + 1],
                                      0x7632);
  }
}

inline __device__ static void convert_int8(half* result, const uint32_t& source) { // 4 int8 each time
  uint32_t* fp16_result_ptr = reinterpret_cast<uint32_t*>(result);
  uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);
  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(fp16_result_ptr[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(fp16_result_ptr[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(fp16_result_ptr[0]) : "r"(fp16_result_ptr[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(fp16_result_ptr[1]) : "r"(fp16_result_ptr[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
}

inline __device__ static void convert_int4(__nv_bfloat16* result, const uint32_t& source) { // 8 int4 each time
  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(result);

  static constexpr uint32_t immLut  = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t MASK = 0x0f0f0f0f;  // 0xf -> 0b1111 select 0,4
  static constexpr uint32_t I4s_TO_FP32s_MAGIC_NUM = 0x43434343;
  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;

  uint32_t tmp1 = source & MASK; // 0 1 2 3
  uint32_t tmp2 = source >> 4 & MASK; // 4 5 6 7

  bf16_result_ptr[0] = __byte_perm(tmp1,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_01); // 0 1
  bf16_result_ptr[1] = __byte_perm(tmp1,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_23); // 2 3
  bf16_result_ptr[2] = __byte_perm(tmp2,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_01); // 4 5
  bf16_result_ptr[3] = __byte_perm(tmp2,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_23); // 6 7
}

inline __device__ static void convert_int4(half* result, const uint32_t& source) { // 7 5 3 1 6 4 2 0
  uint32_t* fp16_result_ptr = reinterpret_cast<uint32_t*>(result);

  static constexpr uint32_t MASK = 0x0f0f0f0f;  // 0xf -> 0b1111 select 0,1;   7 5 3 1 6 4 2 0
  static constexpr uint32_t I4s_TO_FP32s_MAGIC_NUM = 0x64646464;
  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;

  uint32_t tmp1 = source & MASK; // 0 1 2 3
  uint32_t tmp2 = source >> 4 & MASK; // 4 5 6 7
  fp16_result_ptr[0] = __byte_perm(tmp1,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_01); // 0 1
  fp16_result_ptr[1] = __byte_perm(tmp1,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_23); // 2 3
  fp16_result_ptr[2] = __byte_perm(tmp2,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_01); // 4 5
  fp16_result_ptr[3] = __byte_perm(tmp2,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_23); // 6 7
}

inline __device__ static void convert_int4_k_i42i8_f(half *result,
                                                     int8_t *result_i8,
                                                     const uint32_t& source,
                                                     half (*cache_k_scale_frag)[8],
                                                     half (*cache_k_zp_frag)[8],
                                                     int fy) {
  uint32_t* fp16_result_ptr = reinterpret_cast<uint32_t*>(result);
  uint32_t* result_i8_ui32 = reinterpret_cast<uint32_t*>(result_i8);

  static constexpr uint32_t MASK = 0x0f0f0f0f;  // 0xf -> 0b1111 select 0,1;   7 5 3 1 6 4 2 0
  static constexpr uint32_t I4s_TO_FP32s_MAGIC_NUM = 0x64646464;
  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;

  static constexpr uint32_t mask_for_int8 = 0x6420;

  const half2 MAGIC_NUMBER = make_half2(1151.f, 1151.f); // 1024 + 127

  uint32_t tmp1 = source & MASK; // 0 1 2 3
  uint32_t tmp2 = source >> 4 & MASK; // 4 5 6 7
  fp16_result_ptr[0] = __byte_perm(tmp1,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_01); // 0 1
  fp16_result_ptr[1] = __byte_perm(tmp1,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_23); // 2 3
  fp16_result_ptr[2] = __byte_perm(tmp2,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_01); // 4 5
  fp16_result_ptr[3] = __byte_perm(tmp2,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_23); // 6 7
  half2 *result_h2 = reinterpret_cast<half2*>(result);
  half2 *cache_k_zp_frag_h2 = reinterpret_cast<half2*>(cache_k_zp_frag[fy]);
  half2 *cache_k_scale_frag_h2 = reinterpret_cast<half2*>(cache_k_scale_frag[fy]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    result_h2[i] = __hadd2(__hmul2(__hsub2(result_h2[i], cache_k_zp_frag_h2[i]), cache_k_scale_frag_h2[i]), MAGIC_NUMBER);
  }

  result_i8_ui32[0] = __byte_perm(
    fp16_result_ptr[0],
    fp16_result_ptr[1],
    mask_for_int8);
  result_i8_ui32[1] = __byte_perm(
    fp16_result_ptr[2],
    fp16_result_ptr[3],
    mask_for_int8);

  uint8_t *result_ui8 = reinterpret_cast<uint8_t*>(result_i8);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    result_i8[i] = static_cast<int16_t>(result_ui8[i]) - 127;
  }
}

inline __device__ static void convert_int4_v_i42i8_f(half *result,
                                                     int8_t *result_i8,
                                                     const uint32_t& source,
                                                     half (*cache_v_scale_frag)[4],
                                                     half (*cache_v_zp_frag)[4],
                                                     int fy,
                                                     int fzi) {
  uint32_t* fp16_result_ptr = reinterpret_cast<uint32_t*>(result);
  uint32_t* result_i8_ui32 = reinterpret_cast<uint32_t*>(result_i8);

  static constexpr uint32_t MASK = 0x0f0f0f0f;  // 0xf -> 0b1111 select 0,1;   7 5 3 1 6 4 2 0
  static constexpr uint32_t I4s_TO_FP32s_MAGIC_NUM = 0x64646464;
  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;

  static constexpr uint32_t mask_for_int8 = 0x6420;

  const half2 MAGIC_NUMBER = make_half2(1151.f, 1151.f); // 1024 + 127

  uint32_t tmp1 = source & MASK; // 0 1 2 3
  uint32_t tmp2 = source >> 4 & MASK; // 4 5 6 7
  fp16_result_ptr[0] = __byte_perm(tmp1,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_01); // 0 1
  fp16_result_ptr[1] = __byte_perm(tmp1,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_23); // 2 3
  fp16_result_ptr[2] = __byte_perm(tmp2,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_01); // 4 5
  fp16_result_ptr[3] = __byte_perm(tmp2,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_23); // 6 7

  half2 *result_h2 = reinterpret_cast<half2*>(result);
  half2 cache_v_scale_h2 = make_half2(cache_v_scale_frag[fy][fzi], cache_v_scale_frag[fy][fzi]);
  half2 cache_v_zp_h2 = make_half2(cache_v_zp_frag[fy][fzi], cache_v_zp_frag[fy][fzi]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    result_h2[i] = __hadd2(__hmul2(__hsub2(result_h2[i], cache_v_zp_h2), cache_v_scale_h2), MAGIC_NUMBER);
  }

  result_i8_ui32[0] = __byte_perm(
    fp16_result_ptr[0],
    fp16_result_ptr[1],
    mask_for_int8);
  result_i8_ui32[1] = __byte_perm(
    fp16_result_ptr[2],
    fp16_result_ptr[3],
    mask_for_int8);

  uint8_t *result_ui8 = reinterpret_cast<uint8_t*>(result_i8);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    result_i8[i] = static_cast<int16_t>(result_ui8[i]) - 127;
  }
}

inline __device__ static void convert_int4_k_i42i8(int8_t* result,
                                                   const uint32_t& source,
                                                   int (*cache_k_scale_frag)[8],
                                                   int (*cache_k_zp_frag)[8],
                                                   int fy) {
  static constexpr uint32_t MASK = 0x0f0f0f0f;
  uint32_t tmp1 = source & MASK; // 0 1 2 3
  uint32_t tmp2 = source >> 4 & MASK; // 4 5 6 7

  int8_t *tmp1_i8 = reinterpret_cast<int8_t*>(&tmp1);
  int8_t *tmp2_i8 = reinterpret_cast<int8_t*>(&tmp2);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    result[i] = static_cast<int8_t>(((static_cast<int>(tmp1_i8[i]) - cache_k_zp_frag[fy][i]) * cache_k_scale_frag[fy][i]) >> 24);
    result[i + 4] = static_cast<int8_t>(((static_cast<int>(tmp2_i8[i]) - cache_k_zp_frag[fy][i + 4]) * cache_k_scale_frag[fy][i + 4]) >> 24);
  }
}

inline __device__ static void convert_int4_v_i42i8(int8_t* result,
                                                   const uint32_t& source,
                                                   int (*cache_v_scale_frag)[4],
                                                   int (*cache_v_zp_frag)[4],
                                                   int fy,
                                                   int fzi) {
  static constexpr uint32_t MASK = 0x0f0f0f0f;
  uint32_t tmp1 = source & MASK; // 0 1 2 3
  uint32_t tmp2 = source >> 4 & MASK; // 4 5 6 7

  int8_t *tmp1_i8 = reinterpret_cast<int8_t*>(&tmp1);
  int8_t *tmp2_i8 = reinterpret_cast<int8_t*>(&tmp2);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    result[i] = static_cast<int8_t>(((static_cast<int>(tmp1_i8[i]) - cache_v_zp_frag[fy][fzi]) * cache_v_scale_frag[fy][fzi]) >> 24);
    result[i + 4] = static_cast<int8_t>(((static_cast<int>(tmp2_i8[i]) - cache_v_zp_frag[fy][fzi]) * cache_v_scale_frag[fy][fzi]) >> 24);
  }
}

inline __device__ static void convert_int4_k_i42i8_v2(uint32_t* result,
                                                      const uint32_t& source,
                                                      int (*cache_k_scale_frag)[8],
                                                      int (*cache_k_zp_frag)[8],
                                                      int fy) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t mask_for_int8 = 0x6420;
  uint32_t tmp1 = source & MASK; // 0 1
  uint32_t tmp2 = source >> 4 & MASK; // 2 3

  int16_t *tmp1_i16 = reinterpret_cast<int16_t*>(&tmp1);
  int16_t *tmp2_i16 = reinterpret_cast<int16_t*>(&tmp2);

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    // 0 1 2 3
    tmp1_i16[i] = ((tmp1_i16[i] - cache_k_zp_frag[fy][i]) * cache_k_scale_frag[fy][i]) >> 8;
    tmp2_i16[i] = ((tmp2_i16[i] - cache_k_zp_frag[fy][i + 2]) * cache_k_scale_frag[fy][i + 2]) >> 8;
  }
  result[0] = __byte_perm(
    tmp1,
    tmp2,
    mask_for_int8);

  tmp1 = source >> 8 & MASK; // 4 5
  tmp2 = source >> 12 & MASK; // 6 7
  tmp1_i16 = reinterpret_cast<int16_t*>(&tmp1);
  tmp2_i16 = reinterpret_cast<int16_t*>(&tmp2);

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    // 4 5 6 7
    tmp1_i16[i] = ((tmp1_i16[i] - cache_k_zp_frag[fy][i + 4]) * cache_k_scale_frag[fy][i + 4]) >> 8;
    tmp2_i16[i] = ((tmp2_i16[i] - cache_k_zp_frag[fy][i + 6]) * cache_k_scale_frag[fy][i + 6]) >> 8;
  }
  result[1] = __byte_perm(
    tmp1,
    tmp2,
    mask_for_int8);
}

inline __device__ static void convert_int4_v_i42i8_v2(uint32_t* result,
                                                      const uint32_t& source,
                                                      int cache_v_scale_frag,
                                                      int cache_v_zp_frag) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t mask_for_int8 = 0x6420;
  uint32_t tmp1 = source & MASK; // 0 1
  uint32_t tmp2 = source >> 4 & MASK; // 2 3

  int16_t *tmp1_i16 = reinterpret_cast<int16_t*>(&tmp1);
  int16_t *tmp2_i16 = reinterpret_cast<int16_t*>(&tmp2);

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    // 0 1 2 3
    tmp1_i16[i] = ((tmp1_i16[i] - cache_v_zp_frag) * cache_v_scale_frag) >> 8;
    tmp2_i16[i] = ((tmp2_i16[i] - cache_v_zp_frag) * cache_v_scale_frag) >> 8;
  }
  result[0] = __byte_perm(
    tmp1,
    tmp2,
    mask_for_int8);

  tmp1 = source >> 8 & MASK; // 4 5
  tmp2 = source >> 12 & MASK; // 6 7
  tmp1_i16 = reinterpret_cast<int16_t*>(&tmp1);
  tmp2_i16 = reinterpret_cast<int16_t*>(&tmp2);

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    // 4 5 6 7
    tmp1_i16[i] = ((tmp1_i16[i] - cache_v_zp_frag) * cache_v_scale_frag) >> 8;
    tmp2_i16[i] = ((tmp2_i16[i] - cache_v_zp_frag) * cache_v_scale_frag) >> 8;
  }
  result[1] = __byte_perm(
    tmp1,
    tmp2,
    mask_for_int8);
}

/******************* vec_t type cast *******************/

template <typename dst_t, typename src_t, size_t vec_size>
__forceinline__ __host__ __device__  void vec_cast(dst_t* dst, const src_t* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    dst[i] = src[i];
  }
}

template <size_t vec_size>
__forceinline__ __host__ __device__  void vec_cast<float, half>(float* dst, const half* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size / 2; ++i) {
    ((float2*)dst)[i] = __half22float2(((half2*)src)[i]);
  }
}

template <size_t vec_size>
__forceinline__ __host__ __device__  void vec_cast<half, float>(half* dst, const float* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size / 2; ++i) {
    ((half2*)dst)[i] = __float22half2_rn(((float2*)src)[i]);
  }
}

template <size_t vec_size>
__forceinline__ __host__ __device__  void vec_cast<float, nv_bfloat16>(float* dst, const nv_bfloat16* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size / 2; ++i) {
    ((float2*)dst)[i] = __bfloat1622float2(((nv_bfloat162*)src)[i]);
  }
}

template <size_t vec_size>
__forceinline__ __host__ __device__  void vec_cast<nv_bfloat16, float>(nv_bfloat16* dst, const float* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size / 2; ++i) {
    ((nv_bfloat162*)dst)[i] = __float22bfloat162_rn(((float2*)src)[i]);
  }
}

#define DISPATCH_CAUSAL(causal, CAUSAL, ...) \
  if (causal) {                              \
    constexpr bool CAUSAL = true;            \
    __VA_ARGS__                              \
  }


#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
  if (group_size == 4) {                                     \
    constexpr size_t GROUP_SIZE = 4;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 8) {                              \
    constexpr size_t GROUP_SIZE = 8;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 12) {                             \
    constexpr size_t GROUP_SIZE = 12;                        \
    __VA_ARGS__                                              \
  } else {                                \
    PADDLE_THROW(phi::errors::InvalidArgument("not support the group_size")); \
  }
  // } else if (group_size == 4) {                              \
  //   constexpr size_t GROUP_SIZE = 4;                         \
  //   __VA_ARGS__                                              \
  // } else if (group_size == 7) {                              \
  //   constexpr size_t GROUP_SIZE = 7;                         \
  //   __VA_ARGS__                                              \
  // } else if (group_size == 8) {                              \
  //   constexpr size_t GROUP_SIZE = 8;                         \
  //   __VA_ARGS__                                              \


#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)              \
  switch (head_dim) {                                           \
    case 128: {                                                 \
      constexpr size_t HEAD_DIM = 128;                          \
      __VA_ARGS__                                               \
      break;                                                    \
    }                                                           \
    default: {                                                  \
      PADDLE_THROW(phi::errors::InvalidArgument("not support the head_dim")); \
    }                                                           \
  }

#define DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, ...)   \
  if (block_size == 64) {                                  \
    constexpr size_t BLOCK_SIZE = 64;                      \
    __VA_ARGS__                                            \
  } else { \
    PADDLE_THROW(phi::errors::InvalidArgument("not support the head_dim")); \
  }

#define DISPATCH_BLOCKSHAPE_Q(block_shape_q, BLOCK_SHAPE_Q, NUM_WARP_Q, ...)  \
  if (block_shape_q <= 16) {                                                  \
    constexpr size_t BLOCK_SHAPE_Q = 16;                                      \
    constexpr size_t NUM_WARP_Q = 1;                                          \
    __VA_ARGS__                                                               \
  } else if (block_shape_q <= 32) {                                           \
    constexpr size_t BLOCK_SHAPE_Q = 32;                                      \
    constexpr size_t NUM_WARP_Q = 1;                                          \
    __VA_ARGS__                                                               \
  } else if (block_shape_q <= 64) {                                           \
    constexpr size_t BLOCK_SHAPE_Q = 64;                                      \
    constexpr size_t NUM_WARP_Q = 4;                                          \
    __VA_ARGS__                                                               \
  } else {                                                                    \
    constexpr size_t BLOCK_SHAPE_Q = 128;                                     \
    constexpr size_t NUM_WARP_Q = 4;                                          \
    __VA_ARGS__                                                               \
  }

#define DISPATCH_USE_SYSTEM(use_system, USE_SYSTEM, ...)     \
  if (use_system == 1) {                                    \
    constexpr bool USE_SYSTEM = 1;                          \
    __VA_ARGS__                                             \
  } else {                                                  \
    constexpr bool USE_SYSTEM = 0;                          \
    __VA_ARGS__                                             \
  }


#define DISPATCH_IS_ROPE(rope_emb, CAL_ROPE, ...)       \
  if (rope_emb) {                                         \
    constexpr bool CAL_ROPE = 1;                          \
    __VA_ARGS__                                           \
  } else {                                                \
    constexpr bool CAL_ROPE = 0;                          \
    __VA_ARGS__                                           \
  }


template<typename T>
void print_tensor(const phi::DenseTensor& x, const std::string file, const int line, const int num = 100, const int offset = 0) {
  if (VLOG_IS_ON(2)) {
    std::vector<T> tmp(num);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(tmp.data(), x.data<T>() + offset, num * sizeof(T), cudaMemcpyDeviceToHost));
    std::cout << "File " << file << " : " << line << "\n";
    for (int i = 0; i < num; i++) {
      std::cout << (float)tmp[i] << " ";
    }
    std::cout << std::endl;
  }
}

} // namespace fusion
} // namespace phi
