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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace phi {
namespace fusion {

enum class MMAMode {
  kInit = 0U,
  kInplaceUpdate = 1U,
};

template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n32k32_row_col_i8i8i32(int* C, // 16
                                                                   uint32_t* A, // 4
                                                                   uint32_t* B) { // 8
  if constexpr (mma_mode == MMAMode::kInit) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0),
          "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[8]), "=r"(C[9]), "=r"(C[10]), "=r"(C[11])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[4]), "r"(B[5]), "r"(0), "r"(0),
          "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[12]), "=r"(C[13]), "=r"(C[14]), "=r"(C[15])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[6]), "r"(B[7]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]), "r"(C[5]),
          "r"(C[6]), "r"(C[7]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[8]), "=r"(C[9]), "=r"(C[10]), "=r"(C[11])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[4]), "r"(B[5]), "r"(C[8]), "r"(C[9]),
          "r"(C[10]), "r"(C[11]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[12]), "=r"(C[13]), "=r"(C[14]), "=r"(C[15])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[6]), "r"(B[7]), "r"(C[12]), "r"(C[13]),
          "r"(C[14]), "r"(C[15]));
  }
}

template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_i8i8i32(int* C, // 8
                                                                   uint32_t* A, // 4
                                                                   uint32_t* B) { // 4
  if constexpr (mma_mode == MMAMode::kInit) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]), "r"(C[5]),
          "r"(C[6]), "r"(C[7]));
  }
}

template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(float* C,
                                                                     uint32_t* A,
                                                                     uint32_t* B) {
  if constexpr (mma_mode == MMAMode::kInit) {
    if constexpr (std::is_same<T, half>::value) { // fp16
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
    } else { // bf16
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
    }
  } else {
    if constexpr (std::is_same<T, half>::value) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
    }
  }
}

template <typename DType>
__device__ __forceinline__ void rowsum_f16f16f32(float* d, DType* s) {
  static_assert(sizeof(DType) == 2, "DType must be 16bit floating data type");
  uint32_t* s_u32 = (uint32_t*)(s);
  if constexpr (std::is_same<DType, half>::value) {
    asm volatile(
        "{\n"
        ".reg .f32 ph;\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  ph,  %1,  ph},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(1006648320),
          "r"(1006648320), "f"(d[0]), "f"(d[1]));
  } else {
    asm volatile(
        "{\n"
        ".reg .f32 ph;\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,  ph,  %1,  ph},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(1065369472),
          "r"(1065369472), "f"(d[0]), "f"(d[1]));
  }
}

template <typename DType>
__device__ __forceinline__ void rowsum_f16f16f32_m16k32(float* d, DType* s) {
  static_assert(sizeof(DType) == 2, "DType must be 16bit floating data type");
  uint32_t* s_u32 = (uint32_t*)(s);
  if constexpr (std::is_same<DType, half>::value) {
    asm volatile(
        "{\n"
        ".reg .f32 ph;\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  ph,  %1,  ph},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(1006648320),
          "r"(1006648320), "f"(d[0]), "f"(d[1]));
    asm volatile(
        "{\n"
        ".reg .f32 ph;\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  ph,  %1,  ph},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[4]), "r"(s_u32[5]), "r"(s_u32[6]), "r"(s_u32[7]), "r"(1006648320),
          "r"(1006648320), "f"(d[0]), "f"(d[1]));
  } else {
    asm volatile(
        "{\n"
        ".reg .f32 ph;\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,  ph,  %1,  ph},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(1065369472),
          "r"(1065369472), "f"(d[0]), "f"(d[1]));
    asm volatile(
        "{\n"
        ".reg .f32 ph;\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,  ph,  %1,  ph},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[4]), "r"(s_u32[5]), "r"(s_u32[6]), "r"(s_u32[7]), "r"(1065369472),
          "r"(1065369472), "f"(d[0]), "f"(d[1]));
  }
}

}
}
