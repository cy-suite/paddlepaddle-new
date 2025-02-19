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

#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/fusion/gpu/append_attention/mem_util.cuh"
#include "paddle/phi/kernels/fusion/gpu/append_attention/mma_tensor_op.cuh"
#include "paddle/phi/kernels/fusion/gpu/append_attention/utils.cuh"


#define PRINT_TID 0
#define PRINT_WID 0

namespace phi {
namespace fusion {


template <typename T>
__forceinline__ __device__ float fixed_expf(float x1, float x2) {
  if constexpr (std::is_same<T, half>::value) {
    if (x1 == -5e4f) {
      return 0;
    } else {
      return __expf(x1 - x2);
    }
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    if (x1 == -3.0e+30f) {
      return 0;
    } else {
      return __expf(x1 - x2);
    }
  }
}

inline __device__ float fixed_expf(float val) {
  return __expf(val);
}

template <typename T>
__device__ __forceinline__ void apply_rope_frag(uint32_t *b_frag, uint32_t *rope_frag) {
  T *k_frag = reinterpret_cast<T*>(b_frag); // 8 elems
  T *r_frag = reinterpret_cast<T*>(rope_frag); // 8 elems
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const T input_left = k_frag[2 * i];
    const T input_right = k_frag[2 * i + 1];
    const T cos_tmp = r_frag[2 * i];
    const T sin_tmp = r_frag[2 * i + 1];
    k_frag[2 * i] = input_left * cos_tmp - input_right * sin_tmp;
    k_frag[2 * i + 1] = input_right * cos_tmp + input_left * sin_tmp;
  }
}

template <size_t vec_size, typename T>
struct prefill_softmax_state_t {
  phi::AlignedVector<T, vec_size> o;
  float m;
  int m_i;
  float d;

  __device__ __forceinline__ void init() {
    if constexpr (std::is_same<T, half>::value) {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((half2*)(&o) + i) = make_half2(0, 0);
      }
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((nv_bfloat162*)(&o) + i) = make_bfloat162(0, 0);
      }
    }
    d = 1.f;
    if constexpr (std::is_same<T, half>::value) {
      m = -5e4f;
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
      m = -3.38953e38f;
    }
  }

  __device__ __forceinline__ void init(const int mask_value) {
    if constexpr (std::is_same<T, half>::value) {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((half2*)(&o) + i) = make_half2(0, 0);
      }
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((nv_bfloat162*)(&o) + i) = make_bfloat162(0, 0);
      }
    }
    d = 1.f;
    if constexpr (std::is_same<T, half>::value) {
      m = -5e4f;
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
      m = -3.38953e38f;
    }
    m_i = mask_value;
  }

  // __device__ __forceinline__ prefill_softmax_state_t() {
  //   init();
  // }

  __device__ __forceinline__ void merge(const phi::AlignedVector<T, vec_size>& other_o,
                                        const float other_m,
                                        const float other_d) {
    float m_prev = m, d_prev = d;
    m = max(m_prev, other_m);
    const float scale1 = fixed_expf(m_prev - m), scale2 = fixed_expf(other_m - m);
    const T scale1_T = static_cast<T>(scale1), scale2_T = static_cast<T>(scale2);
    d = d_prev * scale1 + other_d * scale2;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = o[i] * scale1_T + other_o[i] * scale2_T;
    }
  }

  __device__ __forceinline__ void merge_int_m(const phi::AlignedVector<T, vec_size>& other_o,
                                              const int other_m,
                                              const float other_d,
                                              const int qk_dq_scale) {
    int m_prev = m_i;
    float d_prev = d;
    m_i = max(m_prev, other_m);
    // const float scale1 = int_exp(m_prev - m_i, exp_scale), scale2 = int_exp(other_m - m_i, exp_scale);
    const float scale1 = fixed_expf((float)(m_prev - m_i) * qk_dq_scale), scale2 = fixed_expf((float)(other_m - m_i) * qk_dq_scale);
    const T scale1_T = static_cast<T>(scale1), scale2_T = static_cast<T>(scale2);
    d = d_prev * scale1 + other_d * scale2;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = o[i] * scale1_T + other_o[i] * scale2_T;
    }
  }

  __device__ __forceinline__ void normalize() {
    const T d_t = static_cast<T>(d);
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] /= d_t;
    }
  }

};

template <typename T, uint32_t num_frags_x, uint32_t num_frags_y>
__device__ __forceinline__ void init_states(float (*o_frag)[num_frags_y][8], float (*m)[2],
                                            float (*d)[2]) {
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fx][fy][reg_id] = 0.f;
      }
    }
  }
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      if constexpr (std::is_same<T, half>::value) {
        m[fx][j] = -5e4f;
      } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        m[fx][j] = -3.0e+30f;
      }
      d[fx][j] = 1.f;
    }
  }
}


template <uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_y, uint32_t HEAD_DIM, typename T>
__device__ __forceinline__ void load_q_global_smem(T* q_ptr_base,
                                                   smem_t* q_smem,
                                                   uint32_t q_idx_base,
                                                   const uint32_t qo_upper_bound,
                                                   const uint32_t qo_n_stride,
                                                   const uint32_t qo_h_stride) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();

  const uint32_t tx = threadIdx.x, ty = threadIdx.y;

  uint32_t q_smem_offset_w = // [NUM_WARP_Q, num_frags_x, 16, head_dim]
      smem_t::get_permuted_offset<num_vecs_per_head>(ty * num_frags_x * 16 + tx / 8, tx % 8); // 4 * 64

  // q_idx_base += (tx / 8) / group_size;
  // q_ptr_base += ((tx / 8) / group_size) * qo_n_stride + ((tx / 8) % group_size) * qo_h_stride;
  const uint32_t tx_offset = tx / 8;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
    // NUM_WARP_Q * num_frags_x * 16 * head_dim
    const uint32_t base_offset = q_idx_base + fx * 16 + tx_offset;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const uint32_t offset_now = base_offset + j * 4;
      const uint32_t n_offset = offset_now / group_size;
      const uint32_t h_offset = offset_now % group_size;
      T* q_ptr = q_ptr_base + n_offset * qo_n_stride + h_offset * qo_h_stride;
#pragma unroll
      for (uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) { // (num_frags_y * 16) / (8 *  num_elems_per_128b<T>())
        // load q from gmem to smem
        q_smem->load_128b_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr,
                                                            n_offset < qo_upper_bound);
        q_smem_offset_w = q_smem->advance_offset_by_column<8>(q_smem_offset_w, fyo);
        q_ptr += 8 * num_elems_per_128b<T>();
      }
      q_smem_offset_w =
          q_smem->advance_offset_by_row<4, num_vecs_per_head>(q_smem_offset_w) - 2 * num_frags_y; // num_frags_y / 4 * 8
    }
  }
}


template <uint32_t num_frags_x, uint32_t num_frags_y, typename T>
__device__ __forceinline__ void q_smem_inplace_multiply_sm_scale(smem_t* q_smem, // [num_frags_x * 16, num_frags_y * 16]
                                                                 const float sm_scale) {
  constexpr int vec_size = 16 / sizeof(T);
  using LoadT = phi::AlignedVector<T, vec_size>;
  LoadT tmp_vec;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();

#pragma unroll
  for (uint32_t i = 0; i < num_frags_x * 16 * head_dim / 256; ++i) { // 32 * 8 per warp
    phi::Load<T, vec_size>(reinterpret_cast<T*>(q_smem->base + ty * num_frags_x * 16 * num_vecs_per_head) + i * 256 + tx * 8, &tmp_vec);
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      tmp_vec[reg_id] *= sm_scale;
    }
    phi::Store<T, vec_size>(tmp_vec, reinterpret_cast<T*>(q_smem->base + ty * num_frags_x * 16 * num_vecs_per_head) + i * 256 + tx * 8);
  }
}

template <SharedMemFillMode fill_mode, uint32_t num_warps, uint32_t block_size, uint32_t num_frags_y,
          uint32_t num_frags_z, uint32_t NUM_WARP_Q, typename T>
__device__ __forceinline__ void produce_kv_blockwise(smem_t smem,
                                                     uint32_t* smem_offset,
                                                     T **gptr, // [max_block_num, num_heads, block_size, head_dim]
                                                     const uint32_t kv_head_idx,
                                                     const uint32_t kv_n_stride,
                                                     const uint32_t kv_h_stride,
                                                     const uint32_t kv_b_stride,
                                                     const uint32_t kv_idx_base,
                                                     const uint32_t kv_len) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t kv_idx = kv_idx_base + ty * 4 + tx / 8; // kv_idx used to check
#pragma unroll
  for (uint32_t i = 0; i < NUM_WARP_KV * num_frags_z * 4 / num_warps; ++i) { // m num_frags_z * 16 / (num_warps * 4)
  // 16 rows each time
#pragma unroll
    for (uint32_t j = 0; j < num_frags_y / 4; ++j) { // k num_frags_y * 16 / 8 / num_elems_per_128b<T>()
      smem.load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
      *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
      *gptr += 8 * num_elems_per_128b<T>();
    }
    kv_idx += num_warps * 4;
    *smem_offset = smem.advance_offset_by_row<num_warps * 4, num_vecs_per_head>(*smem_offset) -
                    2 * num_frags_y; // num_frags_y / 4 * 8
    *gptr += num_warps * 4 * kv_b_stride - 2 * num_frags_y * num_elems_per_128b<T>();
  }
  *gptr -= NUM_WARP_KV * num_frags_z * 16 * kv_b_stride;
  *smem_offset -= NUM_WARP_KV * num_frags_z * 16 * num_vecs_per_head;
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, typename T, bool CAL_ROPE = false>
__device__ __forceinline__ void compute_qk(smem_t* q_smem, uint32_t* q_smem_offset_r,
                                           smem_t* k_smem, uint32_t* k_smem_offset_r,
                                           float (*s_frag)[num_frags_z][8],
                                           smem_t *rope_smem = nullptr) {
  // q [num_warps_q, num_frags_x, 16, head_dim], k [num_warps_kv, num_frags_z, 16, head_dim]
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  uint32_t a_frag[num_frags_x][4], b_frag[4];

  // compute q*k^T
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) { // k
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) { // m
      q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[fx]);
      *q_smem_offset_r = q_smem->advance_offset_by_row<16, num_vecs_per_head>(*q_smem_offset_r);
    }

    *q_smem_offset_r = q_smem->advance_offset_by_column<2>(*q_smem_offset_r, fy) -
                       num_frags_x * 16 * num_vecs_per_head;

#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) { // n
      k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
      if constexpr (CAL_ROPE) {
        uint32_t rope_frag[4];
        rope_smem->ldmatrix_m8n8x4(*k_smem_offset_r, rope_frag);
        apply_rope_frag<T>(b_frag, rope_frag);
      }
      *k_smem_offset_r = k_smem->advance_offset_by_row<16, num_vecs_per_head>(*k_smem_offset_r);
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        if (fy == 0) {
          mma_sync_m16n16k16_row_col_f16f16f32<T, MMAMode::kInit>(s_frag[fx][fz],
                                                                  a_frag[fx], b_frag);
        } else {
          mma_sync_m16n16k16_row_col_f16f16f32<T>(s_frag[fx][fz], a_frag[fx], b_frag);
        }
      }
    }
    *k_smem_offset_r = k_smem->advance_offset_by_column<2>(*k_smem_offset_r, fy) -
                       num_frags_z * 16 * num_vecs_per_head;
  }
  *q_smem_offset_r -= num_frags_y * 2;
  *k_smem_offset_r -= num_frags_y * 2;
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, typename T>
__device__ __forceinline__ void compute_sfm_v(smem_t* v_smem, uint32_t* v_smem_offset_r,
                                              float (*s_frag)[num_frags_z][8],
                                              float (*o_frag)[num_frags_y][8],
                                              float (*d)[2],
                                              const uint32_t layer_id = 0) {
  // [num_frags_x, 16, num_frags_z, 16] [num_frags_z, 16, num_frags_y, 16] -> [num_frags_x, 16, num_frags_y, 16]
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();

  T s_frag_f16[num_frags_x][num_frags_z][8];
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      vec_cast<T, float, 8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
    }
  }
#ifdef DEBUG_ATTN
  if (layer_id == 0 && threadIdx.x == 0 && threadIdx.y == PRINT_WID && blockIdx.x == 0 && blockIdx.z == 0) {
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        for (int i = 0; i < 8; ++i) {
          printf("fx: %d, fz: %d, s_frag_f16[%d][%d][%d]: %f\n", (int)fx, (int)fz, (int)fx, (int)fz, (int)i, (float)s_frag_f16[fx][fz][i]);
        }
      }
    }
  }
  __syncthreads();
#endif

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      rowsum_f16f16f32(d[fx], s_frag_f16[fx][fz]);
    }
  }

#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) { // k: num_warps_kv * num_frags_z * 16
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) { // n: num_frags_y * 16
      // [num_warps * num_frags_z * 16, num_frags_y * 16]
      uint32_t b_frag[4];
      v_smem->ldmatrix_m8n8x4_trans(*v_smem_offset_r, b_frag);
#ifdef DEBUG_ATTN
      if (layer_id == 0 && threadIdx.x == 0 && threadIdx.y == PRINT_WID && blockIdx.x == 0 && blockIdx.z == 0) {
        T *b_frag_T = reinterpret_cast<T*>(b_frag);
        for (int i = 0; i < 8; ++i) {
          printf("bbb fz: %d, fy: %d, b_frag[%d]: %f\n", (int)fz, (int)fy, (int)i, (float)b_frag_T[i]);
        }
      }
      __syncthreads();
#endif
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) { // m: num_frags_x * 16
        mma_sync_m16n16k16_row_col_f16f16f32<T>(
            o_frag[fx][fy], (uint32_t*)(s_frag_f16[fx][fz]), b_frag);
#ifdef DEBUG_ATTN
        if (layer_id == 0 && threadIdx.x == 0 && threadIdx.y == PRINT_WID && blockIdx.x == 0 && blockIdx.z == 0) {
          for (int i = 0; i < 8; ++i) {
            printf("ooo fx: %d, fy: %d, o_frag[%d]: %f\n", (int)fx, (int)fy, (int)i, (float)o_frag[fx][fy][i]);
          }
        }
#endif
      }
      *v_smem_offset_r = v_smem->advance_offset_by_column<2>(*v_smem_offset_r, fy);
    }
    *v_smem_offset_r =
        v_smem->advance_offset_by_row<16, num_vecs_per_head>(*v_smem_offset_r) - 2 * num_frags_y;
  }
  *v_smem_offset_r -= 16 * num_frags_z * num_vecs_per_head;
}

template <uint32_t num_frags_x, uint32_t num_frags_y>
__device__ __forceinline__ void normalize_d(float (*o_frag)[num_frags_y][8], float (*d)[2]) {
  float d_rcp[num_frags_x][2];
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      d_rcp[fx][j] = 1.f / d[fx][j];
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fx][fy][reg_id] = o_frag[fx][fy][reg_id] * d_rcp[fx][(reg_id % 4) / 2];
      }
    }
  }
}

template<typename T, int VEC_SIZE, typename OutT>
struct StoreFunc {
  __device__ __forceinline__ void operator()(const phi::AlignedVector<T, VEC_SIZE>& ori_out_vec,
                                                         const phi::AlignedVector<T, VEC_SIZE>& shift_bias_vec,
                                                         const phi::AlignedVector<T, VEC_SIZE>& smooth_weight_vec,
                                                         phi::AlignedVector<OutT, VEC_SIZE>& out_vec,
                                                         const float in_scale,
                                                         const int i) {
      printf("Fatal!!! Unimplemented StoreFunc for cascade append attention\n");
  }
};

template<typename T, int VEC_SIZE>
struct StoreFunc<T, VEC_SIZE, int8_t> {
  __device__ __forceinline__ void operator()(const phi::AlignedVector<T, VEC_SIZE>& ori_out_vec,
                                                         const phi::AlignedVector<T, VEC_SIZE>& shift_bias_vec,
                                                         const phi::AlignedVector<T, VEC_SIZE>& smooth_weight_vec,
                                                         phi::AlignedVector<int8_t, VEC_SIZE>& out_vec,
                                                         const float in_scale,
                                                         const int i) {
    float quant_value = 127.0f * static_cast<float>((ori_out_vec[i] + shift_bias_vec[i]) * smooth_weight_vec[i]) * in_scale;
    quant_value = rintf(quant_value);
    quant_value = quant_value > 127.0f ? 127.0f : quant_value;
    quant_value = quant_value < -127.0f ? -127.0f : quant_value;
    out_vec[i] = static_cast<int8_t>(quant_value);
  }
};

template<typename T, int VEC_SIZE>
struct StoreFunc<T, VEC_SIZE, T> {
  __device__ __forceinline__ void operator()(const phi::AlignedVector<T, VEC_SIZE>& ori_out_vec,
                                                         phi::AlignedVector<T, VEC_SIZE>& out_vec,
                                                         const int i) {
    out_vec[i] = ori_out_vec[i];
  }
};


template <uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_y, bool partition_kv, typename T, typename OutT>
__device__ __forceinline__ void write_o_reg_gmem_shift_smooth_quant(float (*o_frag)[num_frags_y][8],
                                                                    smem_t* o_smem,
                                                                    OutT* o_ptr_base,
                                                                    uint32_t o_idx_base,
                                                                    const uint32_t q_head_idx_base,
                                                                    const float in_scale,
                                                                    const uint32_t qo_upper_bound,
                                                                    const uint32_t qo_n_stride,
                                                                    const uint32_t qo_h_stride,
                                                                    const uint32_t layer_id = 0) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr int VEC_SIZE = 8;
  phi::AlignedVector<T, VEC_SIZE> ori_out_vec;
//   phi::AlignedVector<T, VEC_SIZE> shift_bias_vec;
//   phi::AlignedVector<T, VEC_SIZE> smooth_weight_vec;
  phi::AlignedVector<OutT, VEC_SIZE> out_vec;
  // [num_warps * num_frags_x * 16, num_frags_y * 16]
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      // 每个fy放16个数，vec size为8(f16/bf16)，所以y轴为2fy
      uint32_t o_frag_f16[4];
      vec_cast<T, float, 8>((T*)o_frag_f16, o_frag[fx][fy]);
      uint32_t o_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>( // num_vecs_per_head = num_frags_y * 16 / 8 = num_frags_y * 2
          (ty * num_frags_x + fx) * 16 + tx / 4, fy * 2);
      ((uint32_t*)(o_smem->base + o_smem_offset_w))[tx % 4] = o_frag_f16[0];
      ((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * num_vecs_per_head))[tx % 4] =
          o_frag_f16[1];
      ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[tx % 4] = o_frag_f16[2]; // 2fy，异或1往右移一位
      ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) + 8 * num_vecs_per_head))[tx % 4] =
          o_frag_f16[3];
    }
  }
  __syncthreads();

  // smem连续存储到gmem上， [num_frags_x * 16, num_frags_y * 16]
  uint32_t o_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_head>(ty * num_frags_x * 16 + tx / 8, tx % 8); // 每个warp一次搬4行，每次搬64个数

  const uint32_t tx_offset = tx / 8;
  // o_idx_base += (tx / 8) / group_size;
  // o_ptr_base += ((tx / 8) / group_size) * qo_n_stride + ((tx / 8) % group_size) * qo_h_stride;
  // uint32_t q_head_idx_now_base = q_head_idx_base + (tx / 8) % group_size;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
    const uint32_t base_offset = o_idx_base + fx * 16 + tx_offset;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) { // 4 * 4 = 16
      const uint32_t offset_now = base_offset + j * 4;
      const uint32_t n_offset = offset_now / group_size;
      const uint32_t h_offset = offset_now % group_size;
      OutT* o_ptr = o_ptr_base + n_offset * qo_n_stride + h_offset * qo_h_stride;
      uint32_t shift_smooth_offset = (q_head_idx_base + h_offset) * head_dim + tx % 8 * num_elems_per_128b<T>();
#pragma unroll
      for (uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) { // num_frags_y * 16 / (8[tid] * num_elems_per_128b<T>()[vec_per_thread])
        if (n_offset < qo_upper_bound) {
          if constexpr (!partition_kv) {
            // phi::Load<T, VEC_SIZE>(shift_bias + shift_smooth_offset, &shift_bias_vec);
            // phi::Load<T, VEC_SIZE>(smooth_weight + shift_smooth_offset, &smooth_weight_vec);
            phi::Load<T, VEC_SIZE>(reinterpret_cast<T*>(o_smem->base + o_smem_offset_w), &ori_out_vec);
#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
              StoreFunc<T, VEC_SIZE, OutT>()(ori_out_vec, out_vec, i);
#ifdef DEBUG_ATTN_C4
              if (layer_id == 0 && threadIdx.x == PRINT_TID && threadIdx.y == PRINT_WID && blockIdx.z == 0) {
                printf("write_o fx: %d, j: %d, fyo: %d, shift_bias[%d] = %f, smooth_weight[%d] = %f, ori_out[%d] = %f, out_vec[%d]: %f\n",
                       (int)fx, (int)j, (int)fyo, i, (float)shift_bias_vec[i], i, (float)smooth_weight_vec[i], i, (float)ori_out_vec[i], (float)out_vec[i]);
              }
              __syncthreads();
#endif
            }
            phi::Store<OutT, VEC_SIZE>(out_vec, o_ptr);
          } else {
            o_smem->store_128b(o_smem_offset_w, o_ptr);
          }
        }
        o_ptr += 8 * num_elems_per_128b<T>();
        shift_smooth_offset += 8 * num_elems_per_128b<T>();
        o_smem_offset_w = o_smem->advance_offset_by_column<8>(o_smem_offset_w, fyo);
      }
      o_smem_offset_w = o_smem->advance_offset_by_row<4, num_vecs_per_head>(o_smem_offset_w) -
                        2 * num_frags_y;
    }
  }
}

template <uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_y, bool partition_kv, typename T, typename OutT>
__device__ __forceinline__ void write_o_reg_gmem_multi_warps_shift_smooth_quant(float (*o_frag)[num_frags_y][8],
                                                                                smem_t* o_smem,
                                                                                OutT* o_ptr_base,
                                                                                uint32_t o_idx_base,
                                                                                const uint32_t q_head_idx_base,
                                                                                const uint32_t qo_upper_bound,
                                                                                const uint32_t qo_n_stride,
                                                                                const uint32_t qo_h_stride,
                                                                                const uint32_t layer_id = 0) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr int VEC_SIZE = 16 / sizeof(T);
  phi::AlignedVector<T, VEC_SIZE> ori_out_vec;
  phi::AlignedVector<OutT, VEC_SIZE> out_vec;
  // [num_warps * num_frags_x * 16, num_frags_y * 16]
  if (ty == 0) {
  // [num_frags_x * 16, num_frags_y * 16]
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        // 16 * 16
        // 每个fy放16个数，vec size为8(f16/bf16)，所以y轴为2fy
        uint32_t o_frag_f16[4];
        vec_cast<T, float, 8>((T*)o_frag_f16, o_frag[fx][fy]);
        uint32_t o_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>( // num_vecs_per_head = num_frags_y * 16 / 8 = num_frags_y * 2
            fx * 16 + tx / 4, fy * 2);
        ((uint32_t*)(o_smem->base + o_smem_offset_w))[tx % 4] = o_frag_f16[0];
        ((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * num_vecs_per_head))[tx % 4] =
            o_frag_f16[1];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[tx % 4] = o_frag_f16[2]; // 2fy，异或1往右移一位
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) + 8 * num_vecs_per_head))[tx % 4] =
            o_frag_f16[3];
      }
    }
  }
  __syncthreads();
#ifdef DEBUG_ATTN
  if (layer_id == 0 && threadIdx.x == PRINT_TID && threadIdx.y == PRINT_WID && blockIdx.z == 0 && blockIdx.x == gridDim.x - 1) {
    printf("o_smem\n");
    T *o_smem_t = reinterpret_cast<T*>(o_smem->base);
    for (uint32_t i = 0; i < num_frags_x * 16; ++i) {
      for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
        printf("o_smem[%d][%d] = %f  ", (int)i, (int)j, (float)o_smem_t[i * num_frags_y * 16 + j]);
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  // smem连续存储到gmem上， [num_frags_x * 16, num_frags_y * 16]
  uint32_t o_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_head>(ty * 4 + tx / 8, tx % 8); // 每个warp一次搬4行，每次搬64个数

  const uint32_t tx_offset = tx / 8;
  // o_idx_base += (tx / 8) / group_size;
  // o_ptr_base += ((tx / 8) / group_size) * qo_n_stride + ((tx / 8) % group_size) * qo_h_stride;
  // uint32_t q_head_idx_now_base = q_head_idx_base + (tx / 8) % group_size;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
    const uint32_t base_offset = o_idx_base + fx * 16 + tx_offset;
#pragma unroll
    const int j = ty;
    const uint32_t offset_now = base_offset + j * 4;
    const uint32_t n_offset = offset_now / group_size;
    const uint32_t h_offset = offset_now % group_size;
    OutT* o_ptr = o_ptr_base + n_offset * qo_n_stride + h_offset * qo_h_stride;
    uint32_t shift_smooth_offset = (q_head_idx_base + h_offset) * head_dim + tx % 8 * num_elems_per_128b<T>();
#pragma unroll
    for (uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) { // num_frags_y * 16 / (8[tid] * num_elems_per_128b<T>()[vec_per_thread])
      if (n_offset < qo_upper_bound) {
        if constexpr (!partition_kv) {
          phi::Load<T, VEC_SIZE>(reinterpret_cast<T*>(o_smem->base + o_smem_offset_w), &ori_out_vec);
#pragma unroll
          for (int i = 0; i < VEC_SIZE; ++i) {
            // float quant_value = 127.0f * static_cast<float>((ori_out_vec[i] + shift_bias_vec[i]) * smooth_weight_vec[i]) * in_scale;
            // quant_value = rintf(quant_value);
            // quant_value = quant_value > 127.0f ? 127.0f : quant_value;
            // quant_value = quant_value < -127.0f ? -127.0f : quant_value;
            // out_vec[i] = static_cast<int8_t>(quant_value);
            StoreFunc<T, VEC_SIZE, OutT>()(ori_out_vec, out_vec, i);
#ifdef DEBUG_ATTN
            if (layer_id == 0 && threadIdx.x == PRINT_TID && threadIdx.y == PRINT_WID && blockIdx.z == 0 && blockIdx.x == gridDim.x - 1 && blockIdx.y == 0) {
              printf("write_o fx: %d, j: %d, fyo: %d, in_scale: %f, i: %d, shift_bias = %f, smooth_weight = %f, ori_out = %f, out_vec: %f\n",
                      (int)fx, (int)j, (int)fyo, (float)in_scale, (int)i, (float)shift_bias_vec[i], (float)smooth_weight_vec[i], (float)ori_out_vec[i], (float)out_vec[i]);
            }
            __syncthreads();
#endif
          }
          phi::Store<OutT, VEC_SIZE>(out_vec, o_ptr);
        } else {
          o_smem->store_128b(o_smem_offset_w, o_ptr);
        }
      }
      o_ptr += 8 * num_elems_per_128b<T>();
      shift_smooth_offset += 8 * num_elems_per_128b<T>();
      o_smem_offset_w = o_smem->advance_offset_by_column<8>(o_smem_offset_w, fyo);
    }
    o_smem_offset_w = o_smem->advance_offset_by_row<16, num_vecs_per_head>(o_smem_offset_w) -
                      2 * num_frags_y;
    // }
  }
}

template <typename T, bool partition_kv, bool causal, uint32_t group_size, uint32_t num_warps,
          uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, bool IS_SYSTEM = false>
__device__ __forceinline__ void mask_s(const uint32_t qo_idx_base, const uint32_t kv_idx_base,
                                       const uint32_t qo_len, const uint32_t kv_len,
                                       const uint32_t chunk_end,
                                       float (*s_frag)[num_frags_z][8],
                                       const uint32_t layer_id = 0) {
  const uint32_t tx = threadIdx.x;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        if constexpr (!IS_SYSTEM) {
          const uint32_t q_idx =
                            (qo_idx_base + fx * 16 + tx / 4 + 8 * ((reg_id % 4) / 2)) / group_size,
                        kv_idx =
                            kv_idx_base + fz * 16 + 2 * (tx % 4) + 8 * (reg_id / 4) + reg_id % 2;
          const bool out_of_boundary =
              (causal ? (kv_idx > kv_len + q_idx - qo_len || (kv_idx >= chunk_end))
                      : kv_idx >= chunk_end);
          // const bool out_of_boundary =
          //     (causal ? (kv_idx > kv_len + q_idx - qo_len || (partition_kv && kv_idx >= chunk_end))
          //             : kv_idx >= chunk_end);
          if constexpr (std::is_same<T, half>::value) {
            s_frag[fx][fz][reg_id] = out_of_boundary ? -5e4f : s_frag[fx][fz][reg_id];
          } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            s_frag[fx][fz][reg_id] = out_of_boundary ? -3.0e+30f : s_frag[fx][fz][reg_id];
          }
        } else { // 共享前缀decoder加速，不增加q_idx，每位置q_idx相同
          const uint32_t q_idx = qo_idx_base,
                        kv_idx =
                            kv_idx_base + fz * 16 + 2 * (tx % 4) + 8 * (reg_id / 4) + reg_id % 2;
          const bool out_of_boundary =
              (causal ? (kv_idx > kv_len + q_idx - qo_len || (kv_idx >= chunk_end))
                      : kv_idx >= chunk_end);
          // const bool out_of_boundary =
          //     (causal ? (kv_idx > kv_len + q_idx - qo_len || (partition_kv && kv_idx >= chunk_end))
          //             : kv_idx >= chunk_end);
#ifdef DEBUG_ATTN_C4
          if (layer_id == 0 && threadIdx.x == PRINT_TID && threadIdx.y == PRINT_WID && blockIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == gridDim.y - 1 && fx == 0 && fz == 3 && reg_id == 4) {
            printf("q_idx: %d, kv_idx: %d, kv_len: %d, qo_len: %d, chunk_end: %d\n",
                    (int)q_idx, (int)kv_idx, (int)kv_len, (int)qo_len, (int)chunk_end);
          }
          __syncthreads();
#endif
          if constexpr (std::is_same<T, half>::value) {
            s_frag[fx][fz][reg_id] = out_of_boundary ? -5e4f : s_frag[fx][fz][reg_id];
          } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            s_frag[fx][fz][reg_id] = out_of_boundary ? -3.0e+30f : s_frag[fx][fz][reg_id];
          }
        }
      }
    }
  }
}


template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z>
__device__ __forceinline__ void update_mdo_states(float (*s_frag)[num_frags_z][8],
                                                  float (*o_frag)[num_frags_y][8],
                                                  float (*m)[2], float (*d)[2],
                                                  const uint32_t layer_id = 0) {
  // [num_warps * num_frags_x * 16, num_frags_z * 16]
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
    // 16 * （num_frags_z * 16）
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) { // 2行
      float m_prev = m[fx][j];
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        float m_local = max(max(s_frag[fx][fz][j * 2 + 0], s_frag[fx][fz][j * 2 + 1]),
                            max(s_frag[fx][fz][j * 2 + 4], s_frag[fx][fz][j * 2 + 5]));
        m[fx][j] = max(m[fx][j], m_local);
      }
      m[fx][j] = max(m[fx][j], __shfl_xor_sync(-1, m[fx][j], 0x2, 32));
      m[fx][j] = max(m[fx][j], __shfl_xor_sync(-1, m[fx][j], 0x1, 32));
      float o_scale = __expf(m_prev - m[fx][j]);
      d[fx][j] *= o_scale;
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        o_frag[fx][fy][j * 2 + 0] *= o_scale;
        o_frag[fx][fy][j * 2 + 1] *= o_scale;
        o_frag[fx][fy][j * 2 + 4] *= o_scale;
        o_frag[fx][fy][j * 2 + 5] *= o_scale;
      }
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        s_frag[fx][fz][j * 2 + 0] = __expf(s_frag[fx][fz][j * 2 + 0] - m[fx][j]);
        s_frag[fx][fz][j * 2 + 1] = __expf(s_frag[fx][fz][j * 2 + 1] - m[fx][j]);
        s_frag[fx][fz][j * 2 + 4] = __expf(s_frag[fx][fz][j * 2 + 4] - m[fx][j]);
        s_frag[fx][fz][j * 2 + 5] = __expf(s_frag[fx][fz][j * 2 + 5] - m[fx][j]);
      }
    }
  }
}


template <uint32_t num_frags_x, uint32_t num_frags_y, typename T>
__device__ __forceinline__ void merge_block_res_v2(float (*o_frag)[num_frags_y][8],
                                                   float *md_smem,
                                                   float (*m)[2],
                                                   float (*d)[2],
                                                   const uint32_t wid,
                                                   const uint32_t tid) {
  // o: [num_warps, num_frags_x, num_frags_y, warp_size(32), 8]
  // md: [num_warps, num_frags_x, 2, warp_size(32), 2 (m/d)]
  float2 *smem_md = reinterpret_cast<float2*>(md_smem + num_frags_x * num_frags_y * 1024); // 4 * 32 * 8
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      smem_md[((wid * num_frags_x + fx) * 2 + j) * 32 + tid] =
          make_float2(m[fx][j], d[fx][j]);
    }
  }
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      *(reinterpret_cast<float4*>(md_smem + (((wid * num_frags_x + fx) * num_frags_y + fy) * 32 + tid) * 8)) = *(reinterpret_cast<float4*>(&o_frag[fx][fy][0]));
      *(reinterpret_cast<float4*>(md_smem + (((wid * num_frags_x + fx) * num_frags_y + fy) * 32 + tid) * 8 + 4)) = *(reinterpret_cast<float4*>(&o_frag[fx][fy][4]));
    }
  }
  __syncthreads();
  float o_scale[4][num_frags_x][2];

  // deal md/scale
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      float m_new;
      float d_new = 1.f;
      if constexpr (std::is_same<T, half>::value) {
        m_new = -5e4f;
      } else {
        m_new = -3.0e+30f;
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        float2 md = smem_md[((i * num_frags_x + fx) * 2 + j) * 32 + tid];
        float m_prev = m_new, d_prev = d_new;
        m_new = max(m_new, md.x);
        d_new = d_prev * __expf(m_prev - m_new) + md.y * __expf(md.x - m_new);
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        float2 md = smem_md[((i * num_frags_x + fx) * 2 + j) * 32 + tid];
        o_scale[i][fx][j] = __expf(md.x - m_new);
      }
      m[fx][j] = m_new;
      d[fx][j] = d_new;
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      // num_warps * 32 * 8 each time
      phi::AlignedVector<float, 8> o_new;
#pragma
      for (uint32_t o_id = 0; o_id < 4; ++o_id) {
        *(reinterpret_cast<float2*>(&o_new[o_id * 2])) = make_float2(0.f, 0.f);
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        phi::AlignedVector<float, 8> oi;
        phi::Load<float, 8>(
          md_smem + (((i * num_frags_x + fx) * num_frags_y + fy) * 32 + tid) * 8,
          &oi
        );
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          o_new[reg_id] += oi[reg_id] * o_scale[i][fx][(reg_id % 4) / 2];
        }
      }
      *(reinterpret_cast<float4*>(&o_frag[fx][fy][0])) = *(reinterpret_cast<float4*>(&o_new[0]));
      *(reinterpret_cast<float4*>(&o_frag[fx][fy][4])) = *(reinterpret_cast<float4*>(&o_new[4]));
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename T>
__device__ __forceinline__ void q_smem_inplace_multiply_sm_scale_multi_warps(smem_t* q_smem, // [num_frags_x * 16, num_frags_y * 16]
                                                                             const float sm_scale) {
  constexpr int vec_size = 16 / sizeof(T);
  using LoadT = phi::AlignedVector<T, vec_size>;
  LoadT tmp_vec;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();

#pragma unroll
  for (uint32_t i = 0; i < num_frags_x * 16 * head_dim / 1024; ++i) { // 32 * 8 * 4 all warp
    const int offset = i * 1024 + ty * 256 + tx * 8;
    phi::Load<T, vec_size>(reinterpret_cast<T*>(q_smem->base) + offset, &tmp_vec);
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      tmp_vec[reg_id] *= sm_scale;
    }
    phi::Store<T, vec_size>(tmp_vec, reinterpret_cast<T*>(q_smem->base) + offset);
  }
}

template <uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_y, uint32_t HEAD_DIM, typename T>
__device__ __forceinline__ void load_q_global_smem_multi_warps(T* q_ptr_base,
                                                               smem_t* q_smem,
                                                               uint32_t q_idx_base,
                                                               const uint32_t qo_upper_bound,
                                                               const uint32_t qo_n_stride,
                                                               const uint32_t qo_h_stride) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();

  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t q_smem_offset_w = // [NUM_WARP_Q, num_frags_x, 16, head_dim]
      smem_t::get_permuted_offset<num_vecs_per_head>(ty * 4 + tx / 8, tx % 8); // 4 * 64

  // q_idx_base += (tx / 8) / group_size;
  // q_ptr_base += ((tx / 8) / group_size) * qo_n_stride + ((tx / 8) % group_size) * qo_h_stride;
  const uint32_t tx_offset = tx / 8;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
    // num_frags_x * 16 * head_dim
    // load 4 row per warp
    const uint32_t base_offset = q_idx_base + fx * 16 + tx_offset;
#pragma unroll
    // for (uint32_t j = 0; j < 4; ++j) {
    const int j = ty;
    const uint32_t offset_now = base_offset + j * 4;
    const uint32_t n_offset = offset_now / group_size;
    const uint32_t h_offset = offset_now % group_size;
    T* q_ptr = q_ptr_base + n_offset * qo_n_stride + h_offset * qo_h_stride;
#pragma unroll
    for (uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) { // (num_frags_y * 16) / (8 *  num_elems_per_128b<T>())
      // load q from gmem to smem
      q_smem->load_128b_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr,
                                                          n_offset < qo_upper_bound);
      q_smem_offset_w = q_smem->advance_offset_by_column<8>(q_smem_offset_w, fyo);
      q_ptr += 8 * num_elems_per_128b<T>();
    }
    q_smem_offset_w =
        q_smem->advance_offset_by_row<16, num_vecs_per_head>(q_smem_offset_w) - 2 * num_frags_y; // num_frags_y / 4 * 8
    // }
  }
}

} // namespace fusion
} // namespace phi
