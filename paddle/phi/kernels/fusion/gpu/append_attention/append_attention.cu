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

#include "paddle/phi/kernels/fusion/gpu/append_attention/append_attention.cuh"
#include "paddle/phi/kernels/fusion/cacade_append_attn.h"

COMMON_DECLARE_int32(speculate_max_draft_token_num);

namespace phi {
namespace fusion {

template <typename T,
          int vec_size,
          uint32_t bdy,
          uint32_t HEAD_DIM,
          typename OutT = T,
          bool ENABLE_PREFILL = true>
__global__ void merge_multi_chunks_for_fusemt_decoder_v2_kernel(
    const T *__restrict__ multi_out,    // [token_num, num_chunks, num_heads,
                                        // head_dim]
    const float *__restrict__ multi_m,  // [token_num, num_chunks, num_heads]
    const float *__restrict__ multi_d,  // [token_num, num_chunks, num_heads]
    const int *__restrict__ seq_lens_q,
    const int *__restrict__ seq_lens_kv,
    const int *__restrict__ seq_lens_encoder,
    const int *__restrict__ cum_offsets,
    OutT *__restrict__ out,
    const int max_seq_len,
    const int num_chunks,
    const int num_heads,
    const int chunk_size,
    const int head_dim) {
  const int vid = threadIdx.x, ty = threadIdx.y;
  const int bid = blockIdx.x, hid = blockIdx.y;
  __shared__ T smem[bdy * HEAD_DIM];
  __shared__ float md_smem[bdy * 2];
  const int start_token_idx = bid * max_seq_len - cum_offsets[bid];
  const int seq_len_q = seq_lens_q[bid];
  if (seq_len_q == 0) return;
  int seq_len_kv = seq_lens_kv[bid];

  if (ENABLE_PREFILL) {
    seq_len_kv += seq_len_q;
    if (seq_len_kv == 0) return;
  } else {
    if (seq_len_kv == 0) return;
    seq_len_kv += seq_len_q;
  }
  const int seq_len_enc = seq_lens_encoder[bid];
  if (seq_len_enc > 0) {
    return;
  }
  const int num_chunks_this_seq = div_up(seq_len_kv, chunk_size);
  if (num_chunks_this_seq <= 1) {
    return;
  }

  using LoadT = phi::AlignedVector<T, vec_size>;
  LoadT load_vec;
  LoadT res_vec;
  if constexpr (std::is_same<T, half>::value) {
#pragma unroll
    for (int i = 0; i < vec_size / 2; ++i) {
      *((half2 *)(&res_vec) + i) = make_half2(0, 0);
    }
  } else {
#pragma unroll
    for (int i = 0; i < vec_size / 2; ++i) {
      *((nv_bfloat162 *)(&res_vec) + i) = make_bfloat162(0, 0);
    }
  }
  float m;
  float d = 1.f;
  if constexpr (std::is_same<T, half>::value) {
    m = -5e4f;
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    m = -3.0e+30f;
  }
#pragma unroll 2
  for (int i = ty; i < num_chunks_this_seq; i += bdy) {
    // uint32_t offset = (start_token_idx * num_chunks + i) * num_heads + hid;
    uint32_t offset = (bid * num_chunks + i) * num_heads + hid;
    float m_prev = m;
    float d_prev = d;
    const float m_now = multi_m[offset];
    const float d_now = multi_d[offset];
    m = max(m_prev, m_now);
    // offset = (start_token_idx * num_chunks * num_heads + i * num_heads + hid)
    // * head_dim + vid * vec_size;
    offset = (bid * num_chunks * num_heads + i * num_heads + hid) * head_dim +
             vid * vec_size;
    phi::Load<T, vec_size>(&multi_out[offset], &load_vec);
    const float scale1 = __expf(m_prev - m), scale2 = __expf(m_now - m);
    const T scale1_T = static_cast<T>(scale1),
            scale2_T = static_cast<T>(scale2);
    d = d * scale1 + d_now * scale2;
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      res_vec[j] = res_vec[j] * scale1_T + load_vec[j] * scale2_T;
    }
  }
  // store ty res
  phi::Store<T, vec_size>(res_vec, &smem[ty * head_dim + vid * vec_size]);
  md_smem[2 * ty] = m;
  md_smem[2 * ty + 1] = d;
  __syncthreads();
  if (ty == 0) {
    // merge bdy
    prefill_softmax_state_t<vec_size, T> st;
    st.init();
#pragma unroll
    for (int i = 0; i < bdy; i++) {
      phi::Load<T, vec_size>(&smem[i * head_dim + vid * vec_size], &load_vec);
      const float m_tmp = md_smem[2 * i], d_tmp = md_smem[2 * i + 1];
      st.merge(load_vec, m_tmp, d_tmp);
    }
    st.normalize();

    phi::AlignedVector<OutT, vec_size> out_vec;
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      StoreFunc<T, vec_size, OutT>()(st.o, out_vec, i);
    }
    phi::Store<OutT, vec_size>(
        out_vec,
        &out[(start_token_idx * num_heads + hid) * head_dim + vid * vec_size]);
  }
}

template <typename T,
          int vec_size,
          uint32_t bdy,
          uint32_t HEAD_DIM,
          typename OutT = T,
          bool ENABLE_PREFILL = true>
__global__ void merge_multi_chunks_for_fusemt_v2_kernel(
    const T *__restrict__ multi_out,    // [token_num, num_chunks, num_heads,
                                        // head_dim]
    const float *__restrict__ multi_m,  // [token_num, num_chunks, num_heads]
    const float *__restrict__ multi_d,  // [token_num, num_chunks, num_heads]
    const int *__restrict__ seq_lens_q,
    const int *__restrict__ seq_lens_kv,
    const int *__restrict__ seq_lens_encoder,
    const int *__restrict__ padding_offsets,
    OutT *__restrict__ out,
    const int max_seq_len,
    const int num_chunks,
    const int num_heads,
    const int chunk_size,
    const int head_dim,
    const int token_num,
    const int speculate_max_draft_token_num = 5) {
  const int vid = threadIdx.x, ty = threadIdx.y;
  // const int qid = blockIdx.x, hid = blockIdx.y;
  const int hid = blockIdx.y;
  __shared__ T smem[bdy * HEAD_DIM];
  __shared__ float md_smem[bdy * 2];
  for (int qid = blockIdx.x; qid < token_num; qid += gridDim.x) {
    const uint32_t ori_token_id = qid + padding_offsets[qid];
    const uint32_t bid = ori_token_id / max_seq_len;
    const uint32_t local_seq_id = ori_token_id % max_seq_len;
    const int seq_len_q = seq_lens_q[bid];
    if (seq_len_q == 0) continue;
    int seq_len_kv = seq_lens_kv[bid];
    if (ENABLE_PREFILL) {
      seq_len_kv += seq_len_q;
      if (seq_len_kv == 0) continue;

      const int seq_len_enc = seq_lens_encoder[bid];
      if (seq_len_enc <= 0) {
        continue;
      }
    } else {
      if (seq_len_kv == 0) continue;
      seq_len_kv += seq_len_q;
    }
    const int num_chunks_this_seq = div_up(seq_len_kv, chunk_size);
    if (num_chunks_this_seq <= 1) {
      continue;
    }

    using LoadT = phi::AlignedVector<T, vec_size>;
    LoadT load_vec;
    LoadT res_vec;
    if constexpr (std::is_same<T, half>::value) {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((half2 *)(&res_vec) + i) = make_half2(0, 0);
      }
    } else {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((nv_bfloat162 *)(&res_vec) + i) = make_bfloat162(0, 0);
      }
    }
    float m;
    float d = 1.f;
    if constexpr (std::is_same<T, half>::value) {
      m = -5e4f;
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      m = -3.0e+30f;
    }
#pragma unroll 2
    for (int i = ty; i < num_chunks_this_seq; i += bdy) {
      uint32_t offset;
      if (ENABLE_PREFILL) {
        offset = (qid * num_chunks + i) * num_heads + hid;
      } else {
        offset =
            ((bid * speculate_max_draft_token_num + local_seq_id) * num_chunks +
             i) *
                num_heads +
            hid;
      }
      float m_prev = m;
      float d_prev = d;
      const float m_now = multi_m[offset];
      const float d_now = multi_d[offset];
      m = max(m_prev, m_now);
      if (ENABLE_PREFILL) {
        offset =
            (qid * num_chunks * num_heads + i * num_heads + hid) * head_dim +
            vid * vec_size;
      } else {
        offset = ((bid * speculate_max_draft_token_num + local_seq_id) *
                      num_chunks * num_heads +
                  i * num_heads + hid) *
                     head_dim +
                 vid * vec_size;
      }
      phi::Load<T, vec_size>(&multi_out[offset], &load_vec);
      const float scale1 = __expf(m_prev - m), scale2 = __expf(m_now - m);
      const T scale1_T = static_cast<T>(scale1),
              scale2_T = static_cast<T>(scale2);
      d = d * scale1 + d_now * scale2;
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        res_vec[j] = res_vec[j] * scale1_T + load_vec[j] * scale2_T;
      }
    }
    // store ty res
    phi::Store<T, vec_size>(res_vec, &smem[ty * head_dim + vid * vec_size]);
    md_smem[2 * ty] = m;
    md_smem[2 * ty + 1] = d;
    __syncthreads();
    if (ty == 0) {
      // merge bdy
      prefill_softmax_state_t<vec_size, T> st;
      st.init();
#pragma unroll
      for (int i = 0; i < bdy; i++) {
        phi::Load<T, vec_size>(&smem[i * head_dim + vid * vec_size], &load_vec);
        const float m_tmp = md_smem[2 * i], d_tmp = md_smem[2 * i + 1];
        st.merge(load_vec, m_tmp, d_tmp);
      }
      st.normalize();

      phi::AlignedVector<OutT, vec_size> out_vec;
#pragma unroll
      for (int i = 0; i < vec_size; ++i) {
        StoreFunc<T, vec_size, OutT>()(st.o, out_vec, i);
      }
      phi::Store<OutT, vec_size>(
          out_vec, &out[(qid * num_heads + hid) * head_dim + vid * vec_size]);
    }

    __syncthreads();
  }
}

template <typename T,
          bool partition_kv,
          uint32_t GROUP_SIZE,
          bool CAUSAL,
          uint32_t NUM_WARPS,
          uint32_t NUM_WARP_Q,
          uint32_t NUM_WARP_KV,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t num_frags_x,
          uint32_t num_frags_z,
          uint32_t num_frags_y,
          typename OutT = T,
          bool CAL_ROPE = false,
          bool ENABLE_PREFILL = true,
          bool USE_SYSTEM = false>
__global__ void multi_query_append_for_fuse_mt_attention_kernel(
    T *__restrict__ q,        // [token_num. num_heads, head_dim]
    T *__restrict__ cache_k,  // [max_block_num, num_heads, block_size,
                              // head_dim]
    T *__restrict__ cache_v,
    const int *__restrict__ seq_lens,
    const int *__restrict__ seq_lens_kv,
    const int *__restrict__ batch_ids,
    const int *__restrict__ tile_ids_per_batch,
    const int *__restrict__ cum_offsets,
    const int *__restrict__ block_table,  // [bsz, block_num_per_seq]
    const int *__restrict__ seq_mapping,
    T *__restrict__ rope_emb,  // [max_seq_len, head_dim] cos1 sin1 cos2 sin2
                               // ...
    const int max_seq_len,
    const int max_dec_len,
    const int max_block_num_per_seq,
    const float scale,
    const uint32_t chunk_size,
    const uint32_t layer_id,
    T *__restrict__ tmp_workspace,  // split kv [token_num, num_chunks,
                                    // num_heads, head_dim]
    float *__restrict__ tmp_m,      // [token_num, num_chunks, num_heads]
    float *__restrict__ tmp_d,      // [token_num, num_chunks, num_heads]
    OutT *__restrict__ out,
    const int speculate_max_draft_token_num = 5) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
  const uint32_t btid = blockIdx.x, kv_head_idx = blockIdx.z;
  const uint32_t kv_num_heads = gridDim.z;
  const uint32_t q_num_heads = kv_num_heads * GROUP_SIZE;
  const uint32_t q_head_idx = kv_head_idx * GROUP_SIZE;
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;
  const uint32_t num_chunks = gridDim.y;
  const uint32_t chunk_idx = blockIdx.y;

  const uint32_t batch_id = batch_ids[btid];
  const uint32_t tile_id = tile_ids_per_batch[btid];
  const uint32_t num_rows_per_block = NUM_WARPS * num_frags_x * 16;
  const int *block_table_now = nullptr;
  if constexpr (USE_SYSTEM) {
    block_table_now =
        block_table + seq_mapping[batch_id] * max_block_num_per_seq;
  } else {
    block_table_now = block_table + batch_id * max_block_num_per_seq;
  }

  const uint32_t q_len = seq_lens[batch_id];
  if (q_len <= 0) {
    return;
  }

  const uint32_t q_end =
      min(q_len, div_up((tile_id + 1) * num_rows_per_block, GROUP_SIZE));
  uint32_t kv_len = seq_lens_kv[batch_id];
  if (ENABLE_PREFILL) {
    kv_len += q_len;  // !!!
    if (kv_len <= 0) {
      return;
    }
  } else {
    if (kv_len <= 0) {
      return;
    }
    kv_len += q_len;
  }

  const uint32_t num_chunks_this_seq = div_up(kv_len, chunk_size);
  if (chunk_idx >= num_chunks_this_seq) {
    return;
  }

  const uint32_t chunk_start = partition_kv ? chunk_idx * chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min(kv_len, chunk_start + chunk_size) : kv_len;
  const uint32_t chunk_len = chunk_end - chunk_start;

  extern __shared__ uint8_t smem[];
  float s_frag[num_frags_x][num_frags_z][8];
  float o_frag[num_frags_x][num_frags_y][8];
  float m_frag[num_frags_x][2];
  float d_frag[num_frags_x][2];
  init_states<T, num_frags_x, num_frags_y>(o_frag, m_frag, d_frag);

  const uint32_t q_n_stride = q_num_heads * HEAD_DIM;
  const uint32_t q_ori_n_stride = (q_num_heads + kv_num_heads * 2) * HEAD_DIM;
  const uint32_t kv_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM;
  const uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM;
  const uint32_t kv_b_stride = HEAD_DIM;
  const uint32_t q_start_seq_id =
      batch_id * max_seq_len - __ldg(&cum_offsets[batch_id]);
  const uint32_t q_base_seq_id_this_block =
      (tile_id * NUM_WARPS + wid) * num_frags_x * 16;
  const uint32_t q_offset = q_start_seq_id * q_ori_n_stride +
                            q_head_idx * HEAD_DIM +
                            tid % 8 * num_elems_per_128b<T>();
  const uint32_t o_offset = q_start_seq_id * q_n_stride +
                            q_head_idx * HEAD_DIM +
                            tid % 8 * num_elems_per_128b<T>();
  T *q_base_ptr = q + q_offset;
  T *o_base_ptr_T = nullptr;
  OutT *o_base_ptr_int8 = nullptr;
  if constexpr (partition_kv) {
    if (ENABLE_PREFILL) {
      o_base_ptr_T = tmp_workspace + q_start_seq_id * num_chunks * q_n_stride +
                     chunk_idx * q_n_stride + q_head_idx * HEAD_DIM +
                     tid % 8 * num_elems_per_128b<T>();
    } else {
      o_base_ptr_T =
          tmp_workspace +
          batch_id * speculate_max_draft_token_num * num_chunks * q_n_stride +
          chunk_idx * q_n_stride + q_head_idx * HEAD_DIM +
          tid % 8 * num_elems_per_128b<T>();
    }
  } else {
    o_base_ptr_int8 = out + o_offset;
  }
  smem_t qo_smem(smem);

  /*
    1 ｜ 3
    ——————
    2 ｜ 4
  */
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_x * 16 + tid % 16, tid / 16);  // 16 * 16
  load_q_global_smem<GROUP_SIZE, num_frags_x, num_frags_y, HEAD_DIM, T>(
      q_base_ptr,
      &qo_smem,
      q_base_seq_id_this_block,
      q_end,
      q_ori_n_stride,
      HEAD_DIM);
  commit_group();
  wait_group<0>();
  __syncthreads();

  //// Debug
  // if (threadIdx.x == PRINT_TID) {
  //   printf("end load q \n");
  // }
  // __syncthreads();
  //// end debug

  q_smem_inplace_multiply_sm_scale<num_frags_x, num_frags_y, T>(&qo_smem,
                                                                scale);

  smem_t k_smem(smem + NUM_WARPS * num_frags_x * 16 * HEAD_DIM * sizeof(T)),
      v_smem(smem + (NUM_WARPS * num_frags_x + num_frags_z) * 16 * HEAD_DIM *
                        sizeof(T));
  smem_t rope_smem;
  if constexpr (CAL_ROPE) {
    rope_smem.base = reinterpret_cast<phi::fusion::b128_t *>(
        smem + (NUM_WARPS * num_frags_x + num_frags_z * 2) * 16 * HEAD_DIM *
                   sizeof(T));
  }

  const uint32_t num_iterations = div_up(
      CAUSAL
          ? (min(chunk_len,
                 sub_if_greater_or_zero(
                     kv_len - q_len +
                         div_up((tile_id + 1) * num_rows_per_block, GROUP_SIZE),
                     chunk_start)))
          : chunk_len,
      num_frags_z * 16);
  const uint32_t mask_check_iteration =
      (CAUSAL ? (min(chunk_len,
                     sub_if_greater_or_zero(
                         kv_len - q_len +
                             tile_id * num_rows_per_block / GROUP_SIZE,
                         chunk_start)))
              : chunk_len) /
      (num_frags_z * 16);
#ifdef DEBUG_ATTN
  if (tid == 0 && threadIdx.y == PRINT_WID && kv_head_idx == 0) {
    printf(
        "batch_id: %d, tile_id: %d, chunk_size: %d, q_len: %d, kv_len: %d, "
        "chunk_start: %d, chunk_end: %d, num_iterations: %d, "
        "mask_check_iteration: %d\n",
        (int)batch_id,
        (int)tile_id,
        (int)chunk_size,
        (int)q_len,
        (int)kv_len,
        (int)chunk_start,
        (int)chunk_end,
        (int)num_iterations,
        (int)mask_check_iteration);
  }
  __syncthreads();
#endif

  /*
    1 ｜ 2
    ——————
    3 ｜ 4
  */
  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      8 * (tid / 16) + tid % 8, (tid % 16) / 8);
  /*
    1 ｜ 3
    ——————
    2 ｜ 4   transpose
  */
  uint32_t v_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_head>(tid % 16, tid / 16);

  uint32_t kv_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * 4 + tid / 8, tid % 8);

  uint32_t kv_idx_base = chunk_start;
  int block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
  const uint32_t const_offset = kv_head_idx * kv_h_stride +
                                (wid * 4 + tid / 8) * kv_b_stride +
                                tid % 8 * num_elems_per_128b<T>();
  T *cache_k_now = cache_k + block_id * kv_n_stride + const_offset;
  T *cache_v_now = cache_v + block_id * kv_n_stride + const_offset;
  T *rope_emb_now;
  if constexpr (CAL_ROPE) {
    rope_emb_now = rope_emb +
                   (kv_idx_base + (wid * 4 + tid / 8)) * kv_b_stride +
                   tid % 8 * num_elems_per_128b<T>();
  }

  produce_kv_blockwise<SharedMemFillMode::kNoFill,
                       NUM_WARPS,
                       BLOCK_SIZE,
                       num_frags_y,
                       num_frags_z,
                       NUM_WARP_Q>(k_smem,
                                   &kv_smem_offset_w,
                                   &cache_k_now,
                                   kv_head_idx,
                                   kv_n_stride,
                                   kv_h_stride,
                                   kv_b_stride,
                                   kv_idx_base,
                                   chunk_end);

  if constexpr (CAL_ROPE) {
    produce_kv_blockwise<SharedMemFillMode::kNoFill,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(rope_smem,
                                     &kv_smem_offset_w,
                                     &rope_emb_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_b_stride,
                                     kv_idx_base,
                                     chunk_end);
  }
  commit_group();
  produce_kv_blockwise<SharedMemFillMode::kFillZero,
                       NUM_WARPS,
                       BLOCK_SIZE,
                       num_frags_y,
                       num_frags_z,
                       NUM_WARP_Q>(v_smem,
                                   &kv_smem_offset_w,
                                   &cache_v_now,
                                   kv_head_idx,
                                   kv_n_stride,
                                   kv_h_stride,
                                   kv_b_stride,
                                   kv_idx_base,
                                   chunk_end);
  commit_group();

  //// Debug
  // if (threadIdx.x == PRINT_TID) {
  //   printf("end prlogue \n");
  // }
  // __syncthreads();
  //// end debug
#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    wait_group<1>();
    __syncthreads();
#ifdef DEBUG_ATTN
    if (layer_id == 0 && tid == PRINT_TID && threadIdx.y == PRINT_WID &&
        blockIdx.z == 0 && blockIdx.x == 0) {
      printf("cache_k_smem\n");
      T *k_smem_t = reinterpret_cast<T *>(k_smem.base);
      for (uint32_t i = 0; i < NUM_WARP_KV * num_frags_z * 16; ++i) {
        for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
          printf("k_smem[%d][%d] = %f  ",
                 (int)i,
                 (int)j,
                 (float)k_smem_t[i * num_frags_y * 16 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    // s = qk
    compute_qk<num_frags_x, num_frags_y, num_frags_z, T, CAL_ROPE>(
        &qo_smem,
        &q_smem_offset_r,
        &k_smem,
        &k_smem_offset_r,
        s_frag,
        &rope_smem);
    // mask according to kv_idx and q_idx
    if (iter >= mask_check_iteration) {
      mask_s<T,
             partition_kv,
             CAUSAL,
             GROUP_SIZE,
             NUM_WARPS,
             num_frags_x,
             num_frags_y,
             num_frags_z>(q_base_seq_id_this_block,
                          kv_idx_base,
                          q_len,
                          kv_len,
                          chunk_end,
                          s_frag,
                          layer_id);
    }

    // update m,d
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
        s_frag, o_frag, m_frag, d_frag);
    __syncthreads();
#ifdef DEBUG_ATTN
    if (layer_id == 0 && threadIdx.y == PRINT_WID && threadIdx.x == 0 &&
        blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          for (int k = 0; k < 8; k++) {
            printf(
                "after_update_mdo_states_tid:%d_mask_s_s_frag[%d][%d][%d]:%f  ",
                (int)threadIdx.x,
                (int)fx,
                (int)fz,
                (int)k,
                s_frag[fx][fz][k]);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    kv_idx_base += num_frags_z * 16;
    block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
    if (block_id < 0) {
      block_id = 0;  // 搬但不算
    }
    cache_k_now = cache_k + block_id * kv_n_stride + const_offset;
    produce_kv_blockwise<SharedMemFillMode::kNoFill,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(k_smem,
                                     &kv_smem_offset_w,
                                     &cache_k_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_b_stride,
                                     kv_idx_base,
                                     chunk_end);
    if constexpr (CAL_ROPE) {
      rope_emb_now += num_frags_z * 16 * kv_b_stride;
      produce_kv_blockwise<SharedMemFillMode::kNoFill,
                           NUM_WARPS,
                           BLOCK_SIZE,
                           num_frags_y,
                           num_frags_z,
                           NUM_WARP_Q>(rope_smem,
                                       &kv_smem_offset_w,
                                       &rope_emb_now,
                                       kv_head_idx,
                                       kv_n_stride,
                                       kv_h_stride,
                                       kv_b_stride,
                                       kv_idx_base,
                                       chunk_end);
    }
    commit_group();
    wait_group<1>();
    __syncthreads();
#ifdef DEBUG_ATTN
    if (layer_id == 0 && tid == PRINT_TID && threadIdx.y == PRINT_WID &&
        blockIdx.z == 0 && blockIdx.x == 0) {
      printf("cache_v_smem\n");
      T *v_smem_t = reinterpret_cast<T *>(v_smem.base);
      for (uint32_t i = 0; i < NUM_WARP_KV * num_frags_z * 16; ++i) {
        for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
          printf("v_smem[%d][%d] = %f  ",
                 (int)i,
                 (int)j,
                 (float)v_smem_t[i * num_frags_y * 16 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    // compute sfm*v
    compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, T>(
        &v_smem, &v_smem_offset_r, s_frag, o_frag, d_frag);

    __syncthreads();
    cache_v_now = cache_v + block_id * kv_n_stride + const_offset;
    produce_kv_blockwise<SharedMemFillMode::kFillZero,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(v_smem,
                                     &kv_smem_offset_w,
                                     &cache_v_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_b_stride,
                                     kv_idx_base,
                                     chunk_end);
    commit_group();
  }
  wait_group<0>();
  __syncthreads();
  //// Debug
  // if (threadIdx.x == PRINT_TID) {
  //   printf("end compute \n");
  // }
  // __syncthreads();
  //// end debug
#ifdef DEBUG_ATTN
  if (layer_id == 0 && threadIdx.x == PRINT_TID && threadIdx.y == PRINT_WID &&
      blockIdx.z == 0) {
    printf("o_res\n");
    for (uint32_t i = 0; i < num_frags_x; ++i) {
      printf("m1: %f, m2: %f\n", m_frag[i][0], m_frag[i][1]);
      printf("d1: %f, d2: %f\n", d_frag[i][0], d_frag[i][1]);
      for (uint32_t j = 0; j < num_frags_y; ++j) {
        for (int r_id = 0; r_id < 8; r_id++) {
          printf("o_frag[%d][%d][%d]: %f ",
                 (int)i,
                 (int)j,
                 r_id,
                 o_frag[i][j][r_id]);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  if constexpr (!partition_kv) {
    normalize_d<num_frags_x, num_frags_y>(o_frag, d_frag);
  }
  //// Debug
  // if (threadIdx.x == PRINT_TID) {
  //   printf("end normalize_d \n");
  // }
  // __syncthreads();
  //// end debug

  // write o
  // [num_frags_x, 16, num_frags_y, 16]
  if constexpr (partition_kv) {
    write_o_reg_gmem_shift_smooth_quant<GROUP_SIZE,
                                        num_frags_x,
                                        num_frags_y,
                                        partition_kv,
                                        T>(
        o_frag,
        &qo_smem,
        o_base_ptr_T,
        q_base_seq_id_this_block,
        q_head_idx,
        q_len,
        partition_kv ? q_n_stride * num_chunks : q_n_stride,
        HEAD_DIM,
        layer_id);
  } else {
    write_o_reg_gmem_shift_smooth_quant<GROUP_SIZE,
                                        num_frags_x,
                                        num_frags_y,
                                        partition_kv,
                                        T>(
        o_frag,
        &qo_smem,
        o_base_ptr_int8,
        q_base_seq_id_this_block,
        q_head_idx,
        q_len,
        partition_kv ? q_n_stride * num_chunks : q_n_stride,
        HEAD_DIM,
        layer_id);
  }
  //// Debug
  // if (threadIdx.x == PRINT_TID) {
  //   printf("write_o_reg_gmem_shift_smooth_quant \n");
  // }
  // __syncthreads();
  //// end debug

  if constexpr (partition_kv) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_idx_now =
            q_base_seq_id_this_block + tid / 4 + j * 8 + fx * 16;
        const uint32_t qo_head_idx = q_head_idx + qo_idx_now % GROUP_SIZE;
        const uint32_t qo_idx = q_start_seq_id + qo_idx_now / GROUP_SIZE;
        if (qo_idx - q_start_seq_id < q_len) {
          uint32_t offset;
          if (ENABLE_PREFILL) {
            offset =
                (qo_idx * num_chunks + chunk_idx) * q_num_heads + qo_head_idx;
          } else {
            offset = ((batch_id * speculate_max_draft_token_num +
                       qo_idx_now / GROUP_SIZE) *
                          num_chunks +
                      chunk_idx) *
                         q_num_heads +
                     qo_head_idx;
          }
          tmp_m[offset] = m_frag[fx][j];
          tmp_d[offset] = d_frag[fx][j];
        }
      }
    }
  }
}

template <typename T,
          bool partition_kv,
          uint32_t GROUP_SIZE,
          bool CAUSAL,
          uint32_t NUM_WARPS,
          uint32_t NUM_WARP_Q,
          uint32_t NUM_WARP_KV,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t num_frags_x,
          uint32_t num_frags_z,
          uint32_t num_frags_y,
          typename OutT = T,
          bool CAL_ROPE = false,
          bool ENABLE_PREFILL = true>
__global__ void multi_query_append_attention_for_fusemt_warp1_4_kernel(
    T *__restrict__ q,        // [token_num. num_heads, head_dim]
    T *__restrict__ cache_k,  // [max_block_num, num_heads, block_size,
                              // head_dim]
    T *__restrict__ cache_v,
    const int *__restrict__ seq_lens,
    const int *__restrict__ seq_lens_kv,
    const int *__restrict__ batch_ids,
    const int *__restrict__ tile_ids_per_batch,
    const int *__restrict__ cum_offsets,
    const int *__restrict__ block_table,  // [bsz, block_num_per_seq]
    T *__restrict__ rope_emb,  // [max_seq_len, head_dim] cos1 sin1 cos2 sin2
                               // ...
    const int max_seq_len,
    const int max_dec_len,
    const int max_block_num_per_seq,
    const float scale,
    const uint32_t chunk_size,
    const uint32_t layer_id,
    T *__restrict__ tmp_workspace,  // split kv [token_num, num_chunks,
                                    // num_heads, head_dim]
    float *__restrict__ tmp_m,      // [token_num, num_chunks, num_heads]
    float *__restrict__ tmp_d,      // [token_num, num_chunks, num_heads]
    OutT *__restrict__ out,
    int speculate_max_draft_token_num = 5) {
  // q_len <= 32, num_frags_x = 1/2, num_frags_z = 4 / 4 * 1/2/4, num_frags_y =
  // HEAD_DIM / 16
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
  static_assert(NUM_WARP_Q == 1, "NUM_WARP_Q must be 1");
  static_assert(NUM_WARP_KV == 4, "NUM_WARP_KV must be 4");
  const uint32_t btid = blockIdx.x, kv_head_idx = blockIdx.z;
  const uint32_t kv_num_heads = gridDim.z;
  const uint32_t q_num_heads = kv_num_heads * GROUP_SIZE;
  const uint32_t q_head_idx = kv_head_idx * GROUP_SIZE;
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;
  const uint32_t num_chunks = gridDim.y;
  const uint32_t chunk_idx = blockIdx.y;

  const uint32_t batch_id = batch_ids[btid];
  const uint32_t tile_id = tile_ids_per_batch[btid];
  const uint32_t num_rows_per_block = num_frags_x * 16;
  const int *block_table_now = block_table + batch_id * max_block_num_per_seq;

  const uint32_t q_len = seq_lens[batch_id];
  if (q_len <= 0) {
    return;
  }
  const uint32_t q_end =
      min(q_len, div_up((tile_id + 1) * num_rows_per_block, GROUP_SIZE));
  uint32_t kv_len = seq_lens_kv[batch_id];
  if (ENABLE_PREFILL) {
    kv_len += q_len;  // !!!
    if (kv_len <= 0) {
      return;
    }
  } else {
    if (kv_len <= 0) {
      return;
    }
    kv_len += q_len;
  }
  const uint32_t num_chunks_this_seq = div_up(kv_len, chunk_size);
  if (chunk_idx >= num_chunks_this_seq) {
    return;
  }

  const uint32_t chunk_start = partition_kv ? chunk_idx * chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min(kv_len, chunk_start + chunk_size) : kv_len;
  const uint32_t chunk_len = chunk_end - chunk_start;

  extern __shared__ uint8_t smem[];
  float s_frag[num_frags_x][num_frags_z][8];
  float o_frag[num_frags_x][num_frags_y][8];
  float m_frag[num_frags_x][2];
  float d_frag[num_frags_x][2];
  init_states<T, num_frags_x, num_frags_y>(o_frag, m_frag, d_frag);

  const uint32_t q_n_stride = q_num_heads * HEAD_DIM;
  const uint32_t q_ori_n_stride = (q_num_heads + kv_num_heads * 2) * HEAD_DIM;
  const uint32_t kv_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM;
  const uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM;
  const uint32_t kv_b_stride = HEAD_DIM;
  const uint32_t q_start_seq_id =
      batch_id * max_seq_len - __ldg(&cum_offsets[batch_id]);
  const uint32_t q_base_seq_id_this_block = tile_id * num_frags_x * 16;
  const uint32_t q_offset = q_start_seq_id * q_ori_n_stride +
                            q_head_idx * HEAD_DIM +
                            tid % 8 * num_elems_per_128b<T>();
  const uint32_t o_offset = q_start_seq_id * q_n_stride +
                            q_head_idx * HEAD_DIM +
                            tid % 8 * num_elems_per_128b<T>();
  T *q_base_ptr = q + q_offset;
  T *o_base_ptr_T = nullptr;
  OutT *o_base_ptr_int8 = nullptr;
  if (num_chunks_this_seq <= 1) {
    o_base_ptr_int8 = out + o_offset;
  } else {
    // if constexpr (partition_kv) {
    if (ENABLE_PREFILL) {
      o_base_ptr_T = tmp_workspace + batch_id * num_chunks * q_n_stride +
                     chunk_idx * q_n_stride + q_head_idx * HEAD_DIM +
                     tid % 8 * num_elems_per_128b<T>();
    } else {
      o_base_ptr_T =
          tmp_workspace +
          batch_id * speculate_max_draft_token_num * num_chunks * q_n_stride +
          chunk_idx * q_n_stride + q_head_idx * HEAD_DIM +
          tid % 8 * num_elems_per_128b<T>();
    }
    // } else {
    //   o_base_ptr_int8 = out + o_offset;
    // }
  }
#ifdef DEBUG_ATTN
  if (layer_id == 0 && tid == PRINT_TID && threadIdx.y == PRINT_WID &&
      blockIdx.z == 0) {
    printf("q_base_seq_id_this_block: %d, q_offset: %d, o_offset: %d\n",
           (int)q_base_seq_id_this_block,
           (int)q_offset,
           (int)o_offset);
  }
  __syncthreads();
#endif

  smem_t qo_smem(smem);

  /*
    1 ｜ 3
    ——————
    2 ｜ 4
  */
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      tid % 16, tid / 16);  // 16 * 16
  load_q_global_smem_multi_warps<GROUP_SIZE,
                                 num_frags_x,
                                 num_frags_y,
                                 HEAD_DIM,
                                 T>(q_base_ptr,
                                    &qo_smem,
                                    q_base_seq_id_this_block,
                                    q_end,
                                    q_ori_n_stride,
                                    HEAD_DIM);
  commit_group();
  wait_group<0>();
  __syncthreads();
#ifdef DEBUG_ATTN
  if (layer_id == 0 && tid == PRINT_TID && threadIdx.y == PRINT_WID &&
      blockIdx.z == 0) {
    printf("before scale\n");
    T *q_smem_t = reinterpret_cast<T *>(qo_smem.base);
    for (uint32_t i = 0; i < num_frags_x * 16; ++i) {
      for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
        if (blockIdx.z == 0) {
          printf("q_smem[%d][%d] = %f  ",
                 (int)i,
                 (int)(j),
                 (float)q_smem_t[i * num_frags_y * 16 + j]);
        } else {
          int res = q_smem_t[i * num_frags_y * 16 + j] + static_cast<T>(1.f);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  q_smem_inplace_multiply_sm_scale_multi_warps<num_frags_x, num_frags_y, T>(
      &qo_smem, scale);
#ifdef DEBUG_ATTN
  if (layer_id == 0 && tid == PRINT_TID && threadIdx.y == PRINT_WID &&
      blockIdx.z == 0) {
    printf("after scale\n");
    T *q_smem_t = reinterpret_cast<T *>(qo_smem.base);
    for (uint32_t i = 0; i < num_frags_x * 16; ++i) {
      for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
        if (blockIdx.z == 0) {
          printf("q_smem[%d][%d] = %f  ",
                 (int)i,
                 (int)(j),
                 (float)q_smem_t[i * num_frags_y * 16 + j]);
        } else {
          int res = q_smem_t[i * num_frags_y * 16 + j] + static_cast<T>(1.f);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  smem_t k_smem(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T)),
      v_smem(smem + (num_frags_x + NUM_WARP_KV * num_frags_z) * 16 * HEAD_DIM *
                        sizeof(T));
  smem_t rope_smem;
  if constexpr (CAL_ROPE) {
    rope_smem.base = reinterpret_cast<phi::fusion::b128_t *>(
        smem + (num_frags_x + NUM_WARP_KV * num_frags_z * 2) * 16 * HEAD_DIM *
                   sizeof(T));
  }

  const uint32_t num_iterations = div_up(
      CAUSAL
          ? (min(chunk_len,
                 sub_if_greater_or_zero(
                     kv_len - q_len +
                         div_up((tile_id + 1) * num_rows_per_block, GROUP_SIZE),
                     chunk_start)))
          : chunk_len,
      NUM_WARP_KV * num_frags_z * 16);
  const uint32_t mask_check_iteration =
      (CAUSAL ? (min(chunk_len,
                     sub_if_greater_or_zero(
                         kv_len - q_len +
                             tile_id * num_rows_per_block / GROUP_SIZE,
                         chunk_start)))
              : chunk_len) /
      (NUM_WARP_KV * num_frags_z * 16);
#ifdef DEBUG_ATTN
  if (tid == 0 && threadIdx.y == PRINT_WID && kv_head_idx == 0) {
    printf(
        "batch_id: %d, tile_id: %d, chunk_size: %d, q_len: %d, kv_len: %d, "
        "chunk_start: %d, chunk_end: %d, num_iterations: %d, "
        "mask_check_iteration: %d\n",
        (int)batch_id,
        (int)tile_id,
        (int)chunk_size,
        (int)q_len,
        (int)kv_len,
        (int)chunk_start,
        (int)chunk_end,
        (int)num_iterations,
        (int)mask_check_iteration);
  }
  __syncthreads();
#endif
  /*
    1 ｜ 2
    ——————
    3 ｜ 4
  */
  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);
  /*
    1 ｜ 3
    ——————
    2 ｜ 4   transpose
  */
  uint32_t v_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + tid % 16, tid / 16);
  uint32_t kv_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * 4 + tid / 8, tid % 8);  // 注意内存访问事务，8 * 128 / 8 = 128B
  // uint32_t kv_smem_offset_w =
  // smem_t::get_permuted_offset<num_vecs_per_head>(wid * num_frags_z * 16 + tid
  // / 8, tid % 8); // 注意内存访问事务，8 * 128 / 8 = 128B

  uint32_t kv_idx_base = chunk_start;
  int block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
  const uint32_t const_offset = kv_head_idx * kv_h_stride +
                                (wid * 4 + tid / 8) * kv_b_stride +
                                tid % 8 * num_elems_per_128b<T>();
  // uint32_t kv_idx_base = chunk_start + wid * num_frags_z * 16;
  // int block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
  // const uint32_t const_offset = kv_head_idx * kv_h_stride + (wid *
  // num_frags_z * 16 % BLOCK_SIZE + tid / 8) * kv_b_stride + tid % 8 *
  // num_elems_per_128b<T>();
  T *cache_k_now = cache_k + block_id * kv_n_stride + const_offset;
  T *cache_v_now = cache_v + block_id * kv_n_stride + const_offset;
  T *rope_emb_now;
  if constexpr (CAL_ROPE) {
    rope_emb_now = rope_emb +
                   (kv_idx_base + (wid * 4 + tid / 8)) * kv_b_stride +
                   tid % 8 * num_elems_per_128b<T>();
  }
#ifdef DEBUG_ATTN
  if (layer_id == 0 && threadIdx.x == PRINT_TID && threadIdx.y == PRINT_WID &&
      blockIdx.z == 0) {
    printf(
        "2108 ori q_smem_offset_r: %d, k_smem_offset_r: %d, v_smem_offset_r: "
        "%d, kv_smem_offset_w: %d\n",
        (int)q_smem_offset_r,
        (int)k_smem_offset_r,
        (int)v_smem_offset_r,
        (int)kv_smem_offset_w);
  }
  __syncthreads();
#endif

  // load BLOCK_SIZE * HEAD_DIM each time
  produce_kv_blockwise<SharedMemFillMode::kNoFill,
                       NUM_WARPS,
                       BLOCK_SIZE,
                       num_frags_y,
                       num_frags_z,
                       NUM_WARP_Q>(k_smem,
                                   &kv_smem_offset_w,
                                   &cache_k_now,
                                   kv_head_idx,
                                   kv_n_stride,
                                   kv_h_stride,
                                   kv_b_stride,
                                   kv_idx_base,
                                   chunk_end);
  if constexpr (CAL_ROPE) {
    produce_kv_blockwise<SharedMemFillMode::kNoFill,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(rope_smem,
                                     &kv_smem_offset_w,
                                     &rope_emb_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_b_stride,
                                     kv_idx_base,
                                     chunk_end);
  }
  commit_group();
#ifdef DEBUG_ATTN
  if (layer_id == 0 && threadIdx.x == PRINT_TID && threadIdx.y == PRINT_WID &&
      blockIdx.z == 0) {
    printf(
        "2139 ori q_smem_offset_r: %d, k_smem_offset_r: %d, v_smem_offset_r: "
        "%d, kv_smem_offset_w: %d\n",
        (int)q_smem_offset_r,
        (int)k_smem_offset_r,
        (int)v_smem_offset_r,
        (int)kv_smem_offset_w);
  }
  __syncthreads();
#endif
  produce_kv_blockwise<SharedMemFillMode::kFillZero,
                       NUM_WARPS,
                       BLOCK_SIZE,
                       num_frags_y,
                       num_frags_z,
                       NUM_WARP_Q>(v_smem,
                                   &kv_smem_offset_w,
                                   &cache_v_now,
                                   kv_head_idx,
                                   kv_n_stride,
                                   kv_h_stride,
                                   kv_b_stride,
                                   kv_idx_base,
                                   chunk_end);
  commit_group();
#ifdef DEBUG_ATTN
  if (layer_id == 0 && threadIdx.x == PRINT_TID && threadIdx.y == PRINT_WID &&
      blockIdx.z == 0) {
    printf(
        "2021 ori q_smem_offset_r: %d, k_smem_offset_r: %d, v_smem_offset_r: "
        "%d, kv_smem_offset_w: %d\n",
        (int)q_smem_offset_r,
        (int)k_smem_offset_r,
        (int)v_smem_offset_r,
        (int)kv_smem_offset_w);
  }
  __syncthreads();
#endif
#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    wait_group<1>();
    __syncthreads();
#ifdef DEBUG_ATTN
    if (layer_id == 0 && tid == PRINT_TID && threadIdx.y == PRINT_WID &&
        blockIdx.z == 0 && blockIdx.x == 0) {
      printf("cache_k_smem\n");
      T *k_smem_t = reinterpret_cast<T *>(k_smem.base);
      for (uint32_t i = 0; i < NUM_WARP_KV * num_frags_z * 16; ++i) {
        for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
          printf("k_smem[%d][%d] = %f  ",
                 (int)i,
                 (int)j,
                 (float)k_smem_t[i * num_frags_y * 16 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    // s = qk
    compute_qk<num_frags_x, num_frags_y, num_frags_z, T, CAL_ROPE>(
        &qo_smem,
        &q_smem_offset_r,
        &k_smem,
        &k_smem_offset_r,
        s_frag,
        &rope_smem);
    // mask according to kv_idx and q_idx
    if (iter >= mask_check_iteration) {
      mask_s<T,
             partition_kv,
             CAUSAL,
             GROUP_SIZE,
             NUM_WARPS,
             num_frags_x,
             num_frags_y,
             num_frags_z>(q_base_seq_id_this_block,
                          kv_idx_base + wid * num_frags_z * 16,
                          q_len,
                          kv_len,
                          chunk_end,
                          s_frag);
    }
#ifdef DEBUG_ATTN
    if (layer_id == 0 && tid == 0 && threadIdx.y == PRINT_WID &&
        blockIdx.z == 0) {
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          for (int k = 0; k < 8; k++) {
            printf("mask_s_s_frag[%d][%d][%d]:%f  ",
                   (int)fx,
                   (int)fz,
                   (int)k,
                   s_frag[fx][fz][k]);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    // update m,d
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
        s_frag, o_frag, m_frag, d_frag);
    __syncthreads();
#ifdef DEBUG_ATTN
    if (layer_id == 0 && tid == 0 && threadIdx.y == PRINT_WID &&
        blockIdx.z == 0) {
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          for (int k = 0; k < 8; k++) {
            printf("after_update_mdo_states, s_frag[%d][%d][%d]:%f  ",
                   (int)fx,
                   (int)fz,
                   (int)k,
                   s_frag[fx][fz][k]);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    kv_idx_base += NUM_WARP_KV * num_frags_z * 16;
    block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
    if (block_id < 0) {
      block_id = 0;  // 搬但不算
    }
    cache_k_now = cache_k + block_id * kv_n_stride + const_offset;
    produce_kv_blockwise<SharedMemFillMode::kNoFill,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(k_smem,
                                     &kv_smem_offset_w,
                                     &cache_k_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_b_stride,
                                     kv_idx_base,
                                     chunk_end);
    if constexpr (CAL_ROPE) {
      rope_emb_now += NUM_WARP_KV * num_frags_z * 16 * kv_b_stride;
      produce_kv_blockwise<SharedMemFillMode::kNoFill,
                           NUM_WARPS,
                           BLOCK_SIZE,
                           num_frags_y,
                           num_frags_z,
                           NUM_WARP_Q>(rope_smem,
                                       &kv_smem_offset_w,
                                       &rope_emb_now,
                                       kv_head_idx,
                                       kv_n_stride,
                                       kv_h_stride,
                                       kv_b_stride,
                                       kv_idx_base,
                                       chunk_end);
    }
    commit_group();
    wait_group<1>();
    __syncthreads();
#ifdef DEBUG_ATTN
    if (layer_id == 0 && tid == PRINT_TID && threadIdx.y == PRINT_WID &&
        blockIdx.z == 0 && blockIdx.x == 0) {
      printf("cache_v_smem\n");
      T *v_smem_t = reinterpret_cast<T *>(v_smem.base);
      for (uint32_t i = 0; i < NUM_WARP_KV * num_frags_z * 16; ++i) {
        for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
          printf("v_smem[%d][%d] = %f  ",
                 (int)i,
                 (int)j,
                 (float)v_smem_t[i * num_frags_y * 16 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    // compute sfm*v
    compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, T>(
        &v_smem, &v_smem_offset_r, s_frag, o_frag, d_frag, layer_id);
    __syncthreads();

    cache_v_now = cache_v + block_id * kv_n_stride + const_offset;
    produce_kv_blockwise<SharedMemFillMode::kFillZero,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(v_smem,
                                     &kv_smem_offset_w,
                                     &cache_v_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_b_stride,
                                     kv_idx_base,
                                     chunk_end);
    commit_group();
  }
  wait_group<0>();
  __syncthreads();
#ifdef DEBUG_ATTN
  if (layer_id == 0 && threadIdx.x == PRINT_TID && threadIdx.y == PRINT_WID &&
      blockIdx.z == 0) {
    printf("before merge z\n");
    for (uint32_t i = 0; i < num_frags_x; ++i) {
      printf("m1: %f, m2: %f\n", m_frag[i][0], m_frag[i][1]);
      printf("d1: %f, d2: %f\n", d_frag[i][0], d_frag[i][1]);
      for (uint32_t j = 0; j < num_frags_y; ++j) {
        for (int r_id = 0; r_id < 8; r_id++) {
          printf("o_frag[%d][%d][%d]: %f ",
                 (int)i,
                 (int)j,
                 r_id,
                 o_frag[i][j][r_id]);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  merge_block_res_v2<num_frags_x, num_frags_y, T>(
      o_frag, reinterpret_cast<float *>(smem), m_frag, d_frag, wid, tid);
#ifdef DEBUG_ATTN
  if (layer_id == 0 && threadIdx.x == PRINT_TID && threadIdx.y == PRINT_WID &&
      blockIdx.z == 0) {
    printf("after merge z\n");
    for (uint32_t i = 0; i < num_frags_x; ++i) {
      printf("m1: %f, m2: %f\n", m_frag[i][0], m_frag[i][1]);
      printf("d1: %f, d2: %f\n", d_frag[i][0], d_frag[i][1]);
      for (uint32_t j = 0; j < num_frags_y; ++j) {
        for (int r_id = 0; r_id < 8; r_id++) {
          printf("o_frag[%d][%d][%d]: %f ",
                 (int)i,
                 (int)j,
                 r_id,
                 o_frag[i][j][r_id]);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif
  if (num_chunks_this_seq <= 1) {
    normalize_d<num_frags_x, num_frags_y>(o_frag, d_frag);
  }
  // if constexpr (!partition_kv){
  //   normalize_d<num_frags_x, num_frags_y>(o_frag, d_frag);
  // }

  // write o
  // [num_frags_x, 16, num_frags_y, 16]
  if (num_chunks_this_seq <= 1) {
    write_o_reg_gmem_multi_warps_shift_smooth_quant<GROUP_SIZE,
                                                    num_frags_x,
                                                    num_frags_y,
                                                    false,
                                                    T>(o_frag,
                                                       &qo_smem,
                                                       o_base_ptr_int8,
                                                       q_base_seq_id_this_block,
                                                       q_head_idx,
                                                       q_len,
                                                       q_n_stride,
                                                       HEAD_DIM,
                                                       layer_id);
  } else {
    // if constexpr (partition_kv) {
    write_o_reg_gmem_multi_warps_shift_smooth_quant<GROUP_SIZE,
                                                    num_frags_x,
                                                    num_frags_y,
                                                    partition_kv,
                                                    T>(o_frag,
                                                       &qo_smem,
                                                       o_base_ptr_T,
                                                       q_base_seq_id_this_block,
                                                       q_head_idx,
                                                       q_len,
                                                       q_n_stride * num_chunks,
                                                       HEAD_DIM,
                                                       layer_id);
    // } else {
    //   write_o_reg_gmem_multi_warps_shift_smooth_quant<GROUP_SIZE,
    //   num_frags_x, num_frags_y, partition_kv>(
    //     o_frag,
    //     &qo_smem,
    //     o_base_ptr_int8,
    //     shift_bias,
    //     smooth_weight,
    //     q_base_seq_id_this_block,
    //     q_head_idx,
    //     in_scale,
    //     q_len,
    //     q_n_stride,
    //     HEAD_DIM,
    //     layer_id
    //   );
    // }
  }
  if (num_chunks_this_seq > 1) {
    // if constexpr (partition_kv) {
    if (threadIdx.y == PRINT_WID) {
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          const uint32_t qo_idx_now =
              q_base_seq_id_this_block + tid / 4 + j * 8 + fx * 16;
          const uint32_t qo_head_idx = q_head_idx + qo_idx_now % GROUP_SIZE;
          const uint32_t qo_idx = q_start_seq_id + qo_idx_now / GROUP_SIZE;
#ifdef DEBUG_ATTN
          if (batch_id == 0) {
            printf(
                "bid: %d, tid: %d, wid: %d, q_base_seq_id_this_block: %d, "
                "qo_idx_now: %d, qo_idx: %d, q_start_seq_id: %d, q_len: %d, m: "
                "%f, d: %f\n",
                (int)batch_id,
                (int)tid,
                (int)wid,
                (int)q_base_seq_id_this_block,
                (int)qo_idx_now,
                (int)qo_idx,
                (int)q_start_seq_id,
                (int)q_len,
                (float)m_frag[fx][j],
                (float)d_frag[fx][j]);
          }
#endif
          if (qo_idx - q_start_seq_id < q_len) {
            uint32_t offset;
            if (ENABLE_PREFILL) {
              offset = (batch_id * num_chunks + chunk_idx) * q_num_heads +
                       qo_head_idx;
            } else {
              offset = ((batch_id * speculate_max_draft_token_num +
                         qo_idx_now / GROUP_SIZE) *
                            num_chunks +
                        chunk_idx) *
                           q_num_heads +
                       qo_head_idx;
            }
            tmp_m[offset] = m_frag[fx][j];
            tmp_d[offset] = d_frag[fx][j];
          }
        }
      }
    }
  }
}

template <typename T,
          typename Context,
          uint32_t GROUP_SIZE,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          bool CAUSAL,
          uint32_t BLOCK_SHAPE_Q,
          uint32_t NUM_WARP_Q,
          typename OutT = T,
          bool ENABLE_PREFILL = true,
          bool USE_SYSTEM = false,
          bool CAL_ROPE = false>
void MultiQueryAppendForFuseMtAttention(const Context &dev_ctx,
                                        cudaStream_t &stream,
                                        const DenseTensor &q,
                                        const DenseTensor &cache_k,
                                        const DenseTensor &cache_v,
                                        const DenseTensor *attn_mask,
                                        const DenseTensor &seq_lens_q,
                                        const DenseTensor &seq_lens_kv,
                                        const DenseTensor &seq_lens_encoder,
                                        const DenseTensor &padding_offsets,
                                        const DenseTensor &cum_offsets,
                                        const DenseTensor &block_table,
                                        const DenseTensor &batch_ids,
                                        const DenseTensor &tile_ids_per_batch,
                                        const DenseTensor *seq_mapping,
                                        const DenseTensor *rope_emb,
                                        const int num_blocks_x_cpu,
                                        const int max_seq_len,
                                        const int max_dec_len,
                                        const int num_heads,
                                        const int kv_num_heads,
                                        const int layer_id,
                                        const bool causal,
                                        const bool is_decoder,
                                        DenseTensor *out) {
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;
  using OUT_NV_TYPE = typename cascade_attn_type_traits<OutT>::type;
  const auto &q_dims = q.dims();
  const auto &k_dims = cache_k.dims();
  const auto &cum_offsets_dims = cum_offsets.dims();
  const uint32_t token_num = q_dims[0];
  const uint32_t bsz = cum_offsets_dims[0];
  const uint32_t max_block_num_per_seq = block_table.dims()[1];
  // VLOG(2) << "bsz: " << bsz << ", token_num: " << token_num << ", num_heads:
  // " << num_heads << ", kv_num_heads: " << kv_num_heads << ", group_size: " <<
  // GROUP_SIZE;

  constexpr uint32_t num_warps = 4;
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  constexpr uint32_t num_frags_x = BLOCK_SHAPE_Q / (16 * NUM_WARP_Q);  // 1 or 2
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_qrow_per_block = NUM_WARP_Q * num_frags_x * 16;
  // VLOG(2) << "num_warps: " << num_warps << ", BLOCK_SHAPE_Q: " <<
  // BLOCK_SHAPE_Q << ", NUM_WARP_Q: " << NUM_WARP_Q; VLOG(2) <<
  // "seq_lens_encoder: " << seq_lens_encoder; VLOG(2) << "is_decoder: " <<
  // is_decoder; VLOG(2) << "num_frags_x: " << num_frags_x << ", num_frags_y: "
  // << num_frags_y;

  // VLOG(2) << "batch_ids: " << batch_ids;
  // VLOG(2) << "tile_ids_per_batch: " << tile_ids_per_batch;

  const float scale = 1.f / sqrt(HEAD_DIM);

  if constexpr (NUM_WARP_Q == 4) {
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16;  // !!!
    // VLOG(2) << "num_frags_z: " << num_frags_z;
    // constexpr uint32_t num_frags_z = 8; // 128 per iter, 4 is better?
    constexpr uint32_t smem_size =
        (num_warps * num_frags_x + NUM_WARP_KV * num_frags_z * 2) * 16 *
            HEAD_DIM * sizeof(T) +
        (CAL_ROPE ? NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(T) : 0);
    // VLOG(2) << "smem_size: " << smem_size / 1024 << " KB";
    auto split_kv_kernel =
        multi_query_append_for_fuse_mt_attention_kernel<NV_TYPE,
                                                        true,
                                                        GROUP_SIZE,
                                                        CAUSAL,
                                                        num_warps,
                                                        NUM_WARP_Q,
                                                        NUM_WARP_KV,
                                                        HEAD_DIM,
                                                        BLOCK_SIZE,
                                                        num_frags_x,
                                                        num_frags_z,
                                                        num_frags_y,
                                                        OUT_NV_TYPE,
                                                        CAL_ROPE,
                                                        ENABLE_PREFILL,
                                                        USE_SYSTEM>;
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(split_kv_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size);
    }
    const int dev_id = 0;
    int sm_count;
    int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &act_blocks_per_sm, split_kv_kernel, num_warps * 32, smem_size);
    assert(act_blocks_per_sm > 1);
    const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
    const int num_blocks_need = num_blocks_x_cpu * kv_num_heads;
    const int max_num_chunks = div_up(num_blocks_per_wave, num_blocks_need);
    const float ratio = static_cast<float>(num_blocks_need) /
                        static_cast<float>(num_blocks_per_wave);

    uint32_t chunk_size =
        static_cast<uint32_t>(FLAGS_cascade_attention_max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(
          FLAGS_cascade_encoder_attention_max_partition_size);
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);
    VLOG(2) << "chunk_size: " << chunk_size;
    // VLOG(2) << "num_blocks_per_wave: " << num_blocks_per_wave << ",
    // num_blocks_need: " << num_blocks_need << ", max_num_chunks: " <<
    // max_num_chunks << ", ratio: " << ratio; VLOG(2) << "num_chunks: " <<
    // num_chunks;

    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    dim3 blocks(32, num_warps);
    VLOG(1) << "grids: " << grids.x << " " << grids.y << " " << grids.z;
    VLOG(1) << "blocks: " << blocks.x << " " << blocks.y;
    if (num_chunks <= 1) {
      VLOG(1) << "nosplit_kv_kernel";
      auto nosplit_kv_kernel =
          multi_query_append_for_fuse_mt_attention_kernel<NV_TYPE,
                                                          false,
                                                          GROUP_SIZE,
                                                          CAUSAL,
                                                          num_warps,
                                                          NUM_WARP_Q,
                                                          NUM_WARP_KV,
                                                          HEAD_DIM,
                                                          BLOCK_SIZE,
                                                          num_frags_x,
                                                          num_frags_z,
                                                          num_frags_y,
                                                          OUT_NV_TYPE,
                                                          CAL_ROPE,
                                                          ENABLE_PREFILL,
                                                          USE_SYSTEM>;
      if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(nosplit_kv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
      }
      VLOG(1) << "max_seq_len " << max_seq_len;
      VLOG(1) << "max_dec_len " << max_dec_len;
      VLOG(1) << "max_block_num_per_seq " << max_block_num_per_seq;
      VLOG(1) << "scale " << scale;
      VLOG(1) << "max_seq_len " << max_seq_len;

      VLOG(1) << "q " << q.dims();
      print_tensor<OutT>(q, __FILE__, __LINE__);
      VLOG(1) << "cache_k.data<T>() " << cache_k.data<T>();
      VLOG(1) << "cache_v.data<T>() " << cache_v.data<T>();
      VLOG(1) << "out " << out->dims();

      nosplit_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(q.data<T>())),
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k.data<T>())),
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v.data<T>())),
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          seq_mapping ? seq_mapping->data<int>() : nullptr,
          rope_emb ? reinterpret_cast<NV_TYPE *>(
                         const_cast<T *>(rope_emb->data<T>()))
                   : nullptr,
          // rope_emb ? reinterpret_cast<const NV_TYPE*>(rope_emb->data<T>()) :
          // nullptr,
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          chunk_size,
          layer_id,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
          FLAGS_speculate_max_draft_token_num);
      print_tensor<OutT>(*out, __FILE__, __LINE__);
    } else {
      VLOG(1) << "split kv";
      phi::DenseTensor tmp_workspace, tmp_m, tmp_d;
      if (ENABLE_PREFILL) {
        tmp_workspace.Resize({token_num, num_chunks, num_heads, HEAD_DIM});
        tmp_m.Resize({token_num, num_chunks, num_heads});
        tmp_d.Resize({token_num, num_chunks, num_heads});
      } else {
        tmp_workspace.Resize({FLAGS_speculate_max_draft_token_num * bsz,
                              num_chunks,
                              num_heads,
                              HEAD_DIM});
        tmp_m.Resize(
            {FLAGS_speculate_max_draft_token_num * bsz, num_chunks, num_heads});
        tmp_d.Resize(
            {FLAGS_speculate_max_draft_token_num * bsz, num_chunks, num_heads});
      }
      dev_ctx.template Alloc<T>(&tmp_workspace);
      dev_ctx.template Alloc<float>(&tmp_m);
      dev_ctx.template Alloc<float>(&tmp_d);
      if (tmp_workspace.dims()[0] <
          6)  // VLOG(2) << "tmp_workspace1: " << tmp_workspace;
        // VLOG(2) << "tmp_m1: " << tmp_m;
        // VLOG(2) << "tmp_d1: " << tmp_d;
        split_kv_kernel<<<grids, blocks, smem_size, stream>>>(
            reinterpret_cast<NV_TYPE *>(const_cast<T *>(q.data<T>())),
            reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k.data<T>())),
            reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v.data<T>())),
            seq_lens_q.data<int>(),
            seq_lens_kv.data<int>(),
            batch_ids.data<int>(),
            tile_ids_per_batch.data<int>(),
            cum_offsets.data<int>(),
            block_table.data<int>(),
            seq_mapping ? seq_mapping->data<int>() : nullptr,
            rope_emb ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(rope_emb->data<T>()))
                     : nullptr,
            // rope_emb ? reinterpret_cast<const NV_TYPE*>(rope_emb->data<T>())
            // : nullptr,
            max_seq_len,
            max_dec_len,
            max_block_num_per_seq,
            scale,
            chunk_size,
            layer_id,
            reinterpret_cast<NV_TYPE *>(tmp_workspace.data<T>()),
            tmp_m.data<float>(),
            tmp_d.data<float>(),
            reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
            FLAGS_speculate_max_draft_token_num);
      // merge
      // if (tmp_workspace.dims()[0] < 6)
      // VLOG(2) << "tmp_workspace: " << tmp_workspace;
      // VLOG(2) << "tmp_m: " << tmp_m;
      // VLOG(2) << "tmp_d: " << tmp_d;
      // PrintMatrix(tmp_workspace.data<T>(), tmp_workspace.numel(),
      // "tmp_workspace_layer_" + std::to_string(layer_id) + "_dev_" +
      // std::to_string(tmp_workspace.place().GetDeviceId()));
      // PrintMatrix(tmp_m.data<float>(), tmp_m.numel(), "tmp_m_layer_" +
      // std::to_string(layer_id) + "_dev_" +
      // std::to_string(tmp_workspace.place().GetDeviceId()));
      // PrintMatrix(tmp_d.data<float>(), tmp_d.numel(), "tmp_d_layer_" +
      // std::to_string(layer_id) + "_dev_" +
      // std::to_string(tmp_workspace.place().GetDeviceId()));

      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_for_fusemt_decoder_v2_kernel<NV_TYPE,
                                                        vec_size,
                                                        blocky,
                                                        HEAD_DIM,
                                                        OUT_NV_TYPE,
                                                        ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE *>(tmp_workspace.data<T>()),
                tmp_m.data<float>(),
                tmp_d.data<float>(),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM);
      } else {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(min(sm_count * 4, token_num),
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_for_fusemt_v2_kernel<NV_TYPE,
                                                vec_size,
                                                blocky,
                                                HEAD_DIM,
                                                OUT_NV_TYPE,
                                                ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE *>(tmp_workspace.data<T>()),
                tmp_m.data<float>(),
                tmp_d.data<float>(),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM,
                token_num,
                FLAGS_speculate_max_draft_token_num);
      }
    }
  } else {
    VLOG(2) << "NUM_WARP_Q: " << NUM_WARP_Q;
    VLOG(2) << "NUM_WARP_KV: " << NUM_WARP_KV;
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / NUM_WARP_KV;  // !!!
    // VLOG(2) << "num_frags_z: " << num_frags_z;
    constexpr uint32_t smem_size =
        (num_frags_x + NUM_WARP_KV * num_frags_z * 2) * 16 * HEAD_DIM *
            sizeof(T) +
        (CAL_ROPE ? NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(T) : 0);
    // VLOG(2) << "smem_size: " << smem_size / 1024 << " KB";
    auto split_kv_kernel =
        multi_query_append_attention_for_fusemt_warp1_4_kernel<NV_TYPE,
                                                               true,
                                                               GROUP_SIZE,
                                                               CAUSAL,
                                                               num_warps,
                                                               NUM_WARP_Q,
                                                               NUM_WARP_KV,
                                                               HEAD_DIM,
                                                               BLOCK_SIZE,
                                                               num_frags_x,
                                                               num_frags_z,
                                                               num_frags_y,
                                                               OUT_NV_TYPE,
                                                               CAL_ROPE,
                                                               ENABLE_PREFILL>;
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(split_kv_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size);
    }
    const int dev_id = 0;
    int sm_count;
    int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &act_blocks_per_sm, split_kv_kernel, num_warps * 32, smem_size);
    assert(act_blocks_per_sm > 1);
    const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
    const int num_blocks_need = num_blocks_x_cpu * kv_num_heads;
    const int max_num_chunks = div_up(num_blocks_per_wave, num_blocks_need);
    const float ratio = static_cast<float>(num_blocks_need) /
                        static_cast<float>(num_blocks_per_wave);

    uint32_t chunk_size =
        static_cast<uint32_t>(FLAGS_cascade_attention_max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(
          FLAGS_cascade_encoder_attention_max_partition_size);
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);
    VLOG(2) << "chunk_size: " << chunk_size;
    // VLOG(2) << "num_blocks_per_wave: " << num_blocks_per_wave << ",
    // num_blocks_need: " << num_blocks_need << ", max_num_chunks: " <<
    // max_num_chunks << ", ratio: " << ratio; VLOG(2) << "num_chunks: " <<
    // num_chunks;

    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    // dim3 grids(num_blocks_x_cpu, num_chunks, 1);
    dim3 blocks(32, num_warps);
    // VLOG(2) << "grids: " << grids.x << " " << grids.y << " " << grids.z;
    // VLOG(2) << "blocks: " << blocks.x << " " << blocks.y;

    if (num_chunks <= 1) {
      VLOG(2) << "nosplit_kv_kernel";
      auto nosplit_kv_kernel =
          multi_query_append_attention_for_fusemt_warp1_4_kernel<
              NV_TYPE,
              false,
              GROUP_SIZE,
              CAUSAL,
              num_warps,
              NUM_WARP_Q,
              NUM_WARP_KV,
              HEAD_DIM,
              BLOCK_SIZE,
              num_frags_x,
              num_frags_z,
              num_frags_y,
              OUT_NV_TYPE,
              CAL_ROPE,
              ENABLE_PREFILL>;
      if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(nosplit_kv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
      }
      VLOG(1) << "seq_lens_q.data<int>()" << seq_lens_q.data<int>();
      VLOG(1) << "seq_lens_kv.data<int>()" << seq_lens_kv.data<int>();
      VLOG(1) << "batch_ids.data<int>()" << batch_ids.data<int>();
      VLOG(1) << "tile_ids_per_batch.data<int>()"
              << tile_ids_per_batch.data<int>();
      VLOG(1) << "cum_offsets.data<int>()" << cum_offsets.data<int>();
      VLOG(1) << "block_table.data<int>()" << block_table.data<int>();

      nosplit_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(q.data<T>())),
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k.data<T>())),
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v.data<T>())),
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          rope_emb ? reinterpret_cast<NV_TYPE *>(
                         const_cast<T *>(rope_emb->data<T>()))
                   : nullptr,
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          chunk_size,
          layer_id,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
          FLAGS_speculate_max_draft_token_num);
    } else {
      VLOG(2) << "split kv";
      phi::DenseTensor tmp_workspace, tmp_m, tmp_d;
      if (is_decoder) {
        tmp_workspace.Resize({bsz, num_chunks, num_heads, HEAD_DIM});
        tmp_m.Resize({bsz, num_chunks, num_heads});
        tmp_d.Resize({bsz, num_chunks, num_heads});
      } else {
        if (ENABLE_PREFILL) {
          tmp_workspace.Resize({token_num, num_chunks, num_heads, HEAD_DIM});
          tmp_m.Resize({token_num, num_chunks, num_heads});
          tmp_d.Resize({token_num, num_chunks, num_heads});
        } else {
          tmp_workspace.Resize({FLAGS_speculate_max_draft_token_num * bsz,
                                num_chunks,
                                num_heads,
                                HEAD_DIM});
          tmp_m.Resize({FLAGS_speculate_max_draft_token_num * bsz,
                        num_chunks,
                        num_heads});
          tmp_d.Resize({FLAGS_speculate_max_draft_token_num * bsz,
                        num_chunks,
                        num_heads});
        }
      }
      dev_ctx.template Alloc<T>(&tmp_workspace);
      dev_ctx.template Alloc<float>(&tmp_m);
      dev_ctx.template Alloc<float>(&tmp_d);
      if (tmp_workspace.dims()[0] <
          6)  // VLOG(2) << "tmp_workspace1: " << tmp_workspace;
        // VLOG(2) << "tmp_m1: " << tmp_m;
        // VLOG(2) << "tmp_d1: " << tmp_d;
        split_kv_kernel<<<grids, blocks, smem_size, stream>>>(
            reinterpret_cast<NV_TYPE *>(const_cast<T *>(q.data<T>())),
            reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k.data<T>())),
            reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v.data<T>())),
            seq_lens_q.data<int>(),
            seq_lens_kv.data<int>(),
            batch_ids.data<int>(),
            tile_ids_per_batch.data<int>(),
            cum_offsets.data<int>(),
            block_table.data<int>(),
            rope_emb ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(rope_emb->data<T>()))
                     : nullptr,
            max_seq_len,
            max_dec_len,
            max_block_num_per_seq,
            scale,
            chunk_size,
            layer_id,
            reinterpret_cast<NV_TYPE *>(tmp_workspace.data<T>()),
            tmp_m.data<float>(),
            tmp_d.data<float>(),
            reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
            FLAGS_speculate_max_draft_token_num);
      // merge
      // if (tmp_workspace.dims()[0] < 6)
      // VLOG(2) << "tmp_workspace: " << tmp_workspace;
      // VLOG(2) << "tmp_m: " << tmp_m;
      // VLOG(2) << "tmp_d: " << tmp_d;
      // PrintMatrix(tmp_workspace.data<T>(), tmp_workspace.numel(),
      // "tmp_workspace_layer_" + std::to_string(layer_id) + "_dev_" +
      // std::to_string(tmp_workspace.place().GetDeviceId()));
      // PrintMatrix(tmp_m.data<float>(), tmp_m.numel(), "tmp_m_layer_" +
      // std::to_string(layer_id) + "_dev_" +
      // std::to_string(tmp_workspace.place().GetDeviceId()));
      // PrintMatrix(tmp_d.data<float>(), tmp_d.numel(), "tmp_d_layer_" +
      // std::to_string(layer_id) + "_dev_" +
      // std::to_string(tmp_workspace.place().GetDeviceId()));

      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_for_fusemt_decoder_v2_kernel<NV_TYPE,
                                                        vec_size,
                                                        blocky,
                                                        HEAD_DIM,
                                                        OUT_NV_TYPE,
                                                        ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE *>(tmp_workspace.data<T>()),
                tmp_m.data<float>(),
                tmp_d.data<float>(),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM);
      } else {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(min(sm_count * 4, token_num),
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_for_fusemt_v2_kernel<NV_TYPE,
                                                vec_size,
                                                blocky,
                                                HEAD_DIM,
                                                OUT_NV_TYPE,
                                                ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE *>(tmp_workspace.data<T>()),
                tmp_m.data<float>(),
                tmp_d.data<float>(),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM,
                token_num,
                FLAGS_speculate_max_draft_token_num);
      }
    }
  }
}

template <typename T, typename Context, typename OutT>
void CascadeAppendAttentionForFuseMtKernel(
    const Context &dev_ctx,
    cudaStream_t &stream,
    const DenseTensor &q,  // [token_num, num_heads, head_dim]
    const DenseTensor
        &cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    const DenseTensor
        &cache_v,  // [max_block_num, num_heads, head_dim, block_size]
    const DenseTensor *attn_mask,
    const DenseTensor &seq_lens_q,
    const DenseTensor &seq_lens_kv,
    const DenseTensor &seq_lens_encoder,
    const DenseTensor &padding_offsets,
    const DenseTensor &cum_offsets,
    const DenseTensor &block_table,
    const DenseTensor &batch_ids,
    const DenseTensor &tile_ids,
    const DenseTensor *seq_mapping,
    const DenseTensor *rope_emb,
    int num_blocks,
    int block_shape_q,
    int max_seq_len,
    int max_dec_len,
    int num_heads,
    int kv_num_heads,
    int head_dim,
    int layer_id,
    bool causal,
    bool is_decoder,
    DenseTensor *out) {
  if (max_dec_len <= 0) {
    return;
  }
  const auto &q_dims = q.dims();
  const auto &k_dims = cache_k.dims();
  const auto &cum_offsets_dims = cum_offsets.dims();
  const uint32_t token_num = q_dims[0];
  const uint32_t block_size = k_dims[2];
  const uint32_t bsz = cum_offsets_dims[0];
  kv_num_heads = kv_num_heads == -1 ? num_heads : kv_num_heads;
  const uint32_t group_size = num_heads / kv_num_heads;
  VLOG(2) << "block_shape_q: " << block_shape_q;

  const uint32_t use_system = seq_mapping ? 1 : 0;
  DISPATCH_CAUSAL(
      causal,
      CAUSAL,
      {DISPATCH_GQA_GROUP_SIZE(
          group_size,
          GROUP_SIZE,
          {DISPATCH_HEAD_DIM(
              head_dim,
              HEAD_DIM,
              {DISPATCH_BLOCK_SIZE(
                  block_size,
                  BLOCK_SIZE,
                  {DISPATCH_BLOCKSHAPE_Q(
                      block_shape_q,
                      BLOCK_SHAPE_Q,
                      NUM_WARP_Q,
                      {DISPATCH_USE_SYSTEM(
                          use_system,
                          USE_SYSTEM,
                          {DISPATCH_IS_ROPE(rope_emb, CAL_ROPE, {
                            MultiQueryAppendForFuseMtAttention<T,
                                                               Context,
                                                               GROUP_SIZE,
                                                               HEAD_DIM,
                                                               BLOCK_SIZE,
                                                               CAUSAL,
                                                               BLOCK_SHAPE_Q,
                                                               NUM_WARP_Q,
                                                               OutT,
                                                               true,
                                                               USE_SYSTEM,
                                                               CAL_ROPE>(
                                dev_ctx,
                                stream,
                                q,
                                cache_k,
                                cache_v,
                                attn_mask,
                                seq_lens_q,
                                seq_lens_kv,
                                seq_lens_encoder,
                                padding_offsets,
                                cum_offsets,
                                block_table,
                                batch_ids,
                                tile_ids,
                                seq_mapping,
                                rope_emb,
                                num_blocks,
                                max_seq_len,
                                max_dec_len,
                                num_heads,
                                kv_num_heads,
                                layer_id,
                                causal,
                                is_decoder,
                                out);
                          })})})})})})})
}

template void CascadeAppendAttentionForFuseMtKernel<phi::dtype::float16,
                                                    phi::GPUContext,
                                                    phi::dtype::float16>(
    const phi::GPUContext &dev_ctx,
    cudaStream_t &stream,
    const DenseTensor &q,  // [token_num, num_heads, head_dim]
    const DenseTensor
        &cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    const DenseTensor
        &cache_v,  // [max_block_num, num_heads, head_dim, block_size]
    const DenseTensor *attn_mask,
    const DenseTensor &seq_lens_q,
    const DenseTensor &seq_lens_kv,
    const DenseTensor &seq_lens_encoder,
    const DenseTensor &padding_offsets,
    const DenseTensor &cum_offsets,
    const DenseTensor &block_table,
    const DenseTensor &batch_ids,
    const DenseTensor &tile_ids,
    const DenseTensor *seq_mapping,
    const DenseTensor *rope_emb,
    int num_blocks,
    int block_shape_q,
    int max_seq_len,
    int max_dec_len,
    int num_heads,
    int kv_num_heads,
    int head_dim,
    int layer_id,
    bool causal,
    bool is_decoder,
    DenseTensor *out);

template void CascadeAppendAttentionForFuseMtKernel<phi::dtype::bfloat16,
                                                    phi::GPUContext,
                                                    phi::dtype::bfloat16>(
    const phi::GPUContext &dev_ctx,
    cudaStream_t &stream,
    const DenseTensor &q,  // [token_num, num_heads, head_dim]
    const DenseTensor
        &cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    const DenseTensor
        &cache_v,  // [max_block_num, num_heads, head_dim, block_size]
    const DenseTensor *attn_mask,
    const DenseTensor &seq_lens_q,
    const DenseTensor &seq_lens_kv,
    const DenseTensor &seq_lens_encoder,
    const DenseTensor &padding_offsets,
    const DenseTensor &cum_offsets,
    const DenseTensor &block_table,
    const DenseTensor &batch_ids,
    const DenseTensor &tile_ids,
    const DenseTensor *seq_mapping,
    const DenseTensor *rope_emb,
    int num_blocks,
    int block_shape_q,
    int max_seq_len,
    int max_dec_len,
    int num_heads,
    int kv_num_heads,
    int head_dim,
    int layer_id,
    bool causal,
    bool is_decoder,
    DenseTensor *out);

}  // namespace fusion
}  // namespace phi
