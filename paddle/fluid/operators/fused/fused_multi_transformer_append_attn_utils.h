/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "cub/cub.cuh"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/fusion/cacade_append_attn.h"

namespace paddle {
namespace operators {

__forceinline__ __host__ __device__ int div_up(int a, int b) {
  return (a + b - 1) / b;
}

inline cudaError_t GetNumBlocks(int64_t n, int *num_blocks) {
  constexpr int kBlockSize = 128;
  constexpr int kNumWaves = 16;

  const int device_id = phi::backends::gpu::GetCurrentDeviceId();
  const int sm_count = phi::backends::gpu::GetGPUMultiProcessors(device_id);
  const int max_thread_per_multiprocessor =
      phi::backends::gpu::GetGPUMultiProcessors(device_id);

  *num_blocks =
      std::max<int>(1,
                    std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                      sm_count * max_thread_per_multiprocessor /
                                          kBlockSize * kNumWaves));
  return cudaSuccess;
}

template <class Func>
inline cudaError_t GetNumBlocks(Func func,
                                int64_t block_size,
                                size_t dynamic_smem_size,
                                int64_t max_blocks,
                                int64_t waves,
                                int *num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int sm_count;
  {
    cudaError_t err =
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int max_active_blocks;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, func, block_size, dynamic_smem_size);
  }
  *num_blocks = std::max<int>(
      1, std::min<int64_t>(max_blocks, sm_count * max_active_blocks * waves));
  return cudaSuccess;
}

template <typename T, int VecSize = 1>
__global__ void VariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const T *qkv_biases,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * last_dim;
  const int offset = 3 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t bias_idx = qkv_id * hidden_size + hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * 3 * hidden_size + bias_idx;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left =
          static_cast<float>(src_vec[2 * i] + bias_vec[2 * i]);
      const float input_right =
          static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1]);
      // const float cos_tmp = cos_emb_vec[i];
      // const float sin_tmp = sin_emb_vec[i];
      // src_vec[2 * i] = static_cast<T>(input_left * cos_tmp - input_right *
      // sin_tmp); src_vec[2 * i + 1] = static_cast<T>(input_right * cos_tmp +
      // input_left * sin_tmp);

      if (qkv_id < 2) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        src_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        src_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        src_vec[2 * i] = static_cast<T>(input_left);
        src_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

// non-quant + prefix cache
template <typename T>
void rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,              // [token_num, 3, num_head, dim_head]
    const T *qkv_input,  // qkv
    const T *qkv_bias,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head) {
  const int elem_nums = token_num * 3 * head_num * dim_head;  // just q and k
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
  VariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                      cos_emb,
                                                      sin_emb,
                                                      padding_offsets,
                                                      seq_lens,
                                                      seq_lens_decoder,
                                                      qkv_bias,
                                                      qkv,
                                                      elem_nums,
                                                      head_num,
                                                      seq_len,
                                                      dim_head);
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const T *qkv_biases,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim,
    const int gqa_group_size) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  // const int hidden_size = num_head * last_dim;
  const int offset = (num_head + 2 * gqa_group_size) * last_dim;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / last_dim;
    const int h_bias = bias % last_dim;

    int ori_seq_id;
    ori_seq_id = ori_token_idx % seq_len + seq_lens_decoder[ori_bi];

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t bias_idx = hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * offset + bias_idx;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left =
          static_cast<float>(src_vec[2 * i] + bias_vec[2 * i]);
      const float input_right =
          static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1]);
      // const float cos_tmp = cos_emb_vec[i];
      // const float sin_tmp = sin_emb_vec[i];
      // src_vec[2 * i] = static_cast<T>(input_left * cos_tmp - input_right *
      // sin_tmp); src_vec[2 * i + 1] = static_cast<T>(input_right * cos_tmp +
      // input_left * sin_tmp);

      if (hi < num_head + gqa_group_size) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        src_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        src_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        src_vec[2 * i] = static_cast<T>(input_left);
        src_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

// non-quant + prefix cache
template <typename T>
void gqa_rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,              // [token_num, 3, num_head, dim_head]
    const T *qkv_input,  // qkv
    const T *qkv_bias,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int *seq_lens_decoder,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    const int gqa_group_size) {
  const int elem_nums =
      token_num * (head_num + 2 * gqa_group_size) * dim_head;  // for all q k v
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
  GQAVariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                      cos_emb,
                                                      sin_emb,
                                                      padding_offsets,
                                                      seq_lens,
                                                      seq_lens_decoder,
                                                      qkv_bias,
                                                      qkv,
                                                      elem_nums,
                                                      head_num,
                                                      seq_len,
                                                      dim_head,
                                                      gqa_group_size);
}

template <typename T, int VecSize = 1, bool USE_SYSTEM>
__global__ void cache_kernel(
    const T *__restrict__ qkv,  // [num_tokens, num_heads + 2 * gqa_group_size,
                                // head_size]
    T *__restrict__ key_cache,  // [num_blocks, gqa_group_size, block_size,
                                // head_size]
    T *__restrict__ value_cache,  // [num_blocks, gqa_group_size, block_size,
                                  // head_size]
    const int *__restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int *__restrict__ padding_offsets,  // [num_tokens]
    const int *__restrict__ cum_offsets,
    const int *__restrict__ seq_lens,          // [bsz]
    const int *__restrict__ seq_lens_decoder,  // [bsz]
    const int *__restrict__ seq_mapping,
    const int *__restrict__ excess_blocks,  // [bsz, excess_num]
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int block_size,
    const uint32_t elem_cnt,
    const int gqa_group_size,
    const int token_num,
    const int excess_num) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  uint32_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t hidden_size = gqa_group_size * head_size;
  const uint32_t offset = 2 * hidden_size;
  for (uint32_t linear_index = global_thread_idx * VecSize,
                step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    uint32_t token_idx = linear_index / offset;
    const uint32_t bias = linear_index % offset;
    const uint32_t qkv_id = bias / hidden_size;  // skip q
    const uint32_t qkv_bias = bias % hidden_size;
    const uint32_t hi = qkv_bias / head_size;
    const uint32_t h_bias = qkv_bias % head_size;

    uint32_t block_idx, block_offset;

    if (token_idx < token_num) {
      const uint32_t ori_token_idx = token_idx + padding_offsets[token_idx];
      const uint32_t ori_bi = ori_token_idx / max_seq_len;
      const uint32_t last_offset = seq_lens[ori_bi] % block_size;
      if (seq_lens[ori_bi] == 0) continue;

      const int32_t *block_table_now = nullptr;
      if constexpr (USE_SYSTEM) {
        block_table_now =
            block_tables + seq_mapping[ori_bi] * max_blocks_per_seq;
      } else {
        block_table_now = block_tables + ori_bi * max_blocks_per_seq;
      }
      const uint32_t ori_seq_id =
          ori_token_idx % max_seq_len + seq_lens_decoder[ori_bi];
      if (ori_seq_id >= seq_lens[ori_bi] - last_offset) continue;

      block_idx = block_table_now[ori_seq_id / block_size];
      block_offset = ori_seq_id % block_size;
    } else {
      const uint32_t excess_token_id = token_idx - token_num;
      const uint32_t ori_bi = excess_token_id / (excess_num * block_size);
      const uint32_t last_offset = seq_lens[ori_bi] % block_size;
      if (seq_lens[ori_bi] == 0) continue;

      const uint32_t excess_id =
          (excess_token_id % (excess_num * block_size)) / block_size;
      const uint32_t excess_token_offset = excess_token_id % block_size;

      if (excess_token_offset < last_offset) {
        token_idx = ori_bi * max_seq_len - cum_offsets[ori_bi] +
                    seq_lens[ori_bi] - last_offset + excess_token_offset;
      } else {
        continue;
      }

      block_idx = excess_blocks[ori_bi * excess_num + excess_id];
      block_offset = excess_token_offset;
    }

    // if (hi == 0 && h_bias == 0 && qkv_id == 0) {
    //   printf("block_idx %d token_idx %d block_offset %d\n", block_idx,
    //   token_idx, block_offset);
    // }

    const uint32_t tgt_idx =
        block_idx * gqa_group_size * block_size * head_size +
        hi * block_size * head_size + block_offset * head_size + h_bias;
    // if (hi == 0 && h_bias == 0 && qkv_id == 0) {
    //   printf("tgt_idx %d\n", tgt_idx);
    // }
    const uint32_t ori_idx =
        token_idx * (num_heads + 2 * gqa_group_size) * head_size +
        num_heads * head_size + qkv_id * hidden_size + hi * head_size + h_bias;
    // if (hi == 0 && h_bias == 0 && qkv_id == 0) {
    //   printf("ori_idx %d\n", ori_idx);
    // }
    phi::Load<T, VecSize>(&qkv[ori_idx], &src_vec);
    if (qkv_id == 0) {
      phi::Store<T, VecSize>(src_vec, &key_cache[tgt_idx]);
    } else {
      phi::Store<T, VecSize>(src_vec, &value_cache[tgt_idx]);
    }
  }
}

template <typename T>
void CacheKernel(const phi::GPUContext &dev_ctx,
                 const phi::DenseTensor
                     &qkv,  // [token_num, 3, num_head, head_dim] ([token_num,
                            // num_head + 2 * gqa_group_size, head_dim] if GQA)
                 const phi::DenseTensor &block_tables,
                 const phi::DenseTensor &padding_offsets,
                 const phi::DenseTensor &cum_offsets,
                 const phi::DenseTensor &seq_lens,
                 const phi::DenseTensor &seq_lens_decoder,
                 const int max_seq_len,
                 phi::DenseTensor *key_cache_out,
                 phi::DenseTensor *value_cache_out,
                 const int num_heads,
                 const int head_size,
                 const int bsz,
                 const phi::DenseTensor *seq_mapping = nullptr,
                 int gqa_group_size = -1,
                 const phi::DenseTensor *excess_blocks = nullptr) {
  typedef phi::PDDataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;

  auto qkv_dims = qkv.dims();
  const int max_blocks_per_seq = block_tables.dims()[1];
  const int num_tokens = qkv_dims[0];
  if (gqa_group_size <= 0) {
    gqa_group_size = num_heads;
  }
  int excess_block_num = 0;
  if (excess_blocks) {
    excess_block_num = excess_blocks->dims()[1];
  }
  VLOG(1) << "excess_block_num " << excess_block_num;

  const int32_t block_size = key_cache_out->dims()[2];
  uint32_t elem_nums = (num_tokens + bsz * excess_block_num * block_size) * 2 *
                       gqa_group_size * head_size;
  // 额外每个bid 多分配excess_block_num * block_size 个
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);

  VLOG(1) << "cache kv not quant";
  VLOG(1) << "num_tokens " << num_tokens;
  VLOG(1) << "num_heads " << num_heads;
  VLOG(1) << "head_size " << head_size;
  VLOG(1) << "gqa_group_size " << gqa_group_size;
  VLOG(1) << "elem_nums " << elem_nums;

  VLOG(2) << "cum_offsets" << cum_offsets;

  // VLOG(1) << "print query";
  // phi::fusion::print_tensor<T>(qkv, __FILE__, __LINE__, 10);
  // VLOG(1) << "print key";
  // phi::fusion::print_tensor<T>(
  //     qkv, __FILE__, __LINE__, 10, num_heads * head_size);
  // VLOG(1) << "print value";
  // phi::fusion::print_tensor<T>(
  // qkv, __FILE__, __LINE__, 10, (num_heads + gqa_group_size) * head_size);
  if (seq_mapping) {
    cache_kernel<DataType_, PackSize, true>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
            reinterpret_cast<DataType_ *>(const_cast<T *>(qkv.data<T>())),
            reinterpret_cast<DataType_ *>(key_cache_out->data<T>()),
            reinterpret_cast<DataType_ *>(value_cache_out->data<T>()),
            block_tables.data<int>(),
            padding_offsets.data<int>(),
            cum_offsets.data<int>(),
            seq_lens.data<int>(),
            seq_lens_decoder.data<int>(),
            seq_mapping->data<int>(),
            excess_blocks->data<int>(),
            max_seq_len,
            max_blocks_per_seq,
            num_heads,
            head_size,
            block_size,
            elem_nums,
            gqa_group_size,
            num_tokens,
            excess_block_num);
  } else {
    cache_kernel<DataType_, PackSize, false>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
            reinterpret_cast<DataType_ *>(const_cast<T *>(qkv.data<T>())),
            reinterpret_cast<DataType_ *>(key_cache_out->data<T>()),
            reinterpret_cast<DataType_ *>(value_cache_out->data<T>()),
            block_tables.data<int>(),
            padding_offsets.data<int>(),
            cum_offsets.data<int>(),
            seq_lens.data<int>(),
            seq_lens_decoder.data<int>(),
            nullptr,
            excess_blocks->data<int>(),
            max_seq_len,
            max_blocks_per_seq,
            num_heads,
            head_size,
            block_size,
            elem_nums,
            gqa_group_size,
            num_tokens,
            excess_block_num);
  }
}

template <typename T, bool USE_SYSTEM = false, int VecSize = 1>
__global__ void append_decode_cache_T_rope_kernel(
    const T
        *__restrict__ qkv,  // [bsz, num_heads + 2 * gqa_group_size, head_size]
    T *__restrict__ key_cache,    // [num_blocks, gqa_group_size, block_size,
                                  // head_size // 2]
    T *__restrict__ value_cache,  // [num_blocks, gqa_group_size, block_size,
                                  // head_size // 2]
    T *__restrict__ qkv_out,
    const int *__restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int *__restrict__ padding_offsets,  // [num_tokens]
    const int *__restrict__ cum_offsets,
    const int *__restrict__ seq_lens,          // [bsz]
    const int *__restrict__ seq_lens_encoder,  // [bsz]
    const float *__restrict__ cos_emb,
    const float *__restrict__ sin_emb,
    const T
        *__restrict__ qkv_biases,  // [num_head + 2 * gqa_group_size, dim_head]
    const int *__restrict__ seq_mapping,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int block_size,
    const uint32_t elem_cnt,
    const int gqa_group_size,
    const int *__restrict__ seq_lens_this_time = nullptr) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  using LoadOutScaleT = phi::AlignedVector<float, VecSize>;
  using LoadKVT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  LoadKVT cache_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int64_t hidden_size = (num_heads + 2 * gqa_group_size) * head_size;
  // const int64_t offset = 2 * hidden_size;
  const int half_head_size = head_size / 2;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int ori_bi = linear_index / hidden_size;
    const int bias = linear_index % hidden_size;
    const int hi = bias / head_size;  // q + k + v
    const int h_bias = bias % head_size;
    const int start_token_idx = ori_bi * max_seq_len - cum_offsets[ori_bi];
    if (seq_lens_encoder[ori_bi] > 0 ||
        (seq_lens_this_time && seq_lens_this_time[ori_bi] <= 0))
      return;
    const int write_seq_id = seq_lens[ori_bi];
    if (write_seq_id == 0) continue;

    const int *block_table_now = nullptr;
    if constexpr (USE_SYSTEM) {
      block_table_now =
          block_tables + __ldg(&seq_mapping[ori_bi]) * max_blocks_per_seq;
    } else {
      block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    }
    const int block_idx = block_table_now[write_seq_id / block_size];
    const int block_offset = write_seq_id % block_size;
    const uint32_t ori_idx =
        start_token_idx * hidden_size + hi * head_size + h_bias;

    const int bias_idx = hi * head_size + h_bias;
    phi::Load<T, VecSize>(&qkv[ori_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    if (hi < num_heads + gqa_group_size) {
      // q k rope
      const uint32_t emb_idx = write_seq_id * half_head_size + h_bias / 2;
      phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      // add_bias + rope
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);
      input_left = input_left + static_cast<float>(bias_vec[2 * i]);
      input_right = input_right + static_cast<float>(bias_vec[2 * i + 1]);
      if (hi < num_heads + gqa_group_size) {
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        bias_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        bias_vec[2 * i] = static_cast<T>(input_left);
        bias_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    if (hi < num_heads) {
      // write q
      phi::Store<T, VecSize>(bias_vec, &qkv_out[ori_idx]);
    } else {
      // quant + write k/v
      const uint32_t kv_head_idx = (hi - num_heads) % gqa_group_size;
      const uint32_t tgt_idx =
          block_idx * gqa_group_size * block_size * head_size +
          kv_head_idx * block_size * head_size + block_offset * head_size +
          h_bias;
      if (hi < num_heads + gqa_group_size) {
        phi::Store<T, VecSize>(bias_vec, &key_cache[tgt_idx]);
      } else {
        phi::Store<T, VecSize>(bias_vec, &value_cache[tgt_idx]);
      }
    }
  }
}

template <typename T, typename QKV_TYPE = int>
void CacheAppendRoPEKernel(
    const phi::GPUContext &dev_ctx,
    cudaStream_t &stream,  // NOLINT
    const phi::DenseTensor
        &qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
               // gqa_group_size, head_dim] if GQA)
    const phi::DenseTensor &block_tables,
    const phi::DenseTensor &rotary_emb,
    const phi::DenseTensor &padding_offsets,
    const phi::DenseTensor &cum_offsets,
    const phi::DenseTensor &seq_lens,
    const phi::DenseTensor &seq_lens_encoder,
    const phi::DenseTensor &qkv_biases,
    phi::DenseTensor *qkv_out,
    phi::DenseTensor *key_cache_out,
    phi::DenseTensor *value_cache_out,
    const int max_seq_len,
    const int num_heads,
    const int head_size,
    const int layer_id = 0,
    const phi::DenseTensor *seq_mapping = nullptr,
    int gqa_group_size = -1,
    const phi::DenseTensor *seq_lens_this_time = nullptr) {
  typedef phi::PDDataTypeTraits<T> traits_;
  typedef phi::PDDataTypeTraits<QKV_TYPE> qkt_nv_type_;
  typedef typename traits_::DataType DataType_;
  typedef typename qkt_nv_type_::DataType QKV_Data_TYPE;
  const QKV_TYPE *qkv_ptr = qkv.data<QKV_TYPE>();
  auto qkv_dims = qkv.dims();
  const int max_blocks_per_seq = block_tables.dims()[1];
  const int bsz = cum_offsets.dims()[0];
  if (gqa_group_size <= 0) {
    gqa_group_size = num_heads;
  }
  VLOG(1) << "gqa_group_size: " << gqa_group_size;
  VLOG(2) << "qkv: " << qkv;
  VLOG(1) << "seq_lens" << seq_lens;
  VLOG(1) << "block_tables" << block_tables;
  VLOG(1) << "padding_offsets" << padding_offsets;
  VLOG(1) << "cum_offsets" << cum_offsets;
  VLOG(1) << "seq_lens_encoder" << seq_lens_encoder;
  // const int32_t block_size = key_cache_out->dims()[2];
  const int32_t block_size = 64;

  VLOG(1) << "num_heads";
  VLOG(1) << "head_size";
  VLOG(1) << "print_q ";
  phi::fusion::print_tensor<T>(qkv, __FILE__, __LINE__, 10);
  VLOG(1) << "print_key ";
  phi::fusion::print_tensor<T>(
      qkv, __FILE__, __LINE__, 10, num_heads * head_size);

  const float *cos_emb = rotary_emb.data<float>();
  const float *sin_emb = rotary_emb.data<float>() + max_seq_len * head_size / 2;

  const uint32_t elem_nums =
      bsz * (num_heads + 2 * gqa_group_size) * head_size;  // just k and v
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (seq_mapping) {
    VLOG(2) << "seq_mapping " << seq_mapping;
    append_decode_cache_T_rope_kernel<DataType_, true, PackSize>
        <<<grid_size, blocksize, 0, stream>>>(
            reinterpret_cast<const QKV_Data_TYPE *>(qkv_ptr),
            reinterpret_cast<DataType_ *>(key_cache_out->data<T>()),
            reinterpret_cast<DataType_ *>(value_cache_out->data<T>()),
            reinterpret_cast<DataType_ *>(qkv_out->data<T>()),
            block_tables.data<int>(),
            padding_offsets.data<int>(),
            cum_offsets.data<int>(),
            seq_lens.data<int>(),
            seq_lens_encoder.data<int>(),
            cos_emb,
            sin_emb,
            reinterpret_cast<DataType_ *>(
                const_cast<T *>(qkv_biases.data<T>())),
            seq_mapping ? seq_mapping->data<int>() : nullptr,
            max_seq_len,
            max_blocks_per_seq,
            num_heads,
            head_size,
            block_size,
            elem_nums,
            gqa_group_size,
            seq_lens_this_time ? seq_lens_this_time->data<int>() : nullptr);
  } else {
    VLOG(1) << "key cache 1";
    phi::fusion::print_tensor<T>(
        *key_cache_out, __FILE__, __LINE__, 10, 9 * head_size);
    append_decode_cache_T_rope_kernel<DataType_, false, PackSize>
        <<<grid_size, blocksize, 0, stream>>>(
            reinterpret_cast<const QKV_Data_TYPE *>(qkv_ptr),
            reinterpret_cast<DataType_ *>(key_cache_out->data<T>()),
            reinterpret_cast<DataType_ *>(value_cache_out->data<T>()),
            reinterpret_cast<DataType_ *>(qkv_out->data<T>()),
            block_tables.data<int>(),
            padding_offsets.data<int>(),
            cum_offsets.data<int>(),
            seq_lens.data<int>(),
            seq_lens_encoder.data<int>(),
            cos_emb,
            sin_emb,
            reinterpret_cast<DataType_ *>(
                const_cast<T *>(qkv_biases.data<T>())),
            seq_mapping ? seq_mapping->data<int>() : nullptr,
            max_seq_len,
            max_blocks_per_seq,
            num_heads,
            head_size,
            block_size,
            elem_nums,
            gqa_group_size,
            seq_lens_this_time ? seq_lens_this_time->data<int>() : nullptr);
    VLOG(1) << "key cache 2";
    phi::fusion::print_tensor<T>(
        *key_cache_out, __FILE__, __LINE__, 10, 9 * head_size);
  }
}

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return max(a, b);
  }
};

template <int THREADBLOCK_SIZE>
__global__ void GetMaxLenKernel(const int *seq_lens,
                                const int *seq_lens_this_time,
                                const int *seq_lens_encoder,
                                const int *seq_lens_this_time_merged,
                                const int *seq_lens_encoder_merged,
                                const int *seq_mapping,
                                const int *system_lens,
                                int *max_lens,
                                const int batch_size) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<int, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int max_len_this_time_this_thread = 0;
  int max_len_encoder_this_thread = 0;
  int max_len_decoder_this_thread = 0;
  int max_len_this_thread = 0;
  int max_just_dec_len_this_thread = 0;
  int max_just_dec_merged_len_this_time_this_thread = 0;
  int max_system_len_this_thread = 0;
  int max_dec_len_without_system_this_thread = 0;
  for (int i = tid; i < batch_size; i += blockDim.x) {
    const int seq_len_this_time = seq_lens_this_time[i];
    max_len_this_time_this_thread =
        max(seq_len_this_time, max_len_this_time_this_thread);
    max_len_encoder_this_thread =
        max(seq_lens_encoder[i], max_len_encoder_this_thread);
    max_len_decoder_this_thread = max(seq_lens[i], max_len_decoder_this_thread);
    if (seq_len_this_time <= 0) continue;
    const int max_just_dec_len_now = seq_lens_encoder[i] > 0 ? 0 : seq_lens[i];
    max_len_this_thread =
        max(seq_lens[i] + seq_len_this_time, max_len_this_thread);
    max_just_dec_len_this_thread =
        max(max_just_dec_len_this_thread, max_just_dec_len_now);
    if (system_lens) {
      const int real_bid = seq_mapping[i];
      const int system_len_now = system_lens[real_bid];
      max_system_len_this_thread =
          max(max_system_len_this_thread, system_len_now);
      max_dec_len_without_system_this_thread =
          max(max_dec_len_without_system_this_thread,
              max_just_dec_len_now - system_len_now);
    }
  }
  if (system_lens) {
    for (int i = tid; i < batch_size; i += blockDim.x) {
      const int ori_seq_len_this_time = seq_lens_this_time_merged[i];
      if (ori_seq_len_this_time <= 0) continue;
      const int max_just_dec_merged_len_this_time_now =
          seq_lens_encoder_merged[i] > 0 ? 0 : ori_seq_len_this_time;
      max_just_dec_merged_len_this_time_this_thread =
          max(max_just_dec_merged_len_this_time_this_thread,
              max_just_dec_merged_len_this_time_now);
    }
  }
  int total_max_len_this_time =
      BlockReduce(temp_storage)
          .Reduce(max_len_this_time_this_thread, MaxOp<int>());
  int total_max_len_encoder =
      BlockReduce(temp_storage)
          .Reduce(max_len_encoder_this_thread, MaxOp<int>());
  int total_max_len_decoder =
      BlockReduce(temp_storage)
          .Reduce(max_len_decoder_this_thread, MaxOp<int>());
  int total =
      BlockReduce(temp_storage).Reduce(max_len_this_thread, MaxOp<int>());
  int total_just_dec = BlockReduce(temp_storage)
                           .Reduce(max_just_dec_len_this_thread, MaxOp<int>());
  int total_just_dec_merged =
      BlockReduce(temp_storage)
          .Reduce(max_just_dec_merged_len_this_time_this_thread, MaxOp<int>());
  int total_system_len = BlockReduce(temp_storage)
                             .Reduce(max_system_len_this_thread, MaxOp<int>());
  int total_dec_len_without_system =
      BlockReduce(temp_storage)
          .Reduce(max_dec_len_without_system_this_thread, MaxOp<int>());
  if (tid == 0) {
    max_lens[0] = total_max_len_this_time;
    max_lens[1] = total_max_len_encoder;
    max_lens[2] = total_max_len_decoder;
    max_lens[3] = total;
    max_lens[4] = total_just_dec;
    max_lens[5] = total_just_dec_merged;
    max_lens[6] = total_system_len;
    max_lens[7] = total_dec_len_without_system;
  }
}

void GetMaxLen(const phi::GPUContext &dev_ctx,
               const phi::DenseTensor &seq_lens_tensor,
               const phi::DenseTensor &seq_lens_this_time,
               const phi::DenseTensor &seq_lens_encoder,
               const phi::DenseTensor *seq_lens_encoder_merged,
               const phi::DenseTensor *seq_lens_this_time_merged,
               const phi::DenseTensor *seq_mapping_tensor,
               const phi::DenseTensor *system_lens_tensor,
               phi::DenseTensor *max_len_tensor,
               const int batch_size,
               int *max_len_this_time,
               int *max_len_encoder,
               int *max_len_decoder,
               int *max_len,
               int *max_dec_len,
               int *max_just_dec_merged_len_this_time,
               int *max_system_len,
               int *max_dec_len_without_system) {
  constexpr int blockSize = 1024;
  std::vector<int> max_len_cpu(max_len_tensor->numel());
  GetMaxLenKernel<blockSize><<<1, blockSize, 0, dev_ctx.stream()>>>(
      seq_lens_tensor.data<int>(),
      seq_lens_this_time.data<int>(),
      seq_lens_encoder.data<int>(),
      seq_lens_this_time_merged ? seq_lens_this_time_merged->data<int>()
                                : nullptr,
      seq_lens_encoder_merged ? seq_lens_encoder_merged->data<int>() : nullptr,
      seq_mapping_tensor ? seq_mapping_tensor->data<int>() : nullptr,
      system_lens_tensor ? system_lens_tensor->data<int>() : nullptr,
      max_len_tensor->data<int>(),
      batch_size);
  VLOG(1) << "max_len_tensor: " << *max_len_tensor;
  memory::Copy(platform::CPUPlace(),
               max_len_cpu.data(),
               dev_ctx.GetPlace(),
               max_len_tensor->data<int>(),
               sizeof(int) * max_len_tensor->numel(),
               dev_ctx.stream());
  *max_len_this_time = max_len_cpu[0];
  *max_len_encoder = max_len_cpu[1];
  *max_len_decoder = max_len_cpu[2];
  *max_len = max_len_cpu[3];
  *max_dec_len = max_len_cpu[4];
  if (system_lens_tensor) {
    *max_just_dec_merged_len_this_time = max_len_cpu[5];
    *max_system_len = max_len_cpu[6];
    *max_dec_len_without_system = max_len_cpu[7];
  }
}

__global__ void split_kv_block(const int *__restrict__ seq_lens_decoder,
                               const int *__restrict__ seq_lens_this_time,
                               int *__restrict__ batch_ids,
                               int *__restrict__ tile_ids_per_batch,
                               int *__restrict__ num_blocks_x,
                               const int bsz,
                               const int pad_len,
                               const int num_rows_per_block) {
  if (threadIdx.x == 0) {
    int gridx = 0;
    int index = 0;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      const int start_len = seq_lens_decoder[bid];
      int seq_len = seq_lens_this_time[bid] + start_len % pad_len;
      if (seq_lens_this_time[bid] == 0) {
        seq_len = 0;
      }
      const int loop_times = div_up(seq_len, num_rows_per_block);
      for (uint32_t tile_id = 0; tile_id < loop_times; tile_id++) {
        batch_ids[index] = bid;
        tile_ids_per_batch[index++] = tile_id;
      }
      gridx += loop_times;
    }
    *num_blocks_x = gridx;
  }
}

__global__ void split_q_block(const int *__restrict__ seq_lens_q,
                              const int *__restrict__ seq_lens_encoder,
                              int *__restrict__ batch_ids,
                              int *__restrict__ tile_ids_per_batch,
                              int *__restrict__ num_blocks_x,
                              const int bsz,
                              const int num_rows_per_block,
                              const int GROUP_SIZE) {
  if (threadIdx.x == 0) {
    int gridx = 0;
    int index = 0;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      int seq_len = seq_lens_q[bid];
      if (seq_lens_encoder && seq_lens_encoder[bid] > 0) {
        seq_len = 0;
      }
      const int loop_times = div_up(seq_len * GROUP_SIZE, num_rows_per_block);
      for (uint32_t tile_id = 0; tile_id < loop_times; tile_id++) {
        batch_ids[index] = bid;
        tile_ids_per_batch[index++] = tile_id;
      }
      gridx += loop_times;
    }
    *num_blocks_x = gridx;
  }
}

void get_block_shape(const phi::GPUContext &dev_ctx,
                     const phi::DenseTensor &sequence_lengths,
                     const phi::DenseTensor *sequence_lengths_encoder,
                     phi::DenseTensor *batch_ids,
                     phi::DenseTensor *tile_ids_per_batch,
                     phi::DenseTensor *num_blocks_x,
                     const int GROUP_SIZE,
                     const int bsz,
                     const int num_qrow_per_block) {
  const int *seq_lens_encoder = nullptr;
  if (sequence_lengths_encoder) {
    seq_lens_encoder = sequence_lengths_encoder->data<int>();
  }
  split_q_block<<<1, 32, 0, dev_ctx.stream()>>>(sequence_lengths.data<int>(),
                                                seq_lens_encoder,
                                                batch_ids->data<int>(),
                                                tile_ids_per_batch->data<int>(),
                                                num_blocks_x->data<int>(),
                                                bsz,
                                                num_qrow_per_block,
                                                GROUP_SIZE);
}

template <typename T, int VecSize, bool USE_SYSTEM = false>
__global__ void RebuildPadding(T *output_data,
                               const T *input_data,
                               const int *cum_offsets,
                               const int *seq_len_decoder,
                               const int *seq_len_encoder,
                               const int *seq_mapping,
                               const int seq_len,
                               const int dim_embed,
                               const size_t elem_nums) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  const int64_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int64_t i = global_idx * VecSize; i < elem_nums;
       i += gridDim.x * blockDim.x * VecSize) {
    const int64_t bi = i / dim_embed;
    const int64_t bias_idx = i % dim_embed;
    int64_t seq_id = 0;
    if (seq_len_decoder[bi] == 0 && seq_len_encoder[bi] == 0) continue;
    // if encoder, get last token; just decoder, get first token.
    if (seq_len_encoder[bi] > 0) seq_id = seq_len_encoder[bi] - 1;
    const int64_t ori_token_idx =
        bi * seq_len - static_cast<int64_t>(cum_offsets[bi]) + seq_id;
    const int64_t src_offset = ori_token_idx * dim_embed + bias_idx;
    int64_t dst_offset;
    if constexpr (USE_SYSTEM) {
      dst_offset = seq_mapping[bi] * dim_embed + bias_idx;
    } else {
      dst_offset = bi * dim_embed + bias_idx;
    }
    phi::Load<T, VecSize>(&input_data[src_offset], &src_vec);
    phi::Store<T, VecSize>(src_vec, &output_data[dst_offset]);
  }
}

constexpr int VEC_16B = 16;

template <typename T>
void InvokeRebuildPadding(const phi::GPUContext &dev_ctx,
                          T *output_data,
                          const T *input_data,
                          const int *cum_offsets,
                          const int *seq_len_decoder,
                          const int *seq_len_encoder,
                          const int *seq_mapping,
                          const int seq_len,
                          const int token_num,
                          const int dim_embed,
                          const int64_t elem_nums) {
  // src: [token_num, dim_embed]
  // dst: [batch_size, 1, dim_embed]
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(dim_embed % PackSize,
                    0,
                    platform::errors::PreconditionNotMet(
                        "dim_embed=%d must be divisible by vec_size=%d",
                        dim_embed,
                        PackSize));
  int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (seq_mapping) {
    RebuildPadding<T, PackSize, true>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(output_data,
                                                        input_data,
                                                        cum_offsets,
                                                        seq_len_decoder,
                                                        seq_len_encoder,
                                                        seq_mapping,
                                                        seq_len,
                                                        dim_embed,
                                                        elem_nums);
  } else {
    RebuildPadding<T, PackSize, false>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(output_data,
                                                        input_data,
                                                        cum_offsets,
                                                        seq_len_decoder,
                                                        seq_len_encoder,
                                                        seq_mapping,
                                                        seq_len,
                                                        dim_embed,
                                                        elem_nums);
  }
}

}  // namespace operators
}  // namespace paddle
