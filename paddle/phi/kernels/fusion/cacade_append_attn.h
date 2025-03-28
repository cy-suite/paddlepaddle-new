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

#include "gpu/append_attention/mem_util.cuh"
#include "gpu/append_attention/utils.cuh"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

COMMON_DECLARE_int64(cascade_attention_max_partition_size);
COMMON_DECLARE_int64(cascade_encoder_attention_max_partition_size);

namespace phi {
namespace fusion {

template <typename T, typename Context, typename OutT>
void CascadeAppendAttentionForFuseMtKernel(
    const Context &dev_ctx,
    cudaStream_t &stream,  // NOLINT
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
