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
namespace phi {
#ifdef PADDLE_WITH_FLASHATTN_V3
template <typename T, typename Context>
void FlashAttnV3Kernel(
    const Context &ctx,
    const DenseTensor &q,
    const DenseTensor &k,
    const DenseTensor &v,
    const paddle::optional<DenseTensor>
        &k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is
                  // cu_seqlens_k_new
    const paddle::optional<DenseTensor>
        &v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is
                  // cu_seqlens_k_new
    const paddle::optional<DenseTensor>
        &q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is
                // cu_seqlens_q
    const paddle::optional<DenseTensor>
        &out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    const paddle::optional<DenseTensor> &cu_seqlens_q_,      // b+1
    const paddle::optional<DenseTensor> &cu_seqlens_k_,      // b+1
    const paddle::optional<DenseTensor> &cu_seqlens_k_new_,  // b+1
    const paddle::optional<DenseTensor>
        &seqused_q_,  // b. If given, only this many elements of each batch
                      // element's queries and outputs are used.
    const paddle::optional<DenseTensor>
        &seqused_k_,  // b. If given, only this many elements of each batch
                      // element's keys are used.
    const paddle::optional<DenseTensor>
        &page_table_,  // (b_k, max_num_pages_per_seq)
    const paddle::optional<DenseTensor>
        &kv_batch_idx_,  // b. indices to index into the KV cache
    const paddle::optional<DenseTensor> &leftpad_k_,  // b
    const paddle::optional<DenseTensor>
        &rotary_cos_,  // seqlen_ro x (rotary_dim / 2)
    const paddle::optional<DenseTensor>
        &rotary_sin_,  // seqlen_ro x (rotary_dim / 2)
    const paddle::optional<DenseTensor> &q_descale_,  // (b, h_k), not (b, h)
    const paddle::optional<DenseTensor> &k_descale_,  // (b, h_k)
    const paddle::optional<DenseTensor> &v_descale_,  // (b, h_k)
    const paddle::optional<DenseTensor> &scheduler_metadata_,  // (b + 1)
    const int
        max_seqlen_q_,  // if max_seqlen_q_ is set to 0, it indicates that it is
                        // uninitialized and should not be referenced
    // TODO(tridao): check if we need max_seqlen_k
    const int
        max_seqlen_k_,  // if max_seqlen_q_ is set to 0, it indicates that it is
                        // uninitialized and should not be referenced
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool is_rotary_interleaved,  // if true, rotary combines indices 0 &
                                       // 1, else indices 0 & rotary_dim / 2
    int num_splits,
    const bool manual_set_pack_gqa,
    const bool
        pack_gqa_,  // the pack_gqa_ will be used only if manual_set_pack_gqa is
                    // set to True; otherwise, the internal heuristic
                    // get_pack_gqa() from fa3 will decide whether to pack gqa
    const int sm_margin,
    DenseTensor *out,
    DenseTensor *softmax_lse,
    DenseTensor *out_accum,
    DenseTensor *softmax_lse_accum);
#endif
}  // namespace phi
