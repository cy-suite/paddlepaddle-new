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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/fused/fused_attention_utils.h"
#include "paddle/fluid/operators/fused/fused_multi_transformer_append_attn_utils.h"
#include "paddle/fluid/platform/device/gpu/gpu_resource_pool.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "paddle/phi/kernels/fusion/cacade_append_attn.h"
#include "paddle/phi/kernels/fusion/gpu/fmha_ref.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_helper.h"
#include "paddle/phi/kernels/fusion/gpu/fused_multi_transformer_helper.cu.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

// #define _DEBUG_FUSED_MULTI_TRANSFORMER
// #define _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR

COMMON_DECLARE_int64(flag_block_shape_q);
COMMON_DECLARE_int64(flag_dec_block_shape_q);

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class FusedMultiTransformerDybatchOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = phi::fusion::LayerNormParamType<T>;
    auto &dev_ctx = ctx.cuda_device_context();

    // 0. input
    auto *input_x = ctx.Input<phi::DenseTensor>("X");
    auto *cum_offsets = ctx.Input<phi::DenseTensor>("CumOffsets");
    auto *padding_offset = ctx.Input<phi::DenseTensor>("PaddingOffset");
    int max_num_blocks_per_seq = -1;
    auto *block_tables = ctx.Input<phi::DenseTensor>("BlockTables");

    const auto input_x_dims = input_x->dims();
    const auto padding_offset_dims = padding_offset->dims();
    auto *rotary_tensor = ctx.Input<phi::DenseTensor>("RotaryPosEmb");
    const int rotary_emb_dims = ctx.Attr<int>("rotary_emb_dims");
    const auto block_size = ctx.Attr<int>("block_size");
    const float inv_compression_ratio =
        ctx.Attr<float>("inv_compression_ratio");

    int bsz = cum_offsets->dims()[0];
    int seq_len = ctx.Attr<int>("max_input_length");
    int token_num = input_x_dims[0];
    int dim_embed = input_x_dims[1];

    const std::string act_method = ctx.Attr<std::string>("act_method");
    bool use_glu = (act_method == "geglu" || act_method == "swiglu");
    const std::string norm_type = ctx.Attr<std::string>("norm_type");
    const bool use_neox_rotary_style = ctx.Attr<bool>("use_neox_rotary_style");

    auto *sequence_lengths = ctx.Input<phi::DenseTensor>(
        "SeqLengthsThisTime");  // seq_len [real_input_len, 1(decoder)]
    auto *sequence_lengths_encoder = ctx.Input<phi::DenseTensor>(
        "SeqLengthsEncoder");  // seq_len_encoder [real_input_len, 0(decoder)]
    auto *sequence_lengths_decoder = ctx.Input<phi::DenseTensor>(
        "SeqLengthsDecoder");  // seq_len_decoder [time_step, 0(encoder)]
    auto *cu_seqlens_q = ctx.Input<phi::DenseTensor>("SeqLengthsEncoderCum");
    auto *cu_seqlens_k = ctx.Input<phi::DenseTensor>("SeqLengthsDecoderCum");

    auto gqa_group_size = ctx.Attr<int>("gqa_group_size");

    // For system decoder accelerate
    auto *cum_offsets_merged = ctx.Input<phi::DenseTensor>("CumOffsetsMerged");
    auto *padding_offsets_merged =
        ctx.Input<phi::DenseTensor>("PaddingOffsetMerged");
    auto *seq_lens_this_time_merged =
        ctx.Input<phi::DenseTensor>("SeqLengthsThisTimeMerged");
    auto *seq_lens_encoder_merged =
        ctx.Input<phi::DenseTensor>("SeqLengthsEncoderMerged");
    auto *seq_lens_decoder_merged =
        ctx.Input<phi::DenseTensor>("SeqLengthsDecoderMerged");
    auto *seq_mapping = ctx.Input<phi::DenseTensor>("SeqMapping");
    auto *system_lens = ctx.Input<phi::DenseTensor>("SystemLens");
    auto *group_ids = ctx.Input<phi::DenseTensor>("GroupIds");
    auto *system_lens_merged = ctx.Input<phi::DenseTensor>("SystemLensMerged");
    bool decoder_use_cascade_inference = false;
    if (seq_mapping) {
      decoder_use_cascade_inference = true;
      VLOG(1) << "seq_lens_this_time_merged: " << *seq_lens_this_time_merged;
      VLOG(1) << "seq_lens_encoder_merged: " << *seq_lens_encoder_merged;
      VLOG(1) << "seq_lens_decoder_merged: " << *seq_lens_decoder_merged;
      VLOG(1) << "seq_mapping: " << *seq_mapping;
      VLOG(1) << "system_lens: " << *system_lens;
      VLOG(1) << "group_ids: " << *group_ids;
    }

    auto rope_theta = ctx.Attr<float>("rope_theta");

    phi::DenseTensor max_len_tensor;
    int max_len_this_time, max_dec_len_this_time, max_enc_len_this_time,
        max_enc_dec_len_this_time, max_just_dec_len_this_time,
        max_just_dec_merged_len_this_time, max_system_len,
        max_just_dec_len_without_system;

    max_len_tensor.Resize({{8}});
    auto *max_len_data = dev_ctx.Alloc<int>(
        &max_len_tensor, max_len_tensor.numel() * sizeof(int));

    GetMaxLen(dev_ctx,
              *sequence_lengths_decoder,
              *sequence_lengths,
              *sequence_lengths_encoder,
              seq_lens_encoder_merged,
              seq_lens_this_time_merged,
              seq_mapping,  // seq_mapping,
              system_lens,
              &max_len_tensor,
              bsz,
              &max_len_this_time,
              &max_enc_len_this_time,
              &max_dec_len_this_time,
              &max_enc_dec_len_this_time,
              &max_just_dec_len_this_time,
              &max_just_dec_merged_len_this_time,
              &max_system_len,
              &max_just_dec_len_without_system);

    VLOG(1) << "max_len_this_time: " << max_len_this_time;
    VLOG(1) << "max_dec_len_this_time: " << max_dec_len_this_time;
    VLOG(1) << "max_enc_len_this_time: " << max_enc_len_this_time;
    VLOG(1) << "max_enc_dec_len_this_time: " << max_enc_dec_len_this_time;
    VLOG(1) << "max_just_dec_len_this_time: " << max_just_dec_len_this_time;
    VLOG(1) << "max_just_dec_merged_len_this_time: "
            << max_just_dec_merged_len_this_time;
    VLOG(1) << "max_system_len: " << max_system_len;
    VLOG(1) << "max_just_dec_len_without_system: "
            << max_just_dec_len_without_system;

    auto *out = ctx.Output<phi::DenseTensor>("Out");
    auto *from_data = dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));
    // InitValue(dev_ctx, from_data, out->numel(), static_cast<T>(0.));
    cudaMemsetAsync(from_data, 0, out->numel() * sizeof(T), dev_ctx.stream());

    if (token_num == 0) return;

    auto *padding_offset_data = padding_offset->data<int>();

    // 1. layer norm
    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    const float epsilon = ctx.Attr<float>("epsilon");
    const float residual_alpha = ctx.Attr<float>("residual_alpha");
    auto ln_scales = ctx.MultiInput<phi::DenseTensor>("LnScale");
    auto ln_biases = ctx.MultiInput<phi::DenseTensor>("LnBias");
    phi::fusion::NormHelper<T> norm_helper(
        dev_ctx, norm_type, token_num, dim_embed, epsilon, residual_alpha);
    phi::DenseTensor ln_mean, ln_var;
    ln_mean.Resize({{token_num}});
    auto *ln_mean_data =
        dev_ctx.Alloc<U>(&ln_mean, ln_mean.numel() * sizeof(U));
    ln_var.Resize({{token_num}});
    auto *ln_var_data = dev_ctx.Alloc<U>(&ln_var, ln_var.numel() * sizeof(U));

    // 2. qkv
    // x: qkv's input [token_num, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed] if not GQA else
    // [num_head + 2 * gqa_group_size, dim_head, dim_embed]
    auto qkv_weights = ctx.MultiInput<phi::DenseTensor>("QKVW");
    auto qkv_biases = ctx.MultiInput<phi::DenseTensor>("QKVBias");
    const bool trans_qkvw = ctx.Attr<bool>("trans_qkvw");
    const auto qkv_w_dims = qkv_weights[0]->dims();

    int num_head, dim_head;

    if (gqa_group_size > 0) {
      num_head = trans_qkvw ? (qkv_w_dims[0] - 2 * gqa_group_size)
                            : (qkv_w_dims[1] - 2 * gqa_group_size);
      dim_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
    } else {
      num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
      dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
    }

    // For append attention
    phi::DenseTensor kv_batch_ids, kv_tile_ids_per_batch, kv_num_blocks_x;
    phi::DenseTensor batch_ids, tile_ids_per_batch, num_blocks_x;
    int kv_num_blocks_x_cpu, num_blocks_x_cpu;

    // For decoder system acceleration
    phi::DenseTensor decoder_batch_ids, decoder_tile_ids_per_batch,
        decoder_num_blocks_x;
    phi::DenseTensor system_decoder_batch_ids,
        system_decoder_tile_ids_per_batch, system_decoder_num_blocks_x;
    int decoder_num_blocks_x_cpu, system_decoder_num_blocks_x_cpu;
    uint32_t encoder_num_qrow_per_block, decoder_num_qrow_per_block;

    const int kv_head_num = gqa_group_size > 0 ? gqa_group_size : num_head;

    // For append attention
    {
      const uint32_t GROUP_SIZE = num_head / kv_head_num;
      if (max_enc_len_this_time > 0) {
        const uint32_t max_tile_size_per_bs =
            div_up(max_enc_dec_len_this_time, block_size);
        kv_batch_ids.Resize({bsz * max_tile_size_per_bs});
        kv_tile_ids_per_batch.Resize({bsz * max_tile_size_per_bs});
        kv_num_blocks_x.Resize({1});
        dev_ctx.template Alloc<int>(&kv_batch_ids);
        dev_ctx.template Alloc<int>(&kv_tile_ids_per_batch);
        dev_ctx.template Alloc<int>(&kv_num_blocks_x);
        split_kv_block<<<1, 32, 0, dev_ctx.stream()>>>(
            sequence_lengths_decoder->data<int>(),
            // sequence_lengths->data<int>(),
            sequence_lengths_encoder->data<int>(),
            kv_batch_ids.data<int>(),
            kv_tile_ids_per_batch.data<int>(),
            kv_num_blocks_x.data<int>(),
            bsz,
            block_size,
            block_size);
        paddle::memory::Copy(paddle::platform::CPUPlace(),
                             &kv_num_blocks_x_cpu,
                             dev_ctx.GetPlace(),
                             kv_num_blocks_x.data<int>(),
                             sizeof(int),
                             dev_ctx.stream());
        VLOG(1) << "kv_batch_ids: " << kv_batch_ids;
        VLOG(1) << "kv_tile_ids_per_batch: " << kv_tile_ids_per_batch;
        VLOG(1) << "kv_num_blocks_x: " << kv_num_blocks_x;

        encoder_num_qrow_per_block =
            FLAGS_flag_block_shape_q;  // get_block_shape_q_fuse_mt(
                                       // max_enc_len_this_time
                                       // * GROUP_SIZE);
        VLOG(1) << "max_enc_len_this_time: " << max_enc_len_this_time
                << ", GROUP_SIZE: " << GROUP_SIZE
                << ", encoder_num_qrow_per_block: "
                << encoder_num_qrow_per_block;
        VLOG(1) << "encoder_num_qrow_per_block: " << encoder_num_qrow_per_block;
        const uint32_t encoder_max_tile_size_per_bs_q =
            div_up((max_enc_dec_len_this_time * GROUP_SIZE),
                   encoder_num_qrow_per_block);
        batch_ids.Resize({bsz * encoder_max_tile_size_per_bs_q});
        tile_ids_per_batch.Resize({bsz * encoder_max_tile_size_per_bs_q});
        num_blocks_x.Resize({1});
        dev_ctx.template Alloc<int>(&batch_ids);
        dev_ctx.template Alloc<int>(&tile_ids_per_batch);
        dev_ctx.template Alloc<int>(&num_blocks_x);
        get_block_shape(dev_ctx,
                        *sequence_lengths_encoder,
                        nullptr,
                        &batch_ids,
                        &tile_ids_per_batch,
                        &num_blocks_x,
                        GROUP_SIZE,
                        bsz,
                        encoder_num_qrow_per_block);
        paddle::memory::Copy(paddle::platform::CPUPlace(),
                             &num_blocks_x_cpu,
                             dev_ctx.GetPlace(),
                             num_blocks_x.data<int>(),
                             sizeof(int),
                             dev_ctx.stream());
        VLOG(1) << "encoder_batch_ids: " << batch_ids;
        VLOG(1) << "encoder_tile_ids_per_batch: " << tile_ids_per_batch;
        VLOG(1) << "encoder_num_blocks_x: " << num_blocks_x;
      }
      if (max_just_dec_len_this_time > 0 && decoder_use_cascade_inference &&
          max_just_dec_merged_len_this_time > 0) {
        decoder_num_qrow_per_block =
            FLAGS_flag_dec_block_shape_q;  // get_block_shape_q_fuse_mt(1 *
                                           // GROUP_SIZE);
        const uint32_t decoder_max_tile_size_per_bs_q =
            div_up((1 * GROUP_SIZE), decoder_num_qrow_per_block);
        decoder_batch_ids.Resize({bsz * decoder_max_tile_size_per_bs_q});
        decoder_tile_ids_per_batch.Resize(
            {bsz * decoder_max_tile_size_per_bs_q});
        decoder_num_blocks_x.Resize({1});
        dev_ctx.template Alloc<int>(&decoder_batch_ids);
        dev_ctx.template Alloc<int>(&decoder_tile_ids_per_batch);
        dev_ctx.template Alloc<int>(&decoder_num_blocks_x);
        get_block_shape(dev_ctx,
                        *sequence_lengths,
                        sequence_lengths_encoder,
                        &decoder_batch_ids,
                        &decoder_tile_ids_per_batch,
                        &decoder_num_blocks_x,
                        GROUP_SIZE,
                        bsz,
                        decoder_num_qrow_per_block);
        paddle::memory::Copy(paddle::platform::CPUPlace(),
                             &decoder_num_blocks_x_cpu,
                             dev_ctx.GetPlace(),
                             decoder_num_blocks_x.data<int>(),
                             sizeof(int),
                             dev_ctx.stream());
        VLOG(1) << "decoder_batch_ids: " << decoder_batch_ids;
        VLOG(1) << "decoder_tile_ids_per_batch: " << decoder_tile_ids_per_batch;
        VLOG(1) << "decoder_num_blocks_x: " << decoder_num_blocks_x;

        const uint32_t decoder_system_max_tile_size_per_bs_q =
            div_up((max_just_dec_merged_len_this_time * GROUP_SIZE),
                   decoder_num_qrow_per_block);
        system_decoder_batch_ids.Resize(
            {bsz * decoder_system_max_tile_size_per_bs_q});
        system_decoder_tile_ids_per_batch.Resize(
            {bsz * decoder_system_max_tile_size_per_bs_q});
        system_decoder_num_blocks_x.Resize({1});
        dev_ctx.template Alloc<int>(&system_decoder_batch_ids);
        dev_ctx.template Alloc<int>(&system_decoder_tile_ids_per_batch);
        dev_ctx.template Alloc<int>(&system_decoder_num_blocks_x);
        get_block_shape(dev_ctx,
                        *seq_lens_this_time_merged,
                        seq_lens_encoder_merged,
                        &system_decoder_batch_ids,
                        &system_decoder_tile_ids_per_batch,
                        &system_decoder_num_blocks_x,
                        GROUP_SIZE,
                        bsz,
                        decoder_num_qrow_per_block);
        paddle::memory::Copy(paddle::platform::CPUPlace(),
                             &system_decoder_num_blocks_x_cpu,
                             dev_ctx.GetPlace(),
                             system_decoder_num_blocks_x.data<int>(),
                             sizeof(int),
                             dev_ctx.stream());
        VLOG(1) << "system_decoder_batch_ids: " << system_decoder_batch_ids;
        VLOG(1) << "system_decoder_tile_ids_per_batch: "
                << system_decoder_tile_ids_per_batch;
        VLOG(1) << "system_decoder_num_blocks_x: "
                << system_decoder_num_blocks_x;
      }
    }

    int hidden_size = num_head * dim_head;
    int output_size = gqa_group_size <= 0
                          ? 3 * hidden_size
                          : (num_head + 2 * gqa_group_size) * dim_head;
    int input_size = dim_embed;

    VLOG(1) << "hidden_size " << hidden_size;
    VLOG(1) << "output_size " << output_size;
    VLOG(1) << "input_size " << input_size;

    // Set a flag whether need to add Matmul / Layernorm bias.
    bool compute_bias = qkv_biases.size() > 0;
    bool compute_ln_bias = ln_biases.size() > 0;

    // (transA, transB, compute_bias) = (false, trans_qkvw, false)
    // Since we fused QKVBias into QKVBiasAddTransposeSplit kernel, here we
    // set compute_bias as false.

    auto qkv_compute = phi::fusion::GEMMHelper<T>(
        dev_ctx, token_num, output_size, input_size, "None", trans_qkvw);

    phi::DenseTensor qkv_out;
    if (gqa_group_size > 0) {
      qkv_out.Resize({{token_num, num_head + 2 * gqa_group_size, dim_head}});
    } else {
      qkv_out.Resize({{token_num, 3, num_head, dim_head}});
    }
    auto *qkv_out_data =
        dev_ctx.Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

    phi::DenseTensor qktv_out, fmha_out;

    fmha_out.Resize({{token_num, num_head, dim_head}});
    auto *fmha_out_data =
        dev_ctx.Alloc<T>(&fmha_out, fmha_out.numel() * sizeof(T));
    // InitValue(dev_ctx, fmha_out_data, fmha_out.numel(), static_cast<T>(0.));
    cudaMemsetAsync(
        fmha_out_data, 0, fmha_out.numel() * sizeof(T), dev_ctx.stream());

    auto cache_kv_outs = ctx.MultiOutput<phi::DenseTensor>("CacheKVOut");
    // 4. out_linear
    auto out_linear_weights = ctx.MultiInput<phi::DenseTensor>("OutLinearW");
    auto out_linear_biases = ctx.MultiInput<phi::DenseTensor>("OutLinearBias");
    int ring_id = ctx.Attr<int>("ring_id");
    // (transA, transB, compute_bias) = (false, false, false)

    auto out_linear_compute = phi::fusion::GEMMHelper<T>(
        dev_ctx, token_num, dim_embed, hidden_size, "None", false);

    // 5. ln(residual + bias)
    auto ffn_ln_scales = ctx.MultiInput<phi::DenseTensor>("FFNLnScale");
    auto ffn_ln_biases = ctx.MultiInput<phi::DenseTensor>("FFNLnBias");
    phi::DenseTensor bias_dropout_residual_out, dropout_mask_out;
    T *bias_dropout_residual_out_data = nullptr;
    if (pre_layer_norm) {
      bias_dropout_residual_out.Resize({{token_num, dim_embed}});
      bias_dropout_residual_out_data =
          dev_ctx.Alloc<T>(&bias_dropout_residual_out,
                           bias_dropout_residual_out.numel() * sizeof(T));
    }
    uint8_t *dropout_mask_out_data = nullptr;

    // 6. ffn matmul1
    auto ffn1_weights = ctx.MultiInput<phi::DenseTensor>("FFN1Weight");
    auto ffn1_biases = ctx.MultiInput<phi::DenseTensor>("FFN1Bias");
    auto ffn1_weight_dim = ffn1_weights[0]->dims();
    // if quant weight,
    // matmul weight is transposed
    int dim_ffn = ffn1_weight_dim[1];
    phi::fusion::FFNHelper<T> ffn1_helper(
        dev_ctx, act_method, token_num, dim_ffn, dim_embed, "None");

    phi::DenseTensor ffn1_out;
    ffn1_out.Resize({{token_num, dim_ffn}});
    auto *ffn1_out_data =
        dev_ctx.Alloc<T>(&ffn1_out, ffn1_out.numel() * sizeof(T));

    // Note(Zhengzekang): It is no need when using FP16 matmul.
    phi::DenseTensor mixgemm_workspace;
    char *mixgemm_workspace_data = nullptr;

    // 7. ffn act + bias
    phi::DenseTensor ffn1_dropout_out;
    int tmp_dim_ffn = dim_ffn;
    if (use_glu) tmp_dim_ffn /= 2;
    ffn1_dropout_out.Resize({{token_num, tmp_dim_ffn}});
    auto *ffn1_dropout_out_data = dev_ctx.Alloc<T>(
        &ffn1_dropout_out, ffn1_dropout_out.numel() * sizeof(T));

    // 8. ffn2 matmul
    auto ffn2_weights = ctx.MultiInput<phi::DenseTensor>("FFN2Weight");
    auto ffn2_biases = ctx.MultiInput<phi::DenseTensor>("FFN2Bias");
    auto ffn2_linear_compute = phi::fusion::GEMMHelper<T>(
        dev_ctx, token_num, dim_embed, tmp_dim_ffn, "None", false);

    // 9. ffn2 residual bias
    phi::fusion::DropoutParam ffn2_dropout_param(
        true, 0, true, true, 0.0, nullptr, 0);
    phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>
        ffn2_fused_dropout_helper(dev_ctx,
                                  token_num,
                                  dim_embed,
                                  ffn2_dropout_param,
                                  epsilon,
                                  residual_alpha);

    phi::DenseTensor tmp_out, tmp_out_rm_padding;
    tmp_out.Resize({{token_num, dim_embed}});
    tmp_out_rm_padding.Resize({{token_num, dim_embed}});
    auto *tmp_out_rm_padding_data = dev_ctx.Alloc<T>(
        &tmp_out_rm_padding, tmp_out_rm_padding.numel() * sizeof(T));
    auto *tmp_out_data =
        dev_ctx.Alloc<T>(&tmp_out, tmp_out.numel() * sizeof(T));

    const T *x_data;
    x_data = input_x->data<T>();
    phi::DenseTensor *buf0 = nullptr;
    phi::DenseTensor *buf1 = nullptr;
    phi::DenseTensor buf2;

    // step0:  x   --> buf1
    // step1: buf1 --> buf0
    // step2: buf0 --> buf1
    int layers = qkv_weights.size();
    // In the case of variable lengths, the padding needs to be rebuilt
    // eventually. So buf0 and buf1 do not need to be changed according to the
    // pre_layer_norm and the number of layers.
    buf0 = &tmp_out;
    buf1 = &tmp_out_rm_padding;

    for (int i = 0; i < layers; ++i) {
      // step1. layer_norm
      if (i == 0 && pre_layer_norm) {
        norm_helper.Norm(x_data,
                         ln_scales[i],
                         compute_ln_bias ? ln_biases[i] : nullptr, /*norm_bias*/
                         &ln_mean,                                 /*mean*/
                         &ln_var,                                  /*var*/
                         buf1);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(1) << "step1";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "ln1_out:" << *buf1;
      PrintMatrix(buf1->data<T>(),
                  buf1->numel(),
                  "ln1_out_" + std::to_string(buf1->place().GetDeviceId()));
#endif
#endif

      // step2. qkv
      // NOTE: In decoder stage, bias is fused in fmha. In encoder stage, bias
      // is fused in QKVBiasAddTransposeSplit
      const phi::DenseTensor *qkv_bias =
          qkv_biases.size() > 0 ? qkv_biases[i] : nullptr;
      if (!pre_layer_norm && i == 0) {
        const phi::DenseTensor *tmp_input_x = input_x;
        VLOG(5) << "Doing !pre_layer_norm&&i==0, qkv gemm, mnk:" << token_num
                << ", " << output_size << ", " << input_size;
        qkv_compute.Compute(tmp_input_x,
                            qkv_weights[i],
                            /*weight_scale*/ nullptr,
                            qkv_bias,
                            &mixgemm_workspace,
                            &qkv_out);
      } else {
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
        VLOG(2) << "qkv_weights:" << *(qkv_weights[i]);
#endif
#endif
        VLOG(1) << "Doing qkv gemm, mnk:" << token_num << ", " << output_size
                << ", " << input_size;
        qkv_compute.Compute(buf1,
                            qkv_weights[i],
                            /*weight_scale*/ nullptr,
                            /*qkv_bias*/ nullptr,
                            &mixgemm_workspace,
                            &qkv_out);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(1) << "step2";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "qkv_out:" << qkv_out;
      PrintMatrix(qkv_out.data<T>(),
                  qkv_out.numel(),
                  "qkv_out_" + std::to_string(qkv_out.place().GetDeviceId()));
#endif
#endif

      // step3. fmha

      phi::DenseTensor *key_cache = cache_kv_outs[2 * i];
      phi::DenseTensor *value_cache = cache_kv_outs[2 * i + 1];
      if (max_enc_len_this_time > 0) {  // generation context stage
        const int *sequence_lengths_data =
            sequence_lengths_encoder->data<int>();
        if (rotary_emb_dims != 0) {
          if (gqa_group_size <= 0) {
            rotary_qk_variable(dev_ctx,
                               qkv_out_data,
                               qkv_out_data,
                               qkv_bias->data<T>(),
                               rotary_tensor->data<float>(),
                               padding_offset_data,
                               sequence_lengths_data,
                               sequence_lengths_decoder->data<int>(),
                               token_num,
                               num_head,
                               seq_len,
                               rotary_tensor->dims()[2],
                               dim_head);
          } else {
            gqa_rotary_qk_variable(dev_ctx,
                                   qkv_out_data,
                                   qkv_out_data,
                                   qkv_bias->data<T>(),
                                   rotary_tensor->data<float>(),
                                   padding_offset_data,
                                   sequence_lengths_data,
                                   sequence_lengths_decoder->data<int>(),
                                   token_num,
                                   num_head,
                                   seq_len,
                                   rotary_tensor->dims()[2],
                                   dim_head,
                                   gqa_group_size);
          }
        }

        CacheKernel<T>(dev_ctx,
                       qkv_out,
                       *block_tables,
                       *padding_offset,
                       *sequence_lengths_encoder,
                       *sequence_lengths_decoder,
                       seq_len,
                       key_cache,
                       value_cache,
                       num_head,
                       dim_head,
                       seq_mapping,  // seq_mapping,
                       gqa_group_size);
        cudaStream_t exec_stream = dev_ctx.stream();
        phi::fusion::
            CascadeAppendAttentionForFuseMtKernel<T, phi::GPUContext, T>(
                dev_ctx,
                exec_stream,  // !!! exec_stream
                qkv_out,
                *key_cache,
                *value_cache,
                nullptr,  // attn_mask
                *sequence_lengths,
                *sequence_lengths_decoder,
                *sequence_lengths_encoder,
                *padding_offset,
                *cum_offsets,
                *block_tables,
                batch_ids,
                tile_ids_per_batch,
                seq_mapping,  // seq_mapping, // USE_SYSTEM
                nullptr,      // rope_emb
                num_blocks_x_cpu,
                encoder_num_qrow_per_block,
                seq_len,
                max_enc_dec_len_this_time,
                num_head,
                gqa_group_size > 0 ? gqa_group_size : num_head,
                dim_head,
                i,
                true,   // causal
                false,  // is_decoder
                &fmha_out);
      }

      if (max_just_dec_len_this_time > 0) {  // generation decoder stage
        cudaStream_t exec_stream = dev_ctx.stream();
        // write cache/kv and rope
        VLOG(1) << "goto CacheAppendRoPEKernel";

        CacheAppendRoPEKernel<T, T>(dev_ctx,
                                    exec_stream,
                                    qkv_out,
                                    *block_tables,
                                    *rotary_tensor,
                                    *padding_offset,
                                    *cum_offsets,
                                    *sequence_lengths_decoder,
                                    *sequence_lengths_encoder,
                                    *qkv_bias,
                                    &qkv_out,
                                    key_cache,
                                    value_cache,
                                    seq_len,
                                    num_head,
                                    dim_head,
                                    i,
                                    seq_mapping,
                                    gqa_group_size,
                                    nullptr);
        // 用append attention 来构建decoder的attn
        phi::fusion::
            CascadeAppendAttentionForFuseMtKernel<T, phi::GPUContext, T>(
                dev_ctx,
                exec_stream,  // !!! exec_stream
                qkv_out,
                *key_cache,
                *value_cache,
                nullptr,  // attn_mask
                *sequence_lengths,
                *sequence_lengths_decoder,
                *sequence_lengths_encoder,
                *padding_offset,
                *cum_offsets,
                *block_tables,
                decoder_batch_ids,
                decoder_tile_ids_per_batch,
                seq_mapping,  // seq_mapping, // USE_SYSTEM
                nullptr,      // rope_emb
                decoder_num_blocks_x_cpu,
                decoder_num_qrow_per_block,
                seq_len,
                max_just_dec_len_this_time,
                num_head,
                gqa_group_size > 0 ? gqa_group_size : num_head,
                dim_head,
                i,
                true,  // causal
                true,  // is_decoder
                &fmha_out);
      }

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(1) << "step3";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "fmha_out:" << fmha_out;
#endif
#endif
      VLOG(5) << "Doing out_linear gemm, mnk:" << token_num << ", " << dim_embed
              << ", " << hidden_size;
      if (pre_layer_norm) {
        out_linear_compute.Compute(&fmha_out,
                                   out_linear_weights[i],
                                   /*weight_scale*/ nullptr,
                                   /*bias*/ nullptr,
                                   &mixgemm_workspace,
                                   buf1);

        phi::fusion::AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
      } else {
        out_linear_compute.Compute(&fmha_out,
                                   out_linear_weights[i],
                                   /*weight_scale*/ nullptr,
                                   /*bias*/ nullptr,
                                   &mixgemm_workspace,
                                   buf0);
        phi::fusion::AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(1) << "step4";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "out_linear_out:" << *buf1;
#endif
#endif
      // step5. ln(residual + dropout(input + bias))
      if (pre_layer_norm) {
        norm_helper.NormResidualBias(
            buf1->data<T>(),
            x_data,
            compute_bias ? out_linear_biases[i] : nullptr, /*skip_bias*/
            ffn_ln_scales[i],
            compute_ln_bias ? ffn_ln_biases[i] : nullptr, /*norm_bias*/
            &ln_mean,                                     /*mean*/
            &ln_var,                                      /*var*/
            &bias_dropout_residual_out,
            buf1);
      } else {
        auto *residual_data = (i == 0 ? x_data : buf1->data<T>());
        norm_helper.NormResidualBias(
            buf0->data<T>(),
            residual_data,
            compute_bias ? out_linear_biases[i] : nullptr, /*skip_bias*/
            ln_scales[i],
            compute_ln_bias ? ln_biases[i] : nullptr, /*norm_bias*/
            &ln_mean,                                 /*mean*/
            &ln_var,                                  /*var*/
            buf0,
            buf1);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(1) << "step5";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "ffn1_input:" << *buf1;
#endif
#endif
      ffn1_helper.Compute(buf1,
                          ffn1_weights[i],
                          /*weight_scale*/ nullptr,
                          compute_bias ? ffn1_biases[i] : nullptr,
                          &mixgemm_workspace,
                          &ffn1_out,
                          &ffn1_dropout_out);
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(1) << "step6";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "ffn1_output:" << ffn1_out;
#endif
#endif
      // step7. ffn2 matmul
      if (pre_layer_norm) {
        ffn2_linear_compute.Compute(&ffn1_dropout_out,
                                    ffn2_weights[i],
                                    nullptr,
                                    /*bias*/ nullptr,
                                    &mixgemm_workspace,
                                    buf1);
      } else {
        ffn2_linear_compute.Compute(&ffn1_dropout_out,
                                    ffn2_weights[i],
                                    nullptr,
                                    /*bias*/ nullptr,
                                    &mixgemm_workspace,
                                    buf0);
      }
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(1) << "step8.0";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      if (pre_layer_norm) {
        VLOG(1) << "ffn2_out, buf1:" << *buf1;
      } else {
        VLOG(1) << "ffn2_out, buf0:" << *buf0;
      }
#endif
#endif
      if (pre_layer_norm) {
        VLOG(4) << "MPAllReduce 4: " << buf1->numel();
        phi::fusion::AllReduce<T>(*buf1, ring_id, buf1->numel(), dev_ctx);
      } else {
        VLOG(4) << "MPAllReduce 4: " << buf0->numel();
        phi::fusion::AllReduce<T>(*buf0, ring_id, buf0->numel(), dev_ctx);
      }

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(1) << "step8.1";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      if (pre_layer_norm) {
        VLOG(2) << "ffn2_out_rd:" << *buf1;
      } else {
        VLOG(2) << "ffn2_out_rd:" << *buf0;
      }
#endif
#endif

      compute_bias = qkv_biases.size() > 0;

      // step8. residual bias
      // TODO(wangxi): remove dropout mask in inference
      if (pre_layer_norm) {
        // TODO(wangxi): remove dropout mask in inference
        if (i < layers - 1) {
          norm_helper.NormResidualBias(
              buf1->data<T>(),
              bias_dropout_residual_out_data,
              compute_bias ? ffn2_biases[i] : nullptr, /*skip_bias*/
              ln_scales[i + 1],
              compute_ln_bias ? ln_biases[i + 1] : nullptr, /*norm_bias*/
              &ln_mean,                                     /*mean*/
              &ln_var,                                      /*var*/
              buf1,
              buf0);
        } else {
          ffn2_fused_dropout_helper.ResidualDropoutBias(
              dev_ctx,
              buf1->data<T>(),
              bias_dropout_residual_out_data,
              compute_bias ? ffn2_biases[i]->data<T>() : nullptr,
              buf1->data<T>(),
              dropout_mask_out_data);
        }
      } else {
        norm_helper.NormResidualBias(
            buf0->data<T>(),
            buf1->data<T>(),
            compute_bias ? ffn2_biases[i] : nullptr, /*skip_bias*/
            ffn_ln_scales[i],
            compute_ln_bias ? ffn_ln_biases[i] : nullptr, /*norm_bias*/
            &ln_mean,                                     /*mean*/
            &ln_var,                                      /*var*/
            buf0,
            buf1);
      }

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
      VLOG(1) << "step9";
#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER_PRINT_TENSOR
      VLOG(2) << "residual_out:" << *buf1;
#endif
#endif
      if (pre_layer_norm) {
        x_data = buf1->data<T>();
        std::swap(buf0, buf1);
      }
      compute_bias = qkv_biases.size() > 0;
    }
    if (pre_layer_norm) {
      InvokeRebuildPadding(dev_ctx,
                           from_data,
                           buf0->data<T>(),
                           cum_offsets->data<int>(),
                           sequence_lengths_decoder->data<int>(),
                           sequence_lengths_encoder->data<int>(),
                           seq_mapping ? seq_mapping->data<int>() : nullptr,
                           seq_len,
                           token_num,
                           dim_embed,
                           out->numel());
    } else {
      InvokeRebuildPadding(dev_ctx,
                           from_data,
                           buf1->data<T>(),
                           cum_offsets->data<int>(),
                           sequence_lengths_decoder->data<int>(),
                           sequence_lengths_encoder->data<int>(),
                           seq_mapping ? seq_mapping->data<int>() : nullptr,
                           seq_len,
                           token_num,
                           dim_embed,
                           out->numel());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
PD_REGISTER_STRUCT_KERNEL(fused_multi_transformer_dybatch,
                          GPU,
                          ALL_LAYOUT,
                          ops::FusedMultiTransformerDybatchOpKernel,
// float,
#if CUDA_VERSION >= 11000 && defined(CUDA_BFLOAT16_AVALIABLE)
                          plat::bfloat16,
#endif
                          plat::float16) {
}
