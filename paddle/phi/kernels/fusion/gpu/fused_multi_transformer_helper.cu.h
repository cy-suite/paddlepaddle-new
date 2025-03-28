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
#include "paddle/phi/kernels/funcs/cublaslt.h"
#include "paddle/phi/kernels/funcs/load_store_util.h"
#include "paddle/phi/kernels/funcs/quant_dequant.h"
#include "paddle/phi/kernels/funcs/weight_only_gemv.h"
#include "paddle/phi/kernels/fusion/gpu/attention_layer.norm.h"
#include "paddle/phi/kernels/fusion/gpu/attn_gemm.h"
#include "paddle/phi/kernels/fusion/gpu/attn_gemm_int8.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_helper.h"
#include "paddle/phi/kernels/fusion/gpu/fused_multi_transformer_op.cu.h"
#include "paddle/phi/kernels/rms_norm_kernel.h"
#if defined(PADDLE_WITH_CUTLASS)
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"
#endif

/*
Note(Zhengzekang):
This header file is to store General Function Helper which has been used in
FusedMultiTransformer.
*/

namespace phi {
namespace fusion {

template <typename T>
class BiasActHelper {
 public:
  BiasActHelper(const phi::GPUContext &dev_ctx,
                const std::string &act_method,
                int rows,
                int cols)
      : dev_ctx_(dev_ctx), act_method_(act_method), rows_(rows), cols_(cols) {}

  // dst = Activation(x + bias(optional))
  void Compute(const phi::DenseTensor *x,
               const phi::DenseTensor *bias,
               phi::DenseTensor *output) {
    const T *bias_data = (bias == nullptr) ? nullptr : bias->data<T>();
    phi::funcs::Load<T> load_func(x->data<T>());
    phi::funcs::Store<T> store_func(output->data<T>());
    ComputeImpl(bias_data, load_func, store_func);
  }

 private:
  template <typename LoadFunc, typename StoreFunc, typename LoadT = T>
  void ComputeImpl(const T *bias_data,
                   LoadFunc load_func,
                   StoreFunc store_func) {
    if (act_method_ == "geglu") {
      // Note(Zhengzekang): For GLU structure, we need divide the cols by 2.
      VLOG(5) << "doing geglu";
      LaunchActFFNGlu<T,
                      phi::fusion::LayerNormParamTypeGeluFunctor<T>,
                      LoadFunc,
                      StoreFunc,
                      LoadT>(
          dev_ctx_, bias_data, rows_, cols_ / 2, load_func, store_func);
    } else if (act_method_ == "swiglu") {
      VLOG(5) << "doing swiglu";
      LaunchActFFNGlu<T, CudaSwishFunctor<T>, LoadFunc, StoreFunc, LoadT>(
          dev_ctx_, bias_data, rows_, cols_ / 2, load_func, store_func);
    } else if (act_method_ == "gelu") {
      if (FLAGS_use_fast_math) {
        VLOG(5) << "doing Fast GELU";
        LaunchBiasAct<T,
                      phi::fusion::FastGeluFunctor<T>,
                      LoadFunc,
                      StoreFunc,
                      LoadT>(
            dev_ctx_, bias_data, rows_, cols_, load_func, store_func);
      } else {
        VLOG(5) << "doing GELU";
        LaunchBiasAct<T,
                      phi::fusion::LayerNormParamTypeGeluFunctor<T>,
                      LoadFunc,
                      StoreFunc,
                      LoadT>(
            dev_ctx_, bias_data, rows_, cols_, load_func, store_func);
      }
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Currently Only Support GeGLU, SwiGLU, GeLU"));
    }
  }
  const phi::GPUContext &dev_ctx_;
  std::string act_method_;
  int rows_;
  int cols_;
};

template <typename T,
          typename nvT = typename phi::PDDataTypeTraits<T>::DataType>
class GEMMHelper {
 public:
  GEMMHelper(const phi::GPUContext &dev_ctx,
             int token_num,
             int dim_ffn,
             int dim_embed,
             const std::string gemm_method,
             phi::CutlassFpAIntBGemmRunner<nvT, uint8_t>
                 *int8_mixed_gemm_runner = nullptr,
             bool transpose_weight = false,
             int group_size = -1)
      : dev_ctx_(dev_ctx),
        token_num_(token_num),
        dim_ffn_(dim_ffn),
        dim_embed_(dim_embed),
        gemm_method_(gemm_method),
        int8_mixed_gemm_runner_(int8_mixed_gemm_runner),
        transpose_weight_(transpose_weight),
        group_size_(group_size) {}

  // dst = act(fc(src[0]) + bias) * src[1]
  void Compute(const phi::DenseTensor *input,
               const phi::DenseTensor *weight,
               const phi::DenseTensor *scale,
               const phi::DenseTensor *bias,
               phi::DenseTensor *workspace,
               phi::DenseTensor *output) {
    VLOG(5) << "GEMMHelper,"
            << " token_num_:" << token_num_ << " dim_ffn_:" << dim_ffn_
            << " dim_embed_:" << dim_embed_;
    bool compute_bias = true;
    if (bias == nullptr) {
      compute_bias = false;
    }
    using NvType = typename phi::PDDataTypeTraits<T>::DataType;

    if (gemm_method_ == "None") {
      auto ffn_linear_compute = phi::fusion::AttnMatMul<T>(dev_ctx_,
                                                           false,
                                                           transpose_weight_,
                                                           token_num_,
                                                           dim_ffn_,
                                                           dim_embed_,
                                                           compute_bias);
      ffn_linear_compute.ComputeForward(weight, input, bias, output, output);
    } else if (gemm_method_ == "weight_only") {
      if (bias) {
        VLOG(5) << "do weight_only int8 gemm with bias";
        int8_mixed_gemm_runner_->gemm_bias_act(
            reinterpret_cast<const NvType *>(input->data<T>()),
            reinterpret_cast<const uint8_t *>(weight->data<int8_t>()),
            reinterpret_cast<const NvType *>(scale->data<T>()),
            reinterpret_cast<const NvType *>(bias->data<T>()),
            reinterpret_cast<NvType *>(output->data<T>()),
            token_num_,
            dim_ffn_,
            dim_embed_,
            group_size_,
            "none",
            reinterpret_cast<char *>(workspace->data<uint8_t>()),
            workspace->numel(),
            dev_ctx_.stream());
      } else {
        VLOG(5) << "do weight_only int8 gemm without bias";
        int8_mixed_gemm_runner_->gemm(
            reinterpret_cast<const NvType *>(input->data<T>()),
            reinterpret_cast<const uint8_t *>(weight->data<int8_t>()),
            reinterpret_cast<const NvType *>(scale->data<T>()),
            reinterpret_cast<NvType *>(output->data<T>()),
            token_num_,
            dim_ffn_,
            dim_embed_,
            group_size_,
            reinterpret_cast<char *>(workspace->data<uint8_t>()),
            workspace->numel(),
            dev_ctx_.stream());
      }
      // VLOG(5) << "weight_only int8 gemm input:" << *input;
      // VLOG(5) << "weight_only int8 gemm output:" << *output;
    } else if (gemm_method_ == "weight_only_gemv") {
      const T *bias_data = bias ? bias->data<T>() : nullptr;
      phi::WeightOnlyGemvWrapper<T, phi::GPUContext>(
          dev_ctx_,
          input->data<T>(),
          weight->data<int8_t>(),
          bias_data,
          scale->data<T>(),
          token_num_,
          dim_embed_,
          dim_ffn_,
          group_size_,
          "int8",
          group_size_ > 1 ? "group_wise" : "per_channel",
          "None",
          output->data<T>());
      // VLOG(5) << "weight_only int8 gemv input:" << *input;
      // VLOG(5) << "weight_only int8 gemv output:" << *output;
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Currently GemmHelper only support `None`. "));
    }
  }

 private:
  const phi::GPUContext &dev_ctx_;
  int token_num_;
  int dim_ffn_;
  int dim_embed_;
  std::string gemm_method_;
  phi::CutlassFpAIntBGemmRunner<nvT, uint8_t> *int8_mixed_gemm_runner_;
  bool transpose_weight_;  // Just For AttnMatmul.
  int group_size_;         // Just For AttnMatmul.
};

template <typename T>
class NormHelper {
 public:
  NormHelper(const phi::GPUContext &dev_ctx,
             const std::string &norm_type,
             const int rows,
             const int cols,
             const float epsilon,
             const float residual_alpha)
      : dev_ctx_(dev_ctx),
        norm_type_(norm_type),
        rows_(rows),
        cols_(cols),
        epsilon_(epsilon),
        residual_alpha_(
            residual_alpha),  // TODO(zhengzekang): currently only available for
                              // Layernorm. Need support rmsnorm.
        layernorm_helper_(dev_ctx_, epsilon_, rows_, cols_) {
    // VLOG(0) << "NormHelper residual_alpha:" << residual_alpha_;
    phi::fusion::DropoutParam dropout_param(
        true, 0, true, true, 0.0, nullptr, 0);
    residual_bias_add_layernorm_helper_ =
        phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>(
            dev_ctx, rows_, cols_, dropout_param, epsilon_);
  }

  /*
  Note(Zhengzekang):
  Since input `X` and `Residual` in FusedMT will be swaped by preallocated
  buffer, I have no choice but to pass the data pointer instead of
  phi::DenseTensor.
  */

  // dst = Norm(x + residual + bias(optional))
  void NormResidualBias(const T *x_data,
                        const T *residual_data,
                        const phi::DenseTensor *bias,
                        const phi::DenseTensor *norm_weight,
                        const phi::DenseTensor *norm_bias,
                        phi::DenseTensor *mean,
                        phi::DenseTensor *var,
                        phi::DenseTensor *bias_residual_out,
                        phi::DenseTensor *output) {
    using U = phi::fusion::LayerNormParamType<T>;
    const T *bias_data = bias ? bias->data<T>() : nullptr;
    U *mean_data = mean ? mean->data<U>() : nullptr;
    U *var_data = var ? var->data<U>() : nullptr;
    T *bias_residual_out_data = bias_residual_out->data<T>();
    T *output_data = output->data<T>();

    if (norm_type_ == "layernorm") {
      // For layernorm, it use FP32 type weight and bias.
      const U *norm_weight_data =
          norm_weight ? norm_weight->data<U>() : nullptr;
      const U *norm_bias_data = norm_bias ? norm_bias->data<U>() : nullptr;
      residual_bias_add_layernorm_helper_.LayernormResidualDropoutBias(
          dev_ctx_,
          x_data,
          residual_data,
          bias_data,
          norm_weight_data,
          norm_bias_data,
          bias_residual_out_data,
          nullptr,
          output_data,
          mean_data,
          var_data);
    } else if (norm_type_ == "rmsnorm") {
      // For rmsnorm, it use Input's type weight and bias.
      const T *norm_weight_data =
          norm_weight ? norm_weight->data<T>() : nullptr;
      const T *norm_bias_data = norm_bias ? norm_bias->data<T>() : nullptr;
      phi::ResidualAddRmsNormWrapper<T, phi::GPUContext>(dev_ctx_,
                                                         x_data,
                                                         residual_data,
                                                         bias_data,
                                                         norm_weight_data,
                                                         norm_bias_data,
                                                         epsilon_,
                                                         rows_,
                                                         cols_,
                                                         bias_residual_out_data,
                                                         output_data);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Currently NormHelper only support `layernorm` and `rmsnorm`. "));
    }
  }

  // dst = Norm(x)
  void Norm(const T *x_data,
            const phi::DenseTensor *norm_weight,
            const phi::DenseTensor *norm_bias,
            phi::DenseTensor *mean,
            phi::DenseTensor *var,
            phi::DenseTensor *output) {
    using U = phi::fusion::LayerNormParamType<T>;
    U *mean_data = mean ? mean->data<U>() : nullptr;
    U *var_data = var ? var->data<U>() : nullptr;
    T *output_data = output->data<T>();

    if (norm_type_ == "layernorm") {
      // For layernorm, it use FP32 type weight and bias.
      const U *norm_weight_data =
          norm_weight ? norm_weight->data<U>() : nullptr;
      const U *norm_bias_data = norm_bias ? norm_bias->data<U>() : nullptr;
      layernorm_helper_.ComputeForward(x_data,
                                       norm_weight_data,
                                       norm_bias_data,
                                       output_data,
                                       mean_data,
                                       var_data);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Currently NormHelper only support `layernorm`. "));
    }
  }

 private:
  const phi::GPUContext &dev_ctx_;
  std::string norm_type_;
  int rows_;
  int cols_;
  float epsilon_;
  float residual_alpha_;
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>
      residual_bias_add_layernorm_helper_;
  AttnLayerNorm<T> layernorm_helper_;
};

template <typename T,
          typename nvT = typename phi::PDDataTypeTraits<T>::DataType>
class FFNHelper {
 public:
  FFNHelper(const phi::GPUContext &dev_ctx,
            const std::string &act_method,
            int token_num,
            int dim_ffn,
            int dim_embed,
            const std::string gemm_method,
            phi::CutlassFpAIntBGemmRunner<nvT, uint8_t>
                *int8_mixed_gemm_runner = nullptr,
            int group_size = -1)
      : dev_ctx_(dev_ctx),
        act_method_(act_method),
        token_num_(token_num),
        dim_ffn_(dim_ffn),
        dim_embed_(dim_embed),
        gemm_method_(gemm_method),
        int8_mixed_gemm_runner_(int8_mixed_gemm_runner),
        group_size_(group_size) {}

  // dst = act(fc(src[0]) + bias) * src[1]
  void Compute(const phi::DenseTensor *input,
               const phi::DenseTensor *weight,
               const phi::DenseTensor *scale,
               const phi::DenseTensor *bias,
               phi::DenseTensor *workspace,
               phi::DenseTensor *bias_out,
               phi::DenseTensor *output) {
    /*
    input's shape [token_num, dim_embed]
    weight's shape [dim_embed, dim_ffn]
    bias' shape [dim_ffn]
    output's shape [token_num, dim_ffn].
    */
    GEMMHelper<T, nvT> gemm_helper(dev_ctx_,
                                   token_num_,
                                   dim_ffn_,
                                   dim_embed_,
                                   gemm_method_,
                                   int8_mixed_gemm_runner_,
                                   false,
                                   group_size_);
    BiasActHelper<T> bias_act_helper(
        dev_ctx_, act_method_, token_num_, dim_ffn_);

    gemm_helper.Compute(input, weight, scale, bias, workspace, bias_out);
    bias_act_helper.Compute(bias_out, nullptr, output);
  }

 private:
  const phi::GPUContext &dev_ctx_;
  std::string act_method_;
  int token_num_;
  int dim_ffn_;
  int dim_embed_;
  std::string gemm_method_;
  phi::CutlassFpAIntBGemmRunner<nvT, uint8_t> *int8_mixed_gemm_runner_;
  int group_size_;
};

}  // namespace fusion
}  // namespace phi
