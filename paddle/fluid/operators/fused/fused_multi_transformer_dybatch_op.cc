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

#include <memory>
#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class FusedMultiTransformerDybatchOp : public framework::OperatorWithKernel {
 private:
  static constexpr const char *OpName = "FusedMultiTransformerDybatchOp";

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
#define CHECK_INPUT(name) \
  OP_INOUT_CHECK(ctx->HasInput(#name), "Input", #name, OpName)
#define CHECK_INPUTS(name) \
  OP_INOUT_CHECK(ctx->HasInputs(#name), "Input", #name, OpName)
#define CHECK_OUTPUT(name) \
  OP_INOUT_CHECK(ctx->HasOutput(#name), "Output", #name, OpName)
#define CHECK_OUTPUTS(name) \
  OP_INOUT_CHECK(ctx->HasOutputs(#name), "Output", #name, OpName)

    CHECK_INPUT(X);

    // attention
    CHECK_INPUTS(QKVW);
    CHECK_INPUTS(OutLinearW);

    if (ctx->HasInputs("CacheKV")) {
      CHECK_OUTPUTS(CacheKVOut);
    }

    // ffn
    CHECK_INPUTS(FFN1Weight);
    CHECK_INPUTS(FFN2Weight);

    CHECK_OUTPUT(Out);

    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto x_dim = ctx->GetInputDim("X");
    auto cum_offsets_dim = ctx->GetInputDim("CumOffsets");
    // auto y_dim = ctx->GetInputsDim("QKVW")[0];
    int max_input_length = ctx->Attrs().Get<int>("max_input_length");
    // bool trans_qkvw = ctx->Attrs().Get<bool>("trans_qkvw");
    PADDLE_ENFORCE_GT(max_input_length,
                      0,
                      platform::errors::InvalidArgument(
                          "max input length should be greater than 0 "
                          "but received input is [%d]",
                          max_input_length));

    PADDLE_ENFORCE_EQ(
        x_dim.size(),
        2,
        platform::errors::InvalidArgument("The dimensions of x must be 2"
                                          "(batch_size, seq_len, dim_embed),"
                                          "but received dimensions of"
                                          "Input is [%d]",
                                          x_dim.size()));
    // PADDLE_ENFORCE_EQ(y_dim.size(),
    //                   4,
    //                   platform::errors::InvalidArgument(
    //                       "The dimensions of qkv_weight must be 4"
    //                       "(3, num_head, dim_head, dim_embed),"
    //                       "but received dimensions of"
    //                       "Input is [%d]",
    //                       y_dim.size()));
    // PADDLE_ENFORCE_EQ(
    //     x_dim[1],
    //     trans_qkvw ? y_dim[3] : y_dim[0],
    //     platform::errors::InvalidArgument(
    //         "ShapeError: the dimension of x_dim[1] and y_dim[3](trans_qkvw is
    //         " "true) or y_dim[0](trans_qkvw is false)" "must be equal. But
    //         received: the shape " "of input x = [%s], and the shape of "
    //         "input qkv_weight = [%s]",
    //         x_dim,
    //         y_dim));

    ctx->SetOutputDim("Out", {cum_offsets_dim[0], x_dim[1]});
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    return phi::KernelKey(
        tensor.place(), tensor.layout(), expected_kernel_type.dtype());
  }
};

class FusedMultiTransformerDybatchOpOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("LnScale",
             "Scale is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDuplicable();
    AddInput("LnBias",
             "Bias is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDispensable()
        .AsDuplicable();
    AddInput("QKVW", "The qkv weight tensor.").AsDuplicable();
    AddInput("QKVBias", "The qkv bias tensor.").AsDispensable().AsDuplicable();
    AddInput("CacheKV", "(optional) The cached KV for generation inference.")
        .AsDispensable()
        .AsDuplicable();
    AddInput("PreCaches",
             "(optional) The prefix caches for generation inference.")
        .AsDispensable()
        .AsDuplicable();
    AddInput("RotaryPosEmb",
             "(optional) The RoPE embeddings for generation inference.")
        .AsDispensable();
    AddInput("CumOffsets", "The cum offsets");
    AddInput("PaddingOffset", "The padding offset");
    AddInput("SeqLengthsThisTime", "The real sequence length.");
    AddInput("SeqLengthsEncoder",
             "The sequence length tensor of inputs in encoder.");
    AddInput("SeqLengthsDecoder",
             "The sequence length tensor of inputs in decoder.");
    AddInput("CumOffsetsMerged", "The cum offsets").AsDispensable();
    AddInput("PaddingOffsetMerged", "The padding offset").AsDispensable();
    AddInput("SeqLengthsThisTimeMerged", "The real sequence length.")
        .AsDispensable();
    AddInput("SeqLengthsEncoderMerged",
             "The sequence length tensor of inputs in encoder.")
        .AsDispensable();
    AddInput("SeqLengthsDecoderMerged",
             "The sequence length tensor of inputs in decoder.")
        .AsDispensable();
    AddInput("SeqMapping", "The seq mapping.").AsDispensable();
    AddInput("SystemLens", "The system lens.").AsDispensable();
    AddInput("SystemLensMerged", "The merged system lens.").AsDispensable();
    AddInput("GroupIds", "The group ids.").AsDispensable();
    AddInput("SeqLengthsEncoderCum",
             "The cum sequence length tensor of inputs in encoder.")
        .AsDispensable();
    AddInput("SeqLengthsDecoderCum",
             "The cum sequence length tensor of inputs in decoder.")
        .AsDispensable();
    AddInput("SrcMask", "(optional) The attention mask tensor in encoder fmha.")
        .AsDispensable();
    AddInput("TgtMask", "(optional) The attention mask tensor in decoder fmha.")
        .AsDispensable();
    AddInput("BlockTables", "(optional) the block_table in page attn.")
        .AsDispensable();
    AddInput("ExcessBlocks", "(optional) the block_table in page attn.")
        .AsDispensable();
    AddInput("OutLinearW", "The out_linear weight tensor.").AsDuplicable();
    AddInput("OutLinearBias", "The out_linear bias tensor.")
        .AsDispensable()
        .AsDuplicable();
    AddInput("FFNLnScale", "The layer_norm scale of FusedFeedForward op")
        .AsDuplicable();
    AddInput("FFNLnBias", "The layer_norm bias of FusedFeedForward op")
        .AsDispensable()
        .AsDuplicable();
    AddInput("FFN1Weight", "The linear1 weight of FusedFeedForward op")
        .AsDuplicable();
    AddInput("FFN1Bias", "The linear1 bias of FusedFeedForward op")
        .AsDispensable()
        .AsDuplicable();
    AddInput("FFN2Weight", "The linear2 weight of FusedFeedForward op")
        .AsDuplicable();
    AddInput("FFN2Bias", "The linear2 bias input of FusedFeedForward op")
        .AsDispensable()
        .AsDuplicable();
    AddOutput("CacheKVOut", "The updated cache KV. Inplace with CacheKV")
        .AsDispensable()
        .AsDuplicable();
    AddOutput("Out", "Result after multi .");
    AddAttr<bool>("pre_layer_norm",
                  "if true, the attention op uses pre_layer_norm architecure, "
                  "else, uses post_layer_norm architecuture. "
                  "[default true].")
        .SetDefault(true);
    AddAttr<int>("rotary_emb_dims",
                 "the Attr(dims) for RotaryPosEmb's Computation  [default 0].")
        .SetDefault(0)
        .AddCustomChecker([](const int &rotary_emb_dims) {
          PADDLE_ENFORCE_EQ(
              rotary_emb_dims >= 0 && rotary_emb_dims <= 2,
              true,
              platform::errors::InvalidArgument(
                  "'rotary_emb_dims' in Op(Rotray) should be between"
                  "0 and 2, But received [%s].",
                  rotary_emb_dims));
        });
    AddAttr<int>("max_input_length", "the max input length of padding_inputs")
        .SetDefault(-1);
    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f,
                            true,
                            platform::errors::InvalidArgument(
                                "'epsilon' in Op(LayerNorm) should be between"
                                "0.0 and 0.001, But received [%s].",
                                epsilon));
        });

    AddAttr<float>("residual_alpha",
                   "Constant for residual_alpha [default 1.0].")
        .SetDefault(1.0f);

    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<std::string>("act_method", "act_method")
        .SetDefault("gelu")
        .AddCustomChecker([](const std::string &act_type) {
          PADDLE_ENFORCE_EQ(act_type == "gelu" || act_type == "geglu" ||
                                act_type == "swiglu" || act_type == "relu" ||
                                act_type == "none",
                            true,
                            platform::errors::InvalidArgument(
                                "Only support `gelu`, `geglu`, `swiglu`, "
                                "`relu`, `none` activation in "
                                "FusedMultiTransformer. "));
        });

    AddAttr<bool>(
        "trans_qkvw",
        "Whether the weights of qkv should be transposed. If true,"
        "the shape eights of qkv should be [3, num_head, dim_head, dim_embed]."
        "Otherwise the shape of weights of qkv should be"
        "[dim_embed, 3, num_head, dim_head]")
        .SetDefault(true);

    AddAttr<int>(
        "ring_id",
        "ring id for tensor model parallel. distributed training and inference")
        .SetDefault(-1);

    AddAttr<std::string>("norm_type", "norm_type")
        .SetDefault("layernorm")
        .AddCustomChecker([](const std::string &norm_type) {
          PADDLE_ENFORCE_EQ(
              norm_type == "layernorm" || norm_type == "rmsnorm",
              true,
              platform::errors::InvalidArgument(
                  "Only support `layernorm`, `rmsnorm` method for in"
                  "FusedMultiTransformerDybatch. "));
        });

    AddAttr<bool>("use_neox_rotary_style",
                  "Whether use neox rotary embedding. ")
        .SetDefault(false);

    AddAttr<int>("block_size", "block_size in page_attn [default 16].")
        .SetDefault(128);

    AddAttr<float>("inv_compression_ratio",
                   "inv_compression_ratio in RoPE pos id, when using 8.0 it is "
                   "PI-ROPE [default 1.0].")
        .SetDefault(1.0f);
    AddAttr<int>("gqa_group_size", "(int, default -1) the group size of GQA")
        .SetDefault(-1);
    AddAttr<float>("rope_theta", "(float, default 10000.0f) the theta in RoPE")
        .SetDefault(10000.0f);

    AddComment(R"DOC(fused multi transformer dybatch layers op)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_multi_transformer_dybatch,
    ops::FusedMultiTransformerDybatchOp,
    ops::FusedMultiTransformerDybatchOpOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_VERSION(fused_multi_transformer_dybatch)
    .AddCheckpoint(
        R"ROC(
              Add a new attribute [trans_qkvw] )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "trans_qkvw",
            "A flag to indicate whether to transpose for weights of qkv.",
            true));
