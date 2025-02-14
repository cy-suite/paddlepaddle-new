// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/xpu/adaptive_layernorm_xpu_fuse_pass.h"
#include <optional>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {
/*
fuse malmul + act to fc_xpu
For example:
graph:

                  x   w   b
                    \ | /          in1     in2
                      |              \     /
                      |           unsqueeze1
                      |               |  factor
                      |               |  /
                    layer_norm      scale
                      |               |
                      |               |
                      |               |            in3   in4
                      |               |             \     /
                      -----------------            unsqueeze2
                             |                         |
                             |                         |
                          multiply                     |
                             |                         |
                             ---------------------------
                                         |
                                        add
                                         |
                                        output
------------------------------------------------------
After the pass is applied:
            x   w  b  in1  in2 in3  in4
            |   |  |   |    |   |    |
            ------------------------
                      |
            adaptive_layernorm_xpu
                      |
                      |
                    Output
*/

class AdaptiveLayernormPattern : public paddle::drr::DrrPatternBase {
 private:
  bool transpose_w_;

 public:
  explicit AdaptiveLayernormPattern(bool transpose_w)
      : transpose_w_(transpose_w) {}
  std::string name() const override { return "AdaptiveLayernormPattern"; }
  uint32_t benefit() const override { return 3; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &layernorm = pat.Op(paddle::dialect::LayerNormOp::name(),
                                   {{"epsilon", pat.Attr("epsilon")}});

    const auto &full_int_array1 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value1", pat.Attr("value")}});

    const auto &unsqueeze1 = pat.Op(paddle::dialect::UnsqueezeOp::name());

    const auto &full = pat.Op(paddle::dialect::FullOp::name(),
                              {{"full_shape", pat.Attr("shape")},
                               {"full_value", pat.Attr("value")},
                               {"full_dtype", pat.Attr("dtype")},
                               {"full_place", pat.Attr("place")}});
    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("scale_bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});

    const auto &multiply = pat.Op(paddle::dialect::MultiplyOp::name());

    const auto &full_int_array2 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value2", pat.Attr("value")}});

    const auto &unsqueeze2 = pat.Op(paddle::dialect::UnsqueezeOp::name());

    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    layernorm({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("b")},
              {&pat.Tensor("layer_norm_out"),
               &pat.Tensor("mean_out_0"),
               &pat.Tensor("variance_out_0")});
    pat.Tensor("full_int_array_out1") = full_int_array1();
    unsqueeze1(
        {&pat.Tensor("unsqueeze_in1"), &pat.Tensor("full_int_array_out1")},
        {&pat.Tensor("unsqueeze_out1")});
    pat.Tensor("full_out") = full();
    scale({&pat.Tensor("unsqueeze_out1"), &pat.Tensor("full_out")},
          {&pat.Tensor("scale_out")});
    multiply({&pat.Tensor("layer_norm_out"), &pat.Tensor("scale_out")},
             {&pat.Tensor("multiply_out")});
    pat.Tensor("full_int_array_out2") = full_int_array2();
    unsqueeze2(
        {&pat.Tensor("unsqueeze_in2"), &pat.Tensor("full_int_array_out2")},
        {&pat.Tensor("unsqueeze_out2")});
    add({&pat.Tensor("multiply_out"), &pat.Tensor("unsqueeze_out2")},
        {&pat.Tensor("output")});

    // (%1222, %1223, %1224) = "pd_op.layer_norm" (%1192, %1167, %1166)
    // {begin_norm_axis:2,epsilon:1e-06,stop_gradient:[true,true,true],struct_name:"/DiTBlockTwoStream/"}
    // : (tensor<2x4096x2560xbf16>, tensor<2560xbf16>, tensor<2560xbf16>) ->
    // tensor<2x4096x2560xbf16>, tensor<2x4096xf32>, tensor<2x4096xf32>
    // (%1225) = "pd_op.full_int_array" ()
    // {dtype:int64,place:Place(cpu),stop_gradient:[true],struct_name:"/DiTBlockTwoStream/",value:[1]}
    // : () -> tensor<1xi64>
    // (%1226) = "pd_op.unsqueeze" (%1204, %1225)
    // {stop_gradient:[true],struct_name:"/DiTBlockTwoStream/"} :
    // (tensor<2x2560xbf16>, tensor<1xi64>) -> tensor<2x1x2560xbf16>
    // (%1227) = "pd_op.full" ()
    // {dtype:float32,place:Place(cpu),shape:[1],stop_gradient:[true],struct_name:"/DiTBlockTwoStream/",value:1}
    // : () -> tensor<1xf32>
    // (%1228) = "pd_op.scale" (%1226, %1227)
    // {bias:1,bias_after_scale:true,stop_gradient:[true],struct_name:"/DiTBlockTwoStream/"}
    // : (tensor<2x1x2560xbf16>, tensor<1xf32>) -> tensor<2x1x2560xbf16>
    // (%1229) = "pd_op.multiply" (%1222, %1228)
    // {stop_gradient:[true],struct_name:"/DiTBlockTwoStream/"} :
    // (tensor<2x4096x2560xbf16>, tensor<2x1x2560xbf16>) ->
    // tensor<2x4096x2560xbf16>
    // (%1230) = "pd_op.full_int_array" ()
    // {dtype:int64,place:Place(cpu),stop_gradient:[true],struct_name:"/DiTBlockTwoStream/",value:[1]}
    // : () -> tensor<1xi64>
    // (%1231) = "pd_op.unsqueeze" (%1203, %1230)
    // {stop_gradient:[true],struct_name:"/DiTBlockTwoStream/"} :
    // (tensor<2x2560xbf16>, tensor<1xi64>) -> tensor<2x1x2560xbf16>
    // (%1232) = "pd_op.add" (%1229, %1231)
    // {stop_gradient:[true],struct_name:"/DiTBlockTwoStream/"} :
    // (tensor<2x4096x2560xbf16>, tensor<2x1x2560xbf16>) ->
    // tensor<2x4096x2560xbf16>

    // Constraints
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      auto b_shape = pir::GetShapeFromValue(match_ctx.Tensor("b"));
      std::cout << "x_shape: " << x_shape.size() << std::endl;
      std::cout << "w_shape: " << w_shape.size() << std::endl;
      std::cout << "b_shape: " << b_shape.size() << std::endl;
      return true;
    });

    // Result pattern
    // paddle::drr::ResultPattern res = pat.ResultPattern();

    // const auto &in_num_col_dims_attr =
    //     res.ComputeAttr([&](const paddle::drr::MatchContext &match_ctx) ->
    //     int {
    //       auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
    //       return x_shape.size() - 1;
    //     });

    // if (!transpose_w_) {
    //   // prepare weight, transpose it if necessary
    //   const auto &perm_attr = res.ComputeAttr(
    //       [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int>
    //       {
    //         auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
    //         if (w_shape.size() == 2) {
    //           return {1, 0};
    //         } else {
    //           PADDLE_THROW(common::errors::Unimplemented(
    //               "Not support convert w_shape.size()(%d).",
    //               w_shape.size()));
    //         }
    //       });
    //   const auto &transpose_op =
    //       res.Op(paddle::dialect::TransposeOp::name(), {{"perm",
    //       perm_attr}});
    //   res.Tensor("w_trans") = transpose_op(res.Tensor("w"));
    //   VLOG(3) << "transpose weight for fc_xpu op";
    // }

    // const auto &out_dtype_attr = res.ComputeAttr(
    //     [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
    //       auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("x"));
    //       // 目前仅支持以下几种非量化的情况
    //       if (x_dtype.isa<pir::Float32Type>()) {
    //         return phi::DataType::FLOAT32;
    //       } else if (x_dtype.isa<pir::Float16Type>()) {
    //         return phi::DataType::FLOAT16;
    //       } else if (x_dtype.isa<pir::BFloat16Type>()) {
    //         return phi::DataType::BFLOAT16;
    //       } else {
    //         return phi::DataType::UNDEFINED;
    //       }
    //     });

    //     const auto &fc_xpu = res.Op(
    //         paddle::dialect::FcXpuOp::name(),
    //         {{
    //             {"in_num_col_dims", in_num_col_dims_attr},
    //             {"transpose_x", pat.Attr("transpose_x")},
    //             {"alpha", res.Float32Attr(1.0f)},
    //             {"beta", res.Float32Attr(0.f)},
    //             {"act_type",
    //             res.Int32Attr(ConvertActivationType("swish_glu"))},
    //             {"act_alpha", res.Float32Attr(0.0f)},
    //             {"out_dtype", out_dtype_attr},
    //         }});
    //     fc_xpu(
    //         {
    //             &res.Tensor("x"),
    //             &res.InputNoneTensor(),
    //             transpose_w_ ? &res.Tensor("w") : &res.Tensor("w_trans"),
    //             &res.InputNoneTensor(),
    //             &res.InputNoneTensor(),
    //             &res.InputNoneTensor(),
    //             &res.InputNoneTensor(),
    //         },
    //         {&res.Tensor("act_out"), &res.Tensor("out_max")});
  }
};

class AdaptiveLayernormXpuFusePass : public pir::PatternRewritePass {
 public:
  AdaptiveLayernormXpuFusePass()
      : pir::PatternRewritePass("adaptive_layernorm_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<AdaptiveLayernormPattern>(context, false));
    ps.Add(paddle::drr::Create<AdaptiveLayernormPattern>(context, true));
    return ps;
  }

  pir::GreedyRewriteConfig InitializeConfig() override {
    pir::GreedyRewriteConfig config;

    config.use_top_down_traversal = false;

    config.max_iterations = 10;
    return config;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateAdaptiveLayernormXpuFusePass() {
  return std::make_unique<AdaptiveLayernormXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(adaptive_layernorm_xpu_fuse_pass,
                 AdaptiveLayernormXpuFusePass);
