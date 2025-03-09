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

                  x   w   b           in1
                  |   |   |           |  axis1
                  ---------           |  /
                      |           unsqueeze1
                      |               |  factor
                      |               |  /
                    layer_norm      scale
                      |               |
                      |               |               in2
                      |               |               |  axis2
                      |               |               | /
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
               x   w  b  in1  in2
               |   |  |   |    |
               -----------------
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
  std::string name() const override { return "AdaptiveLayernormPattern"; }
  uint32_t benefit() const override { return 3; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &layernorm =
        pat.Op(paddle::dialect::LayerNormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")}});

    const auto &full_int_array1 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("unsqueeze_1_axis")}});

    const auto &unsqueeze1 = pat.Op(paddle::dialect::UnsqueezeOp::name());

    const auto &full = pat.Op(paddle::dialect::FullOp::name(),
                              {{"shape", pat.Attr("full_shape")},
                               {"value", pat.Attr("full_value")},
                               {"dtype", pat.Attr("full_dtype")},
                               {"place", pat.Attr("full_place")}});
    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("scale_bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});

    const auto &multiply = pat.Op(paddle::dialect::MultiplyOp::name());

    const auto &full_int_array2 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("unsqueeze_2_axis")}});

    const auto &unsqueeze2 = pat.Op(paddle::dialect::UnsqueezeOp::name());

    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    layernorm({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("b")},
              {&pat.Tensor("layer_norm_out"),
               &pat.Tensor("mean_out_0"),
               &pat.Tensor("variance_out_0")});
    unsqueeze1({&pat.Tensor("unsqueeze_1_in"), &full_int_array1()},
               {&pat.Tensor("unsqueeze_1_out")});
    scale({&pat.Tensor("unsqueeze_1_out"), &full()},
          {&pat.Tensor("scale_out")});
    multiply({&pat.Tensor("layer_norm_out"), &pat.Tensor("scale_out")},
             {&pat.Tensor("multiply_out")});
    unsqueeze2({&pat.Tensor("unsqueeze_2_in"), &full_int_array2()},
               {&pat.Tensor("unsqueeze_2_out")});
    add({&pat.Tensor("multiply_out"), &pat.Tensor("unsqueeze_2_out")},
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
      std::cout << "begin_norm_axis: " << match_ctx.Attr<int>("begin_norm_axis")
                << std::endl;
      std::cout << "epsilon: " << match_ctx.Attr<float>("epsilon") << std::endl;
      std::cout << "factor: " << match_ctx.Attr<double>("full_value")
                << std::endl;
      std::cout << "scale_bias: " << match_ctx.Attr<float>("scale_bias")
                << std::endl;
      std::cout << "bias_after_scale: "
                << match_ctx.Attr<bool>("bias_after_scale") << std::endl;
      auto shape = match_ctx.Attr<std::vector<int64_t>>("unsqueeze_1_axis");
      for (auto i : shape) {
        std::cout << "i: " << i << std::endl;
      }
      // const auto &float_factor_attr = res.ComputeAttr(
      // [](const paddle::drr::MatchContext &match_ctx) -> float {
      //   auto factor = match_ctx.Attr<double>("full_value");
      //   return static_cast<float>(factor);
      // });

      // std::cout << "float_factor_attr: " << float_factor_attr << std::endl;
      //   std::cout << "unsqueeze_1_axis: " <<
      //   match_ctx.Attr<std::vector<int64_t>>("unsqueeze_1_axis") <<
      //   std::endl; std::cout << "unsqueeze_2_axis: " <<
      //   match_ctx.Attr<std::vector<int64_t>>("unsqueeze_2_axis") <<
      //   std::endl;
      return true;
    });

    // Result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_unsqueeze_1_axis = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          std::vector<int> int_array_value;
          auto shape = match_ctx.Attr<std::vector<int64_t>>("unsqueeze_1_axis");
          for (auto i : shape) {
            int_array_value.emplace_back(static_cast<int>(i));
          }
          return int_array_value;
        });

    const auto &fused_unsqueeze_2_axis = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::vector<int> {
          std::vector<int> int_array_value;
          auto shape = match_ctx.Attr<std::vector<int64_t>>("unsqueeze_2_axis");
          for (auto i : shape) {
            int_array_value.emplace_back(static_cast<int>(i));
          }
          return int_array_value;
        });

    const auto &fused_factor = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("full_value");
        });

    const auto &adaptive_layernorm_xpu =
        res.Op(paddle::dialect::AdaptiveLayernormXpuOp::name(),
               {{
                   {"begin_norm_axis", pat.Attr("begin_norm_axis")},
                   {"epsilon", pat.Attr("epsilon")},
                   {"factor", fused_factor},
                   {"scale_bias", pat.Attr("scale_bias")},
                   {"bias_after_scale", pat.Attr("bias_after_scale")},
                   {"unsqueeze_1_axis", fused_unsqueeze_1_axis},
                   {"unsqueeze_2_axis", fused_unsqueeze_2_axis},
               }});
    adaptive_layernorm_xpu(
        {
            &res.Tensor("x"),
            &res.Tensor("w"),
            &res.Tensor("b"),
            &res.Tensor("unsqueeze_1_in"),
            &res.Tensor("unsqueeze_2_in"),
        },
        {&res.Tensor("out")});
  }
};

class AdaptiveLayernormXpuFusePass : public pir::PatternRewritePass {
 public:
  AdaptiveLayernormXpuFusePass()
      : pir::PatternRewritePass("adaptive_layernorm_xpu_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<AdaptiveLayernormPattern>(context));
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
