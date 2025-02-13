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

    // Constraints
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto unsqueeze_1_out_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("unsqueeze_1_out"));
      auto unsqueeze_2_out_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("unsqueeze_2_out"));

      if ((x_shape.size() == unsqueeze_1_out_shape.size()) &&
          (unsqueeze_1_out_shape.size() == unsqueeze_2_out_shape.size())) {
        return true;
      }

      return false;
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
        {&res.Tensor("output")});
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
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateAdaptiveLayernormXpuFusePass() {
  return std::make_unique<AdaptiveLayernormXpuFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(adaptive_layernorm_xpu_fuse_pass,
                 AdaptiveLayernormXpuFusePass);
