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

#include "paddle/fluid/pir/transforms/xpu/fused_adaLN_scale_residual_xpu_pass.h"
#include <optional>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

/*
fuse adaln + scale_residual in to xpu_adalan_scale_residual op
For example:
graph:
                    ele_x
                      |     ele_y
                      |    /
                elementwise_mul
                      |
                      |     ele_z
                      |   /
                elementwise_add
                      |
                      |       ele_u
                      |     /
                adaptive_layer_norm
                      |
                      |
                    out_Out
------------------------------------------------------
After the pass is applied:
                    ele_x
                      |     ele_y
           ele_u      |    /
                \     |   /
            adaln_scale_residual_xpu_kernel ---- ele_z
                      |
                      |
                      |
                    Output

*/

namespace {

template <int act_type>
class FusedAdalnScaleResidualPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedAdalnScaleResidualPattern"; }

  // rewrite pattern operator()
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    // Patterns
    const auto &multiply = pat.Op(paddle::dialect::MultiplyOp::name());
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &layernorm =
        pat.Op(paddle::dialect::LayerNormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")}});
    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("scale_bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});
    const auto &full = pat.Op(paddle::dialect::FullOp::name(),
                              {{"shape", pat.Attr("full_shape")},
                               {"value", pat.Attr("full_value")},
                               {"dtype", pat.Attr("full_dtype")},
                               {"place", pat.Attr("full_place")}});
    // calling pattern
    multiply({&pat.Tensor("x1"), &pat.Tensor("unsqueezed1")},
             {&pat.Tensor("final_output")});
    add({&pat.Tensor("final_output"), &pat.Tensor("x2")},
        {&pat.Tensor("final_output")});
    layernorm({&pat.Tensor("final_output"), &pat.Tensor("w"), &pat.Tensor("b")},
              {&pat.Tensor("final_output"),
               &pat.Tensor("mean_out_0"),
               &pat.Tensor("variance_out_0")});
    scale({&pat.Tensor("unsqueezed2"), &full()}, {&pat.Tensor("scale_out")});
    multiply({&pat.Tensor("final_output"), &pat.Tensor("scale_out")},
             {&pat.Tensor("final_output")});
    add({&pat.Tensor("final_output"), &pat.Tensor("unsqueezed3")},
        {&pat.Tensor("final_output")});

    // Constraints
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x1"));
      auto add_in_shape = pir::GetShapeFromValue(match_ctx.Tensor("x2"));
      auto unsqueezed1_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("unsqueezed1"));
      auto unsqueezed2_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("unsqueezed2"));
      auto unsqueezed3_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("unsqueezed3"));

      // if ((x_shape.size() == scale_in_shape.size()) &&
      //     (scale_in_shape.size() == add_in_shape.size())) {
      //   return true;
      // }else {
      //   return false;
      // }
      return true;
    });

    // result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &scale_weight = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("full_value");
        });

    const auto &fused_adaLN_scale_residual_xpu_kernel =
        res.Op(paddle::dialect::FusedAdalnScaleResidualXpuKernelOp::name(),
               {{
                   {"begin_norm_axis", pat.Attr("begin_norm_axis")},
                   {"epsilon", pat.Attr("epsilon")},
                   {"scale_weight", scale_weight},
                   {"scale_bias", pat.Attr("scale_bias")},
                   {"bias_after_scale", pat.Attr("bias_after_scale")},
               }});
    fused_adaLN_scale_residual_xpu_kernel(
        {
            &res.Tensor("x1"),
            &res.Tensor("x2"),
            &res.Tensor("unsqueeze1"),
            &res.Tensor("unsqueeze2"),
            &res.Tensor("unsqueeze3"),
            &res.Tensor("w"),
            &res.Tensor("b"),
        },
        {&res.Tensor("final_output")});
  }
};

class FusedAdalnScaleResidualXpuPass : public pir::PatternRewritePass {
 public:
  FusedAdalnScaleResidualXpuPass()
      : pir::PatternRewritePass("fused_adaLN_scale_residual_xpu_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FusedAdalnScaleResidualPattern>(context));
    return ps;
  }
};
}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateFusedAdalnScaleResidualXpuPass() {
  return std::make_unique<FusedAdalnScaleResidualXpuPass>();
}
}  // namespace pir

REGISTER_IR_PASS(fused_adaLN_scale_residual_xpu_pass,
                 FusedAdalnScaleResidualXpuPass);
