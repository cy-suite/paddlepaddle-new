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
                      x1
                      |     tensor1
                      |    /
                elementwise_mul
                      |
                      |     x2
                      |   /
                elementwise_add
                      |
           tensor3    |       tensor2
                \     |     /
                adaptive_layer_norm
                      |
                      |
                    out_Out
------------------------------------------------------
After the pass is applied:
                      x1
            x2        |     tensor1
               \      |    /
                \     |   /
  tensor3 ---- adaln_scale_residual_xpu_kernel ---- tensor2
                      |
                      |
                      |
                    Output

*/

namespace {
class FusedAdalnScaleResidualXpuPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "FusedAdalnScaleResidualXpuPattern";
  }
  uint32_t benefit() const override { return 3; }

  // rewrite pattern operator()
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    // source pattern
    const auto &multiply = pat.Op(paddle::dialect::MultiplyOp::name());
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &layernorm =
        pat.Op(paddle::dialect::LayerNormOp::name(),
               {{"epsilon", pat.Attr("epsilon")},
                {"begin_norm_axis", pat.Attr("begin_norm_axis")}});

    const auto &full = pat.Op(paddle::dialect::FullOp::name(),
                              {{"shape", pat.Attr("full_shape")},
                               {"value", pat.Attr("full_value")},
                               {"dtype", pat.Attr("full_dtype")},
                               {"place", pat.Attr("full_place")}});
    const auto &scale =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("scale_bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});

    multiply({&pat.Tensor("input1"), &pat.Tensor("tensor1")},
             {&pat.Tensor("mult_output")});
    add({&pat.Tensor("mult_output"), &pat.Tensor("input2")},
        {&pat.Tensor("add_output")});
    layernorm({&pat.Tensor("add_output"), &pat.Tensor("w"), &pat.Tensor("b")},
              {&pat.Tensor("layer_norm_out"),
               &pat.Tensor("mean_out_0"),
               &pat.Tensor("variance_out_0")});
    scale({&pat.Tensor("tensor2"), &full()}, {&pat.Tensor("scale_out")});
    multiply({&pat.Tensor("layer_norm_out"), &pat.Tensor("scale_out")},
             {&pat.Tensor("multiply_out")});
    add({&pat.Tensor("multiply_out"), &pat.Tensor("tensor3")},
        {&pat.Tensor("final_output")});

    // // Constraints
    // pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
    //   auto input1_shape = pir::GetShapeFromValue(match_ctx.Tensor("input1"));
    //   auto input2_shape = pir::GetShapeFromValue(match_ctx.Tensor("input2"));
    //   auto tensor1_shape =
    //   pir::GetShapeFromValue(match_ctx.Tensor("tensor1")); auto tensor2_shape
    //   = pir::GetShapeFromValue(match_ctx.Tensor("tensor2")); auto
    //   tensor3_shape = pir::GetShapeFromValue(match_ctx.Tensor("tensor3")); if
    //   ((input1_shape.size() == input2_shape.size()) &&
    //       (input2_shape.size() == tensor1_shape.size()) &&
    //       (tensor1_shape.size() == tensor2_shape.size()) &&
    //       (tensor2_shape.size() == tensor3_shape.size())) {
    //     return true;
    //   }else {
    //     return false;
    //   }
    // });

    // result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &scale_weight = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> float {
          return match_ctx.Attr<double>("full_value");
        });

    const auto &fused_adaLN_scale_residual_xpu =
        res.Op(paddle::dialect::FusedAdalnScaleResidualXpuOp::name(),
               {{
                   {"begin_norm_axis", pat.Attr("begin_norm_axis")},
                   {"epsilon", pat.Attr("epsilon")},
                   {"scale_op_weight", scale_weight},
                   {"scale_op_bias", pat.Attr("scale_bias")},
                   {"bias_after_scale", pat.Attr("bias_after_scale")},
               }});
    fused_adaLN_scale_residual_xpu(
        {
            &res.Tensor("input1"),
            &res.Tensor("input2"),
            &res.Tensor("tensor1"),
            &res.Tensor("tensor2"),
            &res.Tensor("tensor3"),
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
    ps.Add(paddle::drr::Create<FusedAdalnScaleResidualXpuPattern>(context));
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
