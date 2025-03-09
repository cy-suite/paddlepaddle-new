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

#include "paddle/fluid/pir/transforms/xpu/fused_multiply_add_xpu_pass.h"
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
fuse elementwise_mul + elementwise_mul to addcmul_xpu
For example:
graph:
              x         y
               \       /
                \     /
            elementwise_mul    w
                    \         /
                     \       /
                  elementwise_add
                        |
                        |
                      output
------------------------------------------------------
After the pass is applied:
               x      y      w
                \     |     /
                 \    |    /
                 addcmul_xpu
                      |
                      |
                    output
*/

namespace {

class FusedMultiplyAddXpuPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
    return "FusedMultiplyAddXpuPattern";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    // Source pattern
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &mul_op = pat.Op(paddle::dialect::MultiplyOp::name());
    const auto &add_op = pat.Op(paddle::dialect::AddOp::name());
    mul_op({&pat.Tensor("x"), &pat.Tensor("y")}, {&pat.Tensor("mul_out")});
    add_op({&pat.Tensor("mul_out"), &pat.Tensor("w")},
           {&pat.Tensor("add_out")});

    auto calculate_element_num =
          [](const std::vector<int64_t> &input) -> int64_t {
        int64_t accu_mul = 1;
        for (auto num : input) {
          accu_mul *= num;
        }
        return accu_mul;
      };
      
    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto x_shape = pir::GetShapeFromValue(match_ctx.Tensor("x"));
      auto y_shape = pir::GetShapeFromValue(match_ctx.Tensor("y"));
      auto w_shape = pir::GetShapeFromValue(match_ctx.Tensor("w"));
      int64_t shape_len_max = std::max<int>(w_shape.size(), std::max<int>(x_shape.size(), y_shape.size()));
      for (int64_t i = 0; i < shape_len_max; i++) {
        int64_t wi = i + w_shape.size() - shape_len_max;
        int64_t xi = i + x_shape.size() - shape_len_max;
        int64_t yi = i + y_shape.size() - shape_len_max;
        wi = ((wi < 0) ? 1 : w_shape[wi]);
        xi = ((xi < 0) ? 1 : x_shape[xi]);
        yi = ((yi < 0) ? 1 : y_shape[yi]);

        if(wi < 0 || xi < 0 || yi < 0){
          return false;
        }

        if((((xi == yi) && (yi == wi)) || ((xi == wi) && ((yi == 1) || (wi == 1))) ||
                    ((yi == wi) && ((xi == 1) || (wi == 1))) || ((xi == yi) && ((wi == 1) || (xi == 1))))){
                      return true;
                    }
    }
      return false;
    });
    // Result pattern
    paddle::drr::ResultPattern res = pat.ResultPattern();
    
    const auto &addcmul_xpu = res.Op(paddle::dialect::FusedMultiplyAddXpuOp::name());
        addcmul_xpu({&res.Tensor("x"), &res.Tensor("y"), &res.Tensor("w")},
                {&res.Tensor("add_out")});
  }
};

class FusedMultiplyAddXpuPass : public pir::PatternRewritePass {
 public:
  FusedMultiplyAddXpuPass()
      : pir::PatternRewritePass("fused_multiply_add_xpu_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FusedMultiplyAddXpuPattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFusedMultiplyAddXpuPass() {
  return std::make_unique<FusedMultiplyAddXpuPass>();
}

}  // namespace pir

REGISTER_IR_PASS(fused_multiply_add_xpu_pass,
                 FusedMultiplyAddXpuPass);
