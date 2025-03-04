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

#include "paddle/cinn/hlir/dialect/operator/transforms/remove_redundant_group_output_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class RemoveRedundantGroupOutputPattern
    : public pir::OpRewritePattern<cinn::dialect::GroupOp> {
 public:
  explicit RemoveRedundantGroupOutputPattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::GroupOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::GroupOp group_op,
                       pir::PatternRewriter& rewriter) const override {
    auto module_op = group_op.block()
                         ->GetParentOp()
                         ->GetParentOp()
                         ->dyn_cast<pir::ModuleOp>();
    if (!module_op) {
      std::cout << "cannot find module op\n";
      return false;
    } else {
      std::cout << "Before remove redundant output from group op: "
                << std::endl;
      module_op->Print(std::cout);
      std::cout << std::endl;
    }

    const auto& input_args = pir::GetUsedExternalValue(*group_op.block());
    const std::unordered_set<pir::Value> inputs_set = {input_args.begin(),
                                                       input_args.end()};
    const auto& yield_op = group_op.block()->back();
    std::vector<pir::Value> new_outputs;
    std::vector<pir::Type> new_out_types;
    std::unordered_map<pir::Value, uint32_t> origin_group_out_2_new_out_idx;
    std::unordered_map<uint32_t, pir::Value>
        redundant_out_idx_map;  // output index -> input value
    for (uint32_t i = 0; i < yield_op.num_operands(); ++i) {
      if (inputs_set.count(yield_op.operand_source(i)) > 0) {
        redundant_out_idx_map.emplace(i, yield_op.operand_source(i));
      } else {
        new_outputs.push_back(yield_op.operand_source(i));
        new_out_types.push_back(yield_op.operand_source(i).type());
        origin_group_out_2_new_out_idx[group_op.result(i)] =
            new_outputs.size() - 1;
      }
    }
    if (redundant_out_idx_map.empty()) {
      VLOG(1) << "No redundant output in group op, skip.";
      return false;
    }

    auto new_group_op = rewriter.Build<cinn::dialect::GroupOp>(new_out_types);
    const std::vector<pir::Operation*> ops_to_move = [](pir::Block* block) {
      std::vector<pir::Operation*> ops;
      for (auto& op : *block) {
        if (op.isa<pir::YieldOp>()) continue;
        ops.push_back(&op);
      }
      return ops;
    }(group_op.block());
    for (auto& op : ops_to_move) {
      op->MoveTo(new_group_op.block(), new_group_op.block()->end());
    }
    rewriter.SetInsertionPointToBlockEnd(new_group_op.block());
    rewriter.Build<pir::YieldOp>(new_outputs);

    for (uint32_t i = 0; i < group_op.num_results(); ++i) {
      if (redundant_out_idx_map.count(i) > 0) {
        rewriter.ReplaceAllUsesWith(group_op.result(i),
                                    redundant_out_idx_map.at(i));
      } else {
        rewriter.ReplaceAllUsesWith(
            group_op.result(i),
            new_group_op.result(
                origin_group_out_2_new_out_idx.at(group_op.result(i))));
      }
    }
    rewriter.EraseOp(group_op);
    VLOG(1) << "Remove redundant output from group op.";
    return true;
  }
};

class RemoveRedundantGroupOutputPass : public pir::PatternRewritePass {
 public:
  RemoveRedundantGroupOutputPass()
      : pir::PatternRewritePass("remove_redundant_group_output_pass",
                                /*opt_level=*/1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);

    ps.Add<RemoveRedundantGroupOutputPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    if (op->isa<cinn::dialect::GroupOp>()) {
      return false;
    }
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateRemoveRedundantGroupOutputPass() {
  return std::make_unique<RemoveRedundantGroupOutputPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
