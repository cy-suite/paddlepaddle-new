#pragma once

#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"

namespace paddle {
namespace dialect {

// 定义乘法分块优化Pass
class MatMulSplitOptimizePass : public pir::PatternRewritePass {
 public:
  MatMulSplitOptimizePass();
  
  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override;
  bool CanApplyOn(pir::Operation *op) const override;

 private:
  static constexpr int kMinSplitSize = 1024; // 最小分块大小
};

IR_API std::unique_ptr<pir::Pass> CreateMatMulSplitOptimizePass();

}  // namespace dialect
}  // namespace paddle

