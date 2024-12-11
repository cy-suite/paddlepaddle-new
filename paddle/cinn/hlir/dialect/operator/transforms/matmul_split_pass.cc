#include "paddle/dialect/operator/transforms/matmul_split_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include <vector>

namespace paddle {
namespace dialect {

class MatMulSplitPattern 
    : public pir::OpRewritePattern<paddle::dialect::MatMulOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::MatMulOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::MatMulOp op,
                      pir::PatternRewriter &rewriter) const override {
    // 获取矩阵维度
    auto a_shape = op->operand(0).type().dyn_cast<pir::ShapedType>();
    auto b_shape = op->operand(1).type().dyn_cast<pir::ShapedType>();
    
    if (!a_shape || !b_shape) return false;
    
    int m = a_shape.getDimSize(0);
    int k = a_shape.getDimSize(1); 
    int n = b_shape.getDimSize(1);

    // 检查是否需要分块
    if (m < kMinSplitSize || n < kMinSplitSize) return false;

    // 计算分块大小和数量
    int block_size = kMinSplitSize;
    int m_blocks = (m + block_size - 1) / block_size;
    int n_blocks = (n + block_size - 1) / block_size;

    // 创建结果tensor
    auto result_type = op.result().type().dyn_cast<pir::ShapedType>();
    Value result = rewriter.create<AllocOp>(op.getLoc(), result_type);
    
    // 分块计算
    for (int i = 0; i < m_blocks; i++) {
      for (int j = 0; j < n_blocks; j++) {
        // 计算当前块的大小
        int cur_m = std::min(block_size, m - i * block_size);
        int cur_n = std::min(block_size, n - j * block_size);
        
        // 提取子矩阵
        Value a_block = rewriter.create<SliceOp>(
            op.getLoc(), 
            op->operand(0),
            /*starts=*/{i * block_size, 0},
            /*ends=*/{i * block_size + cur_m, k},
            /*axes=*/{0, 1});
            
        Value b_block = rewriter.create<SliceOp>(
            op.getLoc(),
            op->operand(1), 
            /*starts=*/{0, j * block_size},
            /*ends=*/{k, j * block_size + cur_n},
            /*axes=*/{0, 1});

        // 计算块矩阵乘法
        Value mul_result = rewriter.create<MatMulOp>(
            op.getLoc(), a_block, b_block);

        // 更新结果
        rewriter.create<UpdateSliceOp>(
            op.getLoc(),
            result,
            mul_result,
            /*starts=*/{i * block_size, j * block_size});
      }
    }

    // 替换原操作的输出
    rewriter.replaceOp(op, result);
    return true;
  }

 private:
  static constexpr int kMinSplitSize = 1024;
};

MatMulSplitOptimizePass::MatMulSplitOptimizePass()
    : pir::PatternRewritePass("matmul_split_optimize", 1) {}

pir::RewritePatternSet MatMulSplitOptimizePass::InitializePatterns(
    pir::IrContext *context) {
  pir::RewritePatternSet patterns(context);
  patterns.Add<MatMulSplitPattern>(context);
  return patterns;
}

bool MatMulSplitOptimizePass::CanApplyOn(pir::Operation *op) const {
  return op->num_regions() > 0;
}

std::unique_ptr<pir::Pass> CreateMatMulSplitOptimizePass() {
  return std::make_unique<MatMulSplitOptimizePass>();
}

}  // namespace dialect
}  // namespace paddle

