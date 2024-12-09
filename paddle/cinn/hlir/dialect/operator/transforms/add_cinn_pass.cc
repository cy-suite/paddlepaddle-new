#include "paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.h"

#include <chrono>
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/pir/include/pass/pass_manager.h"

// Add new headers
#include "paddle/dialect/operator/transforms/matmul_split_pass.h"
#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"

COMMON_DECLARE_bool(cinn_specify_input_dynamic_dim);
COMMON_DECLARE_string(cinn_input_dynamic_dim_spec_file);
COMMON_DECLARE_bool(print_ir);
COMMON_DECLARE_bool(disable_dyshape_in_train);
COMMON_DECLARE_bool(enable_cinn_accuracy_check);
COMMON_DECLARE_bool(enable_fusion_fallback);
COMMON_DECLARE_bool(logging_pir_py_code_dump_symbolic_dims);
PD_DECLARE_bool(group_schedule_tiling_first);

namespace cinn::dialect::ir {

namespace {
bool HasDynamicShape(const pir::Program& program) {
  if (FLAGS_disable_dyshape_in_train) {
    return false;
  }
  for (const auto& op : *program.block()) {
    if (op.isa<pir::CombineOp>()) {
      continue;
    }
    for (uint32_t i = 0; i < op.num_results(); ++i) {
      if (op.result(i) && op.result(i).type()) {
        auto shape_type =
            op.result(i).type().dyn_cast<pir::ShapedTypeInterface>();
        if (shape_type && shape_type.IsDynamicShape()) {
          return true;
        }
      }
    }
  }
  return false;
}
}  // namespace

void ApplyShapeOptimizationPass(::pir::Program* program,
                               const std::function<std::shared_ptr<::pir::PassManager>()>& CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  bool has_dynamic_shape = HasDynamicShape(*program);
  if (has_dynamic_shape) {
    if (FLAGS_cinn_specify_input_dynamic_dim) {
      PADDLE_ENFORCE_NE(
          FLAGS_cinn_input_dynamic_dim_spec_file,
          "",
          ::common::errors::InvalidArgument(
              "'FLAGS_cinn_input_dynamic_dim_spec_file' should not be empty "
              "when using FLAGS_cinn_specify_input_dynamic_dim."));
      SpecifyInputDynamicDimFromFile(program,
                                    FLAGS_cinn_input_dynamic_dim_spec_file);
    }
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  }
  pass_manager->Run(program);
}

void ApplyPdToCinnPass(::pir::Program* program,
                       const std::function<std::shared_ptr<::pir::PassManager>()>& CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  // Add the new MatMul split optimization pass
  pass_manager->AddPass(paddle::dialect::CreateMatMulSplitOptimizePass());
  
  pass_manager->AddPass(cinn::dialect::ir::CreateReduceAsToSumPass());
  pass_manager->AddPass(pir::CreateFusedGemmEpiloguePass());
  pass_manager->AddPass(cinn::dialect::ir::CreateRemoveAssignOutPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateConv2dTransposeFilterPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateConvertMEA2FAPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateConvertFA2QKVMHAPass());
  pass_manager->AddPass(cinn::dialect::ir::CreatePdOpToCinnOpPass());
  pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());
  pass_manager->Run(program);
}

void ApplyCinnPreprocessPass(::pir::Program* program,
                            const std::function<std::shared_ptr<::pir::PassManager>()>& CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  bool has_dynamic_shape = HasDynamicShape(*program);

  if (has_dynamic_shape) {
    pass_manager->AddPass(
        cinn::dialect::ir::CreateFuseShapeOpsIntoGenerateShapeOpPass());
    pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());
  }
  
  pass_manager->AddPass(cinn::dialect::ir::CreateTransposeOptimizePass());
  pass_manager->Run(program);
}

void ApplyBuildGroupOpPass(::pir::Program* program,
                          const std::function<std::shared_ptr<::pir::PassManager>()>& CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(cinn::dialect::ir::CreateFoldManipulationOpsPass());
  pass_manager->AddPass(pir::CreateBuildCinnPass());
  pass_manager->Run(program);
}

void ApplyGroupOpPass(::pir::Program* program,
                     const std::function<std::shared_ptr<::pir::PassManager>()>& CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(cinn::dialect::ir::CreateAddBroadcastToElementwisePass());
  pass_manager->AddPass(cinn::dialect::ir::CreateInsertBroadcastPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateFuseShapeOpsIntoGenerateShapeOpPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateMoveGenerateShapeOpsToProloguePass());
  pass_manager->AddPass(cinn::dialect::ir::CreateDynamicReshapeOpPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateFoldManipulationOpsPass());
  pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());
  pass_manager->Run(program);
}

void ApplyDivideGroupOpToFusionOpPass(::pir::Program* program,
                                     const std::function<std::shared_ptr<::pir::PassManager>()>& CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(cinn::dialect::ir::CreateAddStoreInGroupOpPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateSingleOpFallbackToPhiPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateShapeOpsFallbackToPhiPass());
  pass_manager->Run(program);
}

void ApplyCinnLowerPass(::pir::Program* program,
                       const std::function<std::shared_ptr<::pir::PassManager>()>& CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  bool has_dynamic_shape = HasDynamicShape(*program);

  bool force_static_shape = false;
  if (auto pass = cinn::dialect::ir::CreateConvertDynamicToStaticDimPass()) {
    pass_manager->AddPass(std::move(pass.value()));
    force_static_shape = true;
  }
  if (auto pass = cinn::dialect::ir::CreateConvertStaticDimToDynamicPass()) {
    pass_manager->AddPass(std::move(pass.value()));
  }

  if (FLAGS_enable_cinn_accuracy_check) {
    pass_manager->AddPass(cinn::dialect::ir::CreateAccuarcyCheckPass());
  }
  if (FLAGS_enable_fusion_fallback) {
    pass_manager->AddPass(cinn::dialect::ir::CreateFusionFallbackPass());
  }
  if (has_dynamic_shape && !force_static_shape) {
    pass_manager->AddPass(cinn::dialect::ir::CreateLowerCinnDyShapeFusionOpPass());
  } else {
    pass_manager->AddPass(cinn::dialect::ir::CreateLowerCinnFusionOpPass());
  }
  pass_manager->AddPass(cinn::dialect::ir::CreateSplitGenerateShapeIntoShapeOpsPass());
  pass_manager->Run(program);
}

void ApplyCinnPass(::pir::Program* program,
                   const std::function<std::shared_ptr<::pir::PassManager>()>& CreatePassManager) {
  const uint32_t origin_num_ops = program->num_ops();
  
  ApplyShapeOptimizationPass(program, CreatePassManager);
  ApplyPdToCinnPass(program, CreatePassManager);
  ApplyCinnPreprocessPass(program, CreatePassManager);
  ApplyBuildGroupOpPass(program, CreatePassManager);
  ApplyGroupOpPass(program, CreatePassManager);
  ApplyDivideGroupOpToFusionOpPass(program, CreatePassManager);
  
  LOG(INFO) << "FusionOp count before lowering : *****[ "
            << GetOpCount<cinn::dialect::FusionOp>(program->module_op())
            << " ]*****";

  auto start = std::chrono::high_resolution_clock::now();
  ApplyCinnLowerPass(program, CreatePassManager);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  
  LOG(INFO) << "Time of lowering and compiling program: ***** [ "
            << duration.count() << " ] ***** seconds.";

  const uint32_t new_num_ops = program->num_ops();
  LOG(INFO) << "Number of ops in the original program is: " << origin_num_ops
            << ", after lowering it becomes: " << new_num_ops
            << ". (compression ratio: " << new_num_ops << "/" << origin_num_ops
            << " = " << static_cast<float>(new_num_ops) / origin_num_ops << ")";
}

}  // namespace cinn::dialect::ir
