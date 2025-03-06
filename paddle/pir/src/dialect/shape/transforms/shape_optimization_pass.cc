// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"

#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/dialect/shape/interface/infer_symbolic_shape/cache_grad_op_symbolic_shape.h"
#include "paddle/pir/include/dialect/shape/interface/infer_symbolic_shape/infer_symbolic_shape.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/dialect/shape/utils/original_attributes_filter.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"

constexpr int vlog_level = 3;

// TODO(zhangbopd): Some op results inferred by InferSymbolicShape is NOT
// consist with the result inferred by InferMeta and should be fixed.
namespace {
bool NeedCheckInferSymbolicWithInferMeta(const std::string& op_name,
                                         size_t result_idx) {
  static std::unordered_map<std::string, std::unordered_set<int>> blacklist{
      {"pd_op.reshape", {1}},
      {"pd_op.empty", {0}},
  };
  const auto& iter = blacklist.find(op_name);
  if (iter == blacklist.end()) return true;
  return iter->second.count(result_idx) == 0;
}
}  // namespace

namespace pir {
namespace {

using PassPipelineRunner =
    std::function<bool(pir::PassManager&, pir::ModuleOp)>;

void PrintProgram(pir::ModuleOp m, std::string msg) {
  ShapeConstraintIRAnalysis& shape_analysis =
      ShapeAnalysisManager::Instance().Get(m.program());
  if (VLOG_IS_ON(vlog_level)) {
    std::cerr << "===================== [ShapeDialect]" << msg
              << " =====================\n"
              << pir::CustomPrintHelper(*m.program(),
                                        shape_analysis.PrintHook())
              << std::endl;
  }
}

std::string PrintOperationWithNoRegion(Operation* op) {
  std::ostringstream os;
  pir::IrPrinter printer(os);

  // print OpResults
  os << "(";
  auto num_op_result = op->num_results();
  for (size_t idx = 0; idx < num_op_result; idx++) {
    os << "%op_" << op->id() << "_" << idx;
    if (idx < num_op_result - 1) os << ", ";
  }
  os << ")";

  os << " =";

  // print OpName & OpId
  os << " \"" << op->name() << "(op_" << op->id() << ")"
     << "\"";

  // print OpOperands
  os << " (";
  auto num_op_operands = op->num_operands();
  for (size_t idx = 0; idx < num_op_operands; idx++) {
    const pir::Value& input = op->operand_source(idx);
    if (input.defining_op()) {
      os << "op_" << input.defining_op()->id() << "_"
         << input.dyn_cast<pir::OpResult>().index();
    } else {
      os << "op_NULL";
    }
    if (idx < num_op_operands - 1) os << ", ";
  }
  os << ")";

  printer.PrintAttributeMap(*op);
  os << " :";

  // PrintOpSignature
  printer.PrintOperandsType(*op);
  os << " -> ";

  printer.PrintOpReturnType(*op);

  return os.str();
}

void PrintOpInfo(pir::Operation* op) {
  if (VLOG_IS_ON(vlog_level)) {
    VLOG(vlog_level) << op->name() << "(op_id: op_" << op->id()
                     << ", num_results=" << op->num_results() << ")"
                     << " has InferSymbolicShapeInterface.\n\t"
                     << PrintOperationWithNoRegion(op);
    if (op->name() == "cinn_op.group") {
      std::cerr << "<<<<<<<<<<<<<<<<<<<< " << op->name() << "(op_id: op_"
                << op->id() << ") START..." << std::endl;
    }
  }
}

void DebugPrintOpInfo(pir::Operation* op,
                      pir::InferSymbolicShapeContext* infer_context = nullptr) {
  std::ostringstream print_stream;
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    const auto& res = op->result(i);
    if (!res || !res.type()) {
      continue;
    }

    print_stream << "\tresult(" << res.dyn_cast<pir::OpResult>().index() << ") "
                 << "ShapeOrData: {";

    if (infer_context != nullptr) {
      print_stream << infer_context->GetShapeOrDataForValue(res);
    }
    print_stream << " }\n";
  }
  if (VLOG_IS_ON(vlog_level)) {
    std::cerr << print_stream.str();
  }
}

void CheckInferSymWithInferMeta(
    pir::Operation* op,
    pir::InferSymbolicShapeContext* infer_context = nullptr) {
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    const auto& res = op->result(i);
    if (!res || !res.type() || res.use_empty()) {
      continue;
    }

    std::ostringstream print_stream;

    // InferMeta funcs of some Ops are not correct now, we don't check them.
    if (!NeedCheckInferSymbolicWithInferMeta(op->name(), i)) continue;

    if (res.type().isa<pir::DenseTensorType>()) {
      const std::vector<int64_t>& infer_meta_shape =
          common::vectorize(res.type().dyn_cast<pir::DenseTensorType>().dims());
      const std::vector<symbol::DimExpr>& infer_sym_shape =
          infer_context->GetShapeOrDataForValue(res).shape();

      if (res.type().dyn_cast<pir::DenseTensorType>().dims().size() == -1) {
        LOG(WARNING) << "Warning: For" << op->name() << " [id:" << op->id()
                     << "] 's result(" << i << ")."
                     << " Rank of infer_meta_shape is dynamic. "
                     << "Received infer_sym_shape is " << infer_sym_shape;
        continue;
      }
      // Check rank.
      if (infer_meta_shape.size() != infer_sym_shape.size()) {
        std::ostringstream print_stream;
        print_stream << "Warning : Check InferSymbolicShape for " << op->name()
                     << " [id:" << op->id() << "] "
                     << " carefully! rank of infer_meta_shape is ["
                     << infer_meta_shape.size()
                     << "], but rank of infer_sym_shape is ["
                     << infer_sym_shape.size() << "].";
        LOG(ERROR) << print_stream.str();
        continue;
      }

      // Check each dim.
      for (size_t i = 0; i < infer_meta_shape.size(); ++i) {
        // Check Static shape should NOT be a symbol.
        if (infer_meta_shape[i] != -1) {
          if (!infer_sym_shape[i].isa<int64_t>()) {
            std::ostringstream print_stream;
            print_stream
                << "Warning : Check InferSymbolicShape for " << op->name()
                << " [id:" << op->id() << "] "
                << " carefully! "
                << "shape[" << i
                << "] of infer_sym_shape should be int64_t NOT a symbol!";
            LOG(ERROR) << print_stream.str();
            continue;
          }

          // Check Static shape should be consist.
          if (infer_meta_shape[i] != infer_sym_shape[i].dyn_cast<int64_t>()) {
            std::ostringstream print_stream;
            print_stream << "Warning : Check InferSymbolicShape for "
                         << op->name() << " [id:" << op->id() << "] "
                         << " carefully! "
                         << "infer_sym_shape is [" << infer_meta_shape[i]
                         << "], but infer_meta_shape is ["
                         << infer_sym_shape[i].dyn_cast<int64_t>() << "].";
            LOG(ERROR) << print_stream.str();
          }
        }
      }
    }
  }
}

class ShapeOptimizationPass : public pir::Pass {
 public:
  ShapeOptimizationPass() : pir::Pass("shape_optimization_pass", 0) {}

  void Run(pir::Operation* op) override {
    VLOG(vlog_level)
        << "===================== ShapeOptimizationPass Run start... "
           "=====================";
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    PADDLE_ENFORCE_NOT_NULL(
        module_op,
        common::errors::InvalidArgument(
            "ShapeOptimizationPass should run on module op."));
    PrintProgram(module_op, "Origin Program");

    ::pir::InferSymExprForAllValues(module_op);
    // Runner is for Canonicalizer.
    PassPipelineRunner runner = [](pir::PassManager& pm, pir::ModuleOp m) {
      pm.EnableIRPrinting();
      return pm.Run(m.program());
    };

    PrintProgram(module_op, "ShapeOptimizationPass Program");
    VLOG(vlog_level) << "===================== ShapeOptimizationPass Run End. "
                        "=====================";
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

void InferSymExprForOp(Operation* op,
                       InferSymbolicShapeContext* infer_context,
                       const InferSymbolicShapeCacheKey& op_infer_cache_key) {
  auto infer_symbolic_shape_interface =
      op->dyn_cast<pir::InferSymbolicShapeInterface>();
  if (infer_symbolic_shape_interface) {
    PrintOpInfo(op);
    PADDLE_ENFORCE_EQ(
        infer_symbolic_shape_interface.InferSymbolicShape(infer_context),
        true,
        common::errors::Fatal("InferSymbolicShape for %s failed.", op->name()));

    if (op->num_results() > 0) {
      // TODO(lanxianghit): deal with the ops which have more than 1
      // ACTUAL results
      pir::shape::SetShapeAttrForOp(
          op, infer_context->GetShapeOrDataForValue(op->result(0)));
    }
  } else {
    const bool is_grad_op = [&]() {
      std::string suffix = "_grad";
      const auto& op_name = op->name();
      if (op_name.size() < suffix.size()) return false;
      return op_name.compare(
                 op_name.size() - suffix.size(), suffix.size(), suffix) == 0;
    }();

    const bool is_special_cached_op = [&]() {
      const auto& op_name = op->name();
      std::vector<std::string> special_cached_ops = {
          "cf.tuple_pop",
      };
      return (std::find(special_cached_ops.begin(),
                        special_cached_ops.end(),
                        op_name) != special_cached_ops.end());
    }();

    if (!is_grad_op)
      LOG(WARNING) << op->name()
                   << " DOES NOT have InferSymbolicShapeInterface!";

    const bool all_outs_static_dims = [&] {
      bool all_static_dims = true;
      for (uint32_t i = 0; i < op->num_results(); ++i) {
        if (IsStaticShape(op->result(i))) {
          continue;
        } else {
          all_static_dims = false;
          break;
        }
      }
      return all_static_dims;
    }();

    if (all_outs_static_dims && !is_special_cached_op) {
      for (uint32_t i = 0; i < op->num_results(); ++i) {
        infer_context->SetSymbolForValueByStaticShape(op->result(i));
      }
    } else {
      if (infer_context->GetOpInferSymbolicShapeCache(op_infer_cache_key)
              .has_value()) {
        std::vector<symbol::ShapeOrDataDimExprs> cached_result_shape_or_data =
            infer_context->GetOpInferSymbolicShapeCache(op_infer_cache_key)
                .value();
        PADDLE_ENFORCE_EQ(cached_result_shape_or_data.size(),
                          op->num_results(),
                          common::errors::Fatal(
                              "Cached number of result %u is not equal to the "
                              "given number of output %u",
                              cached_result_shape_or_data.size(),
                              op->num_results()));
        for (uint32_t i = 0; i < op->num_results(); ++i) {
          infer_context->SetShapeOrDataForValue(op->result(i),
                                                cached_result_shape_or_data[i]);
        }
      } else {
        // risk set
        LOG(WARNING) << "Not found symbolic shape cache for " << op->name()
                     << "[id:" << op->id()
                     << "], op_infer_cache_key is :" << op_infer_cache_key;
        for (uint32_t i = 0; i < op->num_results(); ++i) {
          if (!op->result(i) || !op->result(i).type()) {
            continue;
          }
          infer_context->SetSymbolForValueByStaticShape(op->result(i));
        }
      }
    }
  }
}

static const std::set<std::string> new_symbol_op_set = {
    // cf_op.cc
    "cf.stack_create",
    // op_dialect.cc
    "builtin.parameter",
    // manual_op.cc
    "pd_op.array_read",
    "pd_op.array_to_tensor",
    "pd_op.slice_array",
    "pd_op.slice_array_dense",
    "pd_op.array_pop",
    // control_flow_op.cc
    "pd_op.if",
    "pd_op.while",
    "pd_op.select_input",
    // unary_infer_sym.cc
    "pd_op.as_strided",
    "pd_op.class_center_sample",
    "pd_op.decode_jpeg",
    "pd_op.diag",
    "pd_op.distribute_fpn_proposals",
    "pd_op.nonzero",
    "pd_op.one_hot",
    "pd_op.slice",
    "pd_op.unique",
    "pd_op.unique_consecutive",
    "pd_op.unsqueeze",
    "pd_op.arange",
    "pd_op.data",
    "pd_op.eye",
    "pd_op.read_file",
    // multiary_infer_sym.cc
    "pd_op.batch_norm",
    "pd_op.bicubic_interp",
    "pd_op.assign_pos",
    "pd_op.detection_map",
    "pd_op.flash_attn_varlen_qkvpacked",
    "pd_op.generate_proposals",
    "pd_op.graph_khop_sampler",
    "pd_op.graph_sample_neighbors",
    "pd_op.match_matrix_tensor",
    "pd_op.multiclass_nms3",
    "pd_op.pyramid_hash",
    "pd_op.rnn",
    "pd_op.viterbi_decode",
    "pd_op.warpctc",
    "pd_op.weighted_sample_neighbors",
    // binary_infer_sym.cc
    "pd_op.bincount",
    "pd_op.masked_select",
    "pd_op.matrix_nms",
    "pd_op.repeat_interleave_with_tensor_index",
    "pd_op.segment_pool",
    "pd_op.sequence_mask",
    "pd_op.conv3d_transpose",
    "pd_op.conv2d_transpose",
};

enum class OpType {
  kInt,
  kSym,
  kNegative,
  kAdd,
  kMul,
  kDiv,
  kMax,
  kMin,
  kBroadcast
};

OpType GetOpType(const symbol::DimExpr& expr) {
  auto lambdas = common::Overloaded{
      [](std::int64_t dim_expr) { return OpType::kInt; },
      [](const std::string& dim_expr) { return OpType::kSym; },
      [](const symbol::Negative<symbol::DimExpr>& dim_expr) {
        return OpType::kNegative;
      },
      [](const symbol::Add<symbol::DimExpr>& dim_expr) { return OpType::kAdd; },
      [](const symbol::Mul<symbol::DimExpr>& dim_expr) { return OpType::kMul; },
      [](const symbol::Div<symbol::DimExpr>& dim_expr) { return OpType::kDiv; },
      [](const symbol::Max<symbol::DimExpr>& dim_expr) { return OpType::kMax; },
      [](const symbol::Min<symbol::DimExpr>& dim_expr) { return OpType::kMin; },
      [](const symbol::Broadcast<symbol::DimExpr>& dim_expr) {
        return OpType::kBroadcast;
      },
      [](const auto& dim_expr) {
        PADDLE_THROW(::common::errors::InvalidArgument(
            "Unsupported DimExpr for %s", dim_expr));
        return;
      }};
  return std::visit(lambdas, expr.variant());
}

bool PatternMatch(const symbol::DimExpr& lhs,
                  const symbol::DimExpr& rhs,
                  std::unordered_map<std::string, std::string>* map);

template <template <typename> class Op>
bool ListPatternMatch(const symbol::DimExpr& lhs,
                      const symbol::DimExpr& rhs,
                      std::unordered_map<std::string, std::string>* map) {
  const auto& [lhs_operands] = lhs.Get<Op<symbol::DimExpr>>();
  const auto& [rhs_operands] = rhs.Get<Op<symbol::DimExpr>>();
  if (lhs_operands->size() != rhs_operands->size()) {
    return false;
  }
  for (size_t i = 0; i < lhs_operands->size(); ++i) {
    if (!PatternMatch(lhs_operands->at(i), rhs_operands->at(i), map)) {
      return false;
    }
  }
  return true;
}

template <template <typename> class Op>
bool BinaryPatternMatch(const symbol::DimExpr& lhs,
                        const symbol::DimExpr& rhs,
                        std::unordered_map<std::string, std::string>* map) {
  auto lhs_op = lhs.Get<Op<symbol::DimExpr>>();
  auto rhs_op = rhs.Get<Op<symbol::DimExpr>>();
  return PatternMatch(lhs_op->lhs, rhs_op->lhs, map) &&
         PatternMatch(lhs_op->rhs, rhs_op->rhs, map);
}

template <template <typename> class Op>
bool UnaryPatternMatch(const symbol::DimExpr& lhs,
                       const symbol::DimExpr& rhs,
                       std::unordered_map<std::string, std::string>* map) {
  auto lhs_op = lhs.Get<Op<symbol::DimExpr>>();
  auto rhs_op = rhs.Get<Op<symbol::DimExpr>>();
  return PatternMatch(lhs_op->data, rhs_op->data, map);
}

bool PatternMatch(const symbol::DimExpr& lhs,
                  const symbol::DimExpr& rhs,
                  std::unordered_map<std::string, std::string>* map) {
  OpType lhs_type = GetOpType(lhs);
  OpType rhs_type = GetOpType(rhs);
  if (lhs_type != rhs_type) {
    return false;
  }
  switch (rhs_type) {
    case OpType::kAdd:
      return ListPatternMatch<symbol::Add>(lhs, rhs, map);
    case OpType::kMul:
      return ListPatternMatch<symbol::Mul>(lhs, rhs, map);
    case OpType::kMax:
      return ListPatternMatch<symbol::Max>(lhs, rhs, map);
    case OpType::kMin:
      return ListPatternMatch<symbol::Min>(lhs, rhs, map);
    case OpType::kBroadcast:
      return ListPatternMatch<symbol::Broadcast>(lhs, rhs, map);
    case OpType::kDiv:
      return BinaryPatternMatch<symbol::Div>(lhs, rhs, map);
    case OpType::kNegative:
      return UnaryPatternMatch<symbol::Negative>(lhs, rhs, map);
    case OpType::kInt:
      return lhs == rhs;
    case OpType::kSym:
      auto it = map->find(rhs.Get<std::string>());
      if (it != map->end()) {
        return it->second == lhs.Get<std::string>();
      } else {
        map->insert({rhs.Get<std::string>(), lhs.Get<std::string>()});
        return true;
      }
  }
  return false;
}

enum class ShapeOrDataDimType {
  kTensorShapeOrDataDimExprs,
  kTensorListShapeOrDataDimExprs,
  kRankedTensorArrayShapeOrDataDimExprs,
  kNullShapeOrDataDimExpr
};

ShapeOrDataDimType GetShapeOrDataType(
    const symbol::ShapeOrDataDimExprs& shape_or_data) {
  auto lambdas = common::Overloaded{
      [&](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
        return ShapeOrDataDimType::kTensorShapeOrDataDimExprs;
      },
      [&](const symbol::TensorListShapeOrDataDimExprs& tensor_list) {
        return ShapeOrDataDimType::kTensorListShapeOrDataDimExprs;
      },
      [&](const symbol::RankedTensorArrayShapeOrDataDimExprs& tensor_array) {
        return ShapeOrDataDimType::kRankedTensorArrayShapeOrDataDimExprs;
      },
      [&](const symbol::NullShapeOrDataDimExpr& null_shape_or_data) {
        return ShapeOrDataDimType::kNullShapeOrDataDimExpr;
      }};
  return std::visit(lambdas, shape_or_data.variant());
}

bool ShapeOrDataDimExprsPatternMatch(
    const symbol::ShapeOrDataDimExprs& lhs,
    const symbol::ShapeOrDataDimExprs& rhs,
    std::unordered_map<std::string, std::string>* map) {
  auto lhs_type = GetShapeOrDataType(lhs);
  auto rhs_type = GetShapeOrDataType(rhs);
  if (lhs_type != rhs_type) {
    return false;
  }
  switch (rhs_type) {
    case ShapeOrDataDimType::kTensorShapeOrDataDimExprs:
      if (lhs.dyn_cast<symbol::TensorShapeOrDataDimExprs>().data().has_value() ^
          rhs.dyn_cast<symbol::TensorShapeOrDataDimExprs>()
              .data()
              .has_value()) {
        return false;
      } else if (lhs.dyn_cast<symbol::TensorShapeOrDataDimExprs>()
                     .data()
                     .has_value()) {
        auto lhs_data =
            lhs.dyn_cast<symbol::TensorShapeOrDataDimExprs>().data().value();
        auto rhs_data =
            rhs.dyn_cast<symbol::TensorShapeOrDataDimExprs>().data().value();
        if (lhs_data.size() != rhs_data.size()) {
          return false;
        }
        for (size_t i = 0; i < lhs_data.size(); ++i) {
          if (!PatternMatch(lhs_data.at(i), rhs_data.at(i), map)) {
            return false;
          }
        }
      } else {
        auto lhs_shape =
            lhs.dyn_cast<symbol::TensorShapeOrDataDimExprs>().shape();
        auto rhs_shape =
            rhs.dyn_cast<symbol::TensorShapeOrDataDimExprs>().shape();
        if (lhs_shape.size() != rhs_shape.size()) {
          return false;
        }
        for (size_t i = 0; i < lhs_shape.size(); ++i) {
          if (!PatternMatch(lhs_shape.at(i), rhs_shape.at(i), map)) {
            return false;
          }
        }
      }
      return true;
    case ShapeOrDataDimType::kTensorListShapeOrDataDimExprs:
    case ShapeOrDataDimType::kRankedTensorArrayShapeOrDataDimExprs:
      // TODO(ooooo): support list and array pattern match if nessaary.
      return false;
    case ShapeOrDataDimType::kNullShapeOrDataDimExpr:
      return true;
  }
  return false;
}

void CacheForwardOpSymbolicShape(
    Operation* op,
    InferSymbolicShapeContext* infer_context,
    const InferSymbolicShapeCacheKey& op_infer_cache_key) {
  std::vector<symbol::ShapeOrDataDimExprs> result_shape_or_data;
  const auto& CheckInferSymbolicShapeCacheConsistency =
      [&](const InferSymbolicShapeCacheValue& infer_result,
          const InferSymbolicShapeCacheValue& cache_result) {
        if (infer_result.size() != cache_result.size()) {
          LOG(WARNING) << "cached shape is not consistent with real shape";
        } else {
          std::unordered_map<std::string, std::string> map = {};
          for (uint32_t i = 0; i < cache_result.size(); ++i) {
            if (infer_result[i] != cache_result[i]) {
              if (new_symbol_op_set.find(op->name()) !=
                      new_symbol_op_set.end() &&
                  ShapeOrDataDimExprsPatternMatch(
                      infer_result[i], cache_result[i], &map)) {
                continue;
              }
              LOG(WARNING) << "cached shape is not consistent with real shape";
              VLOG(3) << "InferSymbolicShapeCacheKey is: "
                      << op_infer_cache_key;
              VLOG(3) << "cached shape is: " << cache_result[i];
              VLOG(3) << "real shape is: " << infer_result[i];
            }
          }
        }
      };
  for (const auto& result : op->results()) {
    result_shape_or_data.emplace_back(
        infer_context->GetShapeOrDataForValue(result));
  }
  if (infer_context->GetOpInferSymbolicShapeCache(op_infer_cache_key)
          .has_value()) {
    std::vector<symbol::ShapeOrDataDimExprs> cached_result_shape_or_data =
        infer_context->GetOpInferSymbolicShapeCache(op_infer_cache_key).value();
    // TODO(Hongqing-work): delete check and only set cache for op without
    // InferSymbolicShapeInterface after fixing all warnings.
    CheckInferSymbolicShapeCacheConsistency(result_shape_or_data,
                                            cached_result_shape_or_data);
  } else {
    infer_context->SetOpInferSymbolicShapeCache(op_infer_cache_key,
                                                result_shape_or_data);
  }
}

void CacheBackwardOpSymbolicShape(Operation* op,
                                  InferSymbolicShapeContext* infer_context) {
  auto cache_grad_op_symbolic_shape_interface =
      op->dyn_cast<pir::CacheGradOpSymbolicShapeInterface>();
  if (cache_grad_op_symbolic_shape_interface) {
    VLOG(3) << "CacheBackwardOpSymbolicShape for: " << op->name();
    cache_grad_op_symbolic_shape_interface.CacheGradOpSymbolicShape(
        infer_context);
  }
}

void InferSymExprForBlock(const Block& block,
                          InferSymbolicShapeContext* infer_context) {
  for (auto& op : block) {
    std::vector<symbol::ShapeOrDataDimExprs> input_shape_or_data;
    for (auto& input : op.operands_source()) {
      input_shape_or_data.emplace_back(
          infer_context->GetShapeOrDataForValue(input));
    }
    InferSymbolicShapeCacheKey op_infer_cache_key(
        op.name(),
        input_shape_or_data,
        GetOrderedOriginalAttributes(op.name(), op.attributes()));
    InferSymExprForOp(&op, infer_context, op_infer_cache_key);
    CacheForwardOpSymbolicShape(&op, infer_context, op_infer_cache_key);
    CacheBackwardOpSymbolicShape(&op, infer_context);
    DebugPrintOpInfo(&op, infer_context);
    CheckInferSymWithInferMeta(&op, infer_context);
  }
}

void InferSymExprForAllValues(ModuleOp module_op) {
  ShapeConstraintIRAnalysis& shape_analysis =
      ShapeAnalysisManager::Instance().Get(module_op.program());
  auto* infer_context = shape_analysis.MutInferSymbolicShapeContext();

  // hold the kwargs symbol shape info to avoid be cleared when call init.
  const std::unordered_map<pir::Value, symbol::ShapeOrDataDimExprs>
      symbol_shape_map = [&] {
        std::unordered_map<pir::Value, symbol::ShapeOrDataDimExprs>
            symbol_shape_map;
        for (const auto& [_, value] : module_op.block().kwargs()) {
          if (!infer_context->HasShapeOrDataForValue(value)) {
            infer_context->SetSymbolForValueByStaticShape(value);
          }
          symbol_shape_map.emplace(
              value, infer_context->GetShapeOrDataForValue(value));
        }
        return symbol_shape_map;
      }();

  shape_analysis.InitInferContext();
  // init the kwarg symbol shape info
  for (const auto& kv : symbol_shape_map) {
    infer_context->SetShapeOrDataForValue(kv.first, kv.second);
  }

  InferSymExprForBlock(module_op.block(), infer_context);
}

std::unique_ptr<Pass> CreateShapeOptimizationPass() {
  return std::make_unique<ShapeOptimizationPass>();
}

}  // namespace pir

namespace pir::shape {

bool HasDynamicShape(const pir::Program& program) {
  for (const auto& op : *program.block()) {
    if (op.isa<pir::CombineOp>()) {
      continue;
    }
    for (uint32_t i = 0; i < op.num_results(); ++i) {
      if (op.result(i) && op.result(i).type()) {
        auto shape_type =
            op.result(i).type().dyn_cast<pir::ShapedTypeInterface>();
        if (shape_type && shape_type.IsDynamicShape()) {
          VLOG(vlog_level) << "###### HasDynamicShape == true";
          return true;
        }
      }
    }
  }
  VLOG(vlog_level) << "###### HasDynamicShape == false";
  return false;
}

void AddShapeOptimizationPass(
    std::shared_ptr<pir::PassManager>& pass_manager,  // NOLINT
    pir::Program& program) {                          // NOLINT
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  pass_manager->AddPass(pir::CreateShapeOptimizationPass());
}

}  // namespace pir::shape

// REGISTER_IR_PASS(shape_optimization_pass, pir::ShapeOptimizationPass);
