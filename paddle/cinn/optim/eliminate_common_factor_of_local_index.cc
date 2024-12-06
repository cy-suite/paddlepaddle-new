// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/eliminate_common_factor_of_local_index.h"

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {
namespace {

int ExtractMulNumberFromExpr(const ir::Expr& expr) {
  ir::Expr simplied_expr = cinn::common::AutoSimplify(expr);
  if (simplied_expr.is_constant()) {
    return static_cast<int>(simplied_expr.get_constant());
  } else if (expr.As<ir::Mul>()) {
    auto mul = expr.As<ir::Mul>();
    return ExtractMulNumberFromExpr(mul->a()) *
           ExtractMulNumberFromExpr(mul->b());
  } else {
    VLOG(6) << "Not supported for calculating gcd, expr = " << expr;
    return 1;
  }
}

int ExtractAddNumberFromExpr(const ir::Expr& expr) {
  ir::Expr simplied_expr = cinn::common::AutoSimplify(expr);
  if (simplied_expr.is_constant()) {
    return static_cast<int>(simplied_expr.get_constant());
  } else if (expr.As<ir::Add>()) {
    auto add = expr.As<ir::Add>();
    return ExtractAddNumberFromExpr(add->a()) +
           ExtractAddNumberFromExpr(add->b());
  } else {
    VLOG(6) << "Not supported for calculating offset, expr = " << expr;
    return 0;
  }
}

ir::Expr ExtractSymbolicFromExpr(const ir::Expr& expr) {
  ir::Expr simplied_expr = cinn::common::AutoSimplify(expr);
  if (simplied_expr.is_constant()) {
    return ir::Expr(0);
  } else if (expr.As<ir::_Var_>()) {
    auto var = expr.As<ir::_Var_>();
    if (var->is_symbolic_constant) {
      VLOG(6) << "Extract symbolic constant, name = " << var->name;
      return ir::ir_utils::IRCopy(expr);
    }
    return ir::Expr(0);
  } else {
    VLOG(6) << "Not supported for calculating symbolic, expr = " << expr;
    return ir::Expr(0);
  }
  PADDLE_THROW(::common::errors::Fatal(
      "Dead code. Fail to extract symbolic from expression."));
}

class Gcd {};
class Offset {};
class Symbolic {};

template <typename Op>
struct CommonFactorTrait;

template <>
struct CommonFactorTrait<Gcd> {
  static const ir::Expr unit;

  static ir::Expr Calculate(const ir::Expr& expr1, const ir::Expr& expr2) {
    int num1 = ExtractMulNumberFromExpr(expr1);
    int num2 = ExtractMulNumberFromExpr(expr2);
    if (num1 == 0 && num2 == 0) {
      return ir::Expr(1);
    }
    return ir::Expr(std::gcd(num1, num2));
  }

  static ir::Expr Simplify(const ir::Expr& expr, const ir::Expr& factor) {
    if (factor != unit) {
      return cinn::common::AutoSimplify(ir::Div::Make(expr, factor));
    }
    return expr;
  }
};

const ir::Expr CommonFactorTrait<Gcd>::unit = ir::Expr(1);

template <>
struct CommonFactorTrait<Offset> {
  static const ir::Expr unit;

  static ir::Expr Calculate(const ir::Expr& expr1, const ir::Expr& expr2) {
    return ir::Expr(std::min(ExtractAddNumberFromExpr(expr1),
                             ExtractAddNumberFromExpr(expr2)));
  }

  static ir::Expr Simplify(const ir::Expr& expr, const ir::Expr& factor) {
    if (factor != unit) {
      return cinn::common::AutoSimplify(ir::Sub::Make(expr, factor));
    }
    return expr;
  }
};

const ir::Expr CommonFactorTrait<Offset>::unit = ir::Expr(0);

template <>
struct CommonFactorTrait<Symbolic> {
  static const ir::Expr unit;

  static ir::Expr Calculate(const ir::Expr& expr1, const ir::Expr& expr2) {
    auto IsSymbolicNotEqual = [&](const ir::Expr& expr1,
                                  const ir::Expr& expr2) -> bool {
      return cinn::common::AutoSimplify(
                 ir::Sub::Make(ExtractSymbolicFromExpr(expr1),
                               ExtractSymbolicFromExpr(expr2))) != ir::Expr(0);
    };
    if (IsSymbolicNotEqual(expr1, expr2)) {
      return ir::Expr(0);
    }
    return ExtractSymbolicFromExpr(expr1);
  }

  static ir::Expr Simplify(const ir::Expr& expr, const ir::Expr& factor) {
    if (factor != unit) {
      return cinn::common::AutoSimplify(ir::Sub::Make(expr, factor));
    }
    return expr;
  }
};

const ir::Expr CommonFactorTrait<Symbolic>::unit = ir::Expr(0);

template <typename DoEachT>
void VisitEachRowExpr(const std::vector<std::vector<ir::Expr>>& indexes,
                      std::size_t var_idx,
                      DoEachT&& DoEach) {
  for (std::size_t i = 0; i < indexes.size(); ++i) {
    DoEach(indexes[i][var_idx]);
  }
}

template <typename Op>
std::vector<ir::Expr> CalculateIndicesCommonFactor(
    const std::vector<std::vector<ir::Expr>>& indices_list) {
  std::size_t var_index_size = indices_list[0].size();
  std::vector<ir::Expr> common_factors;
  for (std::size_t var_idx = 0; var_idx < var_index_size; ++var_idx) {
    std::optional<ir::Expr> common_factor;
    VisitEachRowExpr(indices_list, var_idx, [&](const ir::Expr& expr) {
      if (common_factor.has_value()) {
        common_factor =
            CommonFactorTrait<Op>::Calculate(common_factor.value(), expr);
      } else {
        common_factor = expr;
      }
    });
    common_factors.push_back(common_factor.value());
  }
  return common_factors;
}

template <typename Op>
void EliminateCommonFactorHelper(
    std::vector<std::vector<ir::Expr>>* indices_list) {
  std::vector<ir::Expr> common_factors =
      CalculateIndicesCommonFactor<Op>(*indices_list);
  for (auto& indices : *indices_list) {
    for (int dim = 0; dim < indices.size(); ++dim) {
      indices[dim] =
          CommonFactorTrait<Op>::Simplify(indices[dim], common_factors[dim]);
    }
  }
}

struct CollectLocalBufferToIndices : public ir::IRMutator<> {
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node = expr->As<ir::Store>();
    Collect(node, &node->indices);
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load* op, Expr* expr) override {
    auto* node = expr->As<ir::Load>();
    Collect(node, &node->indices);
    ir::IRMutator<>::Visit(op, expr);
  }

  void Collect(ir::LoadStoreAddrMnger* mgr, std::vector<ir::Expr>* indices) {
    if (!mgr->is_addr_tensor()) {
      return;
    }
    auto& buffer = mgr->tensor.as_tensor()->buffer;
    if (buffer->memory_type != ir::MemoryType::GPULocal) {
      return;
    }
    auto [old_iter, is_first_insertion] =
        buffers.emplace(buffer->name, buffer.operator->());
    // We expect that all buffers with the same name point to the same _Buffer_
    // in a function. However, if some transforms didn't follow this rule, we
    // only keep the first _Buffer_ and make all the other buffers point to the
    // first _Buffer_.
    if (!is_first_insertion && buffer.operator->() != old_iter->second) {
      buffer = ir::Buffer(old_iter->second);
    }
    buffer_indices[buffer->name].emplace_back(indices);
  }

 public:
  std::unordered_map<std::string, ir::_Buffer_*> buffers;
  std::unordered_map<std::string, std::vector<std::vector<ir::Expr>*>>
      buffer_indices;
};

std::optional<ir::Var> GetSingleVar(const ir::Expr& expr) {
  std::vector<ir::Expr> res =
      ir::ir_utils::CollectIRNodesInOrder(expr, [&](const ir::Expr* x) {
        return x->is_var() && !x->as_var()->is_symbolic_constant;
      });
  PADDLE_ENFORCE_LE(
      res.size(),
      1UL,
      ::common::errors::PreconditionNotMet(
          "There can be at most one loop variable at each dim of local "
          "buffer's indices. Did you allocate more than one spatial inner loop "
          "or more than one reduce inner loop?"));
  if (res.empty()) {
    return std::nullopt;
  }
  return res[0].as_var_ref();
}

std::vector<std::vector<ir::Expr>> FuseDimsWithCommonVar(
    const std::vector<ir::Expr>& shape,
    const std::vector<std::vector<ir::Expr>>& indices_list) {
  std::vector<bool> is_dim_fused(shape.size());
  std::vector<std::vector<ir::Expr>> new_indices_list(indices_list.size());

  // Find the dims that contain the same variable as what the start_dim
  // contains. The result is unified for all indices.
  const auto FindDimsToFuse = [&](int start_dim) {
    std::set<int> dims_to_fuse = {start_dim};
    for (auto& indices : indices_list) {
      std::optional<ir::Var> opt_var = GetSingleVar(indices[start_dim]);
      if (!opt_var.has_value()) {
        continue;
      }
      // Only search dims after `start_dim` because previous dims must have
      // been fused.
      for (int dim = start_dim + 1; dim < shape.size(); ++dim) {
        if (is_dim_fused[dim]) {
          continue;
        }
        std::optional<ir::Var> opt_other_var = GetSingleVar(indices[dim]);
        if (opt_other_var.has_value() &&
            opt_other_var.value() == opt_var.value()) {
          dims_to_fuse.insert(dim);
          is_dim_fused[dim] = true;
        }
      }
    }
    return dims_to_fuse;
  };

  // Fuse these `dims_to_fuse` in each indices, and append the fused dims to
  // the new_indices_list.
  const auto ApplyDimFusion = [&](const std::set<int>& dims_to_fuse) {
    std::vector<ir::Expr> tmp_shape;
    for (int dim : dims_to_fuse) {
      tmp_shape.emplace_back(shape[dim]);
    }
    for (size_t i = 0; i < indices_list.size(); ++i) {
      std::vector<ir::Expr> tmp_indices;
      for (int dim : dims_to_fuse) {
        tmp_indices.emplace_back(indices_list[i][dim]);
      }
      ir::Expr index = common::IndiceToAbsOffset(tmp_shape, tmp_indices);
      new_indices_list[i].emplace_back(index);
    }
  };

  for (int dim = 0; dim < shape.size(); ++dim) {
    if (!is_dim_fused[dim]) {
      is_dim_fused[dim] = true;
      std::set<int> dims_to_fuse = FindDimsToFuse(dim);
      ApplyDimFusion(dims_to_fuse);
    }
  }

  return new_indices_list;
}

}  // namespace

std::string Print(const std::vector<std::vector<ir::Expr>>& indices_list) {
  std::stringstream ss;
  for (auto& indices : indices_list) {
    ss << "\n" << utils::Join(indices, ", ");
  }
  return ss.str();
}

bool Verify(const std::vector<std::vector<ir::Expr>>& indices_list) {
  for (auto& indices : indices_list) {
    for (auto& expr : indices) {
      if (expr != ir::Expr(0) && !expr.as_var()) {
        return true;
      }
    }
  }
  return false;
}

void EliminateCommonFactorOfLocalIndex(ir::Expr* expr) {
  // Step 1.
  CollectLocalBufferToIndices indices_collector;
  indices_collector(expr);

  for (auto& [name, indices_list] : indices_collector.buffer_indices) {
    PADDLE_ENFORCE_GE(
        indices_list.size(),
        2,
        ::common::errors::PreconditionNotMet(
            "Size of the indices_list of local buffer [%s] must be >=2, "
            "because there should be at least one store and one load of it.",
            name));

    // Step 2.
    ir::_Buffer_* buffer = indices_collector.buffers[name];
    std::vector<std::vector<ir::Expr>> indices_list_copy;
    for (auto* indices : indices_list) {
      indices_list_copy.emplace_back(std::move(*indices));
    }
    VLOG(4) << "ECF begin: " << name << Print(indices_list_copy);
    std::vector<std::vector<ir::Expr>> new_indices_list =
        FuseDimsWithCommonVar(buffer->shape, indices_list_copy);
    VLOG(4) << "ECF fused: " << name << Print(new_indices_list);

    // Step 3.
    EliminateCommonFactorHelper<Gcd>(&new_indices_list);
    EliminateCommonFactorHelper<Offset>(&new_indices_list);
    EliminateCommonFactorHelper<Symbolic>(&new_indices_list);
    VLOG(4) << "ECF simpl: " << name << Print(new_indices_list);

    if (Verify(new_indices_list)) {
      VLOG(0) << "ECF Failed: " << name << Print(new_indices_list);
      PADDLE_THROW(::common::errors::PreconditionNotMet(
          "Failed to eliminate common factors!"));
    }

    // Step 4.
    // The buffer's shape here is not important, because this shape will be
    // finally determined in the next pass. Just assign some placeholders.
    size_t new_dim_size = new_indices_list[0].size();
    buffer->shape.assign(new_dim_size, ir::Expr(1));
    for (size_t i = 0; i < indices_list.size(); ++i) {
      *indices_list[i] = std::move(new_indices_list[i]);
    }
  }
}

}  // namespace optim
}  // namespace cinn
