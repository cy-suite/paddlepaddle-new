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

#pragma once
#include "paddle/cinn/optim/if_fold_pass.h"
#include <vector>
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/ir/utils/ir_copy.h"

namespace cinn {
namespace optim {
using ir::stmt::IfThenElse;
using ir::stmt::StmtRef;

bool JudgeIfStmt(const StmtRef& stmt) {
  if (!stmt.isa<IfThenElse>()) return false;
  auto if_stmt = stmt.as<IfThenElse>();
  auto cond = if_stmt->condition().As<ir::EQ>();
  if (if_stmt->false_case()->stmts().size() != 0) return false;
  if (if_stmt->true_case()->stmts().size() != 1) return false;
  if (!cond) return false;
  if (!cond->b().is_constant()) return false;
  if (cond->b().get_constant() != 0) return false;
  return true;
}

bool IsInnerIfWithEqCond(const StmtRef& stmt) {
  if (stmt.isa<IfThenElse>()) {
    auto if_stmt = stmt.as<IfThenElse>();
    if (auto eq = if_stmt->condition().As<ir::EQ>()) {
      if (eq->b().is_constant() && eq->b().get_constant() == 0) {
        return true;
      }
    }
  }
  return false;
}
void AppendContinuousIfCond(const StmtRef& stmt,
                            std::vector<ir::IndexExpr>* cond_vec,
                            StmtRef* inner_op) {
  if (!JudgeIfStmt(stmt)) {
    if (IsInnerIfWithEqCond(stmt)) {
      auto eq_lhs = stmt.as<IfThenElse>()->condition().As<ir::EQ>()->a();
      if (eq_lhs.is_index())
        cond_vec->push_back(common::ChangeSeqOfDivMod(
            ir::ir_utils::IRCopy(eq_lhs).as_index().Normalize()));
    }
    *inner_op = stmt;
    return;
  }

  auto if_stmt = stmt.as<IfThenElse>();
  auto eq_lhs = if_stmt->condition().As<ir::EQ>()->a();
  if (eq_lhs.is_index())
    cond_vec->push_back(common::ChangeSeqOfDivMod(
        ir::ir_utils::IRCopy(eq_lhs).as_index().Normalize()));
  AppendContinuousIfCond(
      if_stmt->true_case()->stmts().at(0), cond_vec, inner_op);
}

LogicalResult IfFoldPass::Run(StmtRef stmt) {
  if (!JudgeIfStmt(stmt)) return LogicalResult::success();

  std::vector<ir::IndexExpr> cond_vec;
  StmtRef inner_op;

  AppendContinuousIfCond(stmt, &cond_vec, &inner_op);

  ir::IndexExpr expr(0);
  int32_t min_len = INT32_MAX;
  VLOG(6) << "-------------cond_vec start--------------";
  for (auto v : cond_vec) {
    VLOG(6) << "v: " << v;
    min_len = std::min(v.length(), min_len);
    if (v.node_type() == ir::IrNodeTy::Div) {
      expr = expr + v * v.operand(1);
    } else {
      expr = expr + v;
    }
  }
  VLOG(6) << "-------------cond_vec end----------------";
  expr = common::MergeMulMod(expr);

  if (expr != ir::IndexExpr(0) && expr.length() < min_len &&
      inner_op.defined()) {
    VLOG(6) << "old stmt: " << stmt;
    auto stmt_if = stmt.as<IfThenElse>();
    stmt_if->set_condition(ir::EQ::Make(expr, ir::IndexExpr(0)));
    if (IsInnerIfWithEqCond(inner_op)) {
      stmt_if->set_true_case(inner_op.as<IfThenElse>()->true_case());
      stmt_if->set_false_case(inner_op.as<IfThenElse>()->false_case());
    } else {
      stmt_if->set_true_case(
          ir::stmt::BlockRef(std::vector<StmtRef>{inner_op}));
    }
    VLOG(6) << "new stmt: " << stmt;
  }

  return LogicalResult::success();
}

std::unique_ptr<StmtPass> CreateIfFoldPass() {
  return std::make_unique<IfFoldPass>();
}
}  // namespace optim
}  // namespace cinn
