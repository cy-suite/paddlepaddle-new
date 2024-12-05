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

#include <string>

#include <functional>
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/pass/pass_adaptor.h"

namespace cinn {
namespace optim {
namespace detail {

#define MAX_REWRITE_LIMIT 10

template <typename PassT>
void PassAdaptor<PassT>::RunPipeline(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<PassT>>& passes,
    bool need_converge) {
  const auto& RunUntilConverge = [&]() {
    bool over_rewrite_limit = true;
    for (int i = 0; i < MAX_REWRITE_LIMIT; ++i) {
      bool changed = RunWithoutConverge(func, passes);
      if (!changed) {
        over_rewrite_limit = false;
        break;
      }
    }
    if (over_rewrite_limit) {
      LOG(WARNING) << "Reach max rewrite limit, pass may not converge.";
    }
  };

  // TODO(Hongqing-work): remove convert after update all the backend passes.
  func->body_block = ir::ConvertExprBlockToStmtBlock(func->body);
  if (need_converge) {
    RunUntilConverge();
  } else {
    RunWithoutConverge(func, passes);
  }
  func->body = ir::ConvertStmtBlockToExprBlock(func->body_block);
}

template void PassAdaptor<FuncPass>::RunPipeline(
    ir::LoweredFunc, const std::vector<std::unique_ptr<FuncPass>>&, bool);

template void PassAdaptor<BlockPass>::RunPipeline(
    ir::LoweredFunc, const std::vector<std::unique_ptr<BlockPass>>&, bool);

template void PassAdaptor<StmtPass>::RunPipeline(
    ir::LoweredFunc, const std::vector<std::unique_ptr<StmtPass>>&, bool);

template void PassAdaptor<ExprPass>::RunPipeline(
    ir::LoweredFunc, const std::vector<std::unique_ptr<ExprPass>>&, bool);

namespace {
template <typename PassT, typename IRScopeRefT>
void RunPasses(const std::vector<std::unique_ptr<PassT>>& passes,
               IRScopeRefT scope,
               bool* changed) {
  for (auto& pass : passes) {
    *changed = pass->Run(scope) || *changed;
  }
}
}  // namespace

bool FuncPassAdaptor::RunWithoutConverge(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<FuncPass>>& passes) {
  bool changed = false;
  RunPasses(passes, func, &changed);
  return changed;
}

namespace {
bool RunPassesOnBlock(ir::stmt::BlockRef block,
                      const std::vector<std::unique_ptr<BlockPass>>& passes) {
  bool changed = false;
  std::vector<ir::stmt::StmtRef> new_stmts = block->stmts();
  for (ir::stmt::StmtRef inner_stmt : new_stmts) {
    std::vector<ir::stmt::BlockRef> inner_blocks = inner_stmt->block_fields();
    for (ir::stmt::BlockRef inner_block : inner_blocks) {
      changed = RunPassesOnBlock(inner_block, passes) || changed;
    }
    inner_stmt->set_block_fields(inner_blocks);
  }
  block->set_stmts(new_stmts);
  RunPasses(passes, block, &changed);
  return changed;
}
}  // namespace

bool FuncToBlockPassAdaptor::RunWithoutConverge(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<BlockPass>>& passes) {
  ir::stmt::BlockRef func_block = func->body_block;
  bool changed = RunPassesOnBlock(func_block, passes);
  func->body_block = func_block;
  return changed;
}

bool FuncToStmtPassAdaptor::RunWithoutConverge(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<StmtPass>>& passes) {
  ir::stmt::BlockRef func_block = func->body_block;
  bool changed = false;
  ir::stmt::Mutate(
      func_block,
      [&](ir::stmt::StmtRef stmt) {},
      [&](ir::stmt::StmtRef stmt) { RunPasses(passes, stmt, &changed); });
  return changed;
}

namespace {
using ExprMutateFuncT = std::function<bool(ir::Expr expr)>;
class StmtToExprPassAdaptor : public StmtPass {
 public:
  explicit StmtToExprPassAdaptor(const ExprMutateFuncT& func)
      : StmtPass("stmt to expr pass adaptor"), mutator_(func) {}
  virtual bool Run(ir::stmt::StmtRef stmt) {
    mutator_.VisitStmt(stmt);
    return false;
  }

 private:
  class LocalExprMutator : public ir::stmt::StmtMutator<> {
   public:
    explicit LocalExprMutator(const ExprMutateFuncT& expr_mutator)
        : expr_mutator_(expr_mutator) {}

    void VisitStmt(ir::stmt::StmtRef stmt) override {
      ir::stmt::StmtMutator<>::VisitStmt(stmt);
    }

   private:
    ExprMutateFuncT expr_mutator_;
#define __(stmt__) void VisitStmt(ir::stmt::stmt__ stmt) override;
    NODETY_FORALL_STMT(__)
#undef __
  };
  LocalExprMutator mutator_;
};

void StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(ir::stmt::Let stmt) {
  ir::Expr symbol = stmt->symbol();
  ir::Expr body = stmt->body();
  expr_mutator_(symbol);
  if (body.defined()) {
    expr_mutator_(body);
  }
  stmt->set_symbol(symbol);
  stmt->set_body(body);
}

void StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(ir::stmt::Store stmt) {
  ir::Expr value = stmt->value();
  ir::Expr tensor = stmt->tensor();
  std::vector<ir::Expr> indices = stmt->indices();
  expr_mutator_(value);
  expr_mutator_(tensor);
  for (ir::Expr indice : indices) {
    expr_mutator_(indice);
  }
  stmt->set_value(value);
  stmt->set_tensor(tensor);
  stmt->set_indices(indices);
}

void StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(ir::stmt::Alloc stmt) {
  std::vector<ir::Expr> extents = stmt->extents();
  ir::Expr condition = stmt->condition();
  ir::Expr body = stmt->body();
  for (ir::Expr extent : extents) {
    expr_mutator_(extent);
  }
  if (condition.defined()) {
    expr_mutator_(condition);
  }
  if (body.defined()) {
    expr_mutator_(body);
  }
  stmt->set_extents(extents);
  stmt->set_condition(condition);
  stmt->set_body(body);
}

void StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(ir::stmt::Free stmt) {
  ir::Expr destination = stmt->destination();
  expr_mutator_(destination);
  stmt->set_destination(destination);
}

void StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(
    ir::stmt::IfThenElse stmt) {
  ir::Expr condition = stmt->condition();
  expr_mutator_(condition);
  stmt->set_condition(condition);
}

void StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(ir::stmt::For stmt) {
  ir::Expr min = stmt->min();
  ir::Expr extent = stmt->extent();
  expr_mutator_(min);
  expr_mutator_(extent);
  stmt->set_min(min);
  stmt->set_extent(extent);
}

void StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(
    ir::stmt::Schedule stmt) {
  std::vector<ir::Var> iter_vars = stmt->iter_vars();
  std::vector<ir::Expr> iter_values = stmt->iter_values();
  std::vector<ir::Expr> read_buffers = stmt->read_buffers();
  std::vector<ir::Expr> write_buffers = stmt->write_buffers();

  for (ir::Var iter_var : iter_vars) {
    if (iter_var->lower_bound.defined()) {
      expr_mutator_(iter_var->lower_bound);
    }
    if (iter_var->upper_bound.defined()) {
      expr_mutator_(iter_var->upper_bound);
    }
  }
  for (ir::Expr iter_value : iter_values) {
    expr_mutator_(iter_value);
  }
  for (ir::Expr read_buffer : read_buffers) {
    expr_mutator_(read_buffer);
  }
  for (ir::Expr write_buffer : write_buffers) {
    expr_mutator_(write_buffer);
  }

  stmt->set_iter_vars(iter_vars);
  stmt->set_iter_values(iter_values);
  stmt->set_read_buffers(read_buffers);
  stmt->set_write_buffers(write_buffers);
}

void StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(
    ir::stmt::Evaluate stmt) {
  ir::Expr value = stmt->value();
  expr_mutator_(value);
  stmt->set_value(value);
}
}  // namespace

bool FuncToExprPassAdaptor::RunWithoutConverge(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<ExprPass>>& passes) {
  bool changed = false;
  StmtToExprPassAdaptor stmt_to_expr_addaptor([&](ir::Expr expr) {
    RunPasses(passes, expr, &changed);
    return false;
  });
  ir::stmt::BlockRef func_block = func->body_block;
  ir::stmt::Mutate(
      func_block,
      [&](ir::stmt::StmtRef stmt) {},
      [&](ir::stmt::StmtRef stmt) { stmt_to_expr_addaptor.Run(stmt); });
  // TODO(Hongqing-work): modify this after expr mutator can return change
  // status.
  return false;
}

}  // namespace detail
}  // namespace optim
}  // namespace cinn
