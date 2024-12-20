// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/replace_var_with_expr.h"

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/replace_const_param_to_integer.h"

namespace cinn {
namespace optim {

struct ReplaceVarWithExprStmtMutator : public ir::stmt::StmtMutator<> {
  ReplaceVarWithExprStmtMutator(const Var& var,
                                const Expr& expr,
                                const std::string& tensor_name)
      : var_(var), expr_(expr), tensor_name_(tensor_name) {}

  void operator()(ir::stmt::StmtRef stmt) {
    VLOG(4) << "Enter ReplaceVarWithExprStmtMutator::operator()";
    if (tensor_name_.empty()) visit_all_ = true;
    ir::stmt::StmtMutator<>::VisitStmt(stmt);
    VLOG(4) << "Exit ReplaceVarWithExprStmtMutator::operator()";
  }

  void operator()(ir::stmt::BlockRef block) {
    VLOG(4) << "Enter ReplaceVarWithExprStmtMutator::operator()";
    if (tensor_name_.empty()) visit_all_ = true;
    ir::stmt::StmtMutator<>::VisitBlock(block);
    VLOG(4) << "Exit ReplaceVarWithExprStmtMutator::operator()";
  }

 private:
  bool ShouldReplaceExpr(const Expr& expr) {
    if (!expr.is_var()) {
      return false;
    }
    if (expr.as_var()->name == var_->name && (do_replace_ || visit_all_)) {
      return true;
    }
    return false;
  }

  void VisitExpr(ir::Expr* expr) {
    VLOG(4) << "Enter VisitExpr(ir::Expr* expr)";
    if (expr->is_var()) {
      VLOG(4) << "Hit var: " << expr->as_var()->name;
      if (expr->as_var()->name == var_->name && (do_replace_ || visit_all_)) {
        auto copied = ir::ir_utils::IRCopy(expr_);
        *expr = copied;
      }
      return;
    }
    for (auto field : (*expr)->expr_fields()) {
      VLOG(4) << "field: " << *field;
      VisitExpr(field);
    }
    VLOG(4) << "Exit VisitExpr(ir::Expr* expr)";
  }

  void VisitStmt(ir::stmt::Let stmt) override {
    VLOG(4) << "Enter VisitStmt(ir::stmt::Let stmt)";
    Expr symbol = stmt->symbol();
    VisitExpr(&symbol);
    stmt->set_symbol(symbol);
    if (stmt->body().defined()) {
      Expr body = stmt->body();
      VisitExpr(&body);
      stmt->set_body(body);
    }
    VLOG(4) << "Exit VisitStmt(ir::stmt::Let stmt)";
  }

  void VisitStmt(ir::stmt::Store stmt) override {
    VLOG(4) << "Enter VisitStmt(ir::stmt::Store stmt)";
    auto* tensor = stmt->tensor().as_tensor();
    if (tensor && tensor->name == tensor_name_) {
      do_replace_ = true;
    } else {
      do_replace_ = false;
    }
    std::vector<Expr> new_indices = stmt->indices();
    for (Expr& index : new_indices) {
      VLOG(4) << "index: " << index;
      VisitExpr(&index);
    }
    stmt->set_indices(new_indices);
    do_replace_ = false;
    VLOG(4) << "stmt->tensor(): " << stmt->tensor();
    Expr tensor_expr = stmt->tensor();
    VisitExpr(&tensor_expr);
    stmt->set_tensor(tensor_expr);

    Expr value = stmt->value();
    VisitExpr(&value);
    stmt->set_value(value);
    VLOG(4) << "Exit VisitStmt(ir::stmt::Store stmt)";
  }

  void VisitStmt(ir::stmt::For stmt) override {
    VLOG(4) << "Enter VisitStmt(ir::stmt::For stmt)";
    VLOG(4) << "stmt->min(): " << stmt->min();
    Expr min = stmt->min();
    VisitExpr(&min);
    VLOG(4) << "stmt->extent(): " << stmt->extent();
    Expr extent = stmt->extent();
    VisitExpr(&extent);
    VLOG(4) << "stmt->body(): " << stmt->body();
    VisitBlock(stmt->body());
    if (stmt->loop_var()->name == var_->name && expr_.as_var() && visit_all_) {
      auto copied = ir::ir_utils::IRCopy(expr_);
      stmt->set_loop_var(copied.as_var_ref());
    }
    VLOG(4) << "Exit VisitStmt(ir::stmt::For stmt)";
  }

  void VisitStmt(ir::stmt::Alloc stmt) override {
    VLOG(4) << "Enter VisitStmt(ir::stmt::Alloc stmt)";
    return;
  }

  void VisitStmt(ir::stmt::Free stmt) override {
    VLOG(4) << "Enter VisitStmt(ir::stmt::Free stmt)";
    return;
  }

  void VisitStmt(ir::stmt::IfThenElse stmt) override {
    VLOG(4) << "Enter VisitStmt(ir::stmt::IfThenElse stmt)";
    Expr condition = stmt->condition();
    VisitExpr(&condition);
    ir::stmt::BlockRef true_case = stmt->true_case();
    VisitBlock(true_case);
    stmt->set_true_case(true_case);
    if (stmt->false_case().defined()) {
      ir::stmt::BlockRef false_case = stmt->false_case();
      VisitBlock(false_case);
      stmt->set_false_case(false_case);
    }
  }

  void VisitStmt(ir::stmt::Evaluate) override {
    VLOG(4) << "Enter VisitStmt(ir::stmt::Evaluate)";
    return;
  }

  void VisitStmt(ir::stmt::Schedule stmt) override {
    VLOG(4) << "Enter VisitStmt(ir::stmt::Schedule stmt)";
    std::vector<Var> vars = stmt->iter_vars();
    for (ir::Var& var : vars) {
      if (var->lower_bound.defined() && ShouldReplaceExpr(var->lower_bound)) {
        auto copied = ir::ir_utils::IRCopy(expr_);
        var->lower_bound = copied;
      }
      if (var->upper_bound.defined() && ShouldReplaceExpr(var->upper_bound)) {
        auto copied = ir::ir_utils::IRCopy(expr_);
        var->upper_bound = copied;
      }
    }
    std::vector<Expr> new_read_buffers = stmt->read_buffers();
    for (Expr& read_buffer : new_read_buffers) {
      VisitExpr(&read_buffer);
    }
    stmt->set_read_buffers(new_read_buffers);

    std::vector<Expr> new_write_buffers = stmt->write_buffers();
    for (Expr& write_buffer : new_write_buffers) {
      VisitExpr(&write_buffer);
    }
    stmt->set_write_buffers(new_write_buffers);
    VisitBlock(stmt->body());
  }

 private:
  bool do_replace_{false};
  bool visit_all_{false};
  const Var& var_;
  const Expr& expr_;
  const std::string& tensor_name_;
};

struct ReplaceVarWithExprMutator : public ir::IRMutator<> {
  ReplaceVarWithExprMutator(const Var& var,
                            const Expr& expr,
                            const std::string& tensor_name)
      : var_(var), expr_(expr), tensor_name_(tensor_name) {}

  void operator()(Expr* expr) {
    if (tensor_name_.empty()) visit_all_ = true;
    IRMutator::Visit(expr, expr);
  }

 private:
  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (expr->name == var_->name && (do_replace_ || visit_all_)) {
      auto copied = ir::ir_utils::IRCopy(expr_);
      *op = copied;
    }
  }

  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    ir::IRMutator<>::Visit(&node->min, &node->min);
    ir::IRMutator<>::Visit(&node->extent, &node->extent);
    ir::IRMutator<>::Visit(&node->body, &node->body);
    if (node->loop_var->name == var_->name && expr_.As<ir::_Var_>() &&
        visit_all_) {
      node->loop_var = expr_.As<ir::_Var_>();
    }
  }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    ir::IRMutator<>::Visit(&node->init, &node->init);
    ir::IRMutator<>::Visit(&node->condition, &node->condition);
    ir::IRMutator<>::Visit(&node->inc, &node->inc);
    ir::IRMutator<>::Visit(&node->body, &node->body);
    if (node->iterator->name == var_->name && expr_.As<ir::_Var_>() &&
        visit_all_) {
      node->iterator = expr_.As<ir::_Var_>();
    }
  }

  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node = expr->As<ir::Store>();
    auto* tensor = node->tensor.as_tensor();

    if (tensor->name == tensor_name_) {
      do_replace_ = true;
    } else {
      do_replace_ = false;
    }
    for (auto& index : node->indices) {
      ir::IRMutator<>::Visit(&index, &index);
    }
    do_replace_ = false;
    ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
    ir::IRMutator<>::Visit(&node->value, &node->value);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    auto* node = op->As<ir::Load>();
    auto* tensor = node->tensor.as_tensor();
    if (tensor->name == tensor_name_) {
      do_replace_ = true;
    } else {
      do_replace_ = false;
    }
    for (auto& idx : node->indices) ir::IRMutator<>::Visit(&idx, &idx);
    do_replace_ = false;
    ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
  }

 private:
  bool do_replace_{false};
  bool visit_all_{false};
  const Var& var_;
  const Expr& expr_;
  const std::string& tensor_name_;
};

void ReplaceVarWithExprInStmt(ir::stmt::StmtRef source,
                              const Var& var,
                              const Expr& expr,
                              const std::string& tensor_name) {
  ReplaceVarWithExprStmtMutator mutator(var, expr, tensor_name);
  mutator(source);
}

void ReplaceVarWithExprInBlock(ir::stmt::BlockRef source,
                               const Var& var,
                               const Expr& expr,
                               const std::string& tensor_name) {
  VLOG(4) << "Enter ReplaceVarWithExprInBlock";
  VLOG(4) << "source: " << source;
  VLOG(4) << "var: " << var;
  VLOG(4) << "expr: " << expr;
  ReplaceVarWithExprStmtMutator mutator(var, expr, tensor_name);
  mutator(source);
}

void ReplaceVarWithExpr(Expr* source,
                        const Var& var,
                        const Expr& expr,
                        const std::string& tensor_name) {
  ReplaceVarWithExprMutator mutator(var, expr, tensor_name);
  mutator(source);
}

struct CollectTensorIndexMutator : public ir::IRMutator<> {
  explicit CollectTensorIndexMutator(const std::string& tensor_name)
      : tensor_name_(tensor_name) {}

  std::vector<std::vector<Expr>> operator()(Expr* expr) {
    IRMutator::Visit(expr, expr);
    return res;
  }

 private:
  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    ir::IRMutator<>::Visit(&node->body, &node->body);
  }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    ir::IRMutator<>::Visit(&node->body, &node->body);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    auto* node = op->As<ir::Load>();
    auto* tensor = node->tensor.as_tensor();
    if (tensor->name == tensor_name_) {
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      res.push_back(node->indices);
    } else {
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      for (auto& idx : node->indices) ir::IRMutator<>::Visit(&idx, &idx);
    }
  }

 private:
  std::vector<std::vector<Expr>> res;
  const std::string& tensor_name_;
};

std::vector<std::vector<Expr>> CollectTensorIndex(
    Expr* source, const std::string& tensor_name) {
  CollectTensorIndexMutator mutator(tensor_name);
  return mutator(source);
}

}  // namespace optim
}  // namespace cinn
