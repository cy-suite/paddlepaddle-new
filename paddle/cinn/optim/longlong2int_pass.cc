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

#include "paddle/cinn/optim/longlong2int_pass.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_utils.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/stmt.h"
#include "paddle/cinn/ir/stmt_visitors.h"

namespace cinn {
namespace optim {
namespace {
using ir::stmt::BlockRef;
using ir::stmt::For;
using ir::stmt::IfThenElse;
using ir::stmt::Schedule;
using ir::stmt::StmtRef;
using ir::stmt::Store;

class CheckOverflow : public ir::stmt::StmtVisitor<> {
 public:
  bool operator()(const StmtRef& stmt) {
    VisitStmt(stmt);
    return is_overflow_;
  }
  bool operator()(const BlockRef& block) {
    VisitBlock(block);
    return is_overflow_;
  }

 private:
  void VisitStmt(const StmtRef& stmt) override {
    if (is_overflow_) return;
    ir::stmt::StmtVisitor<>::VisitStmt(stmt);
  }

  void VisitStmt(const For& for_stmt) override {
    if (!for_stmt->extent().is_constant()) is_overflow_ = true;
    if (!for_stmt->extent().type().is_index_type()) is_overflow_ = true;
    if (curr_product_ > INT_MAX) is_overflow_ = true;

    if (is_overflow_) return;

    curr_product_ *= for_stmt->extent().as_int64();
    VisitBlock(for_stmt->body());
    curr_product_ /= for_stmt->extent().as_int64();
  }

  void VisitStmt(const Schedule& schedule_stmt) override {
    VisitBlock(schedule_stmt->body());
  }

  void VisitStmt(const IfThenElse& stmt) override {
    VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
    }
  }

  void VisitStmt(const ir::stmt::Let& stmt) override { return; }
  void VisitStmt(const ir::stmt::Store& stmt) override { return; }
  void VisitStmt(const ir::stmt::Alloc& stmt) override { return; }
  void VisitStmt(const ir::stmt::Free& stmt) override { return; }
  void VisitStmt(const ir::stmt::Evaluate& stmt) override { return; }

 private:
  int64_t curr_product_ = 1;
  bool is_overflow_ = false;
};

class CastLonglong2Int : public ir::IRMutator<>,
                         public ir::stmt::StmtMutator<> {
 public:
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }
  void operator()(StmtRef stmt) { ir::stmt::StmtMutator<>::VisitStmt(stmt); }
  void operator()(BlockRef block) {
    ir::stmt::StmtMutator<>::VisitBlock(block);
  }

 private:
  void Visit(const ir::_Tensor_* op, Expr* expr) override {
    auto node = expr->As<ir::_Tensor_>();
    std::for_each(node->shape.begin(),
                  node->shape.end(),
                  [&](cinn::ir::Expr& e) { e->convert_int64_to_int32(); });
    CastBufferMeta(node->buffer);
  }
  void Visit(const ir::Load* op, Expr* expr) override {
    auto node = expr->As<ir::Load>();
    std::for_each(node->indices.begin(),
                  node->indices.end(),
                  [&](cinn::ir::Expr& e) { e->convert_int64_to_int32(); });

    ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
  }
  void Visit(const ir::Select* op, Expr* expr) override {
    auto node = expr->As<ir::Select>();
    auto cond = node->condition;
    if (cond.is_cmp()) {
      if (cond->operand(0).is_index())
        cond->operand(0)->convert_int64_to_int32();
      if (cond->operand(1).is_index())
        cond->operand(1)->convert_int64_to_int32();
    }
    ir::IRMutator<>::Visit(&node->true_value, &node->true_value);
    ir::IRMutator<>::Visit(&node->false_value, &node->false_value);
  }
  void VisitStmt(Store stmt) override {
    std::vector<Expr> indices = stmt->indices();
    std::for_each(indices.begin(), indices.end(), [&](cinn::ir::Expr& e) {
      e->convert_int64_to_int32();
    });
    Expr value = stmt->value();
    Expr tensor = stmt->tensor();
    ir::IRMutator<>::Visit(&value, &value);
    ir::IRMutator<>::Visit(&tensor, &tensor);
  }
  void VisitStmt(IfThenElse stmt) override {
    Expr cond = stmt->condition();
    if (cond.is_cmp()) {
      if (cond->operand(0).is_index())
        cond->operand(0)->convert_int64_to_int32();
      if (cond->operand(1).is_index())
        cond->operand(1)->convert_int64_to_int32();
    }
    ir::stmt::StmtMutator<>::VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      ir::stmt::StmtMutator<>::VisitBlock(stmt->false_case());
    }
  }
  void VisitStmt(For stmt) override {
    ir::Var loop_var = stmt->loop_var();
    CastVarWithBound(loop_var);
    stmt->set_loop_var(loop_var);
    stmt->min()->convert_int64_to_int32();
    stmt->extent()->convert_int64_to_int32();
    ir::stmt::StmtMutator<>::VisitBlock(stmt->body());
  }
  void VisitStmt(Schedule stmt) override {
    std::vector<Var> iter_vars = stmt->iter_vars();
    std::for_each(iter_vars.begin(), iter_vars.end(), [&](cinn::ir::Var& v) {
      CastVarWithBound(v);
    });

    for (auto& buffer_range : stmt->read_buffers()) {
      if (auto range = buffer_range.As<ir::_BufferRange_>()) {
        std::vector<Var> ranges = range->ranges;
        std::for_each(ranges.begin(), ranges.end(), [&](cinn::ir::Var& v) {
          CastVarWithBound(v);
        });
        auto bf = range->buffer.as_buffer_ref();
        CastBufferMeta(bf);
      }
    }

    for (auto& buffer_range : stmt->write_buffers()) {
      if (auto range = buffer_range.As<ir::_BufferRange_>()) {
        std::vector<Var> ranges = range->ranges;

        std::for_each(ranges.begin(), ranges.end(), [&](cinn::ir::Var& v) {
          CastVarWithBound(v);
        });
        auto bf = range->buffer.as_buffer_ref();
        CastBufferMeta(bf);
      }
    }
    ir::stmt::StmtMutator<>::VisitBlock(stmt->body());
  }
  void VisitStmt(ir::stmt::Let stmt) override {
    Expr body = stmt->body();
    ir::IRMutator<>::Visit(&body, &body);
  }
  void VisitStmt(ir::stmt::Evaluate stmt) override {
    Expr value = stmt->value();
    ir::IRMutator<>::Visit(&value, &value);
  }

  void VisitStmt(ir::stmt::Alloc stmt) override { return; }
  void VisitStmt(ir::stmt::Free stmt) override { return; }

  void CastVarWithBound(cinn::ir::Var& var) {  // NOLINT
    if (!var.defined()) return;
    var->convert_int64_to_int32();
    auto lb = var->lower_bound;
    auto ub = var->upper_bound;
    if (lb.defined()) lb->convert_int64_to_int32();
    if (ub.defined()) ub->convert_int64_to_int32();
  }
  void CastBufferMeta(cinn::ir::Buffer& bf) {  // NOLINT
    if (!bf.defined()) return;
    std::for_each(bf->shape.begin(), bf->shape.end(), [&](cinn::ir::Expr& e) {
      e->convert_int64_to_int32();
    });
    std::for_each(bf->strides.begin(),
                  bf->strides.end(),
                  [&](cinn::ir::Expr& e) { e->convert_int64_to_int32(); });
    bf->elem_offset->convert_int64_to_int32();
  }
};
}  // namespace

LogicalResult LongLong2IntPass::Run(ir::stmt::StmtRef stmt) {
  CastLonglong2Int narrow;
  narrow(stmt);
  return LogicalResult::success();
}

std::unique_ptr<StmtPass> CreateLongLong2IntPass() {
  return std::make_unique<LongLong2IntPass>();
}

bool CanApplyLongLong2Int(ir::stmt::BlockRef block) {
  CheckOverflow check_overflow;
  return !check_overflow(block);
}
}  // namespace optim
}  // namespace cinn
