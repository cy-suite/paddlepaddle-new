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

#include "paddle/cinn/ir/ir_collector.h"
#include "paddle/cinn/ir/expr_visitors.h"

namespace cinn {
namespace ir {
using stmt::StmtRef;
using ExprSet = std::vector<Expr>;
using StmtSet = std::vector<StmtRef>;

template <typename SourceT, typename TargetT, typename MiddleT>
IRCollector<SourceT, TargetT> operator*(IRCollector<SourceT, MiddleT> x,
                                        IRCollector<MiddleT, TargetT> y) {
  const auto& new_f = [&](const SourceT& source) -> std::vector<TargetT> {
    const auto& x_res_set = x.f_(source);
    std::vector<TargetT> res;
    for (const auto& x_res : x_res_set) {
      const auto& y_res = y.f_(x_res);
      res.insert(res.end(), y_res.begin(), y_res.end());
    }
    return res;
  };
  return IRCollector<SourceT, TargetT>(std::function(new_f),
                                       x.name + "*" + y.name);
}

Stmt2ExprCollector Store2Value = Stmt2ExprCollector(
    [](const StmtRef& s) -> ExprSet {
      if (s.isa<stmt::Store>()) return {s.as<stmt::Store>()->value()};
      return {};
    },
    "Store2Value");

Stmt2ExprCollector Schedule2IterValues = Stmt2ExprCollector(
    [](const StmtRef& s) -> ExprSet {
      if (s.isa<stmt::Schedule>()) return s.as<stmt::Schedule>()->iter_values();
      return {};
    },
    "Schedule2IterValues");

Stmt2StmtCollector ScheduleNotRoot = FilterMaker<StmtRef>(
    [](const StmtRef& s) -> bool {
      return (s.isa<stmt::Schedule>() &&
              s.as<stmt::Schedule>()->name().find("root") == std::string::npos);
    },
    "ScheduleNotRoot");

Stmt2StmtCollector ScheduleIsRoot = FilterMaker<StmtRef>(
    [](const StmtRef& s) -> bool {
      return (s.isa<stmt::Schedule>() &&
              s.as<stmt::Schedule>()->name().find("root") != std::string::npos);
    },
    "ScheduleIsRoot");

Stmt2StmtCollector ScheduleIsNotInit = FilterMaker<StmtRef>(
    [](const StmtRef& s) -> bool {
      return (s.isa<stmt::Schedule>() &&
              s.as<stmt::Schedule>()->name().find("__reduce_init") ==
                  std::string::npos);
    },
    "ScheduleIsNotInit");

Stmt2StmtCollector ScheduleIsInit = FilterMaker<StmtRef>(
    [](const StmtRef& s) -> bool {
      return (s.isa<stmt::Schedule>() &&
              s.as<stmt::Schedule>()->name().find("__reduce_init") !=
                  std::string::npos);
    },
    "ScheduleIsInit");

Stmt2StmtCollector IsFor = FilterMaker<StmtRef>(
    [](const StmtRef& s) -> bool { return s.isa<stmt::For>(); }, "IsFor");

Stmt2StmtCollector ChildSchedules =
    NestedCollectorMaker<StmtRef, StmtRef>(
        [](const StmtRef& s) -> bool { return s.isa<stmt::Schedule>(); },
        "ChildSchedules") *
    ScheduleNotRoot;

Stmt2StmtCollector IsForWithIterVar(const ir::Var& var) {
  return FilterMaker<StmtRef>(
      [&](const StmtRef& s) -> bool {
        return s.isa<stmt::For>() && s.as<stmt::For>()->loop_var() == var;
      },
      "IsForWithIterVar");
}

Stmt2ExprCollector For2Min = Stmt2ExprCollector(
    [](const StmtRef& s) -> ExprSet { return {s.as<stmt::For>()->min()}; },
    "For2Min");

Stmt2ExprCollector For2Max = Stmt2ExprCollector(
    [](const StmtRef& s) -> ExprSet { return {s.as<stmt::For>()->extent()}; },
    "For2Max");

Stmt2StmtCollector ChildStores = NestedCollectorMaker<StmtRef, StmtRef>(
    [](const StmtRef& s) -> bool { return s.isa<stmt::Schedule>(); },
    "ChildStores");

Stmt2ExprCollector ChildExprsWithoutNested = Stmt2ExprCollector(
    [](const StmtRef& s) -> ExprSet {
      std::vector<Expr> res;
      switch (s->stmt_type()) {
#define __(stmt__)                                          \
  case StmtNodeTy::stmt__:                                  \
    VisitExpr(s.as<stmt::stmt__>(),                         \
              [&](const Expr& e) { res.emplace_back(e); }); \
    break;
        NODETY_FORALL_STMT(__)

        default:
          PADDLE_THROW(::common::errors::InvalidArgument(
              "Deadcode, not supported StmtNodeTy"));
#undef __
      }
      return res;
    },
    "ChildExprsWithoutNested");

/*
 * Collect all exprs in a stmt, including nested blocks.
 */
Stmt2ExprCollector ChildExprs = Stmt2ExprCollector(
    [](const StmtRef& s) -> ExprSet {
      std::vector<Expr> rs;
      Visit(
          s,
          [&](const stmt::StmtRef& x) {
            const auto& rs_without_nested = ChildExprsWithoutNested(x);
            rs.insert(
                rs.end(), rs_without_nested.begin(), rs_without_nested.end());
          },
          [&](const stmt::StmtRef& x) {});
      return rs;
    },
    "ChildExprs");

Stmt2ExprCollector ChildTensorLoads = NestedCollectorMaker<StmtRef, Expr>(
    [](const ir::Expr& e) {
      return e.As<ir::Load>() && e.As<ir::Load>()->is_addr_tensor();
    },
    "ChildLoads");

Stmt2StmtCollector ChildTensorStores = NestedCollectorMaker<StmtRef, StmtRef>(
    [](const StmtRef& s) {
      return s.isa<stmt::Store>() && s.as<stmt::Store>()->is_addr_tensor();
    },
    "ChildTensorStores");

Expr2ExprCollector FilterLoadByTensor(const Tensor& tensor) {
  return FilterMaker<Expr>(
      [tensor = tensor](const Expr& e) -> bool {
        return e.As<ir::Load>() &&
               e.As<ir::Load>()->tensor.as_tensor_ref()->name == tensor->name;
      },
      "FilterLoadByTensor(" + tensor->name + ")");
}

Stmt2StmtCollector ChildFors = NestedCollectorMaker<StmtRef, StmtRef>(
    [](const StmtRef& s) { return s.isa<stmt::For>(); }, "ChildFors");

Stmt2StmtCollector ChildIfThenElses = NestedCollectorMaker<StmtRef, StmtRef>(
    [](const StmtRef& s) { return s.isa<stmt::IfThenElse>(); },
    "ChildIfThenElses");

// TODO(Hongqing-work): update father collector after supporting StmtRef to
// record father.
Stmt2StmtCollector FindFather(const StmtRef& root) {
  const auto& f = [root](const StmtRef& child) -> StmtSet {
    std::vector<StmtRef> result;
    std::vector<StmtRef> cur_fathers;
    std::function<stmt::VisitResult(const StmtRef&)> pre_callback =
        [&](const StmtRef& current) -> stmt::VisitResult {
      if (current == child) {
        result = cur_fathers;
        return stmt::VisitResult::interrupt();
      }
      cur_fathers.push_back(current);
      return stmt::VisitResult::advance();
    };
    std::function<stmt::VisitResult(const StmtRef&)> post_callback =
        [&](const StmtRef& current) -> stmt::VisitResult {
      if (current == child) {
        result = cur_fathers;
        return stmt::VisitResult::interrupt();
      }
      cur_fathers.push_back(current);
      return stmt::VisitResult::advance();
    };
    const auto& visit_res = stmt::Visit(root, pre_callback, post_callback);
    return result;
  };
  return Stmt2StmtCollector(f, "FindFather");
}

Stmt2StmtCollector DirectlyFather(const StmtRef& root) {
  const auto& f = [root](const StmtRef& child) -> StmtSet {
    StmtSet result = FindFather(root)(child);
    return {result[result.size() - 1]};
  };
  return Stmt2StmtCollector(f, "DirectlyFather");
}

}  // namespace ir
}  // namespace cinn
