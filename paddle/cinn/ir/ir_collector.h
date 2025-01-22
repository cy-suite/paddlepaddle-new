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

#pragma once

#include "paddle/cinn/ir/stmt.h"
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"

namespace cinn {
namespace ir {

template <typename SourceT, typename TargetT>
struct IRCollector {
  using CollectorFuncT = std::function<std::vector<TargetT>(const SourceT& x)>;
  CollectorFuncT f_;
  std::string name;
  explicit IRCollector(CollectorFuncT f, std::string s = "") : f_(f), name(s) {}

  std::vector<TargetT> operator()(const SourceT& x) const { return f_(x); }
  TargetT GetSingle(const SourceT& x) const {
    const auto& o = f_(x);
    PADDLE_ENFORCE_EQ(
        o.size(),
        1,
        ::common::errors::InvalidArgument(
            "Try to get single result, but we get %d.", o.size()));
    return *o.begin();
  }
};

using Expr2ExprCollector = IRCollector<Expr, Expr>;
using Stmt2StmtCollector = IRCollector<stmt::StmtRef, stmt::StmtRef>;
using Stmt2ExprCollector = IRCollector<stmt::StmtRef, Expr>;

template <typename SourceT, typename TargetT, typename MiddleT>
IRCollector<SourceT, TargetT> operator*(IRCollector<SourceT, MiddleT> x,
                                        IRCollector<MiddleT, TargetT> y);

template Expr2ExprCollector operator*
    <Expr, Expr, Expr>(Expr2ExprCollector, Expr2ExprCollector);
template Stmt2StmtCollector operator*
    <stmt::StmtRef, stmt::StmtRef, stmt::StmtRef>(Stmt2StmtCollector,
                                                  Stmt2StmtCollector);
template Stmt2ExprCollector operator*
    <stmt::StmtRef, Expr, Expr>(Stmt2ExprCollector, Expr2ExprCollector);
template Stmt2ExprCollector operator*
    <stmt::StmtRef, Expr, stmt::StmtRef>(Stmt2StmtCollector,
                                         Stmt2ExprCollector);

template <typename SourceT, typename TargetT>
IRCollector<SourceT, TargetT> NestedCollectorMaker(
    std::function<bool(const TargetT&)> teller, std::string name = "");

// TODO(Hongqing-work): move methods of ir_nodes_collector.h to this file.
template <>
Expr2ExprCollector NestedCollectorMaker(std::function<bool(const Expr&)> teller,
                                        std::string name) {
  return Expr2ExprCollector(
      [=](const Expr& x) -> std::vector<Expr> {
        const auto new_func = [=](const ir::Expr* x) -> bool {
          return teller(*x);
        };
        return cinn::ir::ir_utils::CollectIRNodesInOrder(x, new_func);
      },
      name);
}

template <>
Stmt2StmtCollector NestedCollectorMaker(
    std::function<bool(const stmt::StmtRef&)> teller, std::string name) {
  return Stmt2StmtCollector(
      [=](const stmt::StmtRef& x) -> std::vector<stmt::StmtRef> {
        std::vector<stmt::StmtRef> rs;
        stmt::Visit(
            x,
            [&](const stmt::StmtRef& x) {
              if (teller(x)) rs.push_back(x);
            },
            [&](const stmt::StmtRef& x) {});
        return rs;
      },
      name);
}

extern Stmt2ExprCollector ChildExprsWithoutNested;
extern Stmt2ExprCollector ChildExprs;

template <>
Stmt2ExprCollector NestedCollectorMaker(std::function<bool(const Expr&)> teller,
                                        std::string name) {
  Stmt2ExprCollector res =
      ChildExprs * NestedCollectorMaker<Expr, Expr>(teller);
  res.name = name;
  return res;
}

template <typename TargetT>
IRCollector<TargetT, TargetT> FilterMaker(
    std::function<bool(const TargetT&)> filter, std::string name) {
  return IRCollector<TargetT, TargetT>(
      [=](const TargetT& x) -> std::vector<TargetT> {
        if (filter(x)) return {x};
        return {};
      },
      name);
}

extern Stmt2ExprCollector Store2Value;

extern Stmt2ExprCollector Schedule2IterValues;

extern Stmt2StmtCollector ScheduleNotRoot;

extern Stmt2StmtCollector ScheduleIsRoot;

extern Stmt2StmtCollector ScheduleIsNotInit;

extern Stmt2StmtCollector ScheduleIsInit;

extern Stmt2StmtCollector IsFor;

extern Stmt2StmtCollector ChildSchedules;

extern Stmt2ExprCollector For2Min;

extern Stmt2ExprCollector For2Max;

extern Stmt2StmtCollector ChildStores;

extern Stmt2ExprCollector ChildTensorLoads;

extern Stmt2StmtCollector ChildTensorStores;

extern Stmt2StmtCollector ChildFors;

extern Stmt2StmtCollector ChildIfThenElses;

Stmt2StmtCollector IsForWithIterVar(const Var& var);

Expr2ExprCollector FilterLoadByTensor(const Tensor& tensor);

Stmt2StmtCollector FindFather(const stmt::StmtRef& root);

Stmt2StmtCollector DirectlyFather(const stmt::StmtRef& root);

}  // namespace ir
}  // namespace cinn
