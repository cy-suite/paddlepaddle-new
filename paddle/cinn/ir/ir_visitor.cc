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

#include <unordered_set>

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace ir {

bool operator==(Expr a, Expr b) {
  if (a.get() == b.get()) return true;
  return ir_utils::IRCompare(a, b);
}

template <typename T>
static bool CompareExpressions(const Expr& a, const Expr& b) {
  auto aPart = common::GetFlatternExprs<T>(a);
  auto bPart = common::GetFlatternExprs<T>(b);

  std::sort(aPart.begin(), aPart.end(), common::IsCorrectPriority);
  std::sort(bPart.begin(), bPart.end(), common::IsCorrectPriority);

  if (aPart.size() != bPart.size()) return false;

  for (std::size_t i = 0; i < aPart.size(); ++i) {
    if (aPart.at(i) != bPart.at(i)) return false;
  }

  return true;
}

bool operator!=(Expr a, Expr b) { return !(a == b); }

bool operator==(IndexExpr a, IndexExpr b) {
  if (a.get() == b.get()) return true;
  if (a.node_type() != b.node_type()) return false;
  std::vector<ir::IndexExpr> aPart;
  std::vector<ir::IndexExpr> bPart;
  switch (a.node_type()) {
    case ir::IrNodeTy::IterMark:
      [[fallthrough]];
    case ir::IrNodeTy::IterSplit:
      [[fallthrough]];
    case ir::IrNodeTy::IterSum: {
      return ir_utils::IRCompare(a, b);
    }
    case ir::IrNodeTy::IntImm: {
      return a.as_int64() == b.as_int64();
    }
    case ir::IrNodeTy::_Var_: {
      return a.as_var()->name == b.as_var()->name;
    }
    case ir::IrNodeTy::Div:
    case ir::IrNodeTy::Mod: {
      return a->operand(0).as_index() == b->operand(0).as_index() &&
             a->operand(1).as_index() == b->operand(1).as_index();
    }
    case ir::IrNodeTy::Add:
      return CompareExpressions<ir::Add>(a, b);
    case ir::IrNodeTy::Mul:
      return CompareExpressions<ir::Mul>(a, b);
  }
}

bool operator!=(IndexExpr a, IndexExpr b) { return !(a == b); }

}  // namespace ir
}  // namespace cinn
