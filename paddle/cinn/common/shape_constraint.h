// Copyright (c) 2025 CINN Authors. All Rights Reserved.
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
#include <unordered_set>
#include <vector>
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/common/union_find_set.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"
namespace cinn {
namespace common {

struct IndexExprDirectCompare {
  bool operator()(const ir::IndexExpr& a, const ir::IndexExpr& b) const {
    return a.get() == b.get();
  }
};

class ShapeConstraintManager {
 public:
  static ShapeConstraintManager& Instance();
  void Init(const ::common::UnionFindSet<symbol::DimExpr>& equal_dim_exprs);
  void Init(const ::common::UnionFindSet<ir::IndexExpr, IndexExprDirectCompare>&
                equal_exprs);
  bool IsEqual(const ir::IndexExpr& lhs, const ir::IndexExpr& rhs);
  const ::common::UnionFindSet<symbol::DimExpr>& GetDimExprEqual() const;
  const ::common::UnionFindSet<ir::IndexExpr, IndexExprDirectCompare>&
  GetEqualExprs() const;

 private:
  ::common::UnionFindSet<symbol::DimExpr> equal_dim_exprs_;
  ::common::UnionFindSet<ir::IndexExpr, IndexExprDirectCompare> equal_exprs_;
  ShapeConstraintManager() = default;
  ~ShapeConstraintManager() = default;
  ShapeConstraintManager(const ShapeConstraintManager&) = delete;
  ShapeConstraintManager(ShapeConstraintManager&&) = delete;
  ShapeConstraintManager& operator=(const ShapeConstraintManager&) = delete;
};

std::ostream& operator<<(std::ostream& os,
                         const ShapeConstraintManager& constraints_manager);
}  // namespace common
}  // namespace cinn
