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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/error.h"
#include "paddle/ap/include/index_expr/index_expr.h"
#include "paddle/ap/include/index_expr/index_tuple_expr.h"

namespace ap::index_expr {

using adt::Maybe;

inline Maybe<int64_t> IndexTupleExprGetRank(const IndexTupleExpr& expr) {
  return expr.Match(
      [](const UndefinedIndexTupleExpr&) -> Maybe<int64_t> {
        return adt::Nothing{};
      },
      [](const NothingIndexTupleExpr&) -> Maybe<int64_t> {
        return adt::Nothing{};
      },
      [](const IntArrayLikeIndexTupleExpr&) -> Maybe<int64_t> {
        return adt::Nothing{};
      },
      [](const IndexTupleExprDomain& domain) -> Maybe<int64_t> {
        return domain->ranges->size();
      },
      [](const IndexTupleExprPermute<IndexTupleExpr>& perm) -> Maybe<int64_t> {
        return perm->perms->size();
      },
      [](const IndexTupleExprReshape<IndexTupleExpr>& reshape)
          -> Maybe<int64_t> { return reshape->shape->size(); },
      [](const IndexTupleExprTransform<IndexTupleExpr>& transform)
          -> Maybe<int64_t> { return transform->index_exprs->size(); });
}

inline Maybe<symbol::DimExpr> IndexExprGetRange(const IndexExpr& index_expr) {
  return index_expr.Match(
      [](const UndefinedIndexExpr&) -> Maybe<symbol::DimExpr> {
        return adt::Nothing{};
      },
      [](const PtrGetItem& ptr_get_item) -> Maybe<symbol::DimExpr> {
        return ptr_get_item->range;
      },
      [](const IndexExprDomain& domain) -> Maybe<symbol::DimExpr> {
        return domain->range;
      },
      [](const IndexExprBroadcastMask<IndexExpr>& mask)
          -> Maybe<symbol::DimExpr> { return mask->dim; },
      [](const IndexExprSlice<IndexExpr>& index_slice)
          -> Maybe<symbol::DimExpr> { return index_slice->range; },
      [](const IndexExprAffine<IndexExpr>& index_affine)
          -> Maybe<symbol::DimExpr> { return index_affine->range; },
      [](const DisjointUnion<IndexExpr>& union_expr) -> Maybe<symbol::DimExpr> {
        const auto& opt_lhs_dim_expr = IndexExprGetRange(union_expr->lhs);
        const auto& opt_rhs_dim_expr = IndexExprGetRange(union_expr->rhs);
        return std::visit(
            ::common::Overloaded{
                [](const symbol::DimExpr& lhs, const symbol::DimExpr& rhs)
                    -> Maybe<symbol::DimExpr> { return lhs + rhs; },
                [](const auto&, const auto&) -> Maybe<symbol::DimExpr> {
                  return adt::Nothing{};
                }},
            opt_lhs_dim_expr.variant(),
            opt_rhs_dim_expr.variant());
      });
}

inline Maybe<symbol::DimExpr> IndexExprGetDomain(const IndexExpr& index_expr) {
  return index_expr.Match(
      [](const UndefinedIndexExpr&) -> Maybe<symbol::DimExpr> {
        return adt::Nothing{};
      },
      [](const PtrGetItem& ptr_get_item) -> Maybe<symbol::DimExpr> {
        return ptr_get_item->range;
      },
      [](const IndexExprDomain& domain) -> Maybe<symbol::DimExpr> {
        return domain->range;
      },
      [](const IndexExprBroadcastMask<IndexExpr>& mask)
          -> Maybe<symbol::DimExpr> {
        return IndexExprGetDomain(mask->index_expr);
      },
      [](const IndexExprSlice<IndexExpr>& index_slice)
          -> Maybe<symbol::DimExpr> {
        return IndexExprGetDomain(index_slice->index_expr);
      },
      [](const IndexExprAffine<IndexExpr>& index_affine)
          -> Maybe<symbol::DimExpr> {
        return IndexExprGetDomain(index_affine->index_expr);
      },
      [](const DisjointUnion<IndexExpr>& union_expr) -> Maybe<symbol::DimExpr> {
        const auto& lhs = IndexExprGetDomain(union_expr->lhs);
        const auto& rhs = IndexExprGetDomain(union_expr->rhs);
        const auto& pattern_match = ::common::Overloaded{
            [&](const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) {
              if (lhs == rhs) {
                return Maybe<symbol::DimExpr>{lhs};
              } else {
                return Maybe<symbol::DimExpr>{adt::Nothing{}};
              }
            },
            [&](const auto&, const auto&) {
              return Maybe<symbol::DimExpr>{adt::Nothing{}};
            }};
        return std::visit(pattern_match, lhs.variant(), rhs.variant());
      });
}

inline Maybe<adt::List<symbol::DimExpr>> IndexTupleExprGetRanges(
    const IndexTupleExpr& expr) {
  return expr.Match(
      [](const UndefinedIndexTupleExpr&) -> Maybe<adt::List<symbol::DimExpr>> {
        return adt::Nothing{};
      },
      [](const NothingIndexTupleExpr&) -> Maybe<adt::List<symbol::DimExpr>> {
        return adt::Nothing{};
      },
      [](const IntArrayLikeIndexTupleExpr&)
          -> Maybe<adt::List<symbol::DimExpr>> { return adt::Nothing{}; },
      [](const IndexTupleExprDomain& domain)
          -> Maybe<adt::List<symbol::DimExpr>> { return domain->ranges; },
      [](const IndexTupleExprPermute<IndexTupleExpr>& perm)
          -> Maybe<adt::List<symbol::DimExpr>> {
        const auto& opt_origin_dim_exprs =
            IndexTupleExprGetRanges(perm->indexes_expr);
        if (opt_origin_dim_exprs.Has<adt::Nothing>()) {
          return adt::Nothing{};
        }
        const auto& origin_dim_exprs =
            opt_origin_dim_exprs.Get<adt::List<symbol::DimExpr>>();
        adt::List<symbol::DimExpr> ret;
        ret->reserve(perm->perms->size());
        for (const int idx : *perm->perms) {
          ret->push_back(origin_dim_exprs->at(idx));
        }
        return ret;
      },
      [](const IndexTupleExprReshape<IndexTupleExpr>& reshape)
          -> Maybe<adt::List<symbol::DimExpr>> { return reshape->shape; },
      [](const IndexTupleExprTransform<IndexTupleExpr>& transform)
          -> Maybe<adt::List<symbol::DimExpr>> {
        adt::List<symbol::DimExpr> ret;
        ret->reserve(transform->index_exprs->size());
        for (const auto& index_expr : *transform->index_exprs) {
          const auto& opt_dim_expr = IndexExprGetRange(index_expr);
          if (opt_dim_expr.Has<adt::Nothing>()) {
            return adt::Nothing{};
          }
          ret->push_back(opt_dim_expr.Get<symbol::DimExpr>());
        }
        return ret;
      });
}

}  // namespace ap::index_expr
