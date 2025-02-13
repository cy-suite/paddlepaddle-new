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
#include "paddle/ap/include/index_expr/index_expr_util.h"
#include "paddle/ap/include/index_expr/index_tuple_expr.h"
#include "paddle/ap/include/index_expr/slice.h"

namespace ap::index_expr {

using adt::Result;

class ValidIndexExprBuilder {
 public:
  ValidIndexExprBuilder() {}
  ValidIndexExprBuilder(const ValidIndexExprBuilder&) = delete;
  ValidIndexExprBuilder(ValidIndexExprBuilder&&) = delete;

  Result<IndexExpr> BroadcastMask(const symbol::DimExpr& dim_expr,
                                  const IndexExpr& index_expr) {
    return IndexExprBroadcastMask<IndexExpr>{dim_expr, index_expr};
  }

  Result<IndexExpr> Slice(const ap::index_expr::Slice& slice,
                          const symbol::DimExpr& range,
                          const IndexExpr& index_expr) {
    return IndexExprSlice<IndexExpr>{slice, range, index_expr};
  }

  Result<IndexExpr> Affine(const ap::index_expr::Slice& slice,
                           const symbol::DimExpr& range,
                           const IndexExpr& index_expr) {
    return IndexExprAffine<IndexExpr>{slice, range, index_expr};
  }

  Result<IndexExpr> DisjointUnion(const IndexExpr& lhs_index_expr,
                                  const IndexExpr& rhs_index_expr) {
    const auto& lhs_domain = IndexExprGetDomain(lhs_index_expr);
    const auto& rhs_domain = IndexExprGetDomain(rhs_index_expr);
    const auto& pattern_match = ::common::Overloaded{
        [](const symbol::DimExpr& lhs, const symbol::DimExpr& rhs) {
          return lhs == rhs;
        },
        [](const auto&, const auto&) { return false; }};
    const bool do_equal =
        std::visit(pattern_match, lhs_domain.variant(), rhs_domain.variant());
    if (!do_equal) {
      return adt::errors::TypeError{
          "domain of `lhs_index_expr' does not equal to domain of "
          "`rhs_index_expr'"};
    }
    return index_expr::DisjointUnion<IndexExpr>{lhs_index_expr, rhs_index_expr};
  }

  Result<IndexTupleExpr> Permute(const adt::List<int64_t>& perms,
                                 const IndexTupleExpr& indexes_expr) {
    if (!IsValidPerm(perms)) {
      return adt::errors::InvalidArgumentError{"argument `perms` is not valid"};
    }
    const auto& rank = IndexTupleExprGetRank(indexes_expr);
    if (!rank.Has<int64_t>()) {
      return adt::errors::InvalidArgumentError{
          "wrong indexes_expr argument for IndexTupleExprPermute"};
    }
    if (rank.Get<int64_t>() != perms->size()) {
      return adt::errors::InvalidArgumentError{std::string(
          "the rank of perms does not equal to the rank of "
          "indexes_expr. rank(perm): " +
          std::to_string(perms->size()) +
          ", rank(indexes_expr): " + std::to_string(rank.Get<int64_t>()))};
    }
    return IndexTupleExprPermute<IndexTupleExpr>{perms, indexes_expr};
  }

  Result<IndexTupleExpr> Reshape(const adt::List<symbol::DimExpr>& shape,
                                 const IndexTupleExpr& indexes_expr) {
    if (ContainsNegative(shape)) {
      return adt::errors::InvalidArgumentError{
          "dims in argument `shape` have negative integer"};
    }
    const auto& opt_ranges = IndexTupleExprGetRanges(indexes_expr);
    if (opt_ranges.Has<adt::Nothing>()) {
      return adt::errors::InvalidArgumentError{
          "argument `indexes_expr` is not a ranked IndexTupleExpr"};
    }
    if (!ProductEqual(shape, opt_ranges.Get<adt::List<symbol::DimExpr>>())) {
      return adt::errors::InvalidArgumentError{
          "product of argument `shape` does not equal to elements of "
          "`indexes_expr`"};
    }
    return IndexTupleExprReshape<IndexTupleExpr>{shape, indexes_expr};
  }

  Result<IndexTupleExpr> Transform(
      const adt::List<IndexExpr>& transform_index_exprs,
      const IndexTupleExpr& indexes_expr) {
    const auto& opt_rank = IndexTupleExprGetRank(indexes_expr);
    if (!opt_rank.Has<int64_t>()) {
      return adt::errors::TypeError{
          "The first argument of IndexTupleExprTransform must be a ranked "
          "IndexTupleExpr."};
    }
    const auto& opt_ranges = IndexTupleExprGetRanges(indexes_expr);
    if (!opt_ranges.Has<adt::List<symbol::DimExpr>>()) {
      return adt::errors::RuntimeError{
          "error occurred where calling IndexTupleExprGetDims"};
    }
    const auto& ranges = opt_ranges.Get<adt::List<symbol::DimExpr>>();
    if (opt_rank.Get<int64_t>() != transform_index_exprs->size()) {
      return adt::errors::TypeError{
          "The rank of first argument must equal to number of lambdas."};
    }
    adt::List<symbol::DimExpr> domains{};
    domains->reserve(transform_index_exprs->size());
    for (const auto& index_expr : *transform_index_exprs) {
      const auto& domain = IndexExprGetDomain(index_expr);
      if (!domain.Has<symbol::DimExpr>()) {
        return adt::errors::TypeError{
            "one of transform_index_exprs has no demain."};
      }
      domains->emplace_back(domain.Get<symbol::DimExpr>());
    }
    if (ranges != domains) {
      return adt::errors::TypeError{
          "domain of `transform_index_exprs' does not equal to range of "
          "`indexes_expr'."};
    }
    return IndexTupleExprTransform<IndexTupleExpr>{transform_index_exprs,
                                                   indexes_expr};
  }

  // outer(inner(x)) == (outer . inner)(x)
  Result<IndexTupleExpr> Compose(const IndexTupleExpr& outer,
                                 const IndexTupleExpr& inner) {
    return outer.Match(
        [&](const UndefinedIndexTupleExpr& impl) -> Result<IndexTupleExpr> {
          return impl;
        },
        [&](const NothingIndexTupleExpr& impl) -> Result<IndexTupleExpr> {
          return impl;
        },
        [&](const IntArrayLikeIndexTupleExpr& impl) -> Result<IndexTupleExpr> {
          return impl;
        },
        [&](const IndexTupleExprDomain& domain) -> Result<IndexTupleExpr> {
          const auto& ranges = IndexTupleExprGetRanges(inner);
          if (ranges.Has<adt::Nothing>()) {
            return adt::errors::TypeError{"`inner_indexes_expr' has no range."};
          }
          if (ranges.Get<adt::List<symbol::DimExpr>>() != domain->ranges) {
            return adt::errors::TypeError{
                "the domain of `outer_indexes_expr' does not equal to the "
                "range "
                "of `inner_indexes_expr'."};
          }
          return inner;
        },
        [&](const IndexTupleExprPermute<IndexTupleExpr>& perm)
            -> Result<IndexTupleExpr> {
          const auto& composed_inner = Compose(perm->indexes_expr, inner);
          if (composed_inner.HasError()) {
            return composed_inner.GetError();
          }
          return Permute(perm->perms, composed_inner.Get<IndexTupleExpr>());
        },
        [&](const IndexTupleExprReshape<IndexTupleExpr>& reshape)
            -> Result<IndexTupleExpr> {
          const auto& composed_inner = Compose(reshape->indexes_expr, inner);
          if (composed_inner.HasError()) {
            return composed_inner.GetError();
          }
          return Reshape(reshape->shape, composed_inner.Get<IndexTupleExpr>());
        },
        [&](const IndexTupleExprTransform<IndexTupleExpr>& transform)
            -> Result<IndexTupleExpr> {
          const auto& composed_inner = Compose(transform->indexes_expr, inner);
          if (composed_inner.HasError()) {
            return composed_inner.GetError();
          }
          return Transform(transform->index_exprs,
                           composed_inner.Get<IndexTupleExpr>());
        });
  }

 private:
  template <typename PermsT>
  bool IsValidPerm(const PermsT& perms) {
    std::vector<bool> idx2touched(perms->size(), false);
    for (int64_t perm : *perms) {
      if (perm < 0) {
        return false;
      }
      if (perm >= perms->size()) {
        return false;
      }
      idx2touched[perm] = true;
    }
    for (bool touched : idx2touched) {
      if (!touched) {
        return false;
      }
    }
    return true;
  }

  template <typename ShapeT>
  bool ContainsNegative(const ShapeT& shape) {
    for (const auto& dim : *shape) {
      if (!dim.template Has<int64_t>()) {
        continue;
      }
      if (dim.template Get<int64_t>() < 0) {
        return true;
      }
    }
    return false;
  }

  template <typename DimExprsT>
  symbol::DimExpr Product(const DimExprsT& dim_exprs) {
    symbol::DimExpr ret_expr{1};
    for (const auto& dim : *dim_exprs) {
      ret_expr = ret_expr * dim;
    }
    return ret_expr;
  }

  bool ProductEqual(const auto& lhs, const auto& rhs) {
    return Product(lhs) == Product(rhs);
  }
};

}  // namespace ap::index_expr
