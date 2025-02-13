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

#include <vector>
#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/builtin_class_instance.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/index_expr/slice.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace ap::index_expr {

struct IndexTupleExpr;

std::string IndexTupleExprToString(const std::shared_ptr<IndexTupleExpr>&);

struct UndefinedIndexExprImpl : public std::monostate {
  using std::monostate::monostate;

  std::string ToString() const { return "IndexExpr.Undefined"; }
};

ADT_DEFINE_RC(UndefinedIndexExpr, UndefinedIndexExprImpl);

struct PtrGetItemImpl {
  std::string ptr_var_name;
  std::shared_ptr<IndexTupleExpr> indexes_expr;
  symbol::DimExpr range;

  bool operator==(const PtrGetItemImpl& other) const {
    return (other.ptr_var_name == this->ptr_var_name) &&
           other.indexes_expr == this->indexes_expr &&
           other.range == this->range;
  }

  std::string ToString() const {
    return std::string() + "IndexExpr.PtrGetItem(ptr_var_name=" + ptr_var_name +
           ", indexes_expr=" + IndexTupleExprToString(indexes_expr) +
           ", range=" + symbol::ToString(range) + ")";
  }
};

ADT_DEFINE_RC(PtrGetItem, PtrGetItemImpl);

struct IndexExprDomainImpl {
  symbol::DimExpr range;

  bool operator==(const IndexExprDomainImpl& other) const {
    return other.range == this->range;
  }

  std::string ToString() const {
    return std::string() + "IndexExpr.Domain(" + symbol::ToString(range) + ")";
  }
};

ADT_DEFINE_RC(IndexExprDomain, const IndexExprDomainImpl);

template <typename Expr>
struct IndexExprBroadcastMaskImpl {
  symbol::DimExpr dim;
  Expr index_expr;

  bool operator==(const IndexExprBroadcastMaskImpl& other) const {
    return other.dim == this->dim && other.index_expr == this->index_expr;
  }

  std::string ToString() const {
    return std::string() +
           "IndexExpr.BroadcastMask(dim=" + symbol::ToString(dim) +
           ", index_expr=" + index_expr.ToString() + ")";
  }
};

template <typename Expr>
ADT_DEFINE_RC(IndexExprBroadcastMask, const IndexExprBroadcastMaskImpl<Expr>);

// IndexExprSlice * IndexExprAffine == IdentityFunc if fields are same.
template <typename Expr>
struct IndexExprSliceImpl {
  index_expr::Slice slice;
  symbol::DimExpr range;
  Expr index_expr;

  bool operator==(const IndexExprSliceImpl& other) const {
    return (other.slice == this->slice) && (other.range == this->range) &&
           (other.index_expr == this->index_expr);
  }

  std::string ToString() const {
    return index_expr.ToString() + ".slice(" + slice->ToString() +
           ", range=" + symbol::ToString(range) + ")";
  }
};

template <typename Expr>
ADT_DEFINE_RC(IndexExprSlice, const IndexExprSliceImpl<Expr>);

template <typename Expr>
struct IndexExprAffineImpl {
  index_expr::Slice slice;
  symbol::DimExpr range;
  Expr index_expr;

  bool operator==(const IndexExprAffineImpl& other) const {
    return (other.slice == this->slice) && (other.range == this->range) &&
           (other.index_expr == this->index_expr);
  }

  std::string ToString() const {
    return index_expr.ToString() + ".affine(" + slice->ToString() +
           ", range=" + symbol::ToString(range) + ")";
  }
};

template <typename Expr>
ADT_DEFINE_RC(IndexExprAffine, const IndexExprAffineImpl<Expr>);

template <typename T>
struct DisjointUnionImpl {
  T lhs;
  T rhs;

  bool operator==(const DisjointUnionImpl& other) const {
    return (other.lhs == this->lhs) && (other.rhs == this->rhs);
  }

  std::string ToString() const {
    return std::string() + "IndexExpr.DisjointUnion(" + lhs.ToString() + ", " +
           rhs.ToString() + ")";
  }
};

template <typename T>
ADT_DEFINE_RC(DisjointUnion, const DisjointUnionImpl<T>);

template <typename Expr>
using IndexExprBase = std::variant<UndefinedIndexExpr,
                                   PtrGetItem,
                                   IndexExprDomain,
                                   IndexExprBroadcastMask<Expr>,
                                   IndexExprSlice<Expr>,
                                   IndexExprAffine<Expr>,
                                   DisjointUnion<Expr>>;

struct IndexExpr : public IndexExprBase<IndexExpr> {
  using IndexExprBase<IndexExpr>::IndexExprBase;
  ADT_DEFINE_VARIANT_METHODS(IndexExprBase<IndexExpr>);

  std::string ToString() const {
    return Match([](const auto& impl) { return impl->ToString(); });
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetIndexExprClass();

}  // namespace ap::index_expr
