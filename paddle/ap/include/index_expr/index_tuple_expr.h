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
#include "paddle/ap/include/index_expr/index_expr.h"
#include "paddle/ap/include/index_expr/slice.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace ap::index_expr {

struct UndefinedIndexTupleExprImpl : public std::monostate {
  using std::monostate::monostate;

  std::string ToString() const { return "IndexTupleExpr.Undefined"; }

  const char* TypeName() const { return "UndefinedIndexTupleExpr"; }
};
ADT_DEFINE_RC(UndefinedIndexTupleExpr, UndefinedIndexTupleExprImpl);

struct NothingIndexTupleExprImpl : public std::monostate {
  using std::monostate::monostate;

  std::string ToString() const { return "IndexTupleExpr.Nothing"; }

  const char* TypeName() const { return "NothingIndexTupleExpr"; }
};
ADT_DEFINE_RC(NothingIndexTupleExpr, NothingIndexTupleExprImpl);

struct IntArrayLikeIndexTupleExprImpl : public std::monostate {
  using std::monostate::monostate;

  std::string ToString() const { return "IndexTupleExpr.IntArrayLike"; }

  const char* TypeName() const { return "IntArrayLikeIndexTupleExpr"; }
};
ADT_DEFINE_RC(IntArrayLikeIndexTupleExpr, IntArrayLikeIndexTupleExprImpl);

struct IndexTupleExprDomainImpl {
  adt::List<symbol::DimExpr> ranges;
  bool operator==(const IndexTupleExprDomainImpl& other) const {
    return other.ranges == this->ranges;
  }

  std::string ToString() const {
    std::ostringstream ss;
    ss << "[";
    int i = 0;
    for (const auto& elt : *ranges) {
      if (i++ > 0) {
        ss << ", ";
      }
      ss << symbol::ToString(elt);
    }
    ss << "]";
    return std::string() + "IndexTupleExpr.Domain(" + ss.str() + ")";
  }

  const char* TypeName() const { return "IndexTupleExprDomain"; }
};
ADT_DEFINE_RC(IndexTupleExprDomain, const IndexTupleExprDomainImpl);

template <typename Expr>
struct IndexTupleExprPermuteImpl {
  adt::List<int64_t> perms;
  Expr indexes_expr;

  bool operator==(const IndexTupleExprPermuteImpl& other) const {
    return other.perms == this->perms &&
           other.indexes_expr == this->indexes_expr;
  }

  std::string ToString() const {
    std::ostringstream ss;
    ss << "[";
    int i = 0;
    for (int64_t perm : *perms) {
      if (i++ > 0) {
        ss << ", ";
      }
      ss << perm;
    }
    ss << "]";
    return indexes_expr.ToString() + ".permute(" + ss.str() + ")";
  }

  const char* TypeName() const { return "IndexTupleExprPermute"; }
};

template <typename Expr>
ADT_DEFINE_RC(IndexTupleExprPermute, const IndexTupleExprPermuteImpl<Expr>);

template <typename Expr>
struct IndexTupleExprReshapeImpl {
  adt::List<symbol::DimExpr> shape;
  Expr indexes_expr;

  bool operator==(const IndexTupleExprReshapeImpl& other) const {
    return other.shape == this->shape &&
           other.indexes_expr == this->indexes_expr;
  }

  std::string ToString() const {
    std::ostringstream ss;
    ss << "[";
    int i = 0;
    for (const auto& elt : *shape) {
      if (i++ > 0) {
        ss << ", ";
      }
      ss << symbol::ToString(elt);
    }
    ss << "]";
    return indexes_expr.ToString() + ".reshape(" + ss.str() + ")";
  }

  const char* TypeName() const { return "IndexTupleExprReshape"; }
};
template <typename Expr>
ADT_DEFINE_RC(IndexTupleExprReshape, const IndexTupleExprReshapeImpl<Expr>);

template <typename Expr>
struct IndexTupleExprTransformImpl {
  adt::List<IndexExpr> index_exprs;
  Expr indexes_expr;

  bool operator==(const IndexTupleExprTransformImpl& other) const {
    return other.index_exprs == this->index_exprs &&
           other.indexes_expr == this->indexes_expr;
  }

  std::string ToString() const {
    std::ostringstream ss;
    ss << "[";
    int i = 0;
    for (const auto& elt : *index_exprs) {
      if (i++ > 0) {
        ss << ", ";
      }
      ss << elt.ToString();
    }
    ss << "]";
    return indexes_expr.ToString() + ".transform(" + ss.str() + ")";
  }

  const char* TypeName() const { return "IndexTupleExprTransform"; }
};
template <typename Expr>
ADT_DEFINE_RC(IndexTupleExprTransform, const IndexTupleExprTransformImpl<Expr>);

template <typename Expr>
using IndexTupleExprBase = std::variant<UndefinedIndexTupleExpr,
                                        NothingIndexTupleExpr,
                                        IntArrayLikeIndexTupleExpr,
                                        IndexTupleExprDomain,
                                        IndexTupleExprPermute<Expr>,
                                        IndexTupleExprReshape<Expr>,
                                        IndexTupleExprTransform<Expr>>;

struct IndexTupleExpr : public IndexTupleExprBase<IndexTupleExpr> {
  using IndexTupleExprBase<IndexTupleExpr>::IndexTupleExprBase;
  ADT_DEFINE_VARIANT_METHODS(IndexTupleExprBase<IndexTupleExpr>);

  const char* TypeName() const {
    return Match([](const auto& impl) { return impl->TypeName(); });
  }

  std::string ToString() const {
    return Match([](const auto& impl) { return impl->ToString(); });
  }
};

inline std::string IndexTupleExprToString(
    const std::shared_ptr<IndexTupleExpr>& indexes_expr) {
  return indexes_expr->ToString();
}

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetIndexTupleExprClass();

}  // namespace ap::index_expr
