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

#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/index_expr/index_expr_builtin_functions.h"
#include "paddle/ap/include/index_expr/index_tuple_expr.h"

namespace ap::index_expr {

template <typename ValueT>
struct IndexTupleExprMethodClass {
  using This = IndexTupleExprMethodClass;
  using Self = IndexTupleExpr;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return self.ToString();
  }
};

template <typename ValueT>
struct TypeImplIndexTupleExprMethodClass {
  using This = TypeImplIndexTupleExprMethodClass;
  using Self = axpr::TypeImpl<IndexTupleExpr>;

  static adt::Result<ValueT> StaticConstructIndexTupleExprDomain(
      const ValueT&, const std::vector<ValueT>& args) {
    return This{}.ConstructIndexTupleExprDomain(args);
  }

  adt::Result<ValueT> ConstructIndexTupleExprDomain(
      const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "'IndexTupleExpr.Domain' takes 1 argument but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(list, args.at(0).template TryGet<adt::List<ValueT>>())
        << adt::errors::TypeError{std::string() +
                                  "the argument 1 of 'IndexTupleExpr.Domain' "
                                  "should a list of DimExpr."};
    adt::List<symbol::DimExpr> dim_exprs;
    dim_exprs->reserve(list->size());
    for (const auto& arg : *list) {
      ADT_LET_CONST_REF(dim_expr, CastToDimExpr(arg))
          << adt::errors::TypeError{std::string() +
                                    "the argument 1 of 'IndexTupleExpr.Domain' "
                                    "should a list of DimExpr."};
      dim_exprs->emplace_back(dim_expr);
    }
    IndexTupleExpr index_tuple_expr{IndexTupleExprDomain{dim_exprs}};
    axpr::BuiltinClassInstance<ValueT> instance{
        GetIndexTupleExprClass<ValueT>(), index_tuple_expr};
    return instance;
  }

  adt::Result<symbol::DimExpr> CastToDimExpr(const ValueT& val) {
    ADT_LET_CONST_REF(dim_expr, TryGetDimExpr(val));
    return dim_expr;
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetIndexTupleExprClass() {
  using TypeImplMethods = TypeImplIndexTupleExprMethodClass<ValueT>;
  using ImplMethods = IndexTupleExprMethodClass<ValueT>;
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("IndexTupleExpr", [&](const auto& Define) {
        Define("Domain", &TypeImplMethods::StaticConstructIndexTupleExprDomain);
        Define("__str__", &ImplMethods::ToString);
      }));
  using Self = typename ImplMethods::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::index_expr
