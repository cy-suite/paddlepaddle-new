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
#include "paddle/ap/include/index_expr/index_expr.h"
#include "paddle/ap/include/index_expr/index_expr_builtin_functions.h"

namespace ap::index_expr {

template <typename ValueT>
struct IndexExprMethodClass {
  using This = IndexExprMethodClass;
  using Self = IndexExpr;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return self.ToString();
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetIndexExprClass() {
  using ImplMethods = IndexExprMethodClass<ValueT>;
  static auto cls(axpr::MakeBuiltinClass<ValueT>(
      "IndexExpr",
      [&](const auto& Define) { Define("__str__", &ImplMethods::ToString); }));
  using Self = typename ImplMethods::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::index_expr
