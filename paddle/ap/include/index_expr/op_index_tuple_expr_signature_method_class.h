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
#include "paddle/ap/include/index_expr/op_index_tuple_expr_signature.h"

namespace ap::index_expr {

template <typename ValueT>
struct InIndexTupleExprSignatureMethodClass {
  using Self = index_expr::InIndexTupleExprSignature;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return self->ToString();
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>
GetInIndexTupleExprSignatureClass() {
  using ImplMethods = InIndexTupleExprSignatureMethodClass<ValueT>;
  static auto cls(axpr::MakeBuiltinClass<ValueT>(
      "InIndexTupleExprSignature",
      [&](const auto& Define) { Define("__str__", &ImplMethods::ToString); }));
  using Self = typename ImplMethods::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

template <typename ValueT>
struct OutIndexTupleExprSignatureMethodClass {
  using Self = index_expr::OutIndexTupleExprSignature;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return self->ToString();
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>
GetOutIndexTupleExprSignatureClass() {
  using ImplMethods = OutIndexTupleExprSignatureMethodClass<ValueT>;
  static auto cls(axpr::MakeBuiltinClass<ValueT>(
      "OutIndexTupleExprSignature",
      [&](const auto& Define) { Define("__str__", &ImplMethods::ToString); }));
  using Self = typename ImplMethods::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

template <typename ValueT>
struct OpIndexTupleExprSignatureMethodClass {
  using Self = index_expr::OpIndexTupleExprSignature;

  static adt::Result<ValueT> ToString(const ValueT& self_val,
                                      const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return self->ToString();
  }
};

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>>
GetOpIndexTupleExprSignatureClass() {
  using ImplMethods = OpIndexTupleExprSignatureMethodClass<ValueT>;
  static auto cls(axpr::MakeBuiltinClass<ValueT>(
      "OpIndexTupleExprSignature", [&](const auto& Define) {
        Define("__str__",
               &OpIndexTupleExprSignatureMethodClass<ValueT>::ToString);
      }));
  using Self = typename ImplMethods::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::index_expr
