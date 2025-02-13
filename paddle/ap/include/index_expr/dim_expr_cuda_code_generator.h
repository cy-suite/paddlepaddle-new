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

#include <sstream>
#include "paddle/ap/include/common/unique_id.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace ap::index_expr {

class DimExprCudaCodeGenerator {
 public:
  using ArgName4DimExprT =
      std::function<std::optional<std::string>(const symbol::DimExpr&)>;
  explicit DimExprCudaCodeGenerator(std::ostringstream* ss,
                                    const ArgName4DimExprT& ArgName4DimExprVal,
                                    const std::string& index_type_name)
      : ss_(ss),
        ArgName4DimExpr(ArgName4DimExprVal),
        index_type_name_(index_type_name) {}

  std::ostringstream& ss() { return *ss_; }

  adt::Result<std::string> CodeGen(const symbol::DimExpr& dim_expr) {
    if (const auto& arg_name = ArgName4DimExpr(dim_expr)) {
      return arg_name.value();
    }
    return dim_expr.Match([&](const auto& impl) { return CodeGenImpl(impl); });
  }

 private:
  adt::Result<std::string> CodeGenImpl(int64_t c) { return std::to_string(c); }

  adt::Result<std::string> CodeGenImpl(const std::string& var) {
    return adt::errors::TypeError{
        std::string() + "no kernel argument bound to DimExpr '" + var + "'"};
  }

  adt::Result<std::string> CodeGenImpl(
      const symbol::Negative<symbol::DimExpr>& dim_expr) {
    const auto& [operand] = *dim_expr;
    ADT_LET_CONST_REF(operand_str, CodeGen(operand));
    return std::string() + "(-" + operand_str + ")";
  }

  adt::Result<std::string> CodeGenImpl(
      const symbol::Reciprocal<symbol::DimExpr>&) {
    return adt::errors::ValueError{
        "reciprocal value should be processed in '*'"};
  }

  adt::Result<std::string> CodeGenImpl(
      const symbol::Add<symbol::DimExpr>& dim_expr) {
    ADT_CHECK(dim_expr.operands->size() > 0);
    ADT_LET_CONST_REF(first, CodeGen(dim_expr.operands->at(0)));
    std::string ret = first;
    for (int i = 1; i < dim_expr.operands->size(); ++i) {
      const auto& operand = dim_expr.operands->at(i);
      if (operand.Has<symbol::Negative<symbol::DimExpr>>()) {
        const auto& [negtaive_operand] =
            *operand.Get<symbol::Negative<symbol::DimExpr>>();
        ADT_LET_CONST_REF(operand_str, CodeGen(negtaive_operand));
        ret += " - " + operand_str;
      } else {
        ADT_LET_CONST_REF(operand_str, CodeGen(operand));
        ret += " + " + operand_str;
      }
    }
    return std::string() + "(" + ret + ")";
  }

  adt::Result<std::string> CodeGenImpl(
      const symbol::Mul<symbol::DimExpr>& dim_expr) {
    ADT_CHECK(dim_expr.operands->size() > 0);
    ADT_LET_CONST_REF(first, CodeGen(dim_expr.operands->at(0)));
    std::string ret = first;
    for (int i = 1; i < dim_expr.operands->size(); ++i) {
      const auto& operand = dim_expr.operands->at(i);
      if (operand.Has<symbol::Reciprocal<symbol::DimExpr>>()) {
        const auto& [negtaive_operand] =
            *operand.Get<symbol::Reciprocal<symbol::DimExpr>>();
        ADT_LET_CONST_REF(operand_str, CodeGen(negtaive_operand));
        ret += " / " + operand_str;
      } else {
        ADT_LET_CONST_REF(operand_str, CodeGen(operand));
        ret += " * " + operand_str;
      }
    }
    return std::string() + "(" + ret + ")";
  }

  adt::Result<std::string> CodeGenImpl(
      const symbol::Max<symbol::DimExpr>& dim_expr) {
    ADT_CHECK(dim_expr.operands->size() > 0);
    ADT_LET_CONST_REF(first, CodeGen(dim_expr.operands->at(0)));
    const std::string& var_name = ap::common::NewUniqueId("_ap_sym");
    ss() << index_type_name_ << " " << var_name << " = " << first << ";\n";
    for (int i = 1; i < dim_expr.operands->size(); ++i) {
      const auto& operand = dim_expr.operands->at(i);
      const std::string& operand_var_name = ap::common::NewUniqueId("_ap_sym");
      ADT_LET_CONST_REF(operand_str, CodeGen(operand));
      ss() << index_type_name_ << " " << operand_var_name << " = "
           << operand_str << ";\n";
      ss() << var_name << " = (" << operand_var_name << " > " << var_name
           << " ? " << operand_var_name << " : " << var_name << ");\n";
    }
    return var_name;
  }

  adt::Result<std::string> CodeGenImpl(
      const symbol::Min<symbol::DimExpr>& dim_expr) {
    ADT_CHECK(dim_expr.operands->size() > 0);
    ADT_LET_CONST_REF(first, CodeGen(dim_expr.operands->at(0)));
    const std::string& var_name = ap::common::NewUniqueId("_ap_sym");
    ss() << index_type_name_ << " " << var_name << " = " << first << ";\n";
    for (int i = 1; i < dim_expr.operands->size(); ++i) {
      const auto& operand = dim_expr.operands->at(i);
      const std::string& operand_var_name = ap::common::NewUniqueId("_ap_sym");
      ADT_LET_CONST_REF(operand_str, CodeGen(operand));
      ss() << index_type_name_ << " " << operand_var_name << " = "
           << operand_str << ";\n";
      ss() << var_name << " = (" << operand_var_name << " < " << var_name
           << " ? " << operand_var_name << " : " << var_name << ");\n";
    }
    return var_name;
  }

  adt::Result<std::string> CodeGenImpl(
      const symbol::Broadcast<symbol::DimExpr>& dim_expr) {
    ADT_CHECK(dim_expr.operands->size() > 0);
    ADT_LET_CONST_REF(first, CodeGen(dim_expr.operands->at(0)));
    const std::string& var_name = ap::common::NewUniqueId("_ap_sym");
    ss() << index_type_name_ << " " << var_name << " = " << first << ";\n";
    for (int i = 1; i < dim_expr.operands->size(); ++i) {
      const auto& operand = dim_expr.operands->at(i);
      const std::string& operand_var_name = ap::common::NewUniqueId("_ap_sym");
      ADT_LET_CONST_REF(operand_str, CodeGen(operand));
      ss() << index_type_name_ << " " << operand_var_name << " = "
           << operand_str << ";\n";
      ss() << var_name << " = (" << operand_var_name << " > " << var_name
           << " ? " << operand_var_name << " : " << var_name << ");\n";
    }
    return var_name;
  }

  std::ostringstream* ss_;
  ArgName4DimExprT ArgName4DimExpr;
  std::string index_type_name_;
};

}  // namespace ap::index_expr
