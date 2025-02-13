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
#include <vector>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/common/unique_id.h"
#include "paddle/ap/include/index_expr/dim_expr_cuda_code_generator.h"
#include "paddle/ap/include/index_expr/index_tuple_expr.h"

namespace ap::index_expr {

class IndexTupleExprCudaCodeGenerator {
 public:
  using ArgName4DimExprT =
      std::function<std::optional<std::string>(const symbol::DimExpr&)>;
  IndexTupleExprCudaCodeGenerator(
      std::ostringstream* ss,
      const std::vector<std::string>& loop_var_names,
      const ArgName4DimExprT& ArgName4DimExpr)
      : ss_(ss),
        loop_var_names_(loop_var_names),
        index_type_name_("int64_t"),
        dim_expr_code_gen_(ss, ArgName4DimExpr, "int64_t") {}

  std::ostringstream& ss() { return *ss_; }

  adt::Result<std::string> CodeGen(const IndexTupleExpr& indexes_expr) {
    return indexes_expr.Match(
        [&](const IndexTupleExprDomain& domain) -> adt::Result<std::string> {
          return CodeGenImpl(domain);
        },
        [&](const auto& impl) -> adt::Result<std::string> {
          return adt::errors::NotImplementedError{
              std::string() +
              "IndexTupleExprCudaCodeGenerator::CodeGen not support " +
              impl->TypeName() + " yet."};
        });
  }

 private:
  adt::Result<std::string> CodeGenImpl(const IndexTupleExprDomain& domain) {
    const auto& var_name = NewTmpVarName("_ap_i");
    int i = 0;
    auto DoEachPair = [&](const auto& iter,
                          const auto& stride) -> adt::Result<adt::Ok> {
      if (i++ == 0) {
        ADT_CHECK(stride == symbol::DimExpr{int64_t(1)});
        ss() << index_type_name_ << " " << var_name << " = " << iter << ";\n";
      } else {
        ADT_LET_CONST_REF(stride_var_name, dim_expr_code_gen_.CodeGen(stride));
        ss() << var_name << " += " << iter << " * " << stride_var_name << ";\n";
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitEachIterAndStride(domain->ranges, DoEachPair));
    return var_name;
  }

  template <typename DoEachPairT>
  adt::Result<adt::Ok> VisitEachIterAndStride(
      const adt::List<symbol::DimExpr>& ranges, const DoEachPairT& DoEachPair) {
    symbol::DimExpr stride{int64_t(1)};
    ADT_CHECK(loop_var_names_.size() == ranges->size());
    for (int i = loop_var_names_.size() - 1; i >= 0; --i) {
      const auto& iter_var_name = loop_var_names_.at(i);
      const auto& dim = ranges->at(i);
      ADT_RETURN_IF_ERR(DoEachPair(iter_var_name, stride));
      stride = stride * dim;
    }
    return adt::Ok{};
  }

  std::string NewTmpVarName(const std::string& prefix) {
    return ap::common::NewUniqueId(prefix);
  }

  std::ostringstream* ss_;
  std::vector<std::string> loop_var_names_;
  std::string index_type_name_;
  DimExprCudaCodeGenerator dim_expr_code_gen_;
};

}  // namespace ap::index_expr
