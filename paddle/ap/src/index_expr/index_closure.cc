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

#include "paddle/ap/include/index_expr/index_closure.h"
#include "paddle/ap/include/index_expr/op_index_tuple_expr_signature.h"
#include "paddle/ap/include/index_expr/valid_index_expr_builder.h"

namespace ap::index_expr {

adt::Result<OpIndexTupleExprSignature> OrderedOneofIndexClosureImpl::operator()(
    const IndexTupleExpr& indexes_expr) const {
  size_t count = 0;
  for (const auto& [_, lambdas] : nice2index_lambdas) {
    for (const auto& lambda : lambdas) {
      const auto& res = CallLambda(lambda, indexes_expr);
      if (res.Has<OpIndexTupleExprSignature>()) {
        return res.Get<OpIndexTupleExprSignature>();
      }
      ++count;
    }
  }
  return adt::errors::ValueError{
      std::string() +
      "all index closure failed. tried count: " + std::to_string(count)};
}

adt::Result<OpIndexTupleExprSignature> OrderedOneofIndexClosureImpl::CallLambda(
    const Lambda<CoreExpr>& lambda, const IndexTupleExpr& indexes_expr) const {
  axpr::BuiltinClassInstance<Val> instance{GetIndexTupleExprClass<Val>(),
                                           indexes_expr};
  const std::vector<Val> args{closure_data.ctx,
                              closure_data.inputs_meta,
                              closure_data.outputs_meta,
                              closure_data.in_vars,
                              Val{instance}};
  const auto& opt_ret = (*this->interpreter)(lambda, args);
  ADT_RETURN_IF_ERR(opt_ret);
  const auto& ret = opt_ret.GetOkValue();
  return ret.template CastTo<OpIndexTupleExprSignature>();
}

namespace {

template <typename IndexesTransformApplyT>
adt::Result<OpIndexTupleExprSignature> OpIndexesTransformApply(
    const OpIndexesTransformSignature& indexes_transform_signature,
    const IndexesTransformApplyT& IndexesTransformApply) {
  InIndexTupleExprSignature in_sig;
  for (const auto& transform :
       *indexes_transform_signature.in_signature.descriptors) {
    const auto& converted = IndexesTransformApply(transform);
    ADT_RETURN_IF_ERR(converted);
    in_sig.descriptors->emplace_back(converted.GetOkValue());
  }
  OutIndexTupleExprSignature out_sig;
  for (const auto& transform :
       *indexes_transform_signature.out_signature.descriptors) {
    const auto& converted = IndexesTransformApply(transform);
    ADT_RETURN_IF_ERR(converted);
    out_sig.descriptors->emplace_back(converted.GetOkValue());
  }
  return OpIndexTupleExprSignature{in_sig, out_sig};
}

}  // namespace

adt::Result<OpIndexTupleExprSignature> RecordableIndexClosureImpl::operator()(
    const IndexTupleExpr& indexes_expr) const {
  const auto& ApplyTransform = [&](const TrackedIndexesTransform& transform) {
    return transform.Match(
        [&](const adt::IdentityFunc&) -> adt::Result<IndexTupleExpr> {
          return indexes_expr;
        },
        [&](const IndexTupleExpr& tacked_indexes_expr_as_func)
            -> adt::Result<IndexTupleExpr> {
          return ValidIndexExprBuilder().Compose(tacked_indexes_expr_as_func,
                                                 indexes_expr);
        });
  };
  return OpIndexesTransformApply(this->op_indexes_transform_signature,
                                 ApplyTransform);
}

adt::Result<OpIndexTupleExprSignature> IndexClosure::operator()(
    const IndexTupleExpr& indexes_expr) const {
  return Match([&](const auto& impl) { return (*impl)(indexes_expr); });
}

}  // namespace ap::index_expr
