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

#include <map>
#include <variant>
#include "paddle/ap/include/axpr/core_expr.h"
#include "paddle/ap/include/axpr/error.h"
#include "paddle/ap/include/index_expr/index_expr.h"
#include "paddle/ap/include/index_expr/index_expr_interpreter.h"
#include "paddle/ap/include/index_expr/op_index_tuple_expr_signature.h"
#include "paddle/ap/include/index_expr/value.h"
#include "paddle/ap/include/index_expr/value_method_class.h"

namespace ap::index_expr {

using axpr::CoreExpr;
using axpr::Lambda;

struct IndexClosureData {
  const ap::index_expr::Val ctx;
  const adt::List<ap::index_expr::Val> inputs_meta;
  const adt::List<ap::index_expr::Val> outputs_meta;
  const adt::List<ap::index_expr::Val> in_vars;

  bool operator==(const IndexClosureData& other) const {
    return other.ctx == this->ctx && other.inputs_meta == this->inputs_meta &&
           other.outputs_meta == this->outputs_meta &&
           other.in_vars == this->in_vars;
  }
};

using Nice2IndexLambdas =
    std::map<int64_t, std::vector<axpr::Lambda<axpr::CoreExpr>>>;

struct OrderedOneofIndexClosureImpl {
  std::shared_ptr<IndexExprInterpreter> interpreter;
  IndexClosureData closure_data;
  Nice2IndexLambdas nice2index_lambdas;

  adt::Result<OpIndexTupleExprSignature> operator()(
      const IndexTupleExpr&) const;

  bool operator==(const OrderedOneofIndexClosureImpl& other) const {
    return other.interpreter == this->interpreter &&
           other.closure_data == this->closure_data &&
           other.nice2index_lambdas == this->nice2index_lambdas;
  }

 private:
  adt::Result<OpIndexTupleExprSignature> CallLambda(
      const Lambda<CoreExpr>& lambda, const IndexTupleExpr&) const;
};
ADT_DEFINE_RC(OrderedOneofIndexClosure, OrderedOneofIndexClosureImpl);

using TrackedIndexesTransformImpl =
    std::variant<adt::IdentityFunc, ap::index_expr::IndexTupleExpr>;

struct TrackedIndexesTransform : public TrackedIndexesTransformImpl {
  using TrackedIndexesTransformImpl::TrackedIndexesTransformImpl;
  ADT_DEFINE_VARIANT_METHODS(TrackedIndexesTransformImpl);
};

using OpIndexesTransformSignature =
    ap::index_expr::OpSignature<TrackedIndexesTransform>;

struct RecordableIndexClosureImpl {
  OpIndexesTransformSignature op_indexes_transform_signature;

  adt::Result<OpIndexTupleExprSignature> operator()(
      const IndexTupleExpr&) const;

  bool operator==(const RecordableIndexClosureImpl& other) const {
    return other.op_indexes_transform_signature ==
           this->op_indexes_transform_signature;
  }
};
ADT_DEFINE_RC(RecordableIndexClosure, RecordableIndexClosureImpl);

using IndexClosureImpl =
    std::variant<OrderedOneofIndexClosure, RecordableIndexClosure>;

struct IndexClosure : public IndexClosureImpl {
  using IndexClosureImpl::IndexClosureImpl;
  ADT_DEFINE_VARIANT_METHODS(IndexClosureImpl);

  adt::Result<OpIndexTupleExprSignature> operator()(
      const IndexTupleExpr& indexes_expr) const;
};

}  // namespace ap::index_expr
