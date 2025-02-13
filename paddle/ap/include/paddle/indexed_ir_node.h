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
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/index_expr/index_tuple_expr.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

namespace ap::paddle {

inline void ConvertToAZaz09_(std::string* str) {
  for (int i = 0; i < str->size(); ++i) {
    char* ch = &str->at(i);
    if (*ch >= 'a' && *ch <= 'z') {
      continue;
    }
    if (*ch >= 'A' && *ch <= 'Z') {
      continue;
    }
    if (*ch >= '0' && *ch <= '9') {
      continue;
    }
    *ch = '_';
  }
}

inline std::string GetOpUniqueName(const pir::Operation* op) {
  std::string op_name = op->name();
  ConvertToAZaz09_(&op_name);
  return op_name + "_" + std::to_string(op->id());
}

template <typename NodeT>
struct IndexedIrValueImpl {
  graph::Node<NodeT> node;
  pir::Value value;
  index_expr::IndexTupleExpr indexes_expr;

  std::string GetUniqueNameInsideNodeArena() const {
    if (value.defining_op()) {
      return GetOpUniqueName(value.defining_op()) + "_out_" +
             std::to_string(node.node_id().value());
    } else {
      return std::string() + "non_op_out_" +
             std::to_string(node.node_id().value());
    }
  }

  bool operator==(const IndexedIrValueImpl& other) const {
    return this->value == other.value &&
           this->indexes_expr == other.indexes_expr;
  }
};

template <typename NodeT>
ADT_DEFINE_RC(IndexedIrValue, IndexedIrValueImpl<NodeT>);

template <typename NodeT>
struct IndexedIrOpImpl {
  graph::Node<NodeT> node;
  pir::Operation* op;

  std::string GetUniqueNameInsideNodeArena() const {
    return GetOpUniqueName(op) + +"_" + std::to_string(node.node_id().value());
  }

  bool operator==(const IndexedIrOpImpl& other) const {
    return this->op == other.op;
  }
};

template <typename NodeT>
ADT_DEFINE_RC(IndexedIrOp, IndexedIrOpImpl<NodeT>);

template <typename NodeT>
using IndexedIrNodeImpl =
    std::variant<IndexedIrValue<NodeT>, IndexedIrOp<NodeT>>;

struct IndexedIrNode : public IndexedIrNodeImpl<IndexedIrNode> {
  using IndexedIrNodeImpl<IndexedIrNode>::IndexedIrNodeImpl;

  ADT_DEFINE_VARIANT_METHODS(IndexedIrNodeImpl<IndexedIrNode>);

  const graph::Node<IndexedIrNode>& node() const {
    return Match([](const auto& impl) -> const graph::Node<IndexedIrNode>& {
      return impl->node;
    });
  }
};

}  // namespace ap::paddle
