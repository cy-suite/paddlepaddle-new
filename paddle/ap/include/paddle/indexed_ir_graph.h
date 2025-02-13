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
#include "paddle/ap/include/graph/node_arena.h"
#include "paddle/ap/include/paddle/indexed_ir_node.h"

namespace ap::paddle {

using IndexedIrNodeArena = graph::NodeArena<IndexedIrNode>;
using IndexedIrNodeArenaPtr = std::shared_ptr<IndexedIrNodeArena>;

struct PureElementwiseIndexedIrGraphImpl {
  IndexedIrNodeArenaPtr node_arena;
  // free values in fusion op block.
  std::vector<IndexedIrValue<IndexedIrNode>> inputs;
  // yield values in fusion op block.
  std::vector<IndexedIrValue<IndexedIrNode>> yield_op_inputs;
  // output values of fusion op.
  std::vector<pir::Value> outputs;

  std::unordered_map<pir::Value, IndexedIrValue<IndexedIrNode>>
      pir_value2indexed_ir_value;

  adt::Result<IndexedIrValue<IndexedIrNode>> GetIndexedIrValue(
      pir::Value value) const {
    const auto& iter = this->pir_value2indexed_ir_value.find(value);
    ADT_CHECK(iter != this->pir_value2indexed_ir_value.end());
    return iter->second;
  }

  bool operator==(const PureElementwiseIndexedIrGraphImpl& other) const {
    return this == &other;
  }
};

ADT_DEFINE_RC(PureElementwiseIndexedIrGraph, PureElementwiseIndexedIrGraphImpl);

using IndexedIrGraphImpl = std::variant<PureElementwiseIndexedIrGraph>;

struct IndexedIrGraph : public IndexedIrGraphImpl {
  using IndexedIrGraphImpl::IndexedIrGraphImpl;

  ADT_DEFINE_VARIANT_METHODS(IndexedIrGraphImpl);
};

}  // namespace ap::paddle
