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

#include "paddle/ap/include/index_expr/index_tuple_expr.h"
#include "paddle/ap/include/paddle/indexed_ir_graph.h"
#include "paddle/ap/include/paddle/pir_node.h"
#include "paddle/ap/include/paddle/pir_util.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace ap::paddle {

adt::Result<IndexedIrGraph> CreatePureElementwiseIndexedIrGraph(
    const PackedIrOp& pir_op, const index_expr::IndexTupleExpr& indexes_expr);

adt::Result<adt::Ok> GetPackedIrOpInputsOutputs(
    const PackedIrOp& pir_op,
    std::vector<pir::Value>* inputs,
    std::vector<pir::Value>* yield_op_inputs,
    std::vector<pir::Value>* outputs);

namespace detail {

struct CreatePureElementwiseIndexedIrGraphHelper {
  struct Ctx {
    std::unordered_map<pir::Value, IndexedIrValue<IndexedIrNode>> value2node;

    bool Has(pir::Value value) const {
      return this->value2node.find(value) != this->value2node.end();
    }

    void Insert(pir::Value value, const IndexedIrValue<IndexedIrNode>& node) {
      this->value2node[value] = node;
    }

    adt::Result<IndexedIrValue<IndexedIrNode>> Get(pir::Value value) const {
      const auto& iter = this->value2node.find(value);
      ADT_CHECK(iter != this->value2node.end());
      return iter->second;
    }
  };

  adt::Result<PureElementwiseIndexedIrGraph> Create(
      const PackedIrOp& pir_op,
      const index_expr::IndexTupleExpr& indexes_expr) {
    Ctx ctx{};
    ADT_LET_CONST_REF(node_arena_ptr,
                      CreateNodeArena(&ctx, pir_op, indexes_expr));
    return CreatePureElementwiseIndexedIrGraph(node_arena_ptr, ctx, pir_op);
  }

  adt::Result<PureElementwiseIndexedIrGraph>
  CreatePureElementwiseIndexedIrGraph(const IndexedIrNodeArenaPtr& node_arena,
                                      const Ctx& ctx,
                                      const PackedIrOp& pir_op) {
    std::vector<pir::Value> inputs;
    std::vector<pir::Value> yield_op_inputs;
    std::vector<pir::Value> outputs;
    ADT_RETURN_IF_ERR(GetPackedIrOpInputsOutputs(
        pir_op, &inputs, &yield_op_inputs, &outputs));
    ADT_LET_CONST_REF(input_nodes, GetIndexedIrValues(ctx, inputs));
    ADT_LET_CONST_REF(yield_op_input_nodes,
                      GetIndexedIrValues(ctx, yield_op_inputs));
    return PureElementwiseIndexedIrGraph{
        node_arena, input_nodes, yield_op_input_nodes, outputs, ctx.value2node};
  }

  adt::Result<std::vector<IndexedIrValue<IndexedIrNode>>> GetIndexedIrValues(
      const Ctx& ctx, const std::vector<pir::Value> values) {
    std::vector<IndexedIrValue<IndexedIrNode>> ret;
    ret.reserve(values.size());
    for (const auto& value : values) {
      ADT_LET_CONST_REF(ir_value, ctx.Get(value));
      ret.emplace_back(ir_value);
    }
    return ret;
  }

  adt::Result<IndexedIrNodeArenaPtr> CreateNodeArena(
      Ctx* ctx,
      const PackedIrOp& pir_op,
      const index_expr::IndexTupleExpr& indexes_expr) {
    auto node_arena = std::make_shared<IndexedIrNodeArena>();
    for (auto& op : *pir_op.fusion_op.block()) {
      if (op.isa<pir::YieldOp>()) {
        continue;
      }
      const auto& ir_op = InsertOpNode(node_arena, &op);
      InsertValueNodes(ctx, node_arena, &op, indexes_expr);
      ADT_RETURN_IF_ERR(ConnectOpOperandEdges(ctx, ir_op));
      ADT_RETURN_IF_ERR(ConnectOpResultEdges(ctx, ir_op));
    }
    return node_arena;
  }

  adt::Result<adt::Ok> ConnectOpResultEdges(
      Ctx* ctx, const IndexedIrOp<IndexedIrNode>& ir_op) {
    auto* op = ir_op->op;
    for (int i = 0; i < op->num_results(); ++i) {
      ADT_LET_CONST_REF(ir_value, ctx->Get(op->result(i)));
      ADT_RETURN_IF_ERR(
          ir_op->node.ConnectTo(ir_value->node,
                                graph::IndexedTag<std::monostate>{},
                                graph::UnindexedTag<std::monostate>{}));
    }
    return adt::Ok{};
  }

  adt::Result<adt::Ok> ConnectOpOperandEdges(
      Ctx* ctx, const IndexedIrOp<IndexedIrNode>& ir_op) {
    auto* op = ir_op->op;
    for (int i = 0; i < op->num_operands(); ++i) {
      ADT_LET_CONST_REF(ir_value, ctx->Get(op->operand_source(i)));
      ADT_RETURN_IF_ERR(
          ir_value->node.ConnectTo(ir_op->node,
                                   graph::UnindexedTag<std::monostate>{},
                                   graph::IndexedTag<std::monostate>{}));
    }
    return adt::Ok{};
  }

  void InsertValueNodes(Ctx* ctx,
                        const IndexedIrNodeArenaPtr& node_arena,
                        pir::Operation* op,
                        const index_expr::IndexTupleExpr& indexes_expr) {
    VisitInOutValue(op, [&](pir::Value value) {
      const auto& ir_node = node_arena->New([&](const auto& node) {
        return IndexedIrValue<IndexedIrNode>{node, value, indexes_expr};
      });
      const auto& ir_value =
          ir_node.template Get<IndexedIrValue<IndexedIrNode>>();
      if (!ctx->Has(value)) {
        ctx->Insert(value, ir_value);
      }
    });
  }

  template <typename DoEachT>
  void VisitInOutValue(pir::Operation* op, const DoEachT& DoEach) {
    for (int i = 0; i < op->num_operands(); ++i) {
      DoEach(op->operand_source(i));
    }
    for (int i = 0; i < op->num_results(); ++i) {
      DoEach(op->result(i));
    }
  }

  IndexedIrOp<IndexedIrNode> InsertOpNode(
      const IndexedIrNodeArenaPtr& node_arena, pir::Operation* op) {
    const auto& ir_node = node_arena->New([&](const auto& node) {
      return IndexedIrOp<IndexedIrNode>{node, op};
    });
    return ir_node.template Get<IndexedIrOp<IndexedIrNode>>();
  }
};

}  // namespace detail

inline adt::Result<IndexedIrGraph> CreatePureElementwiseIndexedIrGraph(
    const PackedIrOp& pir_op, const index_expr::IndexTupleExpr& indexes_expr) {
  detail::CreatePureElementwiseIndexedIrGraphHelper helper{};
  ADT_LET_CONST_REF(ir_graph, helper.Create(pir_op, indexes_expr));
  return ir_graph;
}

namespace detail {

struct GetPackedIrOpInputsOutputsHelper {
  adt::Result<adt::Ok> GetPackedIrOpInputsOutputs(
      const PackedIrOp& pir_op,
      std::vector<pir::Value>* inputs,
      std::vector<pir::Value>* yield_op_inputs,
      std::vector<pir::Value>* outputs) {
    *inputs = ap::paddle::GetUsedExternalValue(*pir_op.fusion_op);
    outputs->clear();
    outputs->reserve(pir_op.fusion_op->num_results());
    for (int i = 0; i < pir_op.fusion_op->num_results(); ++i) {
      outputs->emplace_back(pir_op.fusion_op->result(i));
    }
    bool found_yield_op = false;
    for (const auto& op : *pir_op.fusion_op.block()) {
      yield_op_inputs->clear();
      yield_op_inputs->reserve(op.num_operands());
      if (op.isa<pir::YieldOp>()) {
        for (int i = 0; i < op.num_operands(); ++i) {
          yield_op_inputs->emplace_back(op.operand_source(i));
        }
        found_yield_op = true;
      }
    }
    if (found_yield_op) {
      return adt::Ok{};
    } else {
      return adt::errors::ValueError{
          "No yield op have been found in fusion op block."};
    }
  }
};

}  // namespace detail

inline adt::Result<adt::Ok> GetPackedIrOpInputsOutputs(
    const PackedIrOp& pir_op,
    std::vector<pir::Value>* inputs,
    std::vector<pir::Value>* yield_op_inputs,
    std::vector<pir::Value>* outputs) {
  detail::GetPackedIrOpInputsOutputsHelper helper{};
  return helper.GetPackedIrOpInputsOutputs(
      pir_op, inputs, yield_op_inputs, outputs);
}

}  // namespace ap::paddle
