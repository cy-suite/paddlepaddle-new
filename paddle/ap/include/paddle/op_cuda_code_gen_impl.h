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
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/axpr/pointer_type_util.h"
#include "paddle/ap/include/code_gen/code_gen_ctx.h"
#include "paddle/ap/include/code_gen/dim_expr_kernel_arg_id.h"
#include "paddle/ap/include/code_gen/op_code_gen_ctx.h"
#include "paddle/ap/include/code_gen/op_cuda_gen_impl.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/topo_kind.h"
#include "paddle/ap/include/drr/value.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/index_expr/index_tuple_expr_cuda_code_generator.h"
#include "paddle/ap/include/ir_match/native_or_ref_ir_value.h"
#include "paddle/ap/include/paddle/indexed_ir_graph_util.h"
#include "paddle/ap/include/paddle/pir_graph_descriptor.h"
#include "paddle/ap/include/paddle/pir_node.h"
#include "paddle/ap/include/registry/registry.h"
#include "paddle/ap/include/registry/registry_mgr.h"
#include "paddle/ap/include/registry/registry_singleton.h"
#include "paddle/ap/include/registry/value.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/include/core/builtin_type.h"

namespace ap::paddle {

struct OpCudaCodeGenImpl {
  using BirNode = PirNode;
  using OpCodeGenCtx = code_gen::OpCodeGenCtx<BirNode>;
  using IrOp = code_gen::IrOp<BirNode>;

  using DrrValue = drr::Value;
  using DrrNode = drr::Node;
  using DrrGraphNode = graph::Node<DrrNode>;
  using DrrPackedIrOp = drr::PackedIrOp<DrrNode>;
  using DrrOptPackedIrOp = drr::OptPackedIrOp<DrrNode>;
  using DrrOptPackedIrOpOperand = drr::OptPackedIrOpOperand<DrrNode>;
  using DrrOptPackedIrOpResult = drr::OptPackedIrOpResult<DrrNode>;

  using DrrTrivialFusionIrOpImpl =
      std::variant<DrrPackedIrOp, DrrOptPackedIrOp>;
  struct DrrTrivialFusionIrOp : public DrrTrivialFusionIrOpImpl {
    using DrrTrivialFusionIrOpImpl::DrrTrivialFusionIrOpImpl;
    ADT_DEFINE_VARIANT_METHODS(DrrTrivialFusionIrOpImpl);

    DrrGraphNode node() const {
      return Match([](const auto& impl) { return impl->node; });
    }
  };

  using DrrNativeIrValue = drr::NativeIrValue<DrrNode>;
  using DrrPackedIrValue = drr::PackedIrValue<DrrNode>;
  using IndexTupleExpr = index_expr::IndexTupleExpr;

  using GraphMatchCtx = ir_match::GraphMatchCtx<BirNode>;

  using Registry = registry::Registry;

  using ClassAttrs = axpr::ClassAttrs<axpr::SerializableValue>;

  using Function = axpr::Function<axpr::SerializableValue>;

  adt::Result<ClassAttrs> ConvertFusionOpToClassAttrs(
      const OpCodeGenCtx& op_code_gen_ctx, const IrOp& ir_op) {
    using RetT = adt::Result<ClassAttrs>;
    return ir_op.Match(
        [&](const PackedIrOp& packed_ir_op) -> RetT {
          return PackedIrOpConvertFusionOpToClassAttrs(op_code_gen_ctx,
                                                       packed_ir_op);
        },
        [&](const RefIrOp& ref_ir_op) -> RetT {
          return RefIrOpConvertFusionOpToClassAttrs(op_code_gen_ctx, ref_ir_op);
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              std::string() +
              "only packed ir op get supported in ConvertFusionOpToLambda."};
        });
  }

  adt::Result<ClassAttrs> PackedIrOpConvertFusionOpToClassAttrs(
      const OpCodeGenCtx& op_code_gen_ctx, const PackedIrOp& packed_ir_op) {
    ADT_LET_CONST_REF(
        index_tuple_expr,
        GetPureElementwiseLoopIndexTupleExpr(op_code_gen_ctx, packed_ir_op));
    ADT_LET_CONST_REF(
        ir_graph,
        CreatePureElementwiseIndexedIrGraph(packed_ir_op, index_tuple_expr));
    ADT_LET_CONST_REF(init_func,
                      PackedIrOpMakeInitFuncByFusionOp(
                          op_code_gen_ctx, ir_graph, packed_ir_op));
    ADT_LET_CONST_REF(compute_func,
                      PackedIrOpMakeComputeFuncByFusionOp(
                          op_code_gen_ctx, ir_graph, packed_ir_op));
    ADT_LET_CONST_REF(load_from_register_func,
                      PackedIrOpMakeLoadFromRegisterFuncByFusionOp(
                          op_code_gen_ctx, ir_graph, packed_ir_op));
    ADT_LET_CONST_REF(store_to_register_func,
                      PackedIrOpMakeStoreToRegisterFuncByFusionOp(
                          op_code_gen_ctx, ir_graph, packed_ir_op));
    std::string class_name = "PackedIrOpClass";
    adt::List<std::shared_ptr<axpr::ClassAttrsImpl<axpr::SerializableValue>>>
        empty_bases{};
    axpr::AttrMap<axpr::SerializableValue> methods{};
    methods->Set("__init__", init_func);
    methods->Set("compute", compute_func);
    methods->Set("load_from_register", load_from_register_func);
    methods->Set("store_to_register", store_to_register_func);
    return ClassAttrs{class_name, empty_bases, methods};
  }

  adt::Result<index_expr::IndexTupleExpr> GetPureElementwiseLoopIndexTupleExpr(
      const OpCodeGenCtx& op_code_gen_ctx, const PackedIrOp& packed_ir_op) {
    ADT_LET_CONST_REF(
        shape, GetPureElementwiseLoopDimExpr(op_code_gen_ctx, packed_ir_op));
    return index_expr::IndexTupleExprDomain{shape};
  }

  adt::Result<adt::List<symbol::DimExpr>> GetPureElementwiseLoopDimExpr(
      const OpCodeGenCtx& op_code_gen_ctx, const PackedIrOp& packed_ir_op) {
    const auto& input_flags = op_code_gen_ctx->input_index_loop_anchor_flags;
    {
      ADT_LET_CONST_REF(
          num_native_ir_inputs,
          NumNativeIrInputBirValues(op_code_gen_ctx, packed_ir_op));
      ADT_CHECK(input_flags->size() == num_native_ir_inputs)
          << adt::errors::TypeError{
                 std::string() +
                 "len(input_index_loop_anchor_flags) should equal to number of "
                 "native ir inputs of fusion op. (" +
                 std::to_string(input_flags->size()) + " v.s. " +
                 std::to_string(num_native_ir_inputs) + ")"};
    }
    const auto& output_flags = op_code_gen_ctx->output_index_loop_anchor_flags;
    {
      ADT_LET_CONST_REF(
          num_native_ir_outputs,
          NumNativeIrOutputBirValues(op_code_gen_ctx, packed_ir_op));
      ADT_CHECK(output_flags->size() == num_native_ir_outputs)
          << adt::errors::TypeError{
                 std::string() +
                 "len(output_index_loop_anchor_flags) should equal to number "
                 "of native ir outputs of fusion op. (" +
                 std::to_string(output_flags->size()) + " v.s. " +
                 std::to_string(num_native_ir_outputs) + ")"};
    }
    using Shape = adt::List<symbol::DimExpr>;
    auto GetShape = [&](pir::Value value) -> adt::Result<Shape> {
      ADT_LET_CONST_REF(shape_ptr, NativeIrValue{value}.GetShapeDimExprsPtr());
      Shape shape;
      shape->reserve(shape_ptr->size());
      shape->assign(shape_ptr->begin(), shape_ptr->end());
      return shape;
    };
    std::optional<Shape> opt_shape;
    auto InitOrCheckShape = [&](pir::Value value) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(shape, GetShape(value));
      if (opt_shape.has_value()) {
        ADT_CHECK(opt_shape.value() == shape) << adt::errors::TypeError{
            "All loop anchors should have same shapes."};
      } else {
        opt_shape = shape;
      }
      return adt::Ok{};
    };
    {
      int input_idx = 0;
      auto DoEachNativeInput = [&](pir::Value value) -> adt::Result<adt::Ok> {
        if (input_flags->at(input_idx++).value()) {
          ADT_RETURN_IF_ERR(InitOrCheckShape(value));
        }
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(VisitNativeIrInputBirValue(
          op_code_gen_ctx, packed_ir_op, DoEachNativeInput));
    }
    {
      int output_idx = 0;
      auto DoEachNativeOutput = [&](pir::Value value) -> adt::Result<adt::Ok> {
        if (output_flags->at(output_idx++).value()) {
          ADT_RETURN_IF_ERR(InitOrCheckShape(value));
        }
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(VisitNativeIrOutputBirValue(
          op_code_gen_ctx, packed_ir_op, DoEachNativeOutput));
    }
    ADT_CHECK(opt_shape.has_value()) << adt::errors::TypeError{
        "At least one flag should be set in input_index_loop_anchor_flags or "
        "output_index_loop_anchor_flags"};
    return opt_shape.value();
  }

  adt::Result<std::size_t> NumNativeIrInputBirValues(
      const OpCodeGenCtx& op_code_gen_ctx, const PackedIrOp& packed_ir_op) {
    std::size_t num_values = 0;
    auto Acc = [&](pir::Value) -> adt::Result<adt::Ok> {
      ++num_values;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitNativeIrInputBirValue(op_code_gen_ctx, packed_ir_op, Acc));
    return num_values;
  }

  adt::Result<std::size_t> NumNativeIrOutputBirValues(
      const OpCodeGenCtx& op_code_gen_ctx, const PackedIrOp& packed_ir_op) {
    std::size_t num_values = 0;
    auto Acc = [&](pir::Value) -> adt::Result<adt::Ok> {
      ++num_values;
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(
        VisitNativeIrOutputBirValue(op_code_gen_ctx, packed_ir_op, Acc));
    return num_values;
  }

  adt::Result<ClassAttrs> RefIrOpConvertFusionOpToClassAttrs(
      const OpCodeGenCtx& op_code_gen_ctx, const RefIrOp& ref_ir_op) {
    ADT_LET_CONST_REF(
        init_func, RefIrOpMakeInitFuncByFusionOp(op_code_gen_ctx, ref_ir_op));
    ADT_LET_CONST_REF(
        compute_func,
        RefIrOpMakeComputeFuncByFusionOp(op_code_gen_ctx, ref_ir_op));
    ADT_LET_CONST_REF(
        load_from_register_func,
        RefIrOpMakeLoadFromRegisterFuncByFusionOp(op_code_gen_ctx, ref_ir_op));
    ADT_LET_CONST_REF(
        store_to_register_func,
        RefIrOpMakeStoreToRegisterFuncByFusionOp(op_code_gen_ctx, ref_ir_op));
    std::string class_name = "RefIrOpClass";
    adt::List<std::shared_ptr<axpr::ClassAttrsImpl<axpr::SerializableValue>>>
        empty_bases{};
    axpr::AttrMap<axpr::SerializableValue> methods{};
    methods->Set("__init__", init_func);
    methods->Set("compute", compute_func);
    methods->Set("load_from_register", load_from_register_func);
    methods->Set("store_to_register", load_from_register_func);
    return ClassAttrs{class_name, empty_bases, methods};
  }

  adt::Result<Function> PackedIrOpMakeInitFuncByFusionOp(
      const OpCodeGenCtx& op_code_gen_ctx,
      const IndexedIrGraph& ir_graph,
      const PackedIrOp& packed_ir_op) {
    return ir_graph.Match([&](const auto& impl) -> adt::Result<Function> {
      return PackedIrOpMakeInitFuncByFusionOpImpl(
          op_code_gen_ctx, impl, packed_ir_op);
    });
  }

  adt::Result<Function> PackedIrOpMakeStoreToRegisterFuncByFusionOp(
      const OpCodeGenCtx& op_code_gen_ctx,
      const IndexedIrGraph& ir_graph,
      const PackedIrOp& packed_ir_op) {
    return ir_graph.Match([&](const auto& impl) -> adt::Result<Function> {
      return PackedIrOpMakeStoreToRegisterFuncByFusionOpImpl(
          op_code_gen_ctx, impl, packed_ir_op);
    });
  }

  adt::Result<Function> PackedIrOpMakeLoadFromRegisterFuncByFusionOp(
      const OpCodeGenCtx& op_code_gen_ctx,
      const IndexedIrGraph& ir_graph,
      const PackedIrOp& packed_ir_op) {
    return ir_graph.Match([&](const auto& impl) -> adt::Result<Function> {
      return PackedIrOpMakeLoadFromRegisterFuncByFusionOpImpl(
          op_code_gen_ctx, impl, packed_ir_op);
    });
  }

  adt::Result<Function> PackedIrOpMakeLoadFromRegisterFuncByFusionOpImpl(
      const OpCodeGenCtx& op_code_gen_ctx,
      const PureElementwiseIndexedIrGraph& ir_graph,
      const PackedIrOp& packed_ir_op) {
    axpr::LambdaExprBuilder lmbd;
    auto GetMapFunc = [&](auto& ctx) -> axpr::AnfExpr {
      auto& value_class_var =
          ctx.Var("self").Attr("class_factory").Attr("get_value_class").Call();
      auto& name_var = ctx.Var("indexed_ir_node_info_tuple").At(0);
      auto& index_tuple_expr_var = ctx.Var("indexed_ir_node_info_tuple").At(1);
      auto& dtype_var = ctx.Var("indexed_ir_node_info_tuple").At(2);
      auto& input_var = value_class_var.Call(
          index_tuple_expr_var, dtype_var, ctx.Var("input_local_var_name"));
      return ctx.Var(axpr::kBuiltinList()).Call(name_var, input_var);
    };
    using AnfExprs = std::vector<axpr::AnfExpr>;
    auto GetAllInputIndexedIrNodeInfo =
        [&](auto* ctx) -> adt::Result<AnfExprs> {
      AnfExprs ret;
      auto DoEachNativeIrValue =
          [&](pir::Value ir_value) -> adt::Result<adt::Ok> {
        AnfExprs indexed_ir_info_tuple;
        ADT_LET_CONST_REF(dtype, ConvertToDataType(ir_value));
        for (const auto& input : ir_graph->inputs) {
          if (input->value == ir_value) {
            auto& info_var =
                ctx->Var(axpr::kBuiltinList())
                    .Call(ctx->String(input->GetUniqueNameInsideNodeArena()),
                          ctx->Var("self").Attr("loop_index_tuple_expr"),
                          ctx->Var("DataType").Attr(dtype.Name()));
            indexed_ir_info_tuple.emplace_back(
                static_cast<axpr::AnfExpr>(info_var));
          }
        }
        auto& indexed_ir_info_var =
            ctx->Var(axpr::kBuiltinList()).Apply(indexed_ir_info_tuple);
        ret.emplace_back(static_cast<axpr::AnfExpr>(indexed_ir_info_var));
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(VisitNativeIrInputBirValue(
          op_code_gen_ctx, packed_ir_op, DoEachNativeIrValue));
      return ret;
    };
    auto GetBody = [&](auto& ctx) -> adt::Result<axpr::AnfExpr> {
      const auto& map_func_var_name = ctx.NewTmpVarName();
      ctx.Var(map_func_var_name) =
          lmbd.Lambda({"indexed_ir_node_info_tuple"}, GetMapFunc);
      ADT_LET_CONST_REF(indexed_nodes, GetAllInputIndexedIrNodeInfo(&ctx));
      auto& indexed_nodes_var =
          ctx.Var(axpr::kBuiltinList()).Apply(indexed_nodes);
      auto& native_input_indexed_nodes_var =
          indexed_nodes_var.At(ctx.Var("native_input_index"));
      auto& items_var = ctx.Var("map").Call(ctx.Var(map_func_var_name),
                                            native_input_indexed_nodes_var);
      auto& ret = ctx.Var("OrderedDict").Call(items_var);
      return static_cast<axpr::Atomic<axpr::AnfExpr>>(ret);
    };
    ADT_LET_CONST_REF(anf_expr,
                      lmbd.TryLambda({"self",
                                      "code_gen_ctx",
                                      "input_local_var_name",
                                      "native_input_index"},
                                     GetBody));
    const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
    ADT_LET_CONST_REF(
        atomic, core_expr.template TryGet<axpr::Atomic<axpr::CoreExpr>>());
    ADT_LET_CONST_REF(lambda,
                      atomic.template TryGet<axpr::Lambda<axpr::CoreExpr>>());
    return Function{lambda, std::nullopt};
  }

  adt::Result<Function> PackedIrOpMakeStoreToRegisterFuncByFusionOpImpl(
      const OpCodeGenCtx& op_code_gen_ctx,
      const PureElementwiseIndexedIrGraph& ir_graph,
      const PackedIrOp& packed_ir_op) {
    ADT_CHECK(ir_graph->yield_op_inputs.size() == ir_graph->outputs.size());
    auto GetOutputIndex =
        [&](pir::Value output) -> adt::Result<std::optional<int>> {
      for (int i = 0; i < ir_graph->outputs.size(); ++i) {
        if (output == ir_graph->outputs.at(i)) {
          return i;
        }
      }
      return std::nullopt;
    };
    axpr::LambdaExprBuilder lmbd;
    using AnfExprs = std::vector<axpr::AnfExpr>;
    auto GetAllOutputIndexedIrNodeInfo =
        [&](auto* ctx) -> adt::Result<AnfExprs> {
      AnfExprs ret;
      auto DoEachNativeIrValue =
          [&](pir::Value ir_value) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(dtype, ConvertToDataType(ir_value));
        ADT_LET_CONST_REF(opt_idx, GetOutputIndex(ir_value));
        ADT_CHECK(opt_idx.has_value());
        const auto& output = ir_graph->yield_op_inputs.at(opt_idx.value());
        auto& indexed_ir_info_tuple =
            ctx->Var(axpr::kBuiltinList())
                .Call(ctx->String(output->GetUniqueNameInsideNodeArena()),
                      ctx->Var("self").Attr("loop_index_tuple_expr"),
                      ctx->Var("DataType").Attr(dtype.Name()));
        ret.emplace_back(indexed_ir_info_tuple);
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(VisitNativeIrOutputBirValue(
          op_code_gen_ctx, packed_ir_op, DoEachNativeIrValue));
      return ret;
    };

    auto GetBody = [&](auto& ctx) -> adt::Result<axpr::AnfExpr> {
      ADT_LET_CONST_REF(indexed_nodes, GetAllOutputIndexedIrNodeInfo(&ctx));
      auto& indexed_nodes_var =
          ctx.Var(axpr::kBuiltinList()).Apply(indexed_nodes);
      auto& native_output_indexed_node_var =
          indexed_nodes_var.At(ctx.Var("native_output_index"));
      auto& name_var = native_output_indexed_node_var.At(0);
      auto& output_var = ctx.Var("compute_results").At(name_var);
      auto& value_class_var =
          ctx.Var("self").Attr("class_factory").Attr("get_value_class").Call();
      auto& index_tuple_expr_var = native_output_indexed_node_var.At(1);
      auto& dtype_var = native_output_indexed_node_var.At(2);
      auto& store_var = value_class_var.Call(
          index_tuple_expr_var, dtype_var, ctx.Var("out_value_local_var_name"));
      ctx.Var("code_gen_ctx").Attr("assign").Call(store_var, output_var);
      return ctx.None();
    };
    ADT_LET_CONST_REF(anf_expr,
                      lmbd.TryLambda({"self",
                                      "code_gen_ctx",
                                      "compute_results",
                                      "out_value_local_var_name",
                                      "native_output_index"},
                                     GetBody));
    const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
    ADT_LET_CONST_REF(
        atomic, core_expr.template TryGet<axpr::Atomic<axpr::CoreExpr>>());
    ADT_LET_CONST_REF(lambda,
                      atomic.template TryGet<axpr::Lambda<axpr::CoreExpr>>());
    return Function{lambda, std::nullopt};
  }

  adt::Result<Function> PackedIrOpMakeComputeFuncByFusionOp(
      const OpCodeGenCtx& op_code_gen_ctx,
      const IndexedIrGraph& ir_graph,
      const PackedIrOp& packed_ir_op) {
    return ir_graph.Match([&](const auto& impl) -> adt::Result<Function> {
      return PackedIrOpMakeComputeFuncByFusionOpImpl(
          op_code_gen_ctx, impl, packed_ir_op);
    });
  }

  adt::Result<Function> PackedIrOpMakeComputeFuncByFusionOpImpl(
      const OpCodeGenCtx& op_code_gen_ctx,
      const PureElementwiseIndexedIrGraph& ir_graph,
      const PackedIrOp& packed_ir_op) {
    axpr::LambdaExprBuilder lmbd;
    using Ok = adt::Result<adt::Ok>;
    auto UnpackInputs = [&](auto* ctx) -> Ok {
      for (const auto& input : ir_graph->inputs) {
        const auto& name = input->GetUniqueNameInsideNodeArena();
        ctx->Var(name) = ctx->Var("inputs").At(ctx->String(name));
      }
      return adt::Ok{};
    };
    auto ComputeNativeOpCodeGen = [&](auto* ctx,
                                      const auto& indexed_ir_op) -> Ok {
      ADT_LET_CONST_REF(input_var_names, GetInputVarNames(indexed_ir_op));
      const auto& indexed_ir_op_name =
          indexed_ir_op->GetUniqueNameInsideNodeArena();
      ADT_LET_CONST_REF(output_var_names, GetOutputVarNames(indexed_ir_op));
      std::vector<axpr::AnfExpr> args{ctx->Var("code_gen_ctx")};
      args.reserve(input_var_names.size() + 1);
      for (const auto& input_var_name : input_var_names) {
        args.push_back(ctx->Var(input_var_name));
      }
      auto& outputs_var = ctx->Var("self").Attr(indexed_ir_op_name).Apply(args);
      for (int i = 0; i < output_var_names.size(); ++i) {
        const auto& output_var_name = output_var_names.at(i);
        ctx->Var(output_var_name) = outputs_var.At(i);
      }
      return adt::Ok{};
    };
    auto PackedOutputs = [&](auto* ctx) -> adt::Result<axpr::AnfExpr> {
      std::vector<axpr::AnfExpr> yield_op_input_items;
      yield_op_input_items.reserve(ir_graph->yield_op_inputs.size());
      for (const auto& yield_op_input : ir_graph->yield_op_inputs) {
        const auto& name = yield_op_input->GetUniqueNameInsideNodeArena();
        const auto& pair = ctx->Var(axpr::kBuiltinList())
                               .Call(ctx->String(name), ctx->Var(name));
        yield_op_input_items.emplace_back(static_cast<axpr::AnfExpr>(pair));
      }
      const auto& items =
          ctx->Var(axpr::kBuiltinList()).Call(yield_op_input_items);
      return ctx->Call("OrderedDict", items);
    };
    auto GetBody = [&](auto& ctx) -> adt::Result<axpr::AnfExpr> {
      auto* ctx_ptr = &ctx;
      ADT_RETURN_IF_ERR(UnpackInputs(ctx_ptr));
      ADT_RETURN_IF_ERR(
          VisitIndexedIrOp(ir_graph, [&](const auto& indexed_ir_op) -> Ok {
            return ComputeNativeOpCodeGen(ctx_ptr, indexed_ir_op);
          }));
      ADT_LET_CONST_REF(packed_outputs, PackedOutputs(ctx_ptr));
      return packed_outputs;
    };
    std::vector<std::string> arg_names{"self", "code_gen_ctx", "inputs"};
    ADT_LET_CONST_REF(anf_expr, lmbd.TryLambda(arg_names, GetBody));
    const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
    ADT_LET_CONST_REF(
        atomic, core_expr.template TryGet<axpr::Atomic<axpr::CoreExpr>>());
    ADT_LET_CONST_REF(lambda,
                      atomic.template TryGet<axpr::Lambda<axpr::CoreExpr>>());
    return Function{lambda, std::nullopt};
  }

  adt::Result<std::vector<std::string>> GetInputVarNames(
      const IndexedIrOp<IndexedIrNode>& indexed_ir_op) const {
    ADT_LET_CONST_REF(upstreams, indexed_ir_op->node.UpstreamNodes());
    std::vector<std::string> ret{};
    ret.reserve(upstreams.size());
    auto DoEach = [&](const auto& node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(ir_node, node.Get());
      ADT_LET_CONST_REF(
          ir_value, ir_node.template TryGet<IndexedIrValue<IndexedIrNode>>());
      ret.push_back(ir_value->GetUniqueNameInsideNodeArena());
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(upstreams.VisitNodes(DoEach));
    return ret;
  }

  adt::Result<std::vector<std::string>> GetOutputVarNames(
      const IndexedIrOp<IndexedIrNode>& indexed_ir_op) {
    ADT_LET_CONST_REF(downstreams, indexed_ir_op->node.DownstreamNodes());
    std::vector<std::string> ret{};
    ret.reserve(downstreams.size());
    auto DoEach = [&](const auto& node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(ir_node, node.Get());
      ADT_LET_CONST_REF(
          ir_value, ir_node.template TryGet<IndexedIrValue<IndexedIrNode>>());
      ret.push_back(ir_value->GetUniqueNameInsideNodeArena());
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(downstreams.VisitNodes(DoEach));
    return ret;
  }

  adt::Result<Function> PackedIrOpMakeInitFuncByFusionOpImpl(
      const OpCodeGenCtx& op_code_gen_ctx,
      const PureElementwiseIndexedIrGraph& ir_graph,
      const PackedIrOp& packed_ir_op) {
    axpr::LambdaExprBuilder lmbd;
    using Ok = adt::Result<adt::Ok>;
    auto ConstructNativeOpCodeGen = [&](auto* ctx,
                                        const auto& indexed_ir_op) -> Ok {
      const auto& op_name = indexed_ir_op->op->name();
      auto& class_var = ctx->Var("get_native_op_code_generator_class")
                            .Call(ctx->String(op_name));
      {
        std::vector<axpr::AnfExpr> input_dtype_anf_exprs;
        for (int i = 0; i < indexed_ir_op->op->num_operands(); ++i) {
          ADT_LET_CONST_REF(
              dtype, ConvertToDataType(indexed_ir_op->op->operand_source(i)));
          const auto& dtype_var = ctx->Var("DataType").Attr(dtype.Name());
          input_dtype_anf_exprs.emplace_back(
              static_cast<axpr::AnfExpr>(dtype_var));
        }
        ctx->Var("input_dtypes") =
            ctx->Call(axpr::kBuiltinList(), input_dtype_anf_exprs);
      }
      {
        std::vector<axpr::AnfExpr> output_dtype_anf_exprs;
        for (int i = 0; i < indexed_ir_op->op->num_results(); ++i) {
          ADT_LET_CONST_REF(dtype,
                            ConvertToDataType(indexed_ir_op->op->result(i)));
          const auto& dtype_var = ctx->Var("DataType").Attr(dtype.Name());
          output_dtype_anf_exprs.emplace_back(
              static_cast<axpr::AnfExpr>(dtype_var));
        }
        ctx->Var("output_dtypes") =
            ctx->Call(axpr::kBuiltinList(), output_dtype_anf_exprs);
      }
      {
        std::vector<axpr::AnfExpr> input_index_tuple_exprs;
        input_index_tuple_exprs.reserve(indexed_ir_op->op->num_operands());
        for (int i = 0; i < indexed_ir_op->op->num_operands(); ++i) {
          input_index_tuple_exprs.emplace_back(
              ctx->Var("loop_index_tuple_expr"));
        }
        ctx->Var("input_index_tuple_exprs") =
            ctx->Call(axpr::kBuiltinList(), input_index_tuple_exprs);
      }
      {
        std::vector<axpr::AnfExpr> output_index_tuple_exprs;
        output_index_tuple_exprs.reserve(indexed_ir_op->op->num_results());
        for (int i = 0; i < indexed_ir_op->op->num_results(); ++i) {
          output_index_tuple_exprs.emplace_back(
              ctx->Var("loop_index_tuple_expr"));
        }
        ctx->Var("output_index_tuple_exprs") =
            ctx->Call(axpr::kBuiltinList(), output_index_tuple_exprs);
      }
      const auto& indexed_op_name =
          indexed_ir_op->GetUniqueNameInsideNodeArena();
      axpr::AnfExpr indexed_op =
          class_var.Call(ctx->Var("index_expr_code_gen"),
                         ctx->String(indexed_op_name),
                         ctx->Var("input_dtypes"),
                         ctx->Var("output_dtypes"),
                         ctx->Var("input_index_tuple_exprs"),
                         ctx->Var("output_index_tuple_exprs"),
                         /*attrs*/ ctx->None());
      ctx->Var("self").SetAttr(indexed_op_name, indexed_op);
      return adt::Ok{};
    };
    auto GetBody = [&](auto& ctx) -> adt::Result<axpr::AnfExpr> {
      ctx.Var("self").SetAttr("class_factory", ctx.Var("class_factory"));
      ctx.Var("self").SetAttr("loop_index_tuple_expr",
                              ctx.Var("loop_index_tuple_expr"));
      ctx.Var("index_expr_code_generator_class") =
          ctx.Var("class_factory")
              .Attr("get_index_expr_code_generator_class")
              .Call();
      ctx.Var("index_expr_code_gen") =
          ctx.Var("index_expr_code_generator_class")
              .Call(ctx.Var("loop_var_names"));
      ctx.Var("get_native_op_code_generator_class") =
          ctx.Var("class_factory")
              .Attr("get_native_op_code_generator_class")
              .Call();
      auto* ctx_ptr = &ctx;
      ADT_RETURN_IF_ERR(
          VisitIndexedIrOp(ir_graph, [&](const auto& indexed_ir_op) -> Ok {
            return ConstructNativeOpCodeGen(ctx_ptr, indexed_ir_op);
          }));
      return ctx.None();
    };
    ADT_LET_CONST_REF(anf_expr,
                      lmbd.TryLambda({"self",
                                      "class_factory",
                                      "loop_index_tuple_expr",
                                      "loop_var_names"},
                                     GetBody));
    const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
    ADT_LET_CONST_REF(
        atomic, core_expr.template TryGet<axpr::Atomic<axpr::CoreExpr>>());
    ADT_LET_CONST_REF(lambda,
                      atomic.template TryGet<axpr::Lambda<axpr::CoreExpr>>());
    return Function{lambda, std::nullopt};
  }

  template <typename DoEachIndexIrNodeT>
  adt::Result<adt::Ok> VisitIndexedIrOp(
      const PureElementwiseIndexedIrGraph& ir_graph,
      const DoEachIndexIrNodeT& DoEachIndexIrNode) {
    for (const auto& node : ir_graph->node_arena->nodes()) {
      if (node.template Has<IndexedIrOp<IndexedIrNode>>()) {
        ADT_RETURN_IF_ERR(
            DoEachIndexIrNode(node.template Get<IndexedIrOp<IndexedIrNode>>()));
      }
    }
    return adt::Ok{};
  }

  adt::Result<Function> RefIrOpMakeInitFuncByFusionOp(
      const OpCodeGenCtx& op_code_gen_ctx, const RefIrOp& ref_ir_op) {
    axpr::LambdaExprBuilder lmbd;
    auto GetBody = [](auto& ctx) {
      ctx.Var("self").SetAttr("class_factory", ctx.Var("class_factory"));
      ctx.Var("self").SetAttr("loop_index_tuple_expr",
                              ctx.Var("loop_index_tuple_expr"));
      return ctx.None();
    };
    const auto& anf_expr = lmbd.Lambda(
        {"self", "class_factory", "loop_index_tuple_expr", "loop_var_names"},
        GetBody);
    const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
    ADT_LET_CONST_REF(
        atomic, core_expr.template TryGet<axpr::Atomic<axpr::CoreExpr>>());
    ADT_LET_CONST_REF(lambda,
                      atomic.template TryGet<axpr::Lambda<axpr::CoreExpr>>());
    return Function{lambda, std::nullopt};
  }

  adt::Result<Function> RefIrOpMakeComputeFuncByFusionOp(
      const OpCodeGenCtx& op_code_gen_ctx, const RefIrOp& ref_ir_op) {
    axpr::LambdaExprBuilder lmbd;
    auto GetBody = [](auto& ctx) -> axpr::AnfExpr { return ctx.Var("inputs"); };
    const auto& anf_expr =
        lmbd.Lambda({"self", "code_gen_ctx", "inputs"}, GetBody);
    const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
    ADT_LET_CONST_REF(
        atomic, core_expr.template TryGet<axpr::Atomic<axpr::CoreExpr>>());
    ADT_LET_CONST_REF(lambda,
                      atomic.template TryGet<axpr::Lambda<axpr::CoreExpr>>());
    return Function{lambda, std::nullopt};
  }

  adt::Result<Function> RefIrOpMakeLoadFromRegisterFuncByFusionOp(
      const OpCodeGenCtx& op_code_gen_ctx, const RefIrOp& ref_ir_op) {
    pir::Value value = ref_ir_op.ref_node_info->ir_value.value;
    ADT_LET_CONST_REF(dtype, ConvertToDataType(value));
    axpr::LambdaExprBuilder lmbd;
    auto GetBody = [&](auto& ctx) {
      auto& value_class_var =
          ctx.Var("self").Attr("class_factory").Attr("get_value_class").Call();
      auto& index_tuple_expr_var =
          ctx.Var("self").Attr("loop_index_tuple_expr");
      auto& dtype_var = ctx.Var("DataType").Attr(dtype.Name());
      auto& input_var = value_class_var.Call(
          index_tuple_expr_var, dtype_var, ctx.Var("input_local_var_name"));
      return ctx.Var("OrderedDict")
          .Call(ctx.Var(axpr::kBuiltinList())
                    .Call(ctx.Var(axpr::kBuiltinList())
                              .Call(ctx.String("sole_ir_value"), input_var)));
    };
    const auto& anf_expr = lmbd.Lambda(
        {"self", "code_gen_ctx", "input_local_var_name", "native_input_index"},
        GetBody);
    const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
    ADT_LET_CONST_REF(
        atomic, core_expr.template TryGet<axpr::Atomic<axpr::CoreExpr>>());
    ADT_LET_CONST_REF(lambda,
                      atomic.template TryGet<axpr::Lambda<axpr::CoreExpr>>());
    return Function{lambda, std::nullopt};
  }

  adt::Result<Function> RefIrOpMakeStoreToRegisterFuncByFusionOp(
      const OpCodeGenCtx& op_code_gen_ctx, const RefIrOp& ref_ir_op) {
    pir::Value value = ref_ir_op.ref_node_info->ir_value.value;
    ADT_LET_CONST_REF(dtype, ConvertToDataType(value));
    axpr::LambdaExprBuilder lmbd;
    auto GetBody = [&](auto& ctx) {
      auto& value_class_var =
          ctx.Var("self").Attr("class_factory").Attr("get_value_class").Call();
      auto& index_tuple_expr_var =
          ctx.Var("self").Attr("loop_index_tuple_expr");
      auto& dtype_var = ctx.Var("DataType").Attr(dtype.Name());
      auto& output_var = value_class_var.Call(
          index_tuple_expr_var, dtype_var, ctx.Var("out_value_local_var_name"));
      ctx.Var("code_gen_ctx")
          .Attr("assign")
          .Call(output_var,
                ctx.Var("compute_results").At(ctx.String("sole_ir_value")));
      return ctx.None();
    };
    const auto& anf_expr = lmbd.Lambda({"self",
                                        "code_gen_ctx",
                                        "compute_results",
                                        "out_value_local_var_name",
                                        "native_output_index"},
                                       GetBody);
    const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
    ADT_LET_CONST_REF(
        atomic, core_expr.template TryGet<axpr::Atomic<axpr::CoreExpr>>());
    ADT_LET_CONST_REF(lambda,
                      atomic.template TryGet<axpr::Lambda<axpr::CoreExpr>>());
    return Function{lambda, std::nullopt};
  }

  using NativeOrRefIrValue = ir_match::NativeOrRefIrValue<BirNode>;

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitNativeIrInputBirValue(
      const OpCodeGenCtx& op_code_gen_ctx,
      const PackedIrOp& packed_ir_op,
      const DoEachT& DoEach) {
    ADT_LET_CONST_REF(graph_match_ctx, GetGraphMatchCtx(op_code_gen_ctx));
    ADT_LET_CONST_REF(drr_trivial_fusion_ir_op,
                      GetDrrTrivialFusionIrOp(graph_match_ctx, packed_ir_op));
    auto DoEachNativeValue =
        [&](const auto& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(value, GetPirValue(graph_match_ctx, drr_ir_value));
      return DoEach(value);
    };
    auto DoEachPackedValue =
        [&](const auto& drr_ir_value) -> adt::Result<adt::Ok> {
      // Do nothing.
      return adt::Ok{};
    };
    return VisitDrrTrivialFusionIrOpInput(
        drr_trivial_fusion_ir_op, DoEachNativeValue, DoEachPackedValue);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitInputBirNativeIrValue(
      const OpCodeGenCtx& op_code_gen_ctx,
      const PackedIrOp& packed_ir_op,
      const DoEachT& DoEach) {
    ADT_LET_CONST_REF(graph_match_ctx, GetGraphMatchCtx(op_code_gen_ctx));
    ADT_LET_CONST_REF(drr_trivial_fusion_ir_op,
                      GetDrrTrivialFusionIrOp(graph_match_ctx, packed_ir_op));
    auto DoEachNativeValue =
        [&](const auto& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(value, GetPirValue(graph_match_ctx, drr_ir_value));
      return DoEach(value);
    };
    auto DoEachPackedValue =
        [&](const auto& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_RETURN_IF_ERR(
          VisitPackedPirValue(graph_match_ctx, drr_ir_value, DoEach));
      return adt::Ok{};
    };
    return VisitDrrTrivialFusionIrOpInput(
        drr_trivial_fusion_ir_op, DoEachNativeValue, DoEachPackedValue);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitPackedPirValue(const GraphMatchCtx& match_ctx,
                                           const DrrPackedIrValue& drr_ir_value,
                                           const DoEachT& DoEach) {
    auto DoEachPirNode = [&](const PirNode& pir_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(pir_value, pir_node.template TryGet<NativeIrValue>());
      ADT_RETURN_IF_ERR(DoEach(pir_value.value));
      return adt::Ok{};
    };
    const auto& node = drr_ir_value->node;
    ADT_RETURN_IF_ERR(
        match_ctx->VisitPackedBigGraphIrValueNode(node, DoEachPirNode));
    return adt::Ok{};
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitNativeIrOutputBirValue(
      const OpCodeGenCtx& op_code_gen_ctx,
      const PackedIrOp& packed_ir_op,
      const DoEachT& DoEach) {
    ADT_LET_CONST_REF(graph_match_ctx, GetGraphMatchCtx(op_code_gen_ctx));
    ADT_LET_CONST_REF(drr_trivial_fusion_ir_op,
                      GetDrrTrivialFusionIrOp(graph_match_ctx, packed_ir_op));
    auto DoEachNativeValue =
        [&](const auto& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(value, GetPirValue(graph_match_ctx, drr_ir_value));
      return DoEach(value);
    };
    auto DoEachPackedValue =
        [&](const auto& drr_ir_value) -> adt::Result<adt::Ok> {
      // Do nothing.
      return adt::Ok{};
    };
    return VisitDrrTrivialFusionIrOpOutput(
        drr_trivial_fusion_ir_op, DoEachNativeValue, DoEachPackedValue);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitOutputNativeIrValue(
      const OpCodeGenCtx& op_code_gen_ctx,
      const PackedIrOp& packed_ir_op,
      const DoEachT& DoEach) {
    ADT_LET_CONST_REF(graph_match_ctx, GetGraphMatchCtx(op_code_gen_ctx));
    ADT_LET_CONST_REF(drr_trivial_fusion_ir_op,
                      GetDrrTrivialFusionIrOp(graph_match_ctx, packed_ir_op));
    auto DoEachNativeValue =
        [&](const auto& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(value, GetPirValue(graph_match_ctx, drr_ir_value));
      return DoEach(value);
    };
    auto DoEachPackedValue =
        [&](const auto& drr_ir_value) -> adt::Result<adt::Ok> {
      ADT_RETURN_IF_ERR(
          VisitPackedPirValue(graph_match_ctx, drr_ir_value, DoEach));
      return adt::Ok{};
    };
    return VisitDrrTrivialFusionIrOpOutput(
        drr_trivial_fusion_ir_op, DoEachNativeValue, DoEachPackedValue);
  }

  template <typename DoEachNativeValueT, typename DoEachPackedValueT>
  adt::Result<adt::Ok> VisitDrrTrivialFusionIrOpInput(
      const DrrTrivialFusionIrOp& drr_trivial_fusion_ir_op,
      const DoEachNativeValueT& DoEachNativeValue,
      const DoEachPackedValueT DoEachPackedValue) {
    LOG(ERROR) << "drr_trivial_fusion_ir_op: "
               << graph::NodeDescriptor<DrrGraphNode>{}.DebugId(
                      drr_trivial_fusion_ir_op.node());
    auto DoEach = [&](const DrrGraphNode& node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(drr_node, node.Get());
      LOG(ERROR) << "drr_trivial_fusion_ir_op input: "
                 << graph::NodeDescriptor<DrrGraphNode>{}.DebugId(node);
      return drr_node.Match(
          [&](const DrrNativeIrValue& ir_value) -> adt::Result<adt::Ok> {
            return DoEachNativeValue(ir_value);
          },
          [&](const DrrPackedIrValue& ir_value) -> adt::Result<adt::Ok> {
            return DoEachPackedValue(ir_value);
          },
          [&](const auto&) -> adt::Result<adt::Ok> {
            return adt::errors::ValueError{
                "the second connected upstreams of drr packed ir op should be "
                "drr native ir values or drr packed ir values."};
          });
    };
    return VisitSecondConnectedUpstream(drr_trivial_fusion_ir_op.node(),
                                        DoEach);
  }

  template <typename DoEachNativeValueT, typename DoEachPackedValueT>
  adt::Result<adt::Ok> VisitDrrTrivialFusionIrOpOutput(
      const DrrTrivialFusionIrOp& drr_trivial_fusion_ir_op,
      const DoEachNativeValueT& DoEachNativeValue,
      const DoEachPackedValueT DoEachPackedValue) {
    auto DoEach = [&](const DrrGraphNode& node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(drr_node, node.Get());
      return drr_node.Match(
          [&](const DrrNativeIrValue& ir_value) -> adt::Result<adt::Ok> {
            return DoEachNativeValue(ir_value);
          },
          [&](const DrrPackedIrValue& ir_value) -> adt::Result<adt::Ok> {
            return DoEachPackedValue(ir_value);
          },
          [&](const auto&) -> adt::Result<adt::Ok> {
            return adt::errors::ValueError{
                "the second connected upstreams of drr packed ir op should be "
                "drr native ir values or drr packed ir values."};
          });
    };
    return VisitSecondConnectedDownstream(drr_trivial_fusion_ir_op.node(),
                                          DoEach);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitSecondConnectedUpstream(const DrrGraphNode& node,
                                                    const DoEachT& DoEach) {
    auto DoEachUpstream = [&](const auto& upstream) -> adt::Result<adt::Ok> {
      return VisitUpstream(upstream, DoEach);
    };
    return VisitUpstream(node, DoEachUpstream);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitSecondConnectedDownstream(const DrrGraphNode& node,
                                                      const DoEachT& DoEach) {
    auto DoEachUpstream = [&](const auto& downstream) -> adt::Result<adt::Ok> {
      return VisitDownstream(downstream, DoEach);
    };
    return VisitDownstream(node, DoEachUpstream);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitUpstream(const DrrGraphNode& node,
                                     const DoEachT& DoEach) {
    ADT_LET_CONST_REF(upstreams, node.UpstreamNodes());
    return upstreams.VisitNodes(DoEach);
  }

  template <typename DoEachT>
  adt::Result<adt::Ok> VisitDownstream(const DrrGraphNode& node,
                                       const DoEachT& DoEach) {
    ADT_LET_CONST_REF(downstreams, node.DownstreamNodes());
    return downstreams.VisitNodes(DoEach);
  }

  adt::Result<pir::Value> GetPirValue(
      const GraphMatchCtx& graph_match_ctx,
      const DrrNativeIrValue& drr_native_ir_value) {
    const auto& node = drr_native_ir_value->node;
    ADT_LET_CONST_REF(pir_node, graph_match_ctx->GetSoleBigGraphNode(node));
    ADT_LET_CONST_REF(pir_value, pir_node.template TryGet<NativeIrValue>());
    return pir_value.value;
  }

  adt::Result<DrrTrivialFusionIrOp> GetDrrTrivialFusionIrOp(
      const GraphMatchCtx& graph_match_ctx, const PackedIrOp& packed_ir_op) {
    ADT_LET_CONST_REF(node,
                      graph_match_ctx->GetMatchedSmallGraphNode(packed_ir_op));
    ADT_LET_CONST_REF(drr_node, node.Get());
    using RetT = adt::Result<DrrTrivialFusionIrOp>;
    return drr_node.Match(
        [&](const DrrPackedIrOp& impl) -> RetT { return impl; },
        [&](const DrrOptPackedIrOp& impl) -> RetT { return impl; },
        [&](const auto&) -> RetT {
          return adt::errors::NotImplementedError{
              "conversion from DrrNode to DrrTrivialFusionIrOp failed."};
        });
  }

  adt::Result<GraphMatchCtx> GetGraphMatchCtx(
      const OpCodeGenCtx& op_code_gen_ctx) const {
    ADT_LET_CONST_REF(code_gen_ctx,
                      adt::WeakPtrLock(op_code_gen_ctx->code_gen_ctx));
    ADT_CHECK(code_gen_ctx->ir_match_ctx.has_value());
    const auto& ir_match_ctx = code_gen_ctx->ir_match_ctx.value();
    return ir_match_ctx->graph_match_ctx;
  }

  adt::Result<axpr::PointerType> GetConstDataPointerType(pir::Value value) {
    ADT_LET_CONST_REF(data_type, ConvertToDataType(value));
    return axpr::GetConstPointerTypeFromDataType(data_type);
  }

  adt::Result<axpr::PointerType> GetMutableDataPointerType(pir::Value value) {
    ADT_LET_CONST_REF(data_type, ConvertToDataType(value));
    return axpr::GetMutablePointerTypeFromDataType(data_type);
  }

  adt::Result<axpr::DataType> ConvertToDataType(pir::Value value) {
    ADT_LET_CONST_REF(dtype, ConvertToPhiDataType(value));
    return ap::axpr::GetDataTypeFromPhiDataType(dtype);
  }

  adt::Result<phi::DataType> ConvertToPhiDataType(pir::Value value) {
    ADT_LET_CONST_REF(type, GetPirDataType(value));
    try {
      return ::paddle::dialect::TransToPhiDataType(type);
    } catch (const std::exception& e) {
      return adt::errors::TypeError{
          "failed to cast from pir data type to phi data type."};
    }
  }

  adt::Result<pir::Type> GetPirDataType(pir::Value value) {
    if (!value.type().isa<pir::DenseTensorType>()) {
      return adt::errors::NotImplementedError{
          "pir value must be of DenseTensorType"};
    }
    const auto dense_tensor_type =
        value.type().dyn_cast<pir::DenseTensorType>();
    return dense_tensor_type.dtype();
  }
};

}  // namespace ap::paddle

namespace ap::code_gen {

template <>
struct OpCudaCodeGenImpl<ap::paddle::PirNode>
    : public paddle::OpCudaCodeGenImpl {};

}  // namespace ap::code_gen
