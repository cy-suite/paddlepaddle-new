// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/operator_fusion/pir_graph_analyzing/loop_axis_mapping.h"

namespace cinn::fusion {

std::ostream& operator<<(std::ostream& os, const AxisTransform& transform) {
  os << std::visit([](auto&& t) { return t->DebugStr(); }, transform);
  return os;
}
std::ostream& operator<<(std::ostream& os, const AxisTransformRoute& route) {
  os << cinn::utils::Join(route, " -> ");
  return os;
}

AxisTransform DeleteAxisTransform::reverse() {
  return std::make_shared<AppendAxisTransform>(axis, shape);
}
AxisTransform AppendAxisTransform::reverse() {
  return std::make_shared<DeleteAxisTransform>(axis, shape);
}

AxisTransform ReverseTransform(const AxisTransform& transform) {
  return std::visit([](auto&& t) { return t->reverse(); }, transform);
}

AxisTransformRoute ReverseTransformRoute(const AxisTransformRoute& route) {
  AxisTransformRoute result;
  for (auto it = route.rbegin(); it != route.rend(); ++it) {
    result.push_back(ReverseTransform(*it));
  }
  return result;
}

void LoopAxisMapping::SetReverseMapping() {
  loop2input.clear();
  output2loop.clear();
  for (const auto& route : input2loop) {
    PADDLE_ENFORCE(
        !route.empty(),
        ::common::errors::InvalidArgument("input2loop must not be empty."));
    loop2input.push_back(ReverseTransformRoute(route));
  }
  for (const auto& route : loop2output) {
    PADDLE_ENFORCE(
        !route.empty(),
        ::common::errors::InvalidArgument("loop2output must not be empty."));
    output2loop.push_back(ReverseTransformRoute(route));
  }
}

void LoopAxisMapping::EliminateIdentity() {
  auto eliminate_identity = [](AxisTransformRoute& route) {
    if (route.size() < 2) return;
    for (int i = route.size() - 1; i >= 0; --i) {
      if (std::get_if<IdentityTransformPtr>(&route[i])) {
        route.erase(route.begin() + i);
      }
    }
    if (route.empty()) {
      route.push_back(IdentityTransform::InstancePtr());
    }
  };
  for (auto& route : input2loop) {
    eliminate_identity(route);
  }
  for (auto& route : loop2input) {
    eliminate_identity(route);
  }
  for (auto& route : loop2output) {
    eliminate_identity(route);
  }
  for (auto& route : output2loop) {
    eliminate_identity(route);
  }
}

std::string LoopAxisMapping::DebugStr() const {
  std::stringstream ss;
  for (size_t i = 0; i < input_values.size(); ++i) {
    ss << "\n input " << i << " :\t["
       << cinn::utils::Join(GetValueAllDims(input_values[i]), ", ") << "]"
       << ", " << input_values[i].impl();
  }
  ss << "\n  loop   :\t[" << cinn::utils::Join(loop, ", ")
     << "], reduce_axis_num: " << reduce_axis_num;
  for (size_t i = 0; i < output_values.size(); ++i) {
    ss << "\noutput " << i << " :\t["
       << cinn::utils::Join(GetValueAllDims(output_values[i]), ", ") << "]"
       << ", " << output_values[i].impl()
       << ", use_count: " << outputs_use_count.at(output_values[i]);
  }
  for (size_t i = 0; i < input2loop.size(); ++i) {
    ss << "\ninput2loop  " << i << " : " << input2loop[i];
    ss << "\nloop2input  " << i << " : " << loop2input[i];
  }
  for (size_t i = 0; i < loop2output.size(); ++i) {
    ss << "\nloop2output " << i << " : " << loop2output[i];
    ss << "\noutput2loop " << i << " : " << output2loop[i];
  }
  return ss.str();
}

std::pair<AxisTransformRoute, AxisTransformRoute> GetLoopTransformRoute(
    const LoopAxisMapping& upstream, const LoopAxisMapping& downstream) {
  AxisTransformRoute loop_sink_route, loop_lift_route;
  for (size_t i = 0; i < upstream.output_values.size(); ++i) {
    auto value = upstream.output_values[i];
    auto indices = FindPosInVector(downstream.input_values, value);
    for (auto idx : indices) {
      AxisTransformRoute sink_route =
          ConcatVector(upstream.loop2output[i], downstream.input2loop[idx]);
      AxisTransformRoute lift_route =
          ConcatVector(downstream.loop2input[idx], upstream.output2loop[i]);
      if (sink_route.size() < loop_sink_route.size() ||
          loop_sink_route.empty()) {
        loop_sink_route = sink_route;
      }
      if (lift_route.size() < loop_lift_route.size() ||
          loop_lift_route.empty()) {
        loop_lift_route = lift_route;
      }
    }
  }
  return {loop_sink_route, loop_lift_route};
}

LoopAxisMapping LoopMappingMergeImpl(const LoopAxisMapping& upstream,
                                     const LoopAxisMapping& downstream,
                                     bool upstream_is_anchor) {
  const auto& [loop_sink_route, loop_lift_route] =
      GetLoopTransformRoute(upstream, downstream);

  LoopAxisMapping result;
  result.input_values = upstream.input_values;
  for (const auto& trans : upstream.input2loop) {
    result.input2loop.push_back(
        upstream_is_anchor ? trans : ConcatVector(trans, loop_sink_route));
  }
  result.outputs_use_count = upstream.outputs_use_count;
  for (size_t i = 0; i < downstream.input_values.size(); ++i) {
    auto value = downstream.input_values[i];
    if (upstream.outputs_use_count.count(value)) {
      result.outputs_use_count[value]--;
      continue;
    }
    result.input_values.push_back(value);
    result.input2loop.push_back(
        upstream_is_anchor
            ? ConcatVector(downstream.input2loop[i], loop_lift_route)
            : downstream.input2loop[i]);
  }
  for (size_t i = 0; i < upstream.output_values.size(); ++i) {
    auto value = upstream.output_values[i];
    if (result.outputs_use_count[value] > 0) {
      result.output_values.push_back(value);
      result.loop2output.push_back(
          upstream_is_anchor
              ? upstream.loop2output[i]
              : ConcatVector(loop_lift_route, upstream.loop2output[i]));
    } else {
      result.outputs_use_count.erase(value);
    }
  }
  for (size_t i = 0; i < downstream.output_values.size(); ++i) {
    auto value = downstream.output_values[i];
    result.output_values.push_back(value);
    result.outputs_use_count[value] = downstream.outputs_use_count.at(value);
    result.loop2output.push_back(
        upstream_is_anchor
            ? ConcatVector(loop_sink_route, downstream.loop2output[i])
            : downstream.loop2output[i]);
  }
  result.loop = upstream_is_anchor ? upstream.loop : downstream.loop;
  result.reduce_axis_num =
      std::max(upstream.reduce_axis_num, downstream.reduce_axis_num);
  return result;
}

LoopAxisMapping LoopMappingMerge(const LoopAxisMapping& upstream,
                                 const LoopAxisMapping& downstream,
                                 bool upstream_is_anchor) {
  VLOG(4) << "Start LoopMappingMerge: "
          << "\nUpstream: " << upstream.DebugStr()
          << "\nDownstream: " << downstream.DebugStr();
  auto result = LoopMappingMergeImpl(upstream, downstream, upstream_is_anchor);
  result.SetReverseMapping();
  result.EliminateIdentity();
  VLOG(4) << "\nMerged result: " << result.DebugStr();
  return result;
}

LoopAxisMapping ReducePlusTrivialLoopMappingMerge(
    const LoopAxisMapping& upstream, const LoopAxisMapping& downstream) {
  // Signal downstream reduce plus trivial fusion loop is downstream trivial
  // loop plus upstream reduce loop.
  VLOG(4) << "Start LoopMappingMerge: "
          << "\nUpstream: " << upstream.DebugStr()
          << "\nDownstream: " << downstream.DebugStr();
  PADDLE_ENFORCE(
      upstream.reduce_axis_num > 0 && downstream.reduce_axis_num == 0,
      ::common::errors::InvalidArgument(
          "Upstream should be reduce pattern and "
          "downstream should be trivial pattern."));
  auto result = LoopMappingMergeImpl(upstream, downstream, false);
  auto reduce_axis_num = upstream.reduce_axis_num;
  auto reduce_loop = SliceVector(upstream.loop,
                                 upstream.loop.size() - reduce_axis_num,
                                 upstream.loop.size());
  auto reduce_axis = ArangeVector<int64_t>(
      downstream.loop.size(), downstream.loop.size() + reduce_axis_num);
  AxisTransform append_reduce_axis =
      std::make_shared<AppendAxisTransform>(reduce_axis, reduce_loop);
  AxisTransform delete_reduce_axis = ReverseTransform(append_reduce_axis);
  result.loop = ConcatVector(downstream.loop, reduce_loop);
  for (auto& route : result.input2loop) {
    route.push_back(append_reduce_axis);
  }
  for (auto& route : result.loop2output) {
    route.insert(route.begin(), delete_reduce_axis);
  }
  result.SetReverseMapping();
  result.EliminateIdentity();
  VLOG(4) << "\nMerged result: " << result.DebugStr();
  return result;
}

std::optional<AxisTransformRoute> GetValidLoopTransformRoute(
    const LoopAxisMapping& upstream,
    const LoopAxisMapping& downstream,
    bool upstream_is_anchor) {
  VLOG(4) << "Try to get valid loop transform route "
          << (upstream_is_anchor ? "from downstream to upstream."
                                 : "from upstream to downstream.");
  const auto& [loop_sink_route, loop_lift_route] =
      GetLoopTransformRoute(upstream, downstream);
  auto loop_transform_route =
      upstream_is_anchor ? loop_lift_route : loop_sink_route;
  auto source_loop = upstream_is_anchor ? downstream.loop : upstream.loop;

  size_t id = 0;
  auto unique_id = [&]() { return "I" + std::to_string(id++); };

  AxisTransformRoute result;
  std::vector<std::string> axis_ids;
  std::unordered_map<std::string, symbol::DimExpr> axis_symbols;
  std::unordered_set<std::string> deleted_axes;
  std::unordered_set<std::string> unused_axes;
  for (const auto& symbol : source_loop) {
    auto axis_id = unique_id();
    axis_ids.push_back(axis_id);
    axis_symbols[axis_id] = symbol;
  }
  size_t cur_axis_size = axis_ids.size();

  auto apply_transpose = [&](const TransposeTransformPtr& transform) {
    axis_ids = TransposeVector(axis_ids, transform->perm);
    result.push_back(transform);
  };
  auto apply_append_axis = [&](const AppendAxisTransformPtr& transform) {
    for (size_t i = 0; i < transform->axis.size(); ++i) {
      auto axis = transform->axis[i];
      auto symbol = transform->shape[i];
      bool can_reuse = false;
      for (const auto& deleted_axis : deleted_axes) {
        if (axis_symbols.at(deleted_axis) == symbol) {
          // Can reuse deleted axis.
          int deleted_axis_pos =
              std::find(axis_ids.begin(), axis_ids.end(), deleted_axis) -
              axis_ids.begin();
          auto perm = ArangeVector<int32_t>(0, axis_ids.size());
          perm.erase(perm.begin() + deleted_axis_pos);
          perm.insert(perm.begin() + axis, deleted_axis_pos);
          axis_ids.erase(axis_ids.begin() + deleted_axis_pos);
          auto axis_id = unique_id();
          axis_ids.insert(axis_ids.begin() + axis, axis_id);
          axis_symbols[axis_id] = symbol;
          deleted_axes.erase(deleted_axis);
          cur_axis_size++;
          result.push_back(std::make_shared<TransposeTransform>(perm));
          can_reuse = true;
          break;
        }
      }
      // If can not reuse deleted axis, insert new axis and mark it as unused.
      if (!can_reuse) {
        auto axis_id = unique_id();
        axis_ids.insert(axis_ids.begin() + axis, axis_id);
        axis_symbols[axis_id] = symbol;
        unused_axes.insert(axis_id);
        cur_axis_size++;
        result.push_back(std::make_shared<AppendAxisTransform>(
            std::vector<int64_t>{axis}, std::vector<symbol::DimExpr>{symbol}));
      }
    }
  };
  auto apply_delete_axis = [&](const DeleteAxisTransformPtr& transform) {
    for (int i = transform->axis.size() - 1; i >= 0; --i) {
      auto axis = transform->axis[i];
      auto axis_id = axis_ids[axis];
      auto symbol = axis_symbols.at(axis_id);
      if (symbol == symbol::DimExpr(1) || unused_axes.count(axis_id)) {
        // Unused axis or axis with size 1 can be deleted directly.
        axis_ids.erase(axis_ids.begin() + axis);
        unused_axes.erase(axis_id);
        result.push_back(std::make_shared<DeleteAxisTransform>(
            std::vector<int64_t>{axis}, std::vector<symbol::DimExpr>{symbol}));
      } else {
        // Used axis can not be deleted directly, we need to transpose it to
        // the end to ensure accuracy of subsequent transform.
        std::vector<std::string> new_axis_ids;
        std::vector<int> perm;
        for (int i = 0; i < axis_ids.size(); ++i) {
          if (i == axis) continue;
          new_axis_ids.push_back(axis_ids[i]);
          perm.push_back(i);
        }
        deleted_axes.insert(axis_id);
        new_axis_ids.push_back(axis_id);
        axis_ids = new_axis_ids;
        perm.push_back(axis);
        result.push_back(std::make_shared<TransposeTransform>(perm));
      }
      cur_axis_size--;
    }
  };
  auto apply_reshape = [&](const ReshapeTransformPtr& transform) {
    auto in_shape = transform->in_shape;
    auto out_shape = transform->out_shape;
    std::vector<std::string> new_axis_ids;
    if (!ShapeProductEqual(in_shape, out_shape)) {
      for (const auto& symbol : out_shape) {
        auto axis_id = unique_id();
        new_axis_ids.push_back(axis_id);
        axis_symbols[axis_id] = symbol;
      }
    } else {
      const auto& partion_indices = PartionReshapeAxes(in_shape, out_shape);
      for (int idx = 1; idx < partion_indices.size(); ++idx) {
        const auto& [in_start, out_start] = partion_indices[idx - 1];
        const auto& [in_end, out_end] = partion_indices[idx];
        if (in_end == in_start + 1 && out_end == out_start + 1) {
          new_axis_ids.push_back(axis_ids[in_start]);
        } else {
          bool is_unused = true;
          for (int i = in_start; i < in_end; ++i) {
            if (axis_symbols.at(axis_ids[i]) != symbol::DimExpr(1) &&
                !unused_axes.count(axis_ids[i])) {
              is_unused = false;
              break;
            }
          }
          for (int i = out_start; i < out_end; ++i) {
            auto axis_id = unique_id();
            new_axis_ids.push_back(axis_id);
            axis_symbols[axis_id] = out_shape[i];
            if (is_unused) unused_axes.insert(axis_id);
          }
        }
      }
    }
    new_axis_ids = ConcatVector(
        new_axis_ids, SliceVector(axis_ids, cur_axis_size, axis_ids.size()));
    axis_ids = new_axis_ids;
    cur_axis_size = out_shape.size();
    result.push_back(transform);
  };

  auto apply_transform = adt::match{
      [&](const IdentityTransformPtr& trans) {},
      [&](const TransposeTransformPtr& trans) { apply_transpose(trans); },
      [&](const AppendAxisTransformPtr& trans) { apply_append_axis(trans); },
      [&](const DeleteAxisTransformPtr& trans) { apply_delete_axis(trans); },
      [&](const ReshapeTransformPtr& trans) { apply_reshape(trans); },
      [&](const auto& trans) {
        PADDLE_THROW(
            ::common::errors::Unimplemented("Unknown transform type."));
      }};

  for (auto& transform : loop_transform_route) {
    if (std::holds_alternative<UnsupportedTransformPtr>(transform)) {
      VLOG(4) << "Can not find valid loop transform because of unsupported "
                 "transform.";
      return std::nullopt;
    } else {
      std::visit(apply_transform, transform);
    }
  }

  // Check if all deleted axes are used, otherwise the transform is invalid.
  if (!deleted_axes.empty()) {
    VLOG(4) << "Can not find valid loop transform because of unreused deleted "
               "axes.";
    return std::nullopt;
  }
  return result;
}

LoopAxisMapping CreateDefaultAxisMapping(pir::Operation* op) {
  LoopAxisMapping result;
  result.input2loop.resize(op->num_operands());
  result.loop2output.resize(op->num_results());
  for (int i = 0; i < op->num_operands(); ++i) {
    result.input_values.push_back(op->operand_source(i));
    result.input2loop[i].push_back(UnsupportedTransform::InstancePtr());
  }
  for (int i = 0; i < op->num_results(); ++i) {
    result.output_values.push_back(op->result(i));
    result.loop2output[i].push_back(UnsupportedTransform::InstancePtr());
  }
  return result;
}

LoopAxisMapping CreateLoopMappingForElementwise(pir::Operation* op) {
  LoopAxisMapping result;
  result.input2loop.resize(op->num_operands());
  result.loop2output.resize(op->num_results());
  for (int i = 0; i < op->num_operands(); ++i) {
    result.input_values.push_back(op->operand_source(i));
    result.input2loop[i].push_back(IdentityTransform::InstancePtr());
  }
  for (int i = 0; i < op->num_results(); ++i) {
    result.output_values.push_back(op->result(i));
    result.loop2output[i].push_back(IdentityTransform::InstancePtr());
  }
  result.loop = GetValueAllDims(result.output_values[0]);
  return result;
}

LoopAxisMapping CreateLoopMappingForTranspose(pir::Operation* op) {
  PADDLE_ENFORCE(
      op->num_operands() == 1 && op->num_results() == 1,
      ::common::errors::InvalidArgument(
          "num_operands and num_results of transpose_op shall be equal 1."));
  LoopAxisMapping result;
  result.input_values.push_back(op->operand_source(0));
  result.output_values.push_back(op->result(0));
  result.loop = GetValueAllDims(result.output_values[0]);
  result.input2loop.resize(1);
  result.loop2output.resize(1);
  std::vector<int32_t> perm =
      GetInt32ArrayAttributeData(op->attributes().at("perm"));
  result.input2loop[0].push_back(std::make_shared<TransposeTransform>(perm));
  result.loop2output[0].push_back(IdentityTransform::InstancePtr());
  return result;
}

LoopAxisMapping CreateLoopMappingForSlice(pir::Operation* op) {
  PADDLE_ENFORCE(
      op->num_operands() == 1 && op->num_results() == 1,
      ::common::errors::InvalidArgument(
          "num_operands and num_results of slice_op shall be equal 1."));
  LoopAxisMapping result;
  result.input_values.push_back(op->operand_source(0));
  result.output_values.push_back(op->result(0));
  result.loop = GetValueAllDims(result.output_values[0]);
  result.input2loop.resize(1);
  result.loop2output.resize(1);

  std::vector<int64_t> axes =
      GetInt64ArrayAttributeData(op->attributes().at("axes"));
  std::vector<int64_t> decrease_axis =
      GetInt64ArrayAttributeData(op->attributes().at("decrease_axis"));
  std::vector<int64_t> starts =
      GetInt64ArrayAttributeData(op->attributes().at("starts"));
  std::vector<int64_t> ends =
      GetInt64ArrayAttributeData(op->attributes().at("ends"));
  auto decrease_axis_set = ToUnorderedSet(decrease_axis);
  for (int i = axes.size() - 1; i >= 0; --i) {
    int64_t slice_size = ends[i] - starts[i];
    if (slice_size != 1) {
      // TODO(huangjiyi): Support slice size greater than 1.
      result.input2loop[0].push_back(UnsupportedTransform::InstancePtr());
      break;
    }
    std::vector<int64_t> axis = {axes[i]};
    result.input2loop[0].push_back(std::make_shared<DeleteAxisTransform>(
        axis, GetValueDims(op->operand_source(0), axis)));
    if (!decrease_axis_set.count(axes[i])) {
      result.input2loop[0].push_back(
          std::make_shared<AppendAxisTransform>(axis));
    }
  }
  result.loop2output[0].push_back(IdentityTransform::InstancePtr());
  return result;
}

LoopAxisMapping CreateLoopMappingForBroadcast(pir::Operation* op) {
  LoopAxisMapping result;
  result.input_values.push_back(op->operand_source(0));
  result.output_values.push_back(op->result(0));
  result.loop = GetValueAllDims(result.output_values[0]);
  result.input2loop.resize(1);
  result.loop2output.resize(1);

  const auto& broad_cast_value = GetBroadcastOpInputOuputValue(op);
  PADDLE_ENFORCE(broad_cast_value.has_value(),
                 ::common::errors::InvalidArgument(
                     "Required broad_cast_value is not empty."));
  const auto& [input_value, output_value] = broad_cast_value.value();
  const auto& input_shape = GetValueAllDims(input_value);
  const auto& output_shape = GetValueAllDims(output_value);
  std::vector<int64_t> broadcast_axes;
  std::vector<int64_t> input_keepdims;
  int append_size = output_shape.size() - input_shape.size();
  for (int i = 0; i < append_size; i++) {
    broadcast_axes.push_back(i);
  }
  for (int i = append_size; i < output_shape.size(); i++) {
    if (input_shape[i - append_size] == symbol::DimExpr(1) &&
        output_shape[i] != symbol::DimExpr(1)) {
      broadcast_axes.push_back(i);
      input_keepdims.push_back(i - append_size);
    }
  }
  if (!input_keepdims.empty()) {
    result.input2loop[0].push_back(std::make_shared<DeleteAxisTransform>(
        input_keepdims, GatherVector(input_shape, input_keepdims)));
  }
  result.input2loop[0].push_back(std::make_shared<AppendAxisTransform>(
      broadcast_axes, GatherVector(output_shape, broadcast_axes)));
  result.loop2output[0].push_back(IdentityTransform::InstancePtr());
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateLoopMappingForReduce(pir::Operation* op) {
  PADDLE_ENFORCE(
      op->num_operands() == 1 && op->num_results() == 1,
      ::common::errors::InvalidArgument(
          "num_operands and num_results of reduce_op shall be equal 1."));
  LoopAxisMapping result;
  result.input_values.push_back(op->operand_source(0));
  result.output_values.push_back(op->result(0));
  result.loop = GetValueAllDims(result.input_values[0]);
  result.input2loop.resize(1);
  result.loop2output.resize(1);
  const auto& reduce_axis = GetReduceAxisIdx(op);
  result.reduce_axis_num = reduce_axis.size();
  bool keep_dim = GetReduceOpKeepDims(op);
  auto rank = GetCompatibleRank(op->operand_source(0));
  // Input2Loop: Transpose reduce axis to the last dimension if necessary.
  bool need_transpose = false;
  for (int i = reduce_axis.size() - 1, last = rank - 1; i >= 0;) {
    if (reduce_axis[i--] != last--) {
      need_transpose = true;
    }
  }
  if (need_transpose) {
    std::vector<int32_t> perm =
        GatherVectorExcept(ArangeVector<int32_t>(0, rank), reduce_axis);
    for (const auto& axis : reduce_axis) {
      perm.push_back(axis);
    }
    result.input2loop[0].push_back(std::make_shared<TransposeTransform>(perm));
    result.loop = TransposeVector(result.loop, perm);
  }
  // Input2Loop: Insert axis with size 1 for each reduce axis if keep_dim.
  if (keep_dim) {
    result.input2loop[0].push_back(
        std::make_shared<AppendAxisTransform>(reduce_axis));
    for (const auto& axis : reduce_axis) {
      result.loop.insert(result.loop.begin() + axis, symbol::DimExpr(1));
    }
    rank += reduce_axis.size();
  }
  if (result.input2loop[0].empty()) {
    result.input2loop[0].push_back(IdentityTransform::InstancePtr());
  }
  // Loop2Output: Delete reduce axis
  const auto& delete_axis =
      ArangeVector<int64_t>(rank - reduce_axis.size(), rank);
  result.loop2output[0].push_back(std::make_shared<DeleteAxisTransform>(
      delete_axis, GetValueDims(op->operand_source(0), reduce_axis)));
  return result;
}

LoopAxisMapping CreateLoopMappingForReshape(pir::Operation* op) {
  PADDLE_ENFORCE(
      op->num_operands() == 1 && op->num_results() == 1,
      ::common::errors::InvalidArgument(
          "num_operands and num_results of reshape_op shall be equal 1."));
  LoopAxisMapping result;
  result.input_values.push_back(op->operand_source(0));
  result.output_values.push_back(op->result(0));
  result.input2loop.resize(1);
  result.loop2output.resize(1);
  auto in_shape = GetValueAllDims(op->operand_source(0));
  auto out_shape = GetValueAllDims(op->result(0));
  result.loop = out_shape;

  // If Reshape only appends or deletes dims with size 1,
  // we can use DeleteAxisTransform and AppendAxisTransform.
  bool only_append_or_delete_ones = true;
  std::vector<int64_t> input_unique_axis;
  std::vector<int64_t> output_unique_axis;
  for (int i = 0, j = 0; i < in_shape.size() || j < out_shape.size();) {
    if (j >= out_shape.size()) {
      input_unique_axis.push_back(i++);
    } else if (i >= in_shape.size()) {
      output_unique_axis.push_back(j++);
    } else if (in_shape[i] == out_shape[j]) {
      ++i;
      ++j;
    } else if (in_shape[i] == symbol::DimExpr(1)) {
      input_unique_axis.push_back(i++);
    } else if (out_shape[j] == symbol::DimExpr(1)) {
      output_unique_axis.push_back(j++);
    } else {
      only_append_or_delete_ones = false;
      break;
    }
  }
  if (only_append_or_delete_ones) {
    if (!input_unique_axis.empty()) {
      result.input2loop[0].push_back(std::make_shared<DeleteAxisTransform>(
          input_unique_axis, GatherVector(in_shape, input_unique_axis)));
    }
    if (!output_unique_axis.empty()) {
      result.input2loop[0].push_back(std::make_shared<AppendAxisTransform>(
          output_unique_axis, GatherVector(out_shape, output_unique_axis)));
    }
    if (result.input2loop[0].empty()) {
      result.input2loop[0].push_back(IdentityTransform::InstancePtr());
    }
  } else {
    result.input2loop[0].push_back(
        std::make_shared<ReshapeTransform>(in_shape, out_shape));
  }
  result.loop2output[0].push_back(IdentityTransform::InstancePtr());
  return result;
}

LoopAxisMapping CreateLoopMapping(pir::Operation* op) {
  VLOG(4) << "CreateLoopMapping for op: " << OpsDebugStr({op});
  LoopAxisMapping result;
  auto op_kind = GetOpPatternKind(op);
  if (op->name() == "pd_op.transpose") {
    result = CreateLoopMappingForTranspose(op);
  } else if (op->name() == "cinn_op.reshape" || op->name() == "pd_op.reshape") {
    result = CreateLoopMappingForReshape(op);
  } else if (op->name() == "cinn_op.slice") {
    result = CreateLoopMappingForSlice(op);
  } else if (op_kind == hlir::framework::kBroadcast) {
    result = CreateLoopMappingForBroadcast(op);
  } else if (op_kind == hlir::framework::kReduction) {
    result = CreateLoopMappingForReduce(op);
  } else if (op_kind == hlir::framework::kElementWise) {
    result = CreateLoopMappingForElementwise(op);
  } else {
    result = CreateDefaultAxisMapping(op);
  }
  result.SetReverseMapping();
  for (auto value : result.output_values) {
    result.outputs_use_count[value] = value.use_count();
  }
  VLOG(4) << "LoopMapping Result: " << result.DebugStr();
  return result;
}

}  // namespace cinn::fusion
