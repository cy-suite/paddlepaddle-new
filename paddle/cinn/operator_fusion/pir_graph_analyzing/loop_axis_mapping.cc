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

LoopAxisMapping MergeAxisMapping(LoopAxisMapping upstream,
                                 LoopAxisMapping downstream,
                                 bool upstream_is_anchor) {
  LoopAxisMapping result;
  if (upstream_is_anchor) {
    result.input2loop = upstream.input2loop;
    result.loop2input = upstream.loop2input;
    result.loop2output = ConcatVector(
        upstream.loop2output,
        ConcatVector(downstream.input2loop, downstream.loop2output));
    result.output2loop = ConcatVector(
        ConcatVector(downstream.output2loop, downstream.loop2input),
        upstream.output2loop);
  } else {
    result.input2loop =
        ConcatVector(ConcatVector(upstream.input2loop, upstream.loop2output),
                     downstream.input2loop);
    result.loop2input =
        ConcatVector(downstream.loop2input,
                     ConcatVector(upstream.output2loop, upstream.loop2input));
    result.loop2output = downstream.loop2output;
    result.output2loop = downstream.output2loop;
  }
  return result;
}

LoopAxisMapping CreateDefaultAxisMapping(pir::Operation* op) {
  LoopAxisMapping result;
  for (int i = 0; i < op->num_operands(); ++i) {
    result.input2loop.push_back(UnsupportedTransform::InstancePtr());
  }
  for (int i = 0; i < op->num_results(); ++i) {
    result.loop2output.push_back(UnsupportedTransform::InstancePtr());
  }
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMappingForElementwise(pir::Operation* op) {
  LoopAxisMapping result;
  for (int i = 0; i < op->num_operands(); ++i) {
    result.input2loop.push_back(IdentityTransform::InstancePtr());
  }
  for (int i = 0; i < op->num_results(); ++i) {
    result.loop2output.push_back(IdentityTransform::InstancePtr());
  }
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMappingForTranspose(pir::Operation* op) {
  PADDLE_ENFORCE(
      op->num_operands() == 1 && op->num_results() == 1,
      ::common::errors::InvalidArgument(
          "num_operands and num_results of transpose_op shall be equal 1."));
  std::vector<int32_t> perm =
      GetInt32ArrayAttributeData(op->attributes().at("perm"));
  LoopAxisMapping result;
  result.input2loop.push_back(std::make_shared<TransposeTransform>(perm));
  result.loop2output.push_back(IdentityTransform::InstancePtr());
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMappingForSlice(pir::Operation* op) {
  PADDLE_ENFORCE(
      op->num_operands() == 1 && op->num_results() == 1,
      ::common::errors::InvalidArgument(
          "num_operands and num_results of slice_op shall be equal 1."));
  std::vector<int64_t> axes =
      GetInt64ArrayAttributeData(op->attributes().at("axes"));
  std::vector<int64_t> decrease_axis =
      GetInt64ArrayAttributeData(op->attributes().at("decrease_axis"));
  std::vector<int64_t> starts =
      GetInt64ArrayAttributeData(op->attributes().at("starts"));
  std::vector<int64_t> ends =
      GetInt64ArrayAttributeData(op->attributes().at("ends"));
  auto decrease_axis_set = ToUnorderedSet(decrease_axis);

  LoopAxisMapping result;
  for (int i = axes.size() - 1; i >= 0; --i) {
    int64_t slice_size = ends[i] - starts[i];
    if (slice_size != 1) {
      // TODO(huangjiyi): Support slice size greater than 1.
      result.input2loop.push_back(UnsupportedTransform::InstancePtr());
      break;
    }
    std::vector<int64_t> axis = {axes[i]};
    result.input2loop.push_back(std::make_shared<DeleteAxisTransform>(
        axis, GetValueDims(op->operand_source(0), axis)));
    if (!decrease_axis_set.count(axes[i])) {
      result.input2loop.push_back(std::make_shared<AppendAxisTransform>(axis));
    }
  }
  result.loop2output.push_back(IdentityTransform::InstancePtr());
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMappingForBroadcast(pir::Operation* op) {
  const auto& broad_cast_value = GetBroadcastOpInputOuputValue(op);
  PADDLE_ENFORCE(broad_cast_value.has_value(),
                 ::common::errors::InvalidArgument(
                     "Required broad_cast_value is not empty."));
  const auto& [input_value, output_value] = broad_cast_value.value();
  const auto& input_shape = GetValueAllDims(input_value);
  const auto& output_shape = GetValueAllDims(output_value);
  std::vector<int64_t> broadcast_axes;
  for (int input_idx = 0, output_idx = 0; output_idx < output_shape.size();
       output_idx++) {
    if (input_idx < input_shape.size() &&
        input_shape[input_idx] == output_shape[output_idx]) {
      ++input_idx;
    } else {
      broadcast_axes.push_back(output_idx);
    }
  }
  LoopAxisMapping result;
  result.input2loop.push_back(std::make_shared<AppendAxisTransform>(
      broadcast_axes, GetValueDims(output_value, broadcast_axes)));
  result.loop2output.push_back(IdentityTransform::InstancePtr());
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMappingForReduce(pir::Operation* op) {
  PADDLE_ENFORCE(
      op->num_operands() == 1 && op->num_results() == 1,
      ::common::errors::InvalidArgument(
          "num_operands and num_results of reduce_op shall be equal 1."));
  const auto& reduce_axis = GetReduceAxisIdx(op);
  bool keep_dim = GetReduceOpKeepDims(op);
  auto rank = GetCompatibleRank(op->operand_source(0));
  LoopAxisMapping result;
  // Transpose reduce axis to the last dimension.
  std::vector<int32_t> perm =
      GatherVectorExcept(ArangeVector<int32_t>(0, rank), reduce_axis);
  for (const auto& axis : reduce_axis) {
    perm.push_back(axis);
  }
  result.input2loop.push_back(std::make_shared<TransposeTransform>(perm));
  if (keep_dim) {
    result.input2loop.push_back(
        std::make_shared<AppendAxisTransform>(reduce_axis));
    rank += reduce_axis.size();
  }
  const auto& delete_axis =
      ArangeVector<int64_t>(rank - reduce_axis.size(), rank);
  result.loop2output.push_back(std::make_shared<DeleteAxisTransform>(
      delete_axis, GetValueDims(op->operand_source(0), reduce_axis)));
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMappingForReshape(pir::Operation* op) {
  PADDLE_ENFORCE(
      op->num_operands() == 1 && op->num_results() == 1,
      ::common::errors::InvalidArgument(
          "num_operands and num_results of reshape_op shall be equal 1."));
  std::vector<int64_t> shape =
      GetInt64ArrayAttributeData(op->attributes().at("to"));
  auto rank = static_cast<int>(shape.size());

  auto in_shape = GetValueAllDims(op->operand_source(0));
  auto out_shape = GetValueAllDims(op->result(0));
  LoopAxisMapping result;
  result.input2loop.push_back(
      std::make_shared<ReshapeTransform>(in_shape, out_shape));
  result.loop2output.push_back(IdentityTransform::InstancePtr());
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMapping(pir::Operation* op) {
  auto op_kind = GetOpPatternKind(op);
  if (op_kind == hlir::framework::kElementWise) {
    return CreateAxisMappingForElementwise(op);
  } else if (op->name() == "pd_op.transpose") {
    return CreateAxisMappingForTranspose(op);
  } else if (op->name() == "cinn_op.slice") {
    return CreateAxisMappingForSlice(op);
  } else if (op_kind == hlir::framework::kBroadcast) {
    return CreateAxisMappingForBroadcast(op);
  } else if (op_kind == hlir::framework::kReduction) {
    return CreateAxisMappingForReduce(op);
  } else if (op->name() == "cinn_op.reshape" || op->name() == "pd_op.reshape") {
    return CreateAxisMappingForReshape(op);
  } else {
    return CreateDefaultAxisMapping(op);
  }
}

}  // namespace cinn::fusion
