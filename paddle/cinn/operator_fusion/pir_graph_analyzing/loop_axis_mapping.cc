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
  for (auto& route : input2loop) {
    PADDLE_ENFORCE(
        !route.empty(),
        ::common::errors::InvalidArgument("input2loop must not be empty."));
    loop2input.push_back(ReverseTransformRoute(route));
  }
  for (auto& route : loop2output) {
    PADDLE_ENFORCE(
        !route.empty(),
        ::common::errors::InvalidArgument("loop2output must not be empty."));
    output2loop.push_back(ReverseTransformRoute(route));
  }
}

std::string LoopAxisMapping::DebugStr() const {
  std::stringstream ss;
  auto print_route = [](const AxisTransformRoute& route) {
    return cinn::utils::Join(route, " -> ");
  };
  for (size_t i = 0; i < input_values.size(); ++i) {
    ss << "\n input " << i << " :\t["
       << cinn::utils::Join(GetValueAllDims(input_values[i]), ", ") << "]";
  }
  ss << "\n  loop   :\t[" << cinn::utils::Join(loop, ", ") << "]";
  for (size_t i = 0; i < output_values.size(); ++i) {
    ss << "\noutput " << i << " :\t["
       << cinn::utils::Join(GetValueAllDims(output_values[i]), ", ") << "]";
  }
  for (size_t i = 0; i < input2loop.size(); ++i) {
    ss << "\ninput2loop  " << i << " : " << print_route(input2loop[i]);
    ss << "\nloop2input  " << i << " : " << print_route(loop2input[i]);
  }
  for (size_t i = 0; i < loop2output.size(); ++i) {
    ss << "\nloop2output " << i << " : " << print_route(loop2output[i]);
    ss << "\noutput2loop " << i << " : " << print_route(output2loop[i]);
  }
  return ss.str();
}

// LoopAxisMapping MergeAxisMapping(LoopAxisMapping upstream,
//                                  LoopAxisMapping downstream,
//                                  bool upstream_is_anchor) {
//   LoopAxisMapping result;
//   if (upstream_is_anchor) {
//     result.input2loop = upstream.input2loop;
//     result.loop2input = upstream.loop2input;
//     result.loop2output = ConcatVector(
//         upstream.loop2output,
//         ConcatVector(downstream.input2loop, downstream.loop2output));
//     result.output2loop = ConcatVector(
//         ConcatVector(downstream.output2loop, downstream.loop2input),
//         upstream.output2loop);
//   } else {
//     result.input2loop =
//         ConcatVector(ConcatVector(upstream.input2loop, upstream.loop2output),
//                      downstream.input2loop);
//     result.loop2input =
//         ConcatVector(downstream.loop2input,
//                      ConcatVector(upstream.output2loop,
//                      upstream.loop2input));
//     result.loop2output = downstream.loop2output;
//     result.output2loop = downstream.output2loop;
//   }
//   return result;
// }

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
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMappingForElementwise(pir::Operation* op) {
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
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMappingForTranspose(pir::Operation* op) {
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
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMappingForSlice(pir::Operation* op) {
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
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMappingForBroadcast(pir::Operation* op) {
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

LoopAxisMapping CreateAxisMappingForReduce(pir::Operation* op) {
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
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMappingForReshape(pir::Operation* op) {
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
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping CreateAxisMapping(pir::Operation* op) {
  auto op_kind = GetOpPatternKind(op);
  if (op->name() == "pd_op.transpose") {
    return CreateAxisMappingForTranspose(op);
  } else if (op->name() == "cinn_op.reshape" || op->name() == "pd_op.reshape") {
    return CreateAxisMappingForReshape(op);
  } else if (op->name() == "cinn_op.slice") {
    return CreateAxisMappingForSlice(op);
  } else if (op_kind == hlir::framework::kBroadcast) {
    return CreateAxisMappingForBroadcast(op);
  } else if (op_kind == hlir::framework::kReduction) {
    return CreateAxisMappingForReduce(op);
  } else if (op_kind == hlir::framework::kElementWise) {
    return CreateAxisMappingForElementwise(op);
  } else {
    return CreateDefaultAxisMapping(op);
  }
}

}  // namespace cinn::fusion
