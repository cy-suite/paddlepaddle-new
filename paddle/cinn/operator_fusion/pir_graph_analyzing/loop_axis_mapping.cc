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

AxisTransform AppendAxisTransform::reverse() {
  return std::make_shared<DeleteAxisTransform>(axis, shape);
}
AxisTransform DeleteAxisTransform::reverse() {
  return std::make_shared<AppendAxisTransform>(axis, shape);
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

std::string LoopAxisMapping::DebugStr() const {
  std::stringstream ss;
  for (size_t i = 0; i < input_values.size(); ++i) {
    ss << "\n input " << i << " :\t["
       << cinn::utils::Join(GetCompatibleValueAllDims(input_values[i]), ", ")
       << "], " << input_values[i].impl();
  }
  ss << "\n  loop   :\t[" << cinn::utils::Join(loop, ", ")
     << "], reduce_axis_num: " << reduce_axis_num;
  for (size_t i = 0; i < output_values.size(); ++i) {
    ss << "\noutput " << i << " :\t["
       << cinn::utils::Join(GetCompatibleValueAllDims(output_values[i]), ", ")
       << "], " << output_values[i].impl()
       << ", use_count: " << outputs_use_count.at(output_values[i]);
  }
  for (size_t i = 0; i < input2loop.size(); ++i) {
    ss << "\ninput2loop  " << i << " : " << input2loop[i];
  }
  for (size_t i = 0; i < loop2output.size(); ++i) {
    ss << "\nloop2output " << i << " : " << loop2output[i];
  }
  return ss.str();
}

bool HasUnsupportedTransform(const AxisTransformRoute& route) {
  return std::any_of(route.begin(), route.end(), [](const auto& transform) {
    return std::holds_alternative<UnsupportedTransformPtr>(transform);
  });
}

class AxisTransformSimulator {
 public:
  AxisTransformSimulator() = delete;
  AxisTransformSimulator(const AxisTransformRoute& route,
                         const std::vector<symbol::DimExpr>& inshape)
      : route_(route) {
    for (size_t i = 0; i < inshape.size(); ++i) {
      source_ids_.push_back(UniqueAxisId());
      axis_symbols_[source_ids_[i]] = inshape[i];
    }
    target_ids_ = source_ids_;
    Simulate();
  }

  std::set<std::string> GetRelatedAxisIds(const std::vector<std::string>& ids) {
    std::deque<std::string> queue(ids.begin(), ids.end());
    std::set<std::string> related_ids;
    while (!queue.empty()) {
      auto cur_id = queue.front();
      queue.pop_front();
      if (related_ids.count(cur_id)) continue;
      related_ids.insert(cur_id);
      if (axis_relation_map_.count(cur_id)) {
        for (const auto& id : axis_relation_map_.at(cur_id)) {
          queue.push_back(id);
        }
      }
    }
    return related_ids;
  }

  const AxisTransformRoute& route_;
  std::vector<std::string> source_ids_;
  std::vector<std::string> target_ids_;
  std::unordered_map<std::string, symbol::DimExpr> axis_symbols_;
  std::map<std::string, std::set<std::string>> axis_relation_map_;

 private:
  void Simulate() {
    auto simulate_transform = adt::match{
        [&](const IdentityTransformPtr&) {},
        [&](const TransposeTransformPtr& transform) {
          target_ids_ = TransposeVector(target_ids_, transform->perm);
        },
        [&](const AppendAxisTransformPtr& transform) {
          for (int i = 0; i < transform->axis.size(); ++i) {
            auto new_id = UniqueAxisId();
            target_ids_.insert(target_ids_.begin() + transform->axis[i],
                               new_id);
            axis_symbols_[new_id] = transform->shape[i];
          }
        },
        [&](const DeleteAxisTransformPtr& transform) {
          for (int i = transform->axis.size() - 1; i >= 0; --i) {
            auto id_to_delete = target_ids_[transform->axis[i]];
            target_ids_.erase(target_ids_.begin() + transform->axis[i]);
            axis_symbols_[id_to_delete] = transform->shape[i];
          }
        },
        [&](const ReshapeTransformPtr& transform) {
          const auto& in_shape = transform->in_shape;
          const auto& out_shape = transform->out_shape;
          const auto& partition_indices =
              PartitionReshapeAxes(in_shape, out_shape);
          std::vector<std::string> new_ids;
          for (int idx = 1; idx < partition_indices.size(); ++idx) {
            const auto& [in_start, out_start] = partition_indices[idx - 1];
            const auto& [in_end, out_end] = partition_indices[idx];
            if (in_end == in_start + 1 && out_end == out_start + 1) {
              new_ids.push_back(target_ids_[in_start]);
            } else {
              for (int i = out_start; i < out_end; ++i) {
                if (out_shape[i] == symbol::DimExpr(1)) {
                  new_ids.push_back(UniqueAxisId());
                  axis_symbols_[new_ids.back()] = symbol::DimExpr(1);
                } else {
                  std::string axis_id;
                  for (int j = in_start; j < in_end; ++j) {
                    if (in_shape[j] == symbol::DimExpr(1)) {
                      continue;
                    } else if (in_shape[j] == out_shape[i]) {
                      axis_id = target_ids_[j];
                      break;
                    } else {
                      if (axis_id.empty()) axis_id = UniqueAxisId();
                      axis_relation_map_[target_ids_[j]].insert(axis_id);
                    }
                  }
                  new_ids.push_back(axis_id);
                  if (!axis_symbols_.count(axis_id)) {
                    axis_symbols_[axis_id] = out_shape[i];
                  }
                }
              }
            }
          }
          for (int i = in_shape.size(); i < target_ids_.size(); ++i) {
            new_ids.push_back(target_ids_[i]);
          }
          target_ids_ = new_ids;
        },
        [&](const auto& trans) {
          PADDLE_THROW(
              ::common::errors::Unimplemented("Unsupported transform."));
        },
    };
    for (const auto& trans : route_) {
      std::visit(simulate_transform, trans);
    }
  }

  int id_counter_ = 0;
  std::string UniqueAxisId() { return "I" + std::to_string(id_counter_++); }
};

AxisTransformRoute SimplifySimpleTransform(
    const AxisTransformRoute& route,
    const std::vector<symbol::DimExpr>& inshape) {
  if (route.size() <= 1) return route;
  // 1. Simulate transform route
  AxisTransformSimulator simulator(route, inshape);
  // 2. Get Simlplified transform route
  AxisTransformRoute result;
  auto& source_ids = simulator.source_ids_;
  auto& target_ids = simulator.target_ids_;
  auto& axis_symbols = simulator.axis_symbols_;
  if (source_ids == target_ids) {
    result.push_back(IdentityTransform::InstancePtr());
  } else {
    auto [source_unique_ids, source_unique_pos] =
        GatherFirstNotInSecond(source_ids, target_ids);
    auto [target_unique_ids, target_unique_pos] =
        GatherFirstNotInSecond(target_ids, source_ids);
    auto medium_ids = source_ids;
    if (!source_unique_ids.empty()) {
      auto delete_symbols = GatherMapValue(axis_symbols, source_unique_ids);
      result.push_back(std::make_shared<DeleteAxisTransform>(
          CastVector<int32_t, int64_t>(source_unique_pos), delete_symbols));
      medium_ids = GatherVectorExcept(medium_ids, source_unique_pos);
    }
    if (!target_unique_ids.empty()) {
      auto append_symbols = GatherMapValue(axis_symbols, target_unique_ids);
      result.push_back(std::make_shared<AppendAxisTransform>(
          CastVector<int32_t, int64_t>(target_unique_pos), append_symbols));
      for (const auto& pos : target_unique_pos) {
        medium_ids.insert(medium_ids.begin() + pos, target_ids[pos]);
      }
    }
    if (medium_ids != target_ids) {
      auto perm = GetTransposePerm<int32_t>(medium_ids, target_ids);
      result.push_back(std::make_shared<TransposeTransform>(perm));
    }
  }
  return result;
}

AxisTransformRoute SimplifyContinuousReshape(const AxisTransformRoute& route) {
  if (route.size() <= 1) return route;
  const auto simplify_reshape =
      [](const AxisTransformRoute& route) -> AxisTransformRoute {
    if (route.size() <= 1) return route;
    auto in_shape = std::get<ReshapeTransformPtr>(route.front())->in_shape;
    auto out_shape = std::get<ReshapeTransformPtr>(route.back())->out_shape;
    AxisTransformRoute result;
    if (in_shape == out_shape) {
      result.push_back(IdentityTransform::InstancePtr());
    } else {
      result.push_back(std::make_shared<ReshapeTransform>(in_shape, out_shape));
    }
    return result;
  };
  AxisTransformRoute result;
  AxisTransformRoute continuous_reshape;
  for (const auto& trans : route) {
    if (std::holds_alternative<UnsupportedTransformPtr>(trans)) {
      return {trans};
    } else if (std::holds_alternative<IdentityTransformPtr>(trans)) {
      // Do nothing.
    } else if (std::holds_alternative<ReshapeTransformPtr>(trans)) {
      continuous_reshape.push_back(std::get<ReshapeTransformPtr>(trans));
    } else {
      if (!continuous_reshape.empty()) {
        result = ConcatVector(result, simplify_reshape(continuous_reshape));
        continuous_reshape.clear();
      }
      result.push_back(trans);
    }
  }
  if (!continuous_reshape.empty()) {
    result = ConcatVector(result, simplify_reshape(continuous_reshape));
  }
  return result;
}

AxisTransformRoute SimplifyTransformRoute(
    const AxisTransformRoute& route,
    const std::vector<symbol::DimExpr>& input_shape) {
  AxisTransformRoute reshape_simplified = SimplifyContinuousReshape(route);
  if (reshape_simplified.size() <= 1) return reshape_simplified;
  // Simplify continuous non-reshape route.
  AxisTransformRoute result;
  AxisTransformRoute part;
  auto inshape = input_shape;
  for (const auto& trans : reshape_simplified) {
    if (std::holds_alternative<UnsupportedTransformPtr>(trans)) {
      return {trans};
    } else if (std::holds_alternative<IdentityTransformPtr>(trans)) {
      // Do nothing.
    } else if (auto reshape_trans = std::get_if<ReshapeTransformPtr>(&trans)) {
      if (!part.empty()) {
        result = ConcatVector(result, SimplifySimpleTransform(part, inshape));
        part.clear();
      }
      result.push_back(trans);
      // Reshape transform only change the first dims in some cases.
      auto next_shape = (*reshape_trans)->out_shape;
      for (int i = (*reshape_trans)->in_shape.size(); i < inshape.size(); ++i) {
        next_shape.push_back(inshape[i]);
      }
      inshape = next_shape;
    } else {
      part.push_back(trans);
    }
  }
  result = ConcatVector(result, SimplifySimpleTransform(part, inshape));
  return result;
}

void LoopAxisMapping::SimplifyForwardMapping() {
  for (int i = 0; i < input_values.size(); ++i) {
    input2loop[i] = SimplifyTransformRoute(
        input2loop[i], GetCompatibleValueAllDims(input_values[i]));
  }
  for (int i = 0; i < output_values.size(); ++i) {
    loop2output[i] = SimplifyTransformRoute(loop2output[i], loop);
  }
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

void LoopAxisMapping::DisableLoopMapping() {
  for (int i = 0; i < input_values.size(); ++i) {
    input2loop[i].clear();
    input2loop[i].push_back(UnsupportedTransform::InstancePtr());
  }
  for (int i = 0; i < output_values.size(); ++i) {
    loop2output[i].clear();
    loop2output[i].push_back(UnsupportedTransform::InstancePtr());
  }
  loop.clear();
  reduce_axis_num = 0;
  SetReverseMapping();
}

AxisTransformRoute GetLoopSinkRoute(const LoopAxisMapping& upstream,
                                    const LoopAxisMapping& downstream) {
  AxisTransformRoute result;
  for (size_t i = 0; i < upstream.output_values.size(); ++i) {
    auto value = upstream.output_values[i];
    auto indices = FindPosInVector(downstream.input_values, value);
    for (auto idx : indices) {
      AxisTransformRoute route =
          ConcatVector(upstream.loop2output[i], downstream.input2loop[idx]);
      if (HasUnsupportedTransform(route)) continue;
      if (route.size() < result.size() || result.empty()) result = route;
    }
  }
  if (result.empty()) {
    result.push_back(UnsupportedTransform::InstancePtr());
    return result;
  }
  return SimplifyTransformRoute(result, upstream.loop);
}

AxisTransformRoute GetLoopLiftRoute(const LoopAxisMapping& upstream,
                                    const LoopAxisMapping& downstream) {
  AxisTransformRoute result;
  for (size_t i = 0; i < upstream.output_values.size(); ++i) {
    auto value = upstream.output_values[i];
    auto indices = FindPosInVector(downstream.input_values, value);
    for (auto idx : indices) {
      AxisTransformRoute route =
          ConcatVector(downstream.loop2input[idx], upstream.output2loop[i]);
      if (HasUnsupportedTransform(route)) continue;
      if (route.size() < result.size() || result.empty()) result = route;
    }
  }
  if (result.empty()) {
    result.push_back(UnsupportedTransform::InstancePtr());
    return result;
  }
  return SimplifyTransformRoute(result, downstream.loop);
}

LoopAxisMapping LoopMappingMergeImpl(const LoopAxisMapping& upstream,
                                     const LoopAxisMapping& downstream,
                                     bool upstream_is_anchor) {
  const auto& loop_sink_route = GetLoopSinkRoute(upstream, downstream);
  const auto& loop_lift_route = GetLoopLiftRoute(upstream, downstream);

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
  auto result = LoopMappingMergeImpl(upstream, downstream, upstream_is_anchor);
  result.SimplifyForwardMapping();
  result.SetReverseMapping();
  return result;
}

LoopAxisMapping TrivialSinkLoopMappingMerge(const LoopAxisMapping& upstream,
                                            const LoopAxisMapping& downstream) {
  auto result = LoopMappingMergeImpl(upstream, downstream, false);
  auto upstream_out_value = upstream.output_values[0];
  auto indices = FindPosInVector(result.output_values, upstream_out_value);
  if (!indices.empty()) {
    auto idx = indices.front();
    result.output_values.erase(result.output_values.begin() + idx);
    result.loop2output.erase(result.loop2output.begin() + idx);
    result.outputs_use_count.erase(upstream_out_value);
  }
  result.SimplifyForwardMapping();
  result.SetReverseMapping();
  return result;
}

std::vector<int> GetFakeReduceAxisIdx(const std::vector<symbol::DimExpr>& loop,
                                      const AxisTransformRoute& route,
                                      int reduce_axis_num) {
  AxisTransformSimulator simulator(route, loop);
  auto reduce_trivial_related_ids =
      simulator.GetRelatedAxisIds(simulator.source_ids_);
  std::set<std::string> trivial_non_related_ids;
  for (const auto& axis_id : simulator.target_ids_) {
    if (!reduce_trivial_related_ids.count(axis_id)) {
      trivial_non_related_ids.insert(axis_id);
    }
  }
  std::vector<int> fake_reduce_idx;
  for (int i = loop.size() - reduce_axis_num; i < loop.size(); ++i) {
    auto reduce_axis_id = simulator.source_ids_[i];
    auto indices = FindPosInVector(simulator.target_ids_, reduce_axis_id);
    if (!indices.empty()) {
      fake_reduce_idx.push_back(indices.front());
      continue;
    }
    for (const auto& axis_id : trivial_non_related_ids) {
      if (loop[i] == simulator.axis_symbols_.at(axis_id)) {
        fake_reduce_idx.push_back(
            FindPosInVector(simulator.target_ids_, axis_id).front());
        trivial_non_related_ids.erase(axis_id);
        break;
      }
    }
  }
  return fake_reduce_idx;
}

LoopAxisMapping ReducePlusTrivialLoopMappingMerge(
    const LoopAxisMapping& upstream, const LoopAxisMapping& downstream) {
  // Signal downstream reduce plus trivial fusion loop is downstream trivial
  // loop plus upstream reduce loop.
  PADDLE_ENFORCE(
      upstream.reduce_axis_num > 0 && downstream.reduce_axis_num == 0,
      ::common::errors::InvalidArgument(
          "Upstream should be reduce pattern and "
          "downstream should be trivial pattern."));
  auto loop_sink_route = GetLoopSinkRoute(upstream, downstream);
  if (HasUnsupportedTransform(loop_sink_route)) {
    // TODO(huangjiyi): fix unsupported transform in RT fusion
    auto result = LoopMappingMergeImpl(upstream, downstream, false);
    result.DisableLoopMapping();
    return result;
  }
  auto reduce_axis_num = upstream.reduce_axis_num;
  auto reduce_axis = ArangeVector<int64_t>(
      upstream.loop.size() - reduce_axis_num, upstream.loop.size());
  auto reduce_loop = SliceVector(upstream.loop,
                                 upstream.loop.size() - reduce_axis_num,
                                 upstream.loop.size());
  // Check whether downstream trivial can reuse upstream reduce axis.
  auto fake_reduce_idx =
      GetFakeReduceAxisIdx(upstream.loop, loop_sink_route, reduce_axis_num);
  VLOG(4) << "fake_reduce_idx: " << cinn::utils::Join(fake_reduce_idx, ",");
  LoopAxisMapping result;
  if (fake_reduce_idx.empty()) {
    AxisTransform append_reduce_axis =
        std::make_shared<AppendAxisTransform>(reduce_axis, reduce_loop);
    AxisTransform delete_reduce_axis = ReverseTransform(append_reduce_axis);
    auto upstream_copy = upstream;
    for (auto& route : upstream_copy.input2loop) {
      route.push_back(append_reduce_axis);
    }
    upstream_copy.loop.insert(
        upstream_copy.loop.end(), reduce_loop.begin(), reduce_loop.end());
    result = LoopMappingMergeImpl(upstream_copy, downstream, false);
    result.loop = ConcatVector(downstream.loop, reduce_loop);
    for (auto& route : result.loop2output) {
      route.insert(route.begin(), delete_reduce_axis);
    }
    auto fake_reduce_idx = ArangeVector<int64_t>(
        downstream.loop.size(), downstream.loop.size() + reduce_axis_num);
    AxisTransform append_fake_reduce_idx =
        std::make_shared<AppendAxisTransform>(fake_reduce_idx, reduce_loop);
    for (int i = upstream.input2loop.size(); i < result.input2loop.size();
         ++i) {
      result.input2loop[i].push_back(append_fake_reduce_idx);
    }
  } else {
    // Transpose fake reduce axis to the end
    auto perm = ArangeVector<int>(0, downstream.loop.size());
    for (auto index : fake_reduce_idx) {
      perm.push_back(index);
    }
    std::sort(fake_reduce_idx.begin(), fake_reduce_idx.end());
    std::reverse(fake_reduce_idx.begin(), fake_reduce_idx.end());
    for (auto index : fake_reduce_idx) {
      perm.erase(perm.begin() + index);
    }
    result = LoopMappingMergeImpl(upstream, downstream, false);
    AxisTransformRoute fake_reduce_axis_transforms;
    if (perm != ArangeVector<int>(0, downstream.loop.size())) {
      result.loop = TransposeVector(result.loop, perm);
      auto transpose_trans = std::make_shared<TransposeTransform>(perm);
      fake_reduce_axis_transforms.push_back(transpose_trans);
    }
    // Check whether fake reduce axis reuse all reduce axis
    if (fake_reduce_idx.size() < reduce_axis_num) {
      std::vector<int64_t> one_reduce_axis;
      for (int i = 0; i < reduce_loop.size(); ++i) {
        bool has_reuse = false;
        for (const auto& downstream_idx : fake_reduce_idx) {
          if (reduce_loop[i] == downstream.loop[downstream_idx]) {
            has_reuse = true;
            break;
          }
        }
        if (!has_reuse) {
          PADDLE_ENFORCE_EQ(reduce_loop[i],
                            symbol::DimExpr(1),
                            ::common::errors::PreconditionNotMet(
                                "Reduce axis not been reused must be 1."));
          one_reduce_axis.push_back(downstream.loop.size() -
                                    fake_reduce_idx.size() + i);
        }
      }
      auto append_one_reduce_axis =
          std::make_shared<AppendAxisTransform>(one_reduce_axis);
      fake_reduce_axis_transforms.push_back(append_one_reduce_axis);
    }
    for (auto& route : result.input2loop) {
      route.insert(route.end(),
                   fake_reduce_axis_transforms.begin(),
                   fake_reduce_axis_transforms.end());
    }
    for (auto& route : result.loop2output) {
      route.insert(route.begin(),
                   fake_reduce_axis_transforms.begin(),
                   fake_reduce_axis_transforms.end());
    }
  }
  result.SimplifyForwardMapping();
  result.SetReverseMapping();
  return result;
}

std::optional<AxisTransformRoute> GetValidLoopTransformRoute(
    const LoopAxisMapping& upstream,
    const LoopAxisMapping& downstream,
    bool upstream_is_anchor) {
  VLOG(4) << "Try to get valid loop transform route "
          << (upstream_is_anchor ? "from downstream to upstream."
                                 : "from upstream to downstream.");
  auto source = upstream_is_anchor ? downstream : upstream;
  auto target = upstream_is_anchor ? upstream : downstream;
  VLOG(4) << "Source loop: [" << cinn::utils::Join(source.loop, ", ")
          << "], reduce_axis_num: " << source.reduce_axis_num;
  VLOG(4) << "Target loop: [" << cinn::utils::Join(target.loop, ", ")
          << "], reduce_axis_num: " << target.reduce_axis_num;
  if (source.reduce_axis_num > 0 && target.reduce_axis_num == 0) {
    VLOG(4) << "Cannot transform reduce loop to trivial loop.";
    return std::nullopt;
  } else if (source.reduce_axis_num > 0 && target.reduce_axis_num > 0) {
    if (source.reduce_axis_num != target.reduce_axis_num) {
      VLOG(4) << "Cannot transform reduce loop to different reduce axis num.";
      return std::nullopt;
    }
    auto get_reduce_loop = [](const LoopAxisMapping& mapping) {
      return SliceVector(mapping.loop,
                         mapping.loop.size() - mapping.reduce_axis_num,
                         mapping.loop.size());
    };
    auto source_reduce_loop = get_reduce_loop(source);
    auto target_reduce_loop = get_reduce_loop(target);
    for (size_t i = 0; i < source_reduce_loop.size(); ++i) {
      if (source_reduce_loop[i] != target_reduce_loop[i]) {
        VLOG(4) << "Cannot transform reduce loop to unaligned reduce axis.";
        return std::nullopt;
      }
    }
  }
  bool rr_fusion = source.reduce_axis_num > 0 && target.reduce_axis_num > 0;

  const auto& loop_transform_route =
      upstream_is_anchor ? GetLoopLiftRoute(upstream, downstream)
                         : GetLoopSinkRoute(upstream, downstream);
  VLOG(4) << "Loop transform route: " << loop_transform_route;

  size_t id = 0;
  auto unique_id = [&]() { return "I" + std::to_string(id++); };

  AxisTransformRoute result;
  std::vector<std::string> axis_ids;
  std::unordered_map<std::string, symbol::DimExpr> axis_symbols;
  std::set<std::string> deleted_axes;
  std::unordered_set<std::string> unused_axes;
  std::map<std::string, std::set<std::string>> axis_relation_map;
  for (const auto& symbol : source.loop) {
    auto axis_id = unique_id();
    axis_ids.push_back(axis_id);
    axis_symbols[axis_id] = symbol;
  }
  size_t cur_axis_size = axis_ids.size();
  const auto source_ids = axis_ids;

  if (rr_fusion) {
    // Because reduce axis can not be transformed, we need to add
    // same fake axis to substitute reduce axis for transformation.
    std::vector<int64_t> reduce_axis = ArangeVector<int64_t>(
        cur_axis_size - source.reduce_axis_num, cur_axis_size);
    std::vector<symbol::DimExpr> reduce_shape =
        GatherVector(source.loop, reduce_axis);
    result.push_back(
        std::make_shared<AppendAxisTransform>(reduce_axis, reduce_shape));
  }

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
          // Can reuse deleted axis, move deleted axis to the append position.
          int deleted_axis_pos = FindPosInVector(axis_ids, deleted_axis).back();
          auto new_axis_id = unique_id();
          if (deleted_axis_pos == axis) {
            axis_ids[deleted_axis_pos] = new_axis_id;
          } else {
            auto perm = ArangeVector<int32_t>(0, axis_ids.size());
            perm.erase(perm.begin() + deleted_axis_pos);
            perm.insert(perm.begin() + axis, deleted_axis_pos);
            axis_ids.erase(axis_ids.begin() + deleted_axis_pos);
            axis_ids.insert(axis_ids.begin() + axis, new_axis_id);
            result.push_back(std::make_shared<TransposeTransform>(perm));
          }
          axis_symbols[new_axis_id] = symbol;
          deleted_axes.erase(deleted_axis);
          cur_axis_size++;
          can_reuse = true;
          VLOG(4) << "Reuse axis: " << new_axis_id << " -> " << deleted_axis
                  << ", cur deleted_axes: {"
                  << cinn::utils::Join(SetToVector(deleted_axes), ", ") << "}";
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
        VLOG(4) << "Insert new unused axis: " << axis_id
                << ", cur unused_axes: {"
                << cinn::utils::Join(SetToVector(unused_axes), ", ") << "}";
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
        VLOG(4) << "Delete unused or size 1 axis: " << axis_id
                << ", cur unused_axes: {"
                << cinn::utils::Join(SetToVector(unused_axes), ", ") << "}";
      } else {
        // Used axis can not be deleted directly, we need to transpose it to
        // the end to ensure accuracy of subsequent transform.
        std::vector<std::string> new_axis_ids;
        // No need to transpose if the axis is already at the end.
        if (axis != cur_axis_size - 1) {
          std::vector<int> perm;
          for (int idx = 0; idx < axis_ids.size(); ++idx) {
            if (idx == axis) continue;
            new_axis_ids.push_back(axis_ids[idx]);
            perm.push_back(idx);
          }
          new_axis_ids.push_back(axis_id);
          perm.push_back(axis);
          result.push_back(std::make_shared<TransposeTransform>(perm));
        } else {
          new_axis_ids = axis_ids;
        }
        deleted_axes.insert(axis_id);
        axis_ids = new_axis_ids;
        VLOG(4) << "Pretend to delete axis: " << axis_id
                << ", cur deleted_axes: {"
                << cinn::utils::Join(SetToVector(deleted_axes), ", ") << "}";
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
      for (const auto& in_axis : axis_ids) {
        for (const auto& out_axis : new_axis_ids) {
          axis_relation_map[in_axis].insert(out_axis);
        }
      }
    } else {
      const auto& partition_indices = PartitionReshapeAxes(in_shape, out_shape);
      for (int idx = 1; idx < partition_indices.size(); ++idx) {
        const auto& [in_start, out_start] = partition_indices[idx - 1];
        const auto& [in_end, out_end] = partition_indices[idx];
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
            if (out_shape[i] == symbol::DimExpr(1)) {
              auto axis_id = unique_id();
              new_axis_ids.push_back(axis_id);
              axis_symbols[axis_id] = symbol::DimExpr(1);
            } else {
              std::string axis_id;
              for (int j = in_start; j < in_end; ++j) {
                if (in_shape[j] == symbol::DimExpr(1)) {
                  continue;
                } else if (in_shape[j] == out_shape[i]) {
                  axis_id = axis_ids[j];
                  break;
                } else {
                  if (axis_id.empty()) axis_id = unique_id();
                  axis_relation_map[axis_ids[j]].insert(axis_id);
                }
              }
              new_axis_ids.push_back(axis_id);
              if (!axis_symbols.count(axis_id)) {
                axis_symbols[axis_id] = out_shape[i];
                if (is_unused) unused_axes.insert(axis_id);
              }
            }
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

  auto axis_debug_info = [&]() -> std::string {
    std::vector<symbol::DimExpr> shape;
    for (const auto& id : axis_ids) {
      shape.push_back(axis_symbols.at(id));
    }
    return "Axis ids: [" + cinn::utils::Join(axis_ids, ", ") + "], shape: [" +
           cinn::utils::Join(shape, ", ") +
           "], cur_size: " + std::to_string(cur_axis_size);
  };

  VLOG(4) << "Source axis ids: " << axis_debug_info();
  for (auto& transform : loop_transform_route) {
    if (std::holds_alternative<UnsupportedTransformPtr>(transform)) {
      VLOG(4) << "Can not find valid loop transform because of unsupported "
                 "transform.";
      return std::nullopt;
    } else {
      std::visit(apply_transform, transform);
      VLOG(4) << "After Applying " << transform
              << ", axis ids: " << axis_debug_info();
    }
  }

  if (!deleted_axes.empty()) {
    // Check if all deleted axes are used, otherwise the transform is invalid.
    VLOG(4) << "Can not find valid loop transform because of unreused deleted "
               "axes.";
    return std::nullopt;
  }
  if (rr_fusion) {
    // Check if all reduce axes are reused and there is no relationship
    // between reduce axes and non reduce axes.
    auto [source_trivial_ids, source_reduce_ids] =
        SplitVector(source_ids, source_ids.size() - source.reduce_axis_num);

    auto get_related_ids = [&](const std::vector<std::string>& ids) {
      std::deque<std::string> queue(ids.begin(), ids.end());
      std::set<std::string> related_ids;
      while (!queue.empty()) {
        auto cur_id = queue.front();
        queue.pop_front();
        if (related_ids.count(cur_id)) continue;
        related_ids.insert(cur_id);
        if (axis_relation_map.count(cur_id)) {
          for (const auto& id : axis_relation_map.at(cur_id)) {
            queue.push_back(id);
          }
        }
      }
      return related_ids;
    };
    auto source_reduce_related_ids = get_related_ids(source_reduce_ids);
    auto source_trivial_related_ids = get_related_ids(source_trivial_ids);

    auto [target_trivial_ids, target_reduce_ids] =
        SplitVector(axis_ids, axis_ids.size() - target.reduce_axis_num);

    if (!SetIntersection(source_reduce_related_ids, ToSet(target_trivial_ids))
             .empty() ||
        !SetIntersection(source_trivial_related_ids, ToSet(target_reduce_ids))
             .empty()) {
      VLOG(4) << "Can not find valid loop transform because of relationship "
                 "between reduce axis and non reduce axis.";
      return std::nullopt;
    }
    // Remove fake reduce axes.
    std::vector<int64_t> reduce_axis = ArangeVector<int64_t>(
        axis_ids.size() - target.reduce_axis_num, axis_ids.size());
    std::vector<symbol::DimExpr> reduce_shape =
        GatherVector(target.loop, reduce_axis);
    result.push_back(
        std::make_shared<DeleteAxisTransform>(reduce_axis, reduce_shape));
  }
  if (source.reduce_axis_num == 0 && target.reduce_axis_num > 0) {
    // Check whether reduce trivial fusion with larger reduce dims.
    const auto& reduce_to_trivial_route =
        upstream_is_anchor ? GetLoopSinkRoute(target, source)
                           : GetLoopLiftRoute(source, target);
    auto fake_reduce_idx = GetFakeReduceAxisIdx(
        target.loop, reduce_to_trivial_route, target.reduce_axis_num);
    if (!fake_reduce_idx.empty()) {
      const auto reduce_dims_product =
          GetShapeProduct(target.loop,
                          target.loop.size() - target.reduce_axis_num,
                          target.loop.size());
      if (reduce_dims_product.isa<std::int64_t>() &&
          reduce_dims_product.dyn_cast<std::int64_t>() > 1024 * 8) {
        VLOG(4) << "Can not fuse trivial to reduce with large reduce dims: "
                << reduce_dims_product.dyn_cast<std::int64_t>();
        return std::nullopt;
      }
    }
  }
  if (result.empty()) result.push_back(IdentityTransform::InstancePtr());
  result = SimplifyTransformRoute(result, source.loop);
  VLOG(4) << "Found loop transform: " << result;
  return result;
}

LoopAxisMapping CreateDefaultLoopMapping(pir::Operation* op) {
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

LoopAxisMapping CreateDefaultLoopMappingForTrivialOp(pir::Operation* op) {
  auto result = CreateDefaultLoopMapping(op);
  result.loop = GetCompatibleValueAllDims(result.output_values[0]);
  result.loop2output[0].clear();
  result.loop2output[0].push_back(IdentityTransform::InstancePtr());
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
  result.loop = GetCompatibleValueAllDims(result.output_values[0]);
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
  result.loop = GetCompatibleValueAllDims(result.output_values[0]);
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
  result.loop = GetCompatibleValueAllDims(result.output_values[0]);
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
  auto input_shape = GetValueAllDims(op->operand_source(0));
  for (int i = axes.size() - 1; i >= 0; --i) {
    int64_t slice_size = ends[i] - starts[i];
    if (!decrease_axis_set.count(axes[i]) && slice_size != 1) {
      // TODO(huangjiyi): Support slice size greater than 1.
      result.input2loop[0].push_back(UnsupportedTransform::InstancePtr());
      break;
    }
    std::vector<int64_t> axis = {axes[i]};
    result.input2loop[0].push_back(std::make_shared<DeleteAxisTransform>(
        axis, GatherVector(input_shape, axis)));
    if (!decrease_axis_set.count(axes[i])) {
      result.input2loop[0].push_back(
          std::make_shared<AppendAxisTransform>(axis));
    }
  }
  if (GetRank(result.output_values[0]) == 0) {
    result.input2loop[0].push_back(
        std::make_shared<AppendAxisTransform>(std::vector<int64_t>{0}));
  }
  result.input2loop[0] =
      SimplifyTransformRoute(result.input2loop[0], input_shape);
  result.loop2output[0].push_back(IdentityTransform::InstancePtr());
  return result;
}

LoopAxisMapping CreateLoopMappingForBroadcast(pir::Operation* op) {
  LoopAxisMapping result;
  for (int i = 0; i < op->num_operands(); ++i) {
    result.input_values.push_back(op->operand_source(i));
  }
  result.input2loop.resize(op->num_operands());
  for (int i = 1; i < op->num_operands(); ++i) {
    result.input2loop[i].push_back(UnsupportedTransform::InstancePtr());
  }
  result.output_values.push_back(op->result(0));
  result.loop2output.resize(1);
  result.loop = GetValueAllDims(result.output_values[0]);

  const auto& broad_cast_value = GetBroadcastOpInputOutputValue(op);
  PADDLE_ENFORCE(broad_cast_value.has_value(),
                 ::common::errors::InvalidArgument(
                     "Required broad_cast_value is not empty."));
  const auto& [input_value, output_value] = broad_cast_value.value();
  const auto& input_shape = GetCompatibleValueAllDims(input_value);
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
  result.loop = GetCompatibleValueAllDims(result.input_values[0]);
  result.input2loop.resize(1);
  result.loop2output.resize(1);
  const auto& reduce_axis = GetReduceAxisIdx(op);
  result.reduce_axis_num = reduce_axis.size();
  bool keep_dim = GetReduceOpKeepDims(op);
  auto rank = result.loop.size();
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
  // Input2Loop: Insert a axis with size 1 when reduce all without keep_dim.
  if (result.loop.size() == reduce_axis.size()) {
    result.input2loop[0].push_back(
        std::make_shared<AppendAxisTransform>(std::vector<int64_t>{0}));
    result.loop.insert(result.loop.begin(), symbol::DimExpr(1));
  }
  if (result.input2loop[0].empty()) {
    result.input2loop[0].push_back(IdentityTransform::InstancePtr());
  }
  // Loop2Output: Delete reduce axis
  const auto& delete_axis = ArangeVector<int64_t>(
      result.loop.size() - reduce_axis.size(), result.loop.size());
  result.loop2output[0].push_back(std::make_shared<DeleteAxisTransform>(
      delete_axis, GatherVector(result.loop, delete_axis)));
  return result;
}

LoopAxisMapping CreateLoopMappingForReshape(pir::Operation* op) {
  LoopAxisMapping result;
  for (int i = 0; i < op->num_operands(); ++i) {
    result.input_values.push_back(op->operand_source(i));
  }
  result.input2loop.resize(op->num_operands());
  for (int i = 1; i < op->num_operands(); ++i) {
    result.input2loop[i].push_back(UnsupportedTransform::InstancePtr());
  }
  result.output_values.push_back(op->result(0));
  result.loop2output.resize(1);
  auto in_shape = GetCompatibleValueAllDims(op->operand_source(0));
  auto out_shape = GetValueAllDims(op->result(0));
  result.loop = out_shape;

  if (!ShapeProductEqual(in_shape, out_shape)) {
    return CreateDefaultLoopMappingForTrivialOp(op);
  }

  auto has_dynamic_shape = [](const std::vector<symbol::DimExpr>& shape) {
    return std::any_of(
        shape.begin(), shape.end(), [](const symbol::DimExpr& sym) {
          return !sym.isa<std::int64_t>();
        });
  };
  // TODO(huangjiyi): Support dynamic shape for reshape anchor fusion
  if (has_dynamic_shape(in_shape) || has_dynamic_shape(out_shape)) {
    return CreateDefaultLoopMappingForTrivialOp(op);
  }

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
  auto is_special_trivial = [&](const pir::Operation* op) {
    return op->name() == "cinn_op.concat" || op->name() == "pd_op.gather_nd";
  };
  VLOG(4) << "CreateLoopMapping for op: " << OpsDebugStr({op});
  LoopAxisMapping result;
  auto op_kind = GetOpPatternKind(op);
  if (op->name() == "pd_op.transpose") {
    result = CreateLoopMappingForTranspose(op);
  } else if (op->name() == "cinn_op.reshape" || op->name() == "pd_op.reshape") {
    result = CreateLoopMappingForReshape(op);
  } else if (op->name() == "cinn_op.slice") {
    result = CreateLoopMappingForSlice(op);
  } else if (op->name() == "cinn_op.generate_shape") {
    result = CreateDefaultLoopMapping(op);
  } else if (is_special_trivial(op)) {
    result = CreateDefaultLoopMappingForTrivialOp(op);
  } else if (op_kind == hlir::framework::kBroadcast) {
    result = CreateLoopMappingForBroadcast(op);
  } else if (op_kind == hlir::framework::kReduction) {
    result = CreateLoopMappingForReduce(op);
  } else if (op_kind == hlir::framework::kElementWise) {
    result = CreateLoopMappingForElementwise(op);
  } else {
    result = CreateDefaultLoopMapping(op);
  }
  result.SetReverseMapping();
  for (auto value : result.output_values) {
    result.outputs_use_count[value] = value.use_count();
  }
  VLOG(4) << "LoopMapping Result: " << result.DebugStr();
  return result;
}

}  // namespace cinn::fusion
