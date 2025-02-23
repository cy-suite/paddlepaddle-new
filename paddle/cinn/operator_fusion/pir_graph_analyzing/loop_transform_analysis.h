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

#pragma once

#include "paddle/cinn/operator_fusion/pir_graph_analyzing/loop_axis_mapping.h"

namespace cinn::fusion {

LoopAxisMapping LoopMappingMerge(const LoopAxisMapping& upstream,
                                 const LoopAxisMapping& downstream,
                                 bool upstream_is_anchor);
LoopAxisMapping TrivialSinkLoopMappingMerge(const LoopAxisMapping& upstream,
                                            const LoopAxisMapping& downstream);
LoopAxisMapping ReducePlusTrivialLoopMappingMerge(
    const LoopAxisMapping& upstream, const LoopAxisMapping& downstream);

std::optional<AxisTransformRoute> GetValidLoopTransformRoute(
    const LoopAxisMapping& upstream,
    const LoopAxisMapping& downstream,
    bool upstream_is_anchor);

}  // namespace cinn::fusion
