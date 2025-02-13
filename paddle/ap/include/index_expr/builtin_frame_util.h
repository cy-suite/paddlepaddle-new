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
#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/index_expr/value_method_class.h"

namespace ap::index_expr {

template <typename ValueT, typename DoEachT>
void VisitEachBuiltinFrameClass(const DoEachT& DoEach) {
  DoEach(axpr::GetDimExprClass<ValueT>());
  DoEach(GetSliceClass<ValueT>());
  DoEach(GetIndexExprClass<ValueT>());
  DoEach(GetInIndexTupleExprSignatureClass<ValueT>());
  DoEach(GetOutIndexTupleExprSignatureClass<ValueT>());
  DoEach(GetOpIndexTupleExprSignatureClass<ValueT>());
}

template <typename ValueT>
ap::axpr::AttrMap<ValueT> MakeBuiltinFrameAttrMap() {
  ap::axpr::AttrMap<ValueT> attr_map;
  ap::axpr::VisitEachBuiltinFrameAttr<ValueT>(
      [&](const std::string& k, const ValueT& v) { attr_map->Set(k, v); });
  VisitEachBuiltinFrameClass(
      [&](const auto& cls) { attr_map->Set(cls.Name(), cls); });
  return attr_map;
}

}  // namespace ap::index_expr
