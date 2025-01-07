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

#include <variant>
#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {

#define DECLARE_TRANSFORM_PTR(T) \
  struct T;                      \
  using T##Ptr = std::shared_ptr<T>;

DECLARE_TRANSFORM_PTR(UnsupportedTransform);
DECLARE_TRANSFORM_PTR(IdentityTransform);
DECLARE_TRANSFORM_PTR(TransposeTransform);
DECLARE_TRANSFORM_PTR(DeleteAxisTransform);
DECLARE_TRANSFORM_PTR(AppendAxisTransform);
DECLARE_TRANSFORM_PTR(ReshapeTransform);
#undef DECLARE_TRANSFORM_PTR

using AxisTransform = std::variant<UnsupportedTransformPtr,
                                   IdentityTransformPtr,
                                   TransposeTransformPtr,
                                   DeleteAxisTransformPtr,
                                   AppendAxisTransformPtr,
                                   ReshapeTransformPtr>;
using AxisTransformRoute = std::vector<AxisTransform>;

struct UnsupportedTransform
    : public std::enable_shared_from_this<UnsupportedTransform> {
 public:
  static UnsupportedTransformPtr InstancePtr() {
    static UnsupportedTransform instance;
    return instance.shared_from_this();
  }
  AxisTransform reverse() { return UnsupportedTransform::InstancePtr(); }

 private:
  UnsupportedTransform() = default;
};

struct IdentityTransform
    : public std::enable_shared_from_this<IdentityTransform> {
 public:
  static IdentityTransformPtr InstancePtr() {
    static IdentityTransform instance;
    return instance.shared_from_this();
  }
  AxisTransform reverse() { return IdentityTransform::InstancePtr(); }

 private:
  IdentityTransform() = default;
};

struct TransposeTransform {
  explicit TransposeTransform(const std::vector<int32_t>& perm) : perm(perm) {}
  std::vector<int32_t> perm;
  AxisTransform reverse() {
    return std::make_shared<TransposeTransform>(GetReversePerm(perm));
  }
};

struct DeleteAxisTransform {
  explicit DeleteAxisTransform(const std::vector<int64_t>& axis,
                               const std::vector<symbol::DimExpr>& shape)
      : axis(axis), shape(shape) {}
  std::vector<int64_t> axis;
  std::vector<symbol::DimExpr> shape;
  AxisTransform reverse();
};

struct AppendAxisTransform {
  AppendAxisTransform(const std::vector<int64_t>& axis,
                      const std::vector<symbol::DimExpr>& shape)
      : axis(axis), shape(shape) {}
  explicit AppendAxisTransform(const std::vector<int64_t>& axis) : axis(axis) {
    shape = std::vector<symbol::DimExpr>(axis.size(), symbol::DimExpr(1));
  }
  std::vector<int64_t> axis;
  std::vector<symbol::DimExpr> shape;
  AxisTransform reverse();
};

struct ReshapeTransform {
  explicit ReshapeTransform(const std::vector<symbol::DimExpr>& in_shape,
                            const std::vector<symbol::DimExpr>& out_shape)
      : in_shape(in_shape), out_shape(out_shape) {}
  std::vector<symbol::DimExpr> in_shape;
  std::vector<symbol::DimExpr> out_shape;
  AxisTransform reverse() {
    return std::make_shared<ReshapeTransform>(out_shape, in_shape);
  }
};

AxisTransform ReverseTransform(const AxisTransform& transform);
AxisTransformRoute ReverseTransformRoute(const AxisTransformRoute& route);

struct LoopAxisMapping {
  AxisTransformRoute input2loop;
  AxisTransformRoute loop2output;
  AxisTransformRoute loop2input;
  AxisTransformRoute output2loop;

  void SetReverseMapping() {
    PADDLE_ENFORCE(
        !input2loop.empty() && !loop2output.empty(),
        ::common::errors::InvalidArgument(
            "input2loop and loop2output must not be empty before reverse."));
    loop2input = ReverseTransformRoute(input2loop);
    output2loop = ReverseTransformRoute(loop2output);
  }
};

LoopAxisMapping MergeAxisMapping(LoopAxisMapping upstream,
                                 LoopAxisMapping downstream,
                                 bool upstream_is_anchor = true);

LoopAxisMapping CreateAxisMapping(pir::Operation* op);
}  // namespace cinn::fusion
