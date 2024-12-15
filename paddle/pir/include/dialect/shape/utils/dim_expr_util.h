// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace symbol {
namespace {
using ConstRational = std::pair<std::int64_t, std::int64_t>;

std::int64_t GetInteger(const DimExpr& expr) {
  if (expr.Has<Negative<DimExpr>>()) {
    const auto& integer = expr.Get<Negative<DimExpr>>()->data;
    PADDLE_ENFORCE_EQ(integer.Has<std::int64_t>(),
                      true,
                      common::errors::InvalidArgument(
                          "input expression's member `data` has no attribution "
                          "`int64_t`, maybe input dim is empty"));
    return -integer.Get<std::int64_t>();
  }
  PADDLE_ENFORCE_EQ(
      expr.Has<std::int64_t>(),
      true,
      common::errors::InvalidArgument(
          "input has no attribution `int64_t`, maybe input dim is empty"));
  return expr.Get<std::int64_t>();
}

template <typename T>
std::optional<ConstRational> GetConstRationalImpl(const T& expr) {
  PADDLE_THROW(common::errors::Fatal("not supported."));
  return std::nullopt;
}

std::optional<ConstRational> GetConstRationalImpl(std::int64_t value) {
  return ConstRational{value, 1};
}

ConstRational GetConstRational(const DimExpr& expr) {
  return std::visit(
      [&](const auto& impl) {
        std::optional<ConstRational> opt_ret = GetConstRationalImpl(impl);
        PADDLE_ENFORCE_EQ(
            opt_ret.has_value(),
            true,
            common::errors::InvalidArgument("Input is empty, please check"));
        return opt_ret.value();
      },
      expr.variant());
}

ConstRational SimplifiedConstRational(int64_t num, int64_t dem) {
  std::int64_t gcd = std::gcd(num, dem);
  return ConstRational{num / gcd, dem / gcd};
}

ConstRational MulConstRational(const ConstRational& lhs,
                               const ConstRational& rhs) {
  const auto [lhs_num, lhs_dem] = lhs;
  const auto [rhs_num, rhs_dem] = rhs;
  // Crossing is correct.
  const auto [simplified_lhs_num, simplified_rhs_dem] =
      SimplifiedConstRational(lhs_num, rhs_dem);
  const auto [simplified_rhs_num, simplified_lhs_dem] =
      SimplifiedConstRational(rhs_num, lhs_dem);
  return ConstRational{simplified_lhs_num * simplified_rhs_num,
                       simplified_lhs_dem * simplified_rhs_dem};
}

template <template <typename> class Op>
struct FoldOperandTrait;

template <>
struct FoldOperandTrait<Add> {
  using const_value_type = std::int64_t;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    if (dim_expr.Has<Negative<DimExpr>>()) {
      const auto& operand = dim_expr.Get<Negative<DimExpr>>()->data;
      return operand.Has<std::int64_t>();
    }
    return false;
  }

  static const_value_type MakeUnit() { return 0; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    *value = *value + GetInteger(expr);
  }
  static bool IsUnit(const const_value_type& value) { return value == 0; }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 0;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    (*ret)->emplace_back(value);
  }

  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    if (lhs.Has<Negative<DimExpr>>()) {
      const auto& lhs_operand = lhs.Get<Negative<DimExpr>>()->data;
      return lhs_operand == rhs;
    }
    if (rhs.Has<Negative<DimExpr>>()) {
      const auto& [rhs_operand] = *rhs.Get<Negative<DimExpr>>();
      return rhs_operand == lhs;
    }
    return false;
  }
};

template <>
struct FoldOperandTrait<Max> {
  using const_value_type = std::int64_t;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    if (dim_expr.Has<Negative<DimExpr>>()) {
      const auto& operand = dim_expr.Get<Negative<DimExpr>>()->data;
      return operand.Has<std::int64_t>();
    }
    return false;
  }

  static const_value_type MakeUnit() { return INT64_MIN; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    *value = std::max(*value, GetInteger(expr));
  }
  static bool IsUnit(const const_value_type& value) {
    return value == INT64_MIN;
  }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == INT64_MIN;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    (*ret)->emplace_back(value);
  }

  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    return false;
  }
};

template <>
struct FoldOperandTrait<Min> {
  using const_value_type = std::int64_t;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    if (dim_expr.Has<Negative<DimExpr>>()) {
      const auto& operand = dim_expr.Get<Negative<DimExpr>>()->data;
      return operand.Has<std::int64_t>();
    }
    return false;
  }

  static const_value_type MakeUnit() { return INT64_MAX; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    *value = std::min(*value, GetInteger(expr));
  }
  static bool IsUnit(const const_value_type& value) {
    return value == INT64_MAX;
  }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == INT64_MAX;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    (*ret)->emplace_back(value);
  }

  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    return false;
  }
};

template <>
struct FoldOperandTrait<Mul> {
  using const_value_type = ConstRational;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    return false;
  }

  static const_value_type MakeUnit() { return ConstRational{1, 1}; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    *value = MulConstRational(*value, GetConstRational(expr));
  }
  static bool IsUnit(const const_value_type& value) {
    return value.first == 1 && value.second == 1;
  }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 1;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    const auto& [num, dem] = value;
    (*ret)->emplace_back(num);
    PADDLE_ENFORCE_NE(dem,
                      0,
                      common::errors::InvalidArgument(
                          "The denominator of rational can not be zero."));
  }
  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    if (lhs.Has<Div<DimExpr>>()) {
      const auto& lhs_operand = lhs.Get<Div<DimExpr>>().operands;
      if (lhs_operand->size() != 2 || !lhs_operand->at(0).Has<std::int64_t>()) {
        return false;
      }
      const auto& [lhs_num, lhs_dem] = GetConstRational(lhs_operand->at(0));
      if (lhs_dem != 1) {
        return false;
      }
      if (lhs_operand->at(1) == rhs) {
        return true;
      }
    }
    if (rhs.Has<Div<DimExpr>>()) {
      const auto& rhs_operand = rhs.Get<Div<DimExpr>>().operands;
      if (rhs_operand->size() != 2 || !rhs_operand->at(0).Has<std::int64_t>()) {
        return false;
      }
      const auto& [rhs_num, rhs_dem] = GetConstRational(rhs_operand->at(0));
      if (rhs_dem != 1) {
        return false;
      }
      if (rhs_operand->at(1) == lhs) {
        return true;
      }
    }
    return false;
  }
};

template <>
struct FoldOperandTrait<Div> {
  using const_value_type = ConstRational;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    return false;
  }

  static const_value_type MakeUnit() { return ConstRational{1, 1}; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    *value = MulConstRational(
        *value, GetConstRational(expr));  // The expr are in the denominator.
  }
  static bool IsUnit(const const_value_type& value) {
    return value.first == 1 && value.second == 1;
  }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 1;
  }
  static bool IsZeroDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 0;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    const auto& [num, dem] = value;
    (*ret)->emplace_back(num);
    PADDLE_ENFORCE_NE(dem,
                      0,
                      common::errors::InvalidArgument(
                          "The denominator of rational can not be zero."));
  }
};

template <>
struct FoldOperandTrait<Broadcast> {
  using const_value_type = std::int64_t;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    return false;
  }

  static const_value_type MakeUnit() { return 1; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    PADDLE_ENFORCE_EQ(expr.Has<std::int64_t>(),
                      true,
                      common::errors::InvalidArgument(
                          "Input constant `expr`(second argument) has no "
                          "attribution `int64_T`, please check"));
    std::int64_t expr_value = expr.Get<std::int64_t>();
    if (*value == 1) {
      *value = expr_value;
    } else if (expr_value != 1) {
      PADDLE_ENFORCE_EQ(*value,
                        expr_value,
                        common::errors::InvalidArgument(
                            "The value (%d) should be equal to expr "
                            "(%d) when they are both not 1.",
                            *value,
                            expr_value));
    } else {
      // do nothing.
    }
  }
  static bool IsUnit(const const_value_type& value) { return value == 1; }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 1;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    (*ret)->emplace_back(value);
  }
  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    return false;
  }
};
}  // namespace

IR_API DimExpr SimplifyDimExpr(const DimExpr& dim_expr);

IR_API DimExpr SubstituteDimExpr(
    const DimExpr& dim_expr,
    const std::unordered_map<DimExpr, DimExpr>& pattern_to_replacement);

IR_API int GetDimExprPriority(const DimExpr& dim_expr);

enum class PriorityComparisonStatus {
  HIGHER,  // lhs has a higher priority than rhs
  EQUAL,   // lhs and rhs have equal priority
  LOWER    // lhs has a lower priority than rhs
};
IR_API PriorityComparisonStatus CompareDimExprPriority(const DimExpr& lhs,
                                                       const DimExpr& rhs);

IR_API std::unordered_set<std::string> CollectDimExprSymbols(
    const DimExpr& dim_expr);

}  // namespace symbol
