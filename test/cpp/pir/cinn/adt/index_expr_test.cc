// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/op/ir_operators.h"

namespace cinn {
namespace common {
TEST(IndexExpr, IndexExpr_0) {
  ir::IndexExpr a(14);
  ir::IndexExpr b(7);
  Expr d(6);
  ir::Expr c0 = a + b;
  ir::Expr c1 = a - b;
  ir::Expr c2 = a * b;
  ir::Expr c3 = a / b;
  ir::Expr c4 = a % b;

  ir::Expr c5 = a / d.as_index();
  ir::Expr c6 = a % d.as_index();

  EXPECT_EQ(c0, Expr(21));
  EXPECT_EQ(c1, Expr(7));
  EXPECT_EQ(c2, Expr(98));
  EXPECT_EQ(c3, Expr(2));
  EXPECT_EQ(c4, Expr(0));
  EXPECT_EQ(c5, Expr(2));
  EXPECT_EQ(c6, Expr(2));
}

TEST(IndexExpr, IndexExpr_2) {
  auto S4 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S4");
  auto S5 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S5");
  auto S6 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S6");
  auto S7 =
      ir::Var(ir::Expr(static_cast<int64_t>(1)), ir::Expr(INT32_MAX), "S7");

  cas_intervals_t divisible_var_intervals = {
      {"S4", CasInterval(S4->lower_bound, S4->upper_bound)},
      {"S5", CasInterval(S5->lower_bound, S5->upper_bound)},
      {"S6", CasInterval(S6->lower_bound, S6->upper_bound)},
      {"S7", CasInterval(S7->lower_bound, S7->upper_bound)}};
  SymbolicExprAnalyzer divisible_analyzer{divisible_var_intervals};

  ir::Expr q1 = S4;
  ir::Expr q2 = S4;
  ir::Expr q3 = S4 + S5;
  ir::Expr q4 = S5 + S4;
  ir::Expr q5 = S4 * 2 + S5 / 4;
  ir::Expr q6 = S5 / 4 + S4 * 2;

  ir::Expr q7 = S4 + S5 + S6;
  ir::Expr q8 = S5 + (S4 + S6);
  ir::Expr q7 = (S7 + S5) + (S4 + S6);
  ir::Expr q8 = (S4 + S5) + (S6 + S7);
  ir::Expr q9 = S4 + (S5 + S7 / 4 + S6 * 2);
  ir::Expr q10 = S5 + (S4 + S6 * 2 + S7 / 4);

  EXPECT_EQ(q1.as_index().Normalize(), q2.as_index().Normalize());
  EXPECT_EQ(q3.as_index().Normalize(), q4.as_index().Normalize());
  EXPECT_EQ(q5.as_index().Normalize(), q6.as_index().Normalize());
  EXPECT_EQ(q7.as_index().Normalize(), q8.as_index().Normalize());
  EXPECT_EQ(q9.as_index().Normalize(), q10.as_index().Normalize());
}
}  // namespace common
}  // namespace cinn
