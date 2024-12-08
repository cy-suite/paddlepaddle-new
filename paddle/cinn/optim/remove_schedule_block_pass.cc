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
#include "paddle/cinn/optim/remove_schedule_block_pass.h"
#include "paddle/cinn/ir/stmt.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {
using ir::stmt::StmtRef;

bool cinn::optim::RemoveScheduleBlockPass::RunOnBlock(
    ir::stmt::BlockRef block) {
  const std::vector<StmtRef>& stmts = block->stmts();
  if (stmts.size() < 2) return false;
  for (auto& stmt : stmts) {
    if (!stmt.isa<ir::ScheduleBlockRealize>()) {
      continue;
    }
    auto& node = stmt.as<ir::ScheduleBlockRealize>();
    auto& iter_values = node.iter_values;
    auto* schedule_block = node.schedule_block.As<ir::ScheduleBlock>();
    PADDLE_ENFORCE_NOT_NULL(
        schedule_block,
        ::common::errors::InvalidArgument(
            "The schedule block could not be cast to ir::ScheduleBlock. Please "
            "check the schedule block type."));
    auto& iter_vars = schedule_block->iter_vars;
    Expr body = schedule_block->body;
    for (int i = 0; i < iter_vars.size(); i++) {
      optim::ReplaceVarWithExpr(&body, iter_vars[i], iter_values[i]);
    }
    stmt = *(body->as<StmtRef>());
    // TODO(LittleHeroZZZX): 用 body 替换旧 stmt
  }
}
}  // namespace optim
}  // namespace cinn
