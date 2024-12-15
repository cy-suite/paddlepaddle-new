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

#include "paddle/cinn/optim/extern_call_process_pass.h"
#include "paddle/cinn/ir/utils/ir_compare.h"

namespace cinn {
namespace optim {

namespace {

void ProcessMultiOutputStore(BlockRef block) {
  const auto& stmts = block->stmts();
  std::vector<StmtRef> new_stmts;

  for (const auto& stmt : stmts) {
    if (stmt.isa<ir::Store>()) {
      auto* store_op = stmt.as<ir::Store>();
      auto* call = store_op->value.As<ir::Call>();
      if (call && call->is_extern_call() && !call->write_args.empty()) {
        new_stmts.emplace_back(store_op->value);
      } else {
        new_stmts.emplace_back(stmt);
      }
    } else {
      new_stmts.emplace_back(stmt);
    }
  }

  block->set_stmts(new_stmts);
}

void RemoveTupleGetStatements(BlockRef block) {
  const auto& stmts = block->stmts();
  std::vector<StmtRef> new_stmts;

  for (const auto& stmt : stmts) {
    if (stmt.isa<ir::Call>()) {
      auto* call = stmt.as<ir::Call>();
      if (call && call->is_extern_call() && call->is_tuple_get()) {
        continue;
      }
    }
    new_stmts.emplace_back(stmt);
  }

  block->set_stmts(new_stmts);
}

}  // namespace

LogicalResult ExternCallMultiOutputShallowStorePass::Run(ir::stmt::BlockRef block) {
  ProcessMultiOutputStore(block);
  return LogicalResult::success();
}

LogicalResult ExternCallRemoveTupleGetStatementsPass::Run(ir::stmt::BlockRef block) {
  RemoveTupleGetStatements(block);
  return LogicalResult::success();
}

std::unique_ptr<BlockPass> CreateExternCallMultiOutputShallowStorePass() {
  return std::make_unique<ExternCallMultiOutputShallowStorePass>();
}

std::unique_ptr<BlockPass> CreateExternCallRemoveTupleGetStatementsPass() {
  return std::make_unique<ExternCallRemoveTupleGetStatementsPass>();
}

}  // namespace optim
}  // namespace cinn
