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

#include <string>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/stmt.h"

namespace cinn {
namespace optim {

enum PassKind { PK_FUNC, PK_BLOCK, PK_STMT, PK_EXPR };

template <typename IRScopeRefT>
class Pass {
 public:
  explicit Pass(PassKind kind, const std::string& name)
      : kind_(kind), name_(name) {}
  virtual ~Pass() {}

  virtual bool Run(IRScopeRefT scope) = 0;

  PassKind kind() const { return kind_; }
  const std::string& name() const { return name_; }

 private:
  PassKind kind_;
  std::string name_;
};

class FuncPass : public Pass<ir::LoweredFunc> {
 public:
  explicit FuncPass(const std::string& name) : Pass(PK_FUNC, name) {}

  virtual bool Run(ir::LoweredFunc f) = 0;
};

class BlockPass : public Pass<ir::stmt::BlockRef> {
 public:
  explicit BlockPass(const std::string& name) : Pass(PK_BLOCK, name) {}
  virtual bool Run(ir::stmt::BlockRef block) = 0;
};

class StmtPass : public Pass<ir::stmt::StmtRef> {
 public:
  explicit StmtPass(const std::string& name) : Pass(PK_STMT, name) {}
  virtual bool Run(ir::stmt::StmtRef stmt) = 0;
};

class ExprPass : public Pass<ir::Expr> {
 public:
  explicit ExprPass(const std::string& name) : Pass(PK_STMT, name) {}
  virtual bool Run(ir::Expr expr) = 0;
};

}  // namespace optim
}  // namespace cinn
