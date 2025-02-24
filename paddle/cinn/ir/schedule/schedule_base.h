// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/stmt.h"
#include "paddle/cinn/utils/error.h"
#include "paddle/cinn/utils/random_engine.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

PD_DECLARE_int32(cinn_error_message_level);

namespace cinn {
namespace ir {

struct BroadcastInfo {
  std::vector<int64_t> broadcast_axes;
  std::vector<int64_t> output_shape;
  std::vector<symbol::DimExpr> output_dim_expr;

  bool with_constrain{false};
  bool first_broadcast{false};
  bool full_broadcast{false};
  std::string op_name;

  bool split_first{false};
  std::vector<std::pair<int, std::vector<int>>> split_info;
};

/**
 * A struct representing a module that contains BlockRefs. This struct is only
 * used in Schedule process.
 */
class ScheduleModule {
 public:
  ScheduleModule() = default;
  ScheduleModule(const ScheduleModule& sched_module) = default;
  ScheduleModule(ScheduleModule&& sched_module) = default;

  ScheduleModule& operator=(const ScheduleModule& sched_module) = default;

  explicit ScheduleModule(const std::vector<Expr>& blocks) : blocks_(blocks) {}
  explicit ScheduleModule(std::vector<Expr>&& blocks)
      : blocks_(std::move(blocks)) {}

  //! Get all the Block in this ScheduleModule.
  std::vector<Expr> GetBlocks() { return blocks_; }

  std::vector<Expr> GetBlocks() const { return blocks_; }

  void SetBlocks(const std::vector<Expr>& blocks) { blocks_ = blocks; }

 private:
  //! Exprs stored in ScheduleModule. Each one is an AST, representing a
  //! computation kernel.
  std::vector<Expr> blocks_;
};

/**
 * Define the interface for scheduling primitives,
 * with subclasses DyScheduleImpl and StScheduleImpl.
 */
class ScheduleBase {
 public:
  ScheduleBase() = delete;
  explicit ScheduleBase(const ScheduleModule& sched_module,
                        bool debug_flag = false,
                        utils::ErrorMessageLevel err_msg_level =
                            utils::ErrorMessageLevel::kGeneral)
      : sched_module_(sched_module), debug_flag_(debug_flag) {
    err_msg_level_ = static_cast<utils::ErrorMessageLevel>(
        FLAGS_cinn_error_message_level || static_cast<int>(err_msg_level));
  }
  explicit ScheduleBase(ScheduleModule&& sched_module)
      : sched_module_(std::move(sched_module)) {}

  static std::unique_ptr<ScheduleBase> Make(
      const ScheduleModule& sched_module,
      bool debug_flag = false,
      utils::ErrorMessageLevel err_msg_level =
          utils::ErrorMessageLevel::kGeneral,
      bool is_dynamic = false);

  static std::unique_ptr<ScheduleBase> Make(ScheduleModule&& sched_module,
                                            bool is_dynamic = false);

  void SetDebugFlag(bool debug_flag) { debug_flag_ = debug_flag; }

  const ScheduleModule& GetModule() const { return sched_module_; }

  void SetBlocks(const std::vector<Expr>& blocks) {
    sched_module_.SetBlocks(blocks);
  }

  virtual void MergeBlocks() = 0;
  virtual bool HasSchedStmt(const std::string& sched_name) const = 0;
  virtual std::vector<Expr> GetLoops(const Expr& target_sched) const = 0;
  virtual std::vector<Expr> GetLoops(
      const std::string& target_sched_name) const = 0;
  virtual std::vector<Expr> GetAllSchedStmts() const = 0;
  virtual std::vector<Expr> GetChildSchedStmts(const Expr& stmt) const = 0;
  virtual Expr GetSchedStmt(const std::string& sched_name) const = 0;

  virtual std::vector<Expr> Split(const Expr& loop,
                                  const std::vector<int>& factors) = 0;
  virtual std::vector<Expr> Split(const Expr& loop,
                                  const std::vector<Expr>& factors) = 0;
  virtual std::vector<Expr> SamplePerfectTile(
      utils::LinearRandomEngine::StateType* rand_seed,
      const Expr& loop,
      int n,
      int max_innermost_factor) = 0;
  virtual Expr Fuse(const std::vector<Expr>& loops) = 0;
  virtual Expr Fuse(const std::string& sched_name,
                    const std::vector<int>& loops_index) = 0;
  virtual Expr Fuse(const Expr& block, const std::vector<int>& loops_index) = 0;
  virtual void ComputeAt(const Expr& block,
                         const Expr& loop,
                         bool keep_unit_loops) = 0;
  virtual void SimpleComputeAt(const Expr& block, const Expr& loop) = 0;
  virtual void ReverseComputeAt(const Expr& block,
                                const Expr& loop,
                                bool keep_unit_loops) = 0;
  virtual Expr GetRootSchedStmt(const Expr& expr) const = 0;
  virtual Expr CacheRead(const Expr& block,
                         int read_buffer_index,
                         const std::string& memory_type) = 0;
  virtual Expr CacheWrite(const Expr& block,
                          int write_buffer_index,
                          const std::string& memory_type) = 0;
  virtual void SyncThreads(const Expr& ir_node, bool after_node = true) = 0;
  virtual void SetBuffer(Expr& block,  // NOLINT
                         const std::string& memory_type,
                         bool fixed = false) = 0;
  virtual Expr Reorder(const std::vector<Expr>& loops) = 0;
  virtual Expr Reorder(const std::string& sched_name,
                       const std::vector<int>& loops_index) = 0;
  virtual Expr Reorder(const Expr& block,
                       const std::vector<int>& loops_index) = 0;
  virtual DeviceAPI GetDeviceAPI() const = 0;
  virtual void MutateForType(const Expr& loop,
                             ForType for_type,
                             int factor = -1) = 0;
  virtual void Parallel(const Expr& loop) = 0;
  virtual void Vectorize(const Expr& loop, int factor) = 0;
  virtual void Unroll(const Expr& loop) = 0;
  virtual void ComputeInline(const Expr& schedule_block) = 0;
  virtual void ReverseComputeInline(const Expr& schedule_block) = 0;
  virtual void Bind(const Expr& loop, const std::string& thread_axis) = 0;
  virtual Expr Rfactor(const Expr& rf_loop, int rf_axis) = 0;
  virtual Expr FactorizeReduction(const Expr& rf_loop,
                                  int rf_axis,
                                  bool with_write_back_block_init = true) = 0;
  virtual Expr AddUnitLoop(const Expr& block) const = 0;
  virtual void Annotate(const Expr& block,
                        const std::string& key,
                        const attr_t& value) = 0;
  virtual void Unannotate(Expr& block, const std::string& key) = 0;  // NOLINT
  virtual void FlattenLoops(const std::vector<Expr>& loops,
                            const bool force_flat = false) = 0;
  virtual void CopyTransformAndLoopInfo(const Expr& block,
                                        const Expr& block_target) = 0;
  virtual void CopyTransformAndLoopInfo(
      const std::string& sched_name, const std::string& block_target_name) = 0;
  virtual Expr SampleCategorical(
      utils::LinearRandomEngine::StateType* rand_seed,
      const std::vector<int>& candidates,
      const std::vector<float>& probs) = 0;

 protected:
  void Replace(const Expr& src_stmt, const Expr& tgt_stmt);

  ScheduleModule sched_module_;
  bool debug_flag_{false};
  utils::ErrorMessageLevel err_msg_level_ = utils::ErrorMessageLevel::kGeneral;
};

}  // namespace ir
}  // namespace cinn
