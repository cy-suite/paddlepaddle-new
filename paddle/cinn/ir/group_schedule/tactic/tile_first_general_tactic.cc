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

#include "paddle/cinn/ir/group_schedule/tactic/tile_first_general_tactic.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"

namespace cinn {
namespace ir {

using cinn::ir::analyzer::IsReductionSBlock;

class TileFirstGeneralTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context, ir::IRSchedule* sch) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;

  std::string TacticName() const override { return "TileFirstGeneralTactic"; }

 private:
  void MergeFlattenAxis(ir::IRSchedule* sch, const std::string& block_id);
  void MergeReduceAxis(ir::IRSchedule* sch, const std::string& block_id);
  void VariableTypeAssignment(ir::IRSchedule* sch, const std::string& block_id);
  void SetReduceType(ir::IRSchedule* sch, const std::string& block_id);

 private:
  ScheduleContext* context_;
  bool can_apply_;
  std::vector<int32_t> vec_flatten_axis_;
  std::vector<int32_t> vec_reduce_axis_;
  std::unordered_map<std::string, std::string> map_rf_block_;
};

void TileFirstGeneralTactic::Init(ScheduleContext* context,
                                  ir::IRSchedule* sch) {
  context_ = context;
  can_apply_ = false;

  // Check whether this group has been tiled by previous tactic.
  ir::Expr module_root = sch->GetModule().GetExprs().front();
  ir::Expr root_block = ir::analyzer::GetRootSBlock(module_root);
  auto* root_node = root_block.As<ir::ScheduleBlockRealize>()
                        ->schedule_block.As<ir::ScheduleBlock>();
  if (root_node->attrs.count(kTileMethod) > 0) {
    return;
  }
  can_apply_ = true;
  root_node->attrs[kTileMethod] = TacticName();

  // reduce axes have been re-ordered to the last
  vec_flatten_axis_.clear();
  vec_reduce_axis_.clear();
  int data_rank = context_->config.base_info->loop_ranges.size();
  int32_t reduce_start_idx =
      data_rank - context_->config.base_info->reduce_axis.size();
  for (int32_t i = 0; i < data_rank; ++i) {
    if (i >= reduce_start_idx) {
      vec_reduce_axis_.push_back(i);
    } else {
      vec_flatten_axis_.push_back(i);
    }
  }
  map_rf_block_.clear();
}

void TileFirstGeneralTactic::Apply(ir::IRSchedule* sch,
                                   const std::string& block_id) {
  if (!can_apply_) return;
  if (ir::IsReduceInitTensorName(block_id)) return;

  VLOG(4) << "Using ApplyContinuousDataTile";
  const auto sp_thread = context_->config.tile_config.warp_num * 32 /
                         context_->config.tile_config.tree_reduce_num;
  const auto sp_loop = context_->config.tile_config.spatial_inner_num;
  const auto rd_thread = context_->config.tile_config.tree_reduce_num;
  const auto rd_block = context_->config.tile_config.grid_reduce_num;
  VLOG(4) << "ApplyContinuousDataTile sp_thread=" << sp_thread;
  VLOG(4) << "ApplyContinuousDataTile sp_loop=" << sp_loop;
  VLOG(4) << "ApplyContinuousDataTile rd_thread=" << rd_thread;
  VLOG(4) << "ApplyContinuousDataTile rd_block=" << rd_block;
  VLOG(4) << "ApplyContinuousDataTile vec_flatten_axis: "
          << utils::Join(vec_flatten_axis_, ", ");
  VLOG(4) << "ApplyContinuousDataTile vec_reduce_axis: "
          << utils::Join(vec_reduce_axis_, ", ");

  // Merge reduce axes
  MergeReduceAxis(sch, block_id);
  VLOG(4) << "After MergeReduceAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Merge spatial axes
  MergeFlattenAxis(sch, block_id);
  VLOG(4) << "After MergeFlattenAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Split spatial axes -> [sp_block, sp_loop, sp_thread]
  int current_reduce_axis = 0;
  if (vec_flatten_axis_.size() > 0) {
    auto loops = sch->GetLoops(block_id);
    if (sp_loop > 1 && sp_thread > 1) {
      // [S, R] => [S(-1), S(inner_loop), S(thread), R]
      sch->Split(loops[0], {-1, sp_loop, sp_thread});
      current_reduce_axis = 3;
    } else if (sp_loop > 1 || sp_thread > 1) {
      // [S, R] => [S(-1), S(thread), R]
      sch->Split(loops[0], {-1, sp_loop > 1 ? sp_loop : sp_thread});
      current_reduce_axis = 2;
    } else {
      // [S, R] => [S, R]
      current_reduce_axis = 1;
    }
  }
  VLOG(4) << "After SplitSptial on block: [" << block_id << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Split reduce axes -> [rd_loop, rd_block, rd_thread]
  std::string global_rf_block;
  if (vec_reduce_axis_.size() > 0) {
    auto loops = sch->GetLoops(block_id);
    sch->Split(loops[current_reduce_axis], {-1, rd_block * rd_thread});

    loops = sch->GetLoops(block_id);
    sch->Reorder({loops[current_reduce_axis + 1], loops[current_reduce_axis]});

    loops = sch->GetLoops(block_id);
    if (IsReductionSBlock(sch->GetBlock(block_id)) &&
        ir::GetLoopExtent(loops[current_reduce_axis]) != 1) {
      ir::Expr rf_tensor =
          sch->FactorizeReduction(loops[current_reduce_axis],
                                  /* rf_axis = */ 0,
                                  /* with_write_back_block_init = */ false);
      map_rf_block_[block_id] = rf_tensor.as_tensor_ref()->name;
    }

    if (rd_block > 1) {
      loops = sch->GetLoops(block_id);
      sch->Split(loops[current_reduce_axis], {rd_block, rd_thread});

      if (IsReductionSBlock(sch->GetBlock(block_id))) {
        loops = sch->GetLoops(map_rf_block_[block_id]);
        sch->Split(loops[current_reduce_axis], {rd_block, rd_thread});

        loops = sch->GetLoops(block_id);
        ir::Expr rf_tensor =
            sch->FactorizeReduction(loops[current_reduce_axis],
                                    /* rf_axis = */ 0,
                                    /* with_write_back_block_init = */ false);
        global_rf_block = rf_tensor.as_tensor_ref()->name;
        rf_tensor.as_tensor_ref()->WithBuffer("global", "_" + global_rf_block);
      }
    }
  }
  VLOG(4) << "After SplitReduce on block: [" << block_id << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Bind CUDA info
  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    std::string sp_axis_type = "threadIdx.y";
    std::string rd_axis_type = "threadIdx.x";
    sch->Bind(loops[0], "blockIdx.x");
    if (!vec_flatten_axis_.empty() && sp_thread > 1) {
      if (vec_reduce_axis_.empty()) {
        // [S..S] => [S(blockIdx.x), optional(inner_loop), S(threadIdx.x)]
        sch->Bind(loops[current_reduce_axis - 1], rd_axis_type);
      } else {
        // [S..S, R..R] =>
        // [S(blockIdx.x), optional(inner_loop), S(threadIdx.y), R..R]
        sch->Bind(loops[current_reduce_axis - 1], sp_axis_type);
      }
    }
    if (!vec_reduce_axis_.empty() && current_reduce_axis > 0) {
      if (rd_block > 1) {
        sch->Bind(loops[current_reduce_axis], "blockIdx.y");
        if (loops.size() > current_reduce_axis + 1) {
          sch->Bind(loops[current_reduce_axis + 1], rd_axis_type);
        }
      } else {
        sch->Bind(loops[current_reduce_axis], rd_axis_type);
      }
    }
  };
  DoBind(sch->GetLoops(block_id));
  if (map_rf_block_.count(block_id) > 0) {
    DoBind(sch->GetLoops(map_rf_block_[block_id]));
  }
  if (!global_rf_block.empty()) {
    DoBind(sch->GetLoops(global_rf_block));
  }
  VLOG(4) << "After BindCudaInfo on block: [" << block_id << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  VariableTypeAssignment(sch, block_id);
  SetReduceType(sch, block_id);
  return;
}

void TileFirstGeneralTactic::MergeFlattenAxis(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  if (vec_flatten_axis_.size() >= 2) {
    sch->Fuse(block_id, vec_flatten_axis_);
  }
}

void TileFirstGeneralTactic::MergeReduceAxis(ir::IRSchedule* sch,
                                             const std::string& block_id) {
  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  int32_t max_loop_idx = 0;
  for (int32_t idx : vec_reduce_axis_) {
    max_loop_idx = std::max(max_loop_idx, idx);
    PADDLE_ENFORCE_EQ(idx < loops.size() || loops.size() == 1,
                      true,
                      ::common::errors::InvalidArgument(
                          "The reduce axis should meet: axis's idx < "
                          "loops.size() or loops.size() == 1, but received "
                          "idx= %d ,loops.size() = %d",
                          idx,
                          loops.size()));
  }
  if (max_loop_idx < loops.size() && vec_reduce_axis_.size() >= 2) {
    sch->Fuse(block_id, vec_reduce_axis_);
  }
}

void TileFirstGeneralTactic::VariableTypeAssignment(
    ir::IRSchedule* sch, const std::string& block_id) {
  const auto IsOutputTensor = [&](const std::string& tensor_name) -> bool {
    return context_->output_names.count(tensor_name) > 0;
  };
  const auto HasConsumers = [&](const ir::Expr& block) -> bool {
    return !ir::analyzer::GetConsumerSBlocks(block, sch->GetRootBlock(block))
                .empty();
  };

  auto block = sch->GetBlock(block_id);
  if (!IsOutputTensor(block_id) && HasConsumers(block)) {
    sch->SetBuffer(block, "local", false);
  }

  if (map_rf_block_.count(block_id) > 0) {
    auto block = sch->GetBlock(map_rf_block_[block_id]);
    sch->SetBuffer(block, "local", false);
  }
}

void TileFirstGeneralTactic::SetReduceType(ir::IRSchedule* sch,
                                           const std::string& block_id) {
  if (IsReductionSBlock(sch->GetBlock(block_id))) {
    auto block = sch->GetBlock(block_id)
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = context_->config.tile_config.reduce_method;
  }
}

std::unique_ptr<ScheduleTactic> CreateTileFirstGeneralTactic() {
  return std::make_unique<TileFirstGeneralTactic>();
}

}  // namespace ir
}  // namespace cinn
