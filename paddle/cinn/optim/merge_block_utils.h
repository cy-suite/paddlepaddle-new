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

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {

struct ForTreeNode {
  const ir::For* val;
  std::vector<ForTreeNode> children;
};

using ForEqualFunc =
    std::function<bool(const ForTreeNode&, const ForTreeNode&)>;

/**
 * Determines if two blocks of code with nested for-loops have identical loop
 extents and can be merged.

 * This pass is applicable in scenarios where there are multiple code blocks
 with nested for-loops,
 * and we need to determine if these blocks can be consolidated to simplify the
 code structure.

 * When applied, this pass will not directly modify the IR but serves as a
 prerequisite check
 * to ensure that loop extents match. If they do, a separate merging process can
 be safely conducted
 * to combine the blocks into a single block with shared loop structures.

 * Performance impact: This pass itself does not directly impact performance but
 enables further
 * optimizations by identifying mergeable loop structures, which can reduce code
 size and potentially
 * improve cache efficiency by consolidating similar data processing tasks.

 * Examples:
 * 1. Simple identical loops:
 *    Input IR:
 *      block(var_B)
 *        for(i, 0, 10)
 *          for(j, 0, 10)
 *            B[i,j] = A[i,j]
 *
 *      block(var_C)
 *        for(i, 0, 10)
 *          for(j, 0, 10)
 *            C[i,j] = A[i,j]
 *    Output IR:
 *      Can be merged since loop extents are identical.
 *
 * 2. Different loop extents:
 *    Input IR:
 *      block(var_B)
 *        for(i, 0, 10)
 *          for(j, 0, 10)
 *            B[i,j] = A[i,j]
 *
 *      block(var_C)
 *        for(i, 0, 3)
 *          for(j, 0, 4)
 *            C[i,j] = A[i,j]
 *    Output IR:
 *      Cannot be merged due to differing loop extents.
 */
bool CanMergeBlocks(const ir::For* first,
                    const ir::For* second,
                    const ForEqualFunc& IsEqual);

/**
 * \brief Move schedule block src after schedule block dst, without checking
 * dependency.
 * @param src The src ScheduleBlock.
 * @param dst The dst ScheduleBlock.
 * @param root The root Expr.
 */
void MoveScheduleBlock(const ir::ScheduleBlock* src,
                       const ir::ScheduleBlock* dst,
                       ir::Expr* root);

/**
 * \brief Fuse two loop, `src` -> `dst`, remains ScheduleBlock order. Return
 * `nullptr` if not supported. Currently support loop struct having exactly the
 * same extents without IfThenElse.
 * @param src The first loop.
 * @param dst The second loop.
 * @return Return fused loop or `nullptr` if not supported.
 */
/**
 * Example 1: LoopFusion(loop_src, loop_dst)
 * loop(loop_dst)
 *   for(i, 0, 10)
 *     for(j, 0, 10)
 *        B[i,j] = A[i,j]
 *
 * loop(loop_src)
 *   for(i, 0, 10)
 *     for(j, 0, 10)
 *        C[i,j] = A[i,j]
 * =>
 * Return value:
 * loop(loop_fused)
 *   for(i, 0, 10)
 *     for(j, 0, 10)
 *        B[i,j] = A[i,j]
 *        C[i,j] = A[i,j]
 *
 * Example 2: LoopFusion(loop_src, loop_dst)
 * loop(loop_dst)
 *   for(i, 0, 10)
 *     for(j, 0, 10)
 *        B[i,j] = A[i,j]
 *
 * loop(loop_src)
 *   for(i, 0, 3)
 *     for(j, 0, 4)
 *        C[i,j] = A[i,j]
 * =>
 * Return value:
 * `nullptr`
 */
ir::Expr LoopFusion(const ir::For* src, const ir::For* dst);

}  // namespace optim
}  // namespace cinn
