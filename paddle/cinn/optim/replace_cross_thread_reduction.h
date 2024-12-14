// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

/**
 * This file implements the strategy to remove the unnecessary nested block.
 */
#pragma once
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * Replace cross thread reduction to external call.
 */
void ReplaceCrossThreadReduction(ir::LoweredFunc fn);
/**
 * A pass that optimizes cross-thread reduction operations on GPU by replacing them with more efficient implementations.
 *
 * [Detailed application scenario]
 * This pass is applicable in scenarios where multiple GPU threads need to perform reduction operations (like sum, max, min)
 * across thread boundaries. These scenarios are common in deep learning workloads, particularly in operations like:
 * - Computing sum/mean across feature dimensions
 * - Global pooling operations
 * - Softmax normalization
 * - Gradient aggregation in distributed training
 *
 * [IR modifications]
 * When applied, this pass will:
 * 1. Identify reduction operations in GPU-bound loops
 * 2. Replace the original reduction operation with an optimized external function call
 * 3. Create shared memory buffers for intermediate results
 * 4. Transform the reduction pattern based on the selected method (None/Warp/Block/Discrete)
 * 
 * The pass supports the following reduction operations:
 * - Addition (ir::Add)
 * - Multiplication (ir::Mul)
 * - Maximum (ir::Max)
 * - Minimum (ir::Min)
 * - Logical AND (ir::And)
 * - Logical OR (ir::Or)
 *
 * [Performance impact]
 * Performance impact: This pass addresses several performance bottlenecks:
 * - Reduces thread synchronization overhead
 * - Optimizes memory access patterns through shared memory usage
 * - Enables efficient parallel reduction at different granularities (warp/block)
 * - Minimizes global memory access during reduction operations
 *
 * [Risks and limitations]
 * Risks and limitations:
 * - Limited to reduction operations with thread-bound loops
 * - Maximum thread block size constraint of 1024
 * - Requires sufficient shared memory resources
 * - May increase register pressure due to shared memory usage
 *
 * TODO:
 * - Support more reduction operations (e.g., custom reduction functions)
 * - Add dynamic selection of reduction methods based on input size
 * - Optimize shared memory allocation for better bank conflicts avoidance
 * - Add support for multi-warp reductions within a block
 *
 * [Examples]
 * 1. Sum Reduction:
 *    Input IR:
 *      for (i = 0; i < 1024; i++) {
 *        if (i < n) {
 *          sum += data[i];
 *        }
 *      }
 *
 *    Output IR:
 *      buffer shm32_float_reduce[32];
 *      sum = __cinn_cuda_reduce_sum(data, shm32_float_reduce, false);
 *
 * 2. Max Reduction with Warp:
 *    Input IR:
 *      for (i = 0; i < 32; i++) {
 *        max_val = max(max_val, data[i]);
 *      }
 *
 *    Output IR:
 *      buffer shm32_float_reduce[32];
 *      max_val = __cinn_cuda_reduce_max(data, shm32_float_reduce, true);
 *
 * [Counter-examples]
 * 1. Non-reduction loop:
 *    for (i = 0; i < n; i++) {
 *      output[i] = input[i] + 1;
 *    }
 *    Reason: No reduction operation present, pass should not be applied
 *
 * 2. Large reduction with thread count > 1024:
 *    for (i = 0; i < 2048; i++) {
 *      sum += data[i];
 *    }
 *    Reason: Exceeds maximum supported thread block size
 */
}  // namespace optim
}  // namespace cinn
