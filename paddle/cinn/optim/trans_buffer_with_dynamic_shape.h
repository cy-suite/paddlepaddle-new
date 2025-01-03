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
#include <string>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/pass/pass.h"

namespace cinn {
namespace optim {

class TransBufferWithDynamicShapePass : public FuncPass {
 public:
  TransBufferWithDynamicShapePass()
      : FuncPass("trans_buffer_with_dynamic_shape") {}
  LogicalResult Run(ir::LoweredFunc func) override;
};

/**
 * Transforms buffers' dynamic shapes to constant shapes and perform shared
 * memory usage checks.
 *
 * This pass is applicable in scenarios where tensor buffers have dynamic
 * shapes, especially in GPU computations. It's crucial for ensuring correct
 * memory allocation and preventing buffer overflows in shared memory usage on
 * GPUs.
 *
 * When applied, this pass will analyze tensor buffers and their shapes,
 * calculating the required memory size. For GPU local memory, it will attempt
 * to determine upper bounds for dynamic shapes. For GPU shared memory, it will
 * calculate the total shared memory usage and verify it against hardware
 * limits.
 *
 * Risks and limitations:
 * - Currently only checks shared memory usage against hardware limits for
 * NVIDIA GPUs and Hygon DCU.
 *
 * TODO:
 * - Extend support for ARM and X86 architectures.
 * - Improve shape simplification for scenarios where runtime-dynamic shapes
 *   have highly complex expressions.
 * - Incorporate additional performance metrics and runtime validation in debug
 * builds.
 *
 * Examples:
 * 1. Dynamic Shape Simplification (Local Memory):
 *    Input IR:
 *      `Tensor A[128, dynamic_dim];`
 *    Output IR:
 *      `Tensor A[128, upper_bound_dynamic_dim];`
 *    Explanation:
 *      The pass calculates an upper bound for `dynamic_dim` and replaces the
 *      dynamic dimension with a constant value for memory allocation.
 *
 * 2. Shared Memory Size Check (Shared Memory):
 *    Input IR:
 *      `Tensor B[64, 256]; `
 *    Output IR:
 *      `Shared memory check: Ensure total size <= max_shared_memory_per_block`
 *    Explanation:
 *      The pass verifies that the total size of `B` does not exceed the shared
 *      memory limit for the target GPU architecture. If it does, an error is
 * raised.
 *
 * Counter-Examples:
 * 1. Static Shapes:
 *    Input IR:
 *      `Tensor C[64, 128];`
 *    Explanation:
 *      If all dimensions are static constants, the pass does not perform any
 * transformations.
 *
 * 2. Unsupported Architectures (e.g., ARM):
 *    Input IR:
 *      `Tensor D[512, dynamic_dim];`
 *    Explanation:
 *      The pass does not run on unsupported architectures like ARM.
 *
 */
std::unique_ptr<FuncPass> CreateTransBufferWithDynamicShapePass();

}  // namespace optim
}  // namespace cinn
