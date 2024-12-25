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

namespace cinn {
namespace optim {

/** 
 * CudaTransBufferWithDynamicShape optimizes buffer size handling for dynamic shapes on CUDA-based devices.
 *
 * This pass is applicable when working with buffers that have dynamic shapes, particularly in CUDA kernels. 
 * It is useful in scenarios where buffers are allocated on GPU (either local or shared memory) and their sizes 
 * depend on runtime parameters or conditions, such as when the shape is not constant and needs to be dynamically 
 * evaluated and adjusted.
 *
 * When applied, this pass calculates the buffer size based on dynamic shape expressions and replaces them 
 * with upper bounds for better memory usage estimation. It also checks that the shared memory size used by 
 * the kernel does not exceed the maximum allowable size for the current device architecture. 
 * The pass ensures that the buffer's shape is simplified to constant expressions where possible.
 *
 * Performance impact: This pass improves memory management by ensuring that GPU buffers are allocated with 
 * accurate sizes based on dynamic shape analysis. It also prevents potential issues related to exceeding 
 * the available shared memory on the device.
 *
 * Risks and limitations:
 * - The pass assumes that dynamic shapes can be upper-bounded to constant expressions, which may not always 
 *   be accurate in all cases.
 * - The current implementation is limited to CUDA and HygonDCU architectures, and other architectures (e.g., ARM) 
 *   may not be fully supported.
 * TODO:
 * - Extend support for additional GPU architectures.
 *
 * Examples:
 * 1. Buffer size calculation for a tensor with dynamic shape:
 *    Input IR:
 *      `Tensor A[128, dynamic_dim];`  
 *    Output IR:
 *      `Tensor A[128, dynamic_dim]; 
 *    Explanation: The dynamic dimension `dynamic_dim` will be replaced with an upper bound derived from the 
 *    analysis, simplifying the dynamic expression for memory estimation.
 *
 * 2. Shared memory size check for a kernel using GPU shared memory:
 *    Input IR:
 *      `Tensor B[64, 256];` 
 *    Output IR:
 *      `Shared memory check: Ensures total size of Tensor B is within max_shared_memory_per_block (e.g., 48KB)`
 *    Explanation: The pass ensures that the total memory used by the tensor `B` does not exceed the available 
 *    shared memory on the GPU, ensuring the kernel will run without memory overflows.
 *
 * Counter-examples:
 * 1. Static shapes:
 *    Input IR:
 *      `Tensor C[64, 128];` 
 *    Explanation: The pass should not be applied here since both dimensions are constants, and no dynamic shape 
 *    handling is required.
 *
 * 2. Unsupported architecture (ARM):
 *    Input IR:
 *      `Tensor D[512, dynamic_dim];`  
 *    Explanation: The pass should not be applied if the target architecture is ARM, as it currently only supports 
 *    CUDA and HygonDCU architectures.
 */
void CudaTransBufferWithDynamicShape(ir::Expr* expr);

}  // namespace optim
}  // namespace cinn
