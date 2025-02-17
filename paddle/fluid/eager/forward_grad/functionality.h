// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/phi/api/include/tensor.h"

namespace egr {

/**
 * @brief Enter a new dual level.
 *
 * This function increments the dual level, which is used to manage
 * the nesting of dual tensors in automatic differentiation.
 *
 * @return The current dual level after incrementing.
 */
TEST_API uint64_t enter_dual_level();

/**
 * @brief Exit a dual level.
 *
 * This function decrements the dual level. The dual level should
 * correspond to a previously entered level to maintain proper
 * nesting of dual tensors.
 *
 * @param[in] level The dual level to exit. It should be equal to or
 *                  less than the current level.
 */
TEST_API void exit_dual_level(uint64_t level);

/**
 * @brief Create a dual tensor from primal and tangent components.
 *
 * This function combines a primal tensor and a tangent tensor into
 * a dual tensor at the specified dual level. This is useful in
 * automatic differentiation for tracking derivatives.
 *
 * @param[in] primal The primal component of the dual tensor.
 * @param[in] tangent The tangent component of the dual tensor.
 * @param[in] level The dual level at which this dual tensor is valid.
 * @return A new dual tensor with the specified primal and tangent
 *         components.
 */
TEST_API paddle::Tensor make_dual(const paddle::Tensor& primal,
                                  const paddle::Tensor& tangent,
                                  int64_t level);

/**
 * @brief Unpack a dual tensor into its primal and tangent components.
 *
 * This function extracts the primal and tangent components from a
 * dual tensor at the specified dual level. This allows access to
 * the underlying components for further computation or analysis.
 *
 * @param[in] tensor The dual tensor to be unpacked.
 * @param[in] level The dual level at which this tensor is unpacked.
 * @return A tuple containing the primal and tangent components of
 *         the dual tensor. The first element is the primal, and the
 *         second one is the tangent.
 * @note Ensure that the dual tensor is at the specified level before
 *       calling this function.
 */
TEST_API std::tuple<paddle::Tensor, paddle::Tensor> unpack_dual(
    const paddle::Tensor& tensor, int64_t level);

}  // namespace egr
