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

#include "paddle/fluid/eager/api/generated/eager_generated/backwards/identity_node.h"

#include "glog/logging.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/platform/device_context.h"

namespace egr {

paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
GradNodeIdentity::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>&
        grads,  // NOLINT
    bool create_graph,
    bool is_new_grad) {
  // 1. Check Output Size
  VLOG(6) << "grad size is: " << grads.size();
  PADDLE_ENFORCE(
      ((grads.size() == 1) && (grads[0].size() == 1)),
      common::errors::Fatal(
          "IdentityGradNode takes exactly 1 grad tensor."
          "However received: %d",
          "This indicates an issue with Eager Dygraph Backward logic",
          grads.size()));
  paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize> outs;

  // 2. Create needed out parttern
  paddle::Tensor out;
  // Apply Gradient Hooks
  if (GradientHooksRegistered()) {
    // TODO(jiabin): Shall we apply hook slot by slot here or accept
    // vector<vector<phi::tensor>> to apply all hooks?
    paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
        hooked_grads = ApplyGradientHooks(grads);
    out = hooked_grads[0][0];
  } else {
    out = grads[0][0];
  }

  return {{out}};
}

}  // namespace egr
