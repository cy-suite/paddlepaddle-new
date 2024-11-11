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
#include <unordered_set>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_factory.h"

namespace paddle {

namespace framework {

bool OpSupportPrecision(const std::string& op_type,
                        phi::Backend backend,
                        phi::DataType precision,
                        const std::unordered_set<std::string>& blacklist = {},
                        const std::unordered_set<std::string>& whitelist = {});

bool KernelSupportPrecision(
    const std::string& op_type,
    phi::Backend backend,
    phi::DataType precision,
    phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT);

bool PhiKernelSupportPrecision(const std::string& op_type,
                               phi::Backend backend,
                               phi::DataType data_type,
                               phi::DataLayout layout);

inline phi::Backend ConvertPlaceToBackend(const phi::Place& place) {
  switch (place.GetType()) {
    case phi::AllocationType::CPU:
      return phi::Backend::CPU;
    case phi::AllocationType::GPU:
      return phi::Backend::GPU;
    case phi::AllocationType::XPU:
      return phi::Backend::XPU;
    case phi::AllocationType::CUSTOM:
      return phi::Backend::CUSTOM;
    default:
      return phi::Backend::UNDEFINED;
  }
  return phi::Backend::UNDEFINED;
}

}  // namespace framework
}  // namespace paddle
