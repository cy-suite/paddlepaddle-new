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

#include "paddle/fluid/custom_engine/custom_engine_manager.h"

namespace paddle {
namespace custom_engine {

C_CustomEngineInterface* CustomEngineManager::GetCustomEngineInterface() {
  return interface_.get();
}

void CustomEngineManager::SetCustomEngineInterface(
    std::unique_ptr<C_CustomEngineInterface> device_interface) {
  if (device_interface) {
    interface_ = std::move(device_interface);
  }
}

}  // namespace custom_engine
}  // namespace paddle
