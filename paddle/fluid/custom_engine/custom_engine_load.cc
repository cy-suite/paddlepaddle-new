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
#include <glog/logging.h>

#include "paddle/fluid/custom_engine/custom_engine_manager.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/include/core/ir_context.h"

bool ValidCustomCustomEngineParams(const CustomEngineParams* params) {
#define CHECK_INTERFACE(ptr, required)                                  \
  if (params->interface->ptr == nullptr && required) {                  \
    LOG(WARNING) << "CustomEngine pointer: " << #ptr << " is not set."; \
    return false;                                                       \
  }

  CHECK_INTERFACE(graph_engine_build, true);
  CHECK_INTERFACE(graph_engine_execute, true);
  CHECK_INTERFACE(custom_engine_op_lower, true);

  return true;
#undef CHECK_INTERFACE
}

typedef bool (*RegisterDevicePluginEngineFn)(CustomEngineParams* engine_params);

void LoadCustomEngineLib(
    const CustomEngineParams& engine_params,
    std::unique_ptr<C_CustomEngineInterface> engine_interface,
    const std::string& dso_lib_path,
    void* dso_handle) {
  if (ValidCustomCustomEngineParams(&engine_params)) {
    // pir::IrContext* ctx = pir::IrContext::Instance();
    // paddle::dialect::CustomEngineDialect* custom_engine_dialect =
    //   ctx->GetOrRegisterDialect<paddle::dialect::CustomEngineDialect>();
    // custom_dialect->RegisterCustomOp<CustomEngineOp>();

    paddle::custom_engine::CustomEngineManager::SetCustomEngineInterface(
        std::move(engine_interface));

  } else {
    LOG(WARNING) << "Skipped lib [" << dso_lib_path
                 << "]. Wrong engine parameters!!! please check the version "
                    "compatibility between PaddlePaddle and Custom Engine.";
  }
}
