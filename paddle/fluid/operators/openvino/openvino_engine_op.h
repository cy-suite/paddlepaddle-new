/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_OPENVINO
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/common/errors.h"
// #include "paddle/fluid/framework/data_device_transform.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/openvino/engine.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle {
namespace inference {
namespace openvino {}  // namespace openvino
template <typename T>
struct Singleton;
}  // namespace inference
}  // namespace paddle

namespace paddle {

namespace operators {

using inference::Singleton;
using inference::openvino::OpenVINOEngine;

class OpenVINOEngineOp : public framework::OperatorBase {
 private:
  std::vector<std::string> input_names_;

 public:
  OpenVINOEngineOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    input_names_ = Inputs("Xs");
  }

 protected:
  void RunNativeImpl(const framework::Scope &scope,
                     const phi::Place &dev_place) const {
    framework::Executor executor(dev_place);
    auto *block = Attr<framework::BlockDesc *>("sub_block");
    auto *program = block->Program();
    auto &current_scope = scope.NewScope();
    auto ctx = executor.Prepare(*program, block->ID());
    executor.RunPreparedContext(ctx.get(), &current_scope, false, true, true);
  }

  void RunImpl(const framework::Scope &scope,
               const phi::Place &dev_place) const override {}
};

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_WITH_CUDA
