/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/openvino/engine.h"
#include <glog/logging.h>
#include <string>

namespace paddle::inference::openvino {

void OpenVINOEngine::BuildEngine() {
  core_ = ovCoreCreate();
  model_ = ovCoreReadModel(
      core_, params_.model_program_path, params_.model_params_path);
  compiled_model_ =
      ovCoreCompileModel(core_, model_, "CPU", params_.model_opt_cache_dir);

  size_t inputs_size = ovCompiledModelInputsSize(compiled_model_);
  for (size_t i = 0; i < inputs_size; ++i) {
    ov_include::ov_output_const_port_t* input_port =
        ovCompiledModelInputByIndex(compiled_model_, i);
    std::string tensor_name = ovPortGetAnyName(input_port);
    std::cout << "input id = " << i << " name = " << tensor_name << std::endl;
    ovOutputConstPortFree(input_port);
  }
  size_t outputs_size = ovCompiledModelOutputsSize(compiled_model_);
  std::cout << "outputs_size = " << outputs_size << std::endl;
  for (size_t i = 0; i < outputs_size; ++i) {
    ov_include::ov_output_const_port_t* output_port =
        ovCompiledModelOutputByIndex(compiled_model_, i);
    std::string tensor_name = ovPortGetAnyName(output_port);
    std::cout << "output id = " << i << " tensor_name = " << tensor_name
              << std::endl;
    ovOutputConstPortFree(output_port);
  }
  infer_request_ = ovCompileModelCreateInferRequest(compiled_model_);
}

}  // namespace paddle::inference::openvino
