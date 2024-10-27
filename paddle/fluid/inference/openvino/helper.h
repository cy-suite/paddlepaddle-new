/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <glog/logging.h>

#include <string>
#include <utility>
#include <vector>

#include "openvino/core/version.hpp"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/backends/dynload/openvino.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/framework/framework.pb.h"
#include "paddle/phi/core/utils/data_type.h"

namespace paddle {
namespace inference {
namespace openvino {

namespace dy = phi::dynload;

inline std::string GetOpenVINOVersion() {
  return std::to_string(OPENVINO_VERSION_MAJOR) + "." +
         std::to_string(OPENVINO_VERSION_MINOR) + "." +
         std::to_string(OPENVINO_VERSION_PATCH);
}

static ov_include::ov_core_t* ovCoreCreate() {
  ov_include::ov_core_t* core = NULL;
  ov_include::ov_status_e ov_status = dy::ov_core_create(&core);
  PADDLE_ENFORCE_NOT_NULL(
      core,
      common::errors::Unavailable("ovCoreCreate failed, ov_status_e = %d",
                                  ov_status));
  return core;
}

static ov_include::ov_model_t* ovCoreReadModel(
    const ov_include::ov_core_t* core,
    const std::string& model_path,
    const std::string& bin_path) {
  ov_include::ov_model_t* model = NULL;
  ov_include::ov_status_e ov_status = dy::ov_core_read_model(
      core, model_path.c_str(), bin_path.c_str(), &model);
  PADDLE_ENFORCE_NOT_NULL(
      model,
      common::errors::Unavailable("ovCoreReadModel failed, ov_status_e = %d",
                                  ov_status));
  return model;
}

static ov_include::ov_compiled_model_t* ovCoreCompileModel(
    const ov_include::ov_core_t* core,
    ov_include::ov_model_t* model,
    const std::string& device_name,
    const std::string& cache_dir) {
  ov_include::ov_compiled_model_t* compiled_model = NULL;
  ov_include::ov_status_e ov_status =
      dy::ov_core_compile_model(core,
                                model,
                                device_name.c_str(),
                                2,
                                &compiled_model,
                                "CACHE_DIR",
                                cache_dir.c_str());
  PADDLE_ENFORCE_NOT_NULL(
      compiled_model,
      common::errors::Unavailable("ovCoreCompileModel failed, ov_status_e = %d",
                                  ov_status));
  return compiled_model;
}

static ov_include::ov_infer_request_t* ovCompileModelCreateInferRequest(
    const ov_include::ov_compiled_model_t* compiled_model) {
  ov_include::ov_infer_request_t* infer_request = NULL;
  ov_include::ov_status_e ov_status =
      dy::ov_compiled_model_create_infer_request(compiled_model,
                                                 &infer_request);
  PADDLE_ENFORCE_NOT_NULL(
      infer_request,
      common::errors::Unavailable(
          "ovCompileModelCreateInferRequest failed, ov_status_e = %d",
          ov_status));
  return infer_request;
}

static void ovCoreFree(ov_include::ov_core_t* core) {
  dy::ov_core_free(core);
  return;
}

static void ovModelFree(ov_include::ov_model_t* model) {
  dy::ov_model_free(model);
  return;
}

static void ovCompileModelFree(
    ov_include::ov_compiled_model_t* compiled_model) {
  dy::ov_compiled_model_free(compiled_model);
  return;
}

static void ovInferRequestFree(ov_include::ov_infer_request_t* infer_request) {
  dy::ov_infer_request_free(infer_request);
  return;
}

static ov_include::ov_tensor_t* ovTensorCreate(
    const ov_include::ov_element_type_e type,
    const ov_include::ov_shape_t shape) {
  ov_include::ov_tensor_t* tensor = NULL;
  ov_include::ov_status_e ov_status =
      dy::ov_tensor_create(type, shape, &tensor);
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      common::errors::Unavailable("ovTensorCreate failed, ov_status_e = %d",
                                  ov_status));
  return tensor;
}

static void ovTensorFree(ov_include::ov_tensor_t* tensor) {
  dy::ov_tensor_free(tensor);
  return;
}

static std::string ovPortGetAnyName(
    const ov_include::ov_output_const_port_t* port) {
  char* c_tensor_name = nullptr;
  ov_include::ov_status_e ov_status =
      dy::ov_port_get_any_name(port, &c_tensor_name);
  PADDLE_ENFORCE_NOT_NULL(
      c_tensor_name,
      common::errors::Unavailable("ovPortGetAnyName failed, ov_status_e = %d",
                                  ov_status));
  std::string tensor_name(c_tensor_name);
  delete[] c_tensor_name;
  return tensor_name;
}

static size_t ovCompiledModelInputsSize(
    const ov_include::ov_compiled_model_t* compiled_model) {
  size_t size = 0;
  ov_include::ov_status_e ov_status =
      dy::ov_compiled_model_inputs_size(compiled_model, &size);
  PADDLE_ENFORCE_EQ(
      ov_status,
      ov_include::ov_status_e::OK,
      common::errors::Unavailable(
          "ovCompiledModelInputsSize failed, ov_status_e = %d", ov_status));
  return size;
}

static size_t ovCompiledModelOutputsSize(
    const ov_include::ov_compiled_model_t* compiled_model) {
  size_t size = 0;
  ov_include::ov_status_e ov_status =
      dy::ov_compiled_model_outputs_size(compiled_model, &size);
  PADDLE_ENFORCE_EQ(
      ov_status,
      ov_include::ov_status_e::OK,
      common::errors::Unavailable(
          "ovCompiledModelOutputsSize failed, ov_status_e = %d", ov_status));
  return size;
}

static ov_include::ov_output_const_port_t* ovCompiledModelInputByIndex(
    const ov_include::ov_compiled_model_t* compiled_model, const size_t index) {
  ov_include::ov_output_const_port_t* input_port = NULL;
  ov_include::ov_status_e ov_status =
      dy::ov_compiled_model_input_by_index(compiled_model, index, &input_port);
  PADDLE_ENFORCE_NOT_NULL(
      input_port,
      common::errors::Unavailable(
          "ovCompiledModelInputByIndex failed, ov_status_e = %d", ov_status));
  return input_port;
}

static ov_include::ov_output_const_port_t* ovCompiledModelOutputByIndex(
    const ov_include::ov_compiled_model_t* compiled_model, const size_t index) {
  ov_include::ov_output_const_port_t* output_port = NULL;
  ov_include::ov_status_e ov_status = dy::ov_compiled_model_output_by_index(
      compiled_model, index, &output_port);
  PADDLE_ENFORCE_NOT_NULL(
      output_port,
      common::errors::Unavailable(
          "ovCompiledModelOutputByIndex failed, ov_status_e = %d", ov_status));
  return output_port;
}

static void ovOutputConstPortFree(ov_include::ov_output_const_port_t* port) {
  dy::ov_output_const_port_free(port);
  return;
}

// shape
static ov_include::ov_shape_t ovPortGetShape(
    const ov_include::ov_output_port_t* port) {
  ov_include::ov_shape_t shape;
  ov_include::ov_status_e ov_status = dy::ov_port_get_shape(port, &shape);
  PADDLE_ENFORCE_EQ(ov_status,
                    ov_include::ov_status_e::OK,
                    common::errors::Unavailable(
                        "ovPortGetShape failed, ov_status_e = %d", ov_status));
  return shape;
}

// type
static ov_include::ov_element_type_e ovPortGetElementType(
    const ov_include::ov_output_const_port_t* port) {
  ov_include::ov_element_type_e type;
  ov_include::ov_status_e ov_status = dy::ov_port_get_element_type(port, &type);
  PADDLE_ENFORCE_EQ(
      ov_status,
      ov_include::ov_status_e::OK,
      common::errors::Unavailable(
          "ovPortGetElementType failed, ov_status_e = %d", ov_status));
  return type;
}

inline bool CanConvertToOpenvino(const std::string& model_path,
                                 const std::string& param_path,
                                 const std::string& opt_cache_dir) {
  std::cout << "CanConvertToOpenvino = true" << std::endl;
  ov_include::ov_core_t* core = ovCoreCreate();
  ov_include::ov_model_t* model = ovCoreReadModel(core, model_path, param_path);
  ov_include::ov_compiled_model_t* compiled_model =
      ovCoreCompileModel(core, model, "CPU", opt_cache_dir);
  size_t inputs_size = ovCompiledModelInputsSize(compiled_model);
  std::cout << "inputs_size = " << inputs_size << std::endl;
  for (size_t i = 0; i < inputs_size; ++i) {
    ov_include::ov_output_const_port_t* input_port =
        ovCompiledModelInputByIndex(compiled_model, i);
    std::string tensor_name = ovPortGetAnyName(input_port);
    std::cout << "input id = " << i << " tensor_name = " << tensor_name
              << std::endl;
    ovOutputConstPortFree(input_port);
  }

  size_t outputs_size = ovCompiledModelOutputsSize(compiled_model);
  std::cout << "outputs_size = " << outputs_size << std::endl;
  for (size_t i = 0; i < outputs_size; ++i) {
    ov_include::ov_output_const_port_t* output_port =
        ovCompiledModelOutputByIndex(compiled_model, i);
    std::string tensor_name = ovPortGetAnyName(output_port);
    std::cout << "output id = " << i << " tensor_name = " << tensor_name
              << std::endl;
    ovOutputConstPortFree(output_port);
  }
  ov_include::ov_infer_request_t* infer_request =
      ovCompileModelCreateInferRequest(compiled_model);

  ovInferRequestFree(infer_request);
  ovCompileModelFree(compiled_model);
  ovModelFree(model);
  ovCoreFree(core);
  return true;
}

}  // namespace openvino
}  // namespace inference
}  // namespace paddle
