/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/core/enforce.h"

namespace ov_include {
extern "C" {
#include "openvino/c/openvino.h"
#include "openvino/c/ov_property.h"
}
}  // namespace ov_include
namespace phi {
namespace dynload {

void* GetOpenVINOHandle();

extern std::once_flag openvino_dso_flag;
extern void* openvino_dso_handle;

#define DECLARE_DYNAMIC_LOAD_OPENVINO_VOID_WRAP(__name)                     \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    void operator()(Args... args) {                                         \
      std::call_once(openvino_dso_flag, []() {                              \
        openvino_dso_handle = phi::dynload::GetOpenVINOHandle();            \
      });                                                                   \
      static void* p_##__name = dlsym(openvino_dso_handle, #__name);        \
      PADDLE_ENFORCE_NOT_NULL(p_##__name,                                   \
                              common::errors::Unavailable(                  \
                                  "Load openvino api %s failed", #__name)); \
      using openvino_func = decltype(&ov_include::__name);                  \
      reinterpret_cast<openvino_func>(p_##__name)(args...);                 \
      return;                                                               \
    }                                                                       \
  };                                                                        \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_OPENVINO_NON_POINTER_WRAP(__name)              \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    auto operator()(Args... args) {                                         \
      std::call_once(openvino_dso_flag, []() {                              \
        openvino_dso_handle = phi::dynload::GetOpenVINOHandle();            \
      });                                                                   \
      static void* p_##__name = dlsym(openvino_dso_handle, #__name);        \
      PADDLE_ENFORCE_NOT_NULL(p_##__name,                                   \
                              common::errors::Unavailable(                  \
                                  "Load openvino api %s failed", #__name)); \
      using openvino_func = decltype(&ov_include::__name);                  \
      auto ret = reinterpret_cast<openvino_func>(p_##__name)(args...);      \
      return ret;                                                           \
    }                                                                       \
  };                                                                        \
  extern DynLoad__##__name __name

#define OPENVINO_RAND_ROUTINE_EACH_VOID(__macro) \
  __macro(ov_core_free);                         \
  __macro(ov_compiled_model_free);               \
  __macro(ov_model_free);                        \
  __macro(ov_infer_request_free);                \
  __macro(ov_tensor_free);                       \
  __macro(ov_output_const_port_free);

#define OPENVINO_RAND_ROUTINE_EACH_NON_POINTER(__macro) \
  __macro(ov_core_create);                              \
  __macro(ov_core_read_model);                          \
  __macro(ov_core_compile_model);                       \
  __macro(ov_compiled_model_create_infer_request);      \
  __macro(ov_tensor_create);                            \
  __macro(ov_compiled_model_inputs_size);               \
  __macro(ov_compiled_model_input_by_index);            \
  __macro(ov_compiled_model_outputs_size);              \
  __macro(ov_compiled_model_output_by_index);           \
  __macro(ov_port_get_any_name);                        \
  __macro(ov_port_get_shape);                           \
  __macro(ov_port_get_element_type);

OPENVINO_RAND_ROUTINE_EACH_VOID(DECLARE_DYNAMIC_LOAD_OPENVINO_VOID_WRAP)
OPENVINO_RAND_ROUTINE_EACH_NON_POINTER(
    DECLARE_DYNAMIC_LOAD_OPENVINO_NON_POINTER_WRAP)

}  // namespace dynload
}  // namespace phi
