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

#include <mutex>  // NOLINT

#include "flux/include/flux_api.h"
#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

extern std::once_flag flux_dso_flag;
extern void* flux_dso_handle;

#define DYNAMIC_LOAD_FLUX_WRAP(__name)                               \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using fluxFunc = decltype(&::__name);                          \
      std::call_once(flux_dso_flag, []() {                           \
        flux_dso_handle = phi::dynload::GetFluxDsoHandle();          \
      });                                                            \
      static void* p_##__name = dlsym(flux_dso_handle, #__name);     \
      return reinterpret_cast<fluxFunc>(p_##__name)(args...);        \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_FLUX_WRAP(__name) DYNAMIC_LOAD_FLUX_WRAP(__name)

#ifdef PADDLE_WITH_CUDA
#define FLUX_ROUTINE_EACH(__macro)                  \
  __macro(gemm_rs);                                 \
  __macro(ensure_nvml_init_capi);                   \
  __macro(get_gpu_device_name_capi);                \
  __macro(cudaipc_barrier_all_on_stream_impl_capi); \
  __macro(ag_gemm);                                 \
  __macro(set_ready);
#endif

FLUX_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_FLUX_WRAP);

#undef DYNAMIC_LOAD_FLUX_WRAP

}  // namespace dynload
}  // namespace phi
