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

#include "paddle/fluid/pir/transforms/tensorrt/auto_mix_precision.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {

namespace framework {

bool PhiKernelSupportPrecision(
    const std::string& op_type,
    phi::Backend backend,
    phi::DataType data_type,
    phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) {
  const auto& kernels = phi::KernelFactory::Instance().kernels();
  if (kernels.count(op_type) == 0) {
    return false;
  }
  phi::KernelKey kernel_key(backend, layout, data_type);
  return phi::KernelFactory::Instance().HasKernel(op_type, kernel_key);
}

bool KernelSupportPrecision(const std::string& op_type,
                            phi::Backend backend,
                            phi::DataType precision,
                            phi::DataLayout layout) {
  auto& phi_op_type = op_type;

  bool support =
      PhiKernelSupportPrecision(phi_op_type, backend, precision, layout);
  if (backend == phi::Backend::GPU) {
    support |= PhiKernelSupportPrecision(
        phi_op_type, phi::Backend::GPUDNN, precision, layout);
  }

  if (!support) {
    const auto& all_kernels =
        paddle::framework::OperatorWithKernel::AllOpKernels();
    auto it = all_kernels.find(op_type);
    if (it != all_kernels.end()) {
      for (const auto& kern_pair : it->second) {
        if (ConvertPlaceToBackend(kern_pair.first.place_) == backend &&
            kern_pair.first.data_type_ == phi::TransToProtoVarType(precision)) {
          support = true;
          break;
        }
      }
    }
  }
  return support;
}

bool OpSupportPrecision(const std::string& kernel_fn_str,
                        phi::Backend backend,
                        phi::DataType precision,
                        const std::unordered_set<std::string>& blacklist,
                        const std::unordered_set<std::string>& whitelist) {
  // 如果操作在白名单中，直接返回 true
  if (whitelist.count(kernel_fn_str)) {
    return true;
  }

  // 如果操作在黑名单中，直接返回 false
  if (blacklist.count(kernel_fn_str)) {
    return false;
  }

  return KernelSupportPrecision(kernel_fn_str, backend, precision);
}

}  // namespace framework
}  // namespace paddle
