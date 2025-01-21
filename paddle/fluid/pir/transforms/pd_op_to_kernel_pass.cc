// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"

#include <iostream>
#include <regex>
#include <string>
#include <unordered_set>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/new_executor/collect_shape_manager.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/interface/parse_kernel_key.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_pylayer_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/tensorrt_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/backends/custom/custom_device_op_list.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

#ifdef PADDLE_WITH_CINN
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/fluid/custom_engine/custom_engine_manager.h"
#endif

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_onednn_dialect.h"
#include "paddle/fluid/pir/dialect/operator/trait/onednn.h"
#include "paddle/phi/core/framework/framework.pb.h"
COMMON_DECLARE_bool(use_mkldnn);
#endif

COMMON_DECLARE_bool(print_ir);
COMMON_DECLARE_bool(enable_collect_shape);

namespace paddle::dialect {

std::unique_ptr<pir::Program> PdOpLowerToKernelPass(pir::Program* prog,
                                                    phi::Place place) {
  auto program = std::make_unique<pir::Program>(pir::IrContext::Instance());
  if (FLAGS_print_ir) {
    std::cout << "IR before lowering = " << *prog << std::endl;
  }
  auto block = prog->block();

  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<KernelDialect>();
  ctx->GetOrRegisterDialect<CustomKernelDialect>();

#ifdef PADDLE_WITH_DNNL
  ctx->GetOrRegisterDialect<OneDNNOperatorDialect>();
  ctx->GetOrRegisterDialect<OneDNNKernelDialect>();
#endif
  std::unordered_map<pir::Operation*, pir::Operation*> map_op_pair;
  std::unordered_map<pir::Value, pir::Value> map_value_pair;

  ProcessBlock(
      place, block, program->block(), ctx, &map_op_pair, &map_value_pair);

  if (FLAGS_enable_collect_shape) {
    paddle::framework::CollectShapeManager::Instance().SetValueMap(
        map_value_pair);
  }

  if (FLAGS_print_ir) {
    std::cout << "IR after lowering = " << *program << std::endl;
  }

  return program;
}
}  // namespace paddle::dialect
