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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_dialect.h"
#include "paddle/common/flags.h"

#include "paddle/fluid/pir/dialect/distributed/ir/attribute_storage.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_op.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/distributed/ir/type_storage.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

REGISTER_FILE_SYMBOLS(dist_dialect);
COMMON_DECLARE_string(disable_logging_tensor_dist_attr_list);
namespace paddle::dialect {

DistDialect::DistDialect(pir::IrContext *context)
    : pir::Dialect(name(), context, pir::TypeId::get<DistDialect>()) {
  initialize();
}

void DistDialect::initialize() {
  RegisterAttributes<ProcessMeshAttribute,
                     TensorDistAttribute,
                     OperationDistAttribute>();
  RegisterTypes<DistDenseTensorType>();
  RegisterOps<ShardTensorOp,
              ReshardOp,
              MoESubMeshTensorsOp,
              MoEGlobalMeshTensorOp>();
}

void DistDialect::PrintType(pir::Type type, std::ostream &os) const {
  if (auto dist_dense_tensor_type = type.dyn_cast<DistDenseTensorType>()) {
    // Todo: Design the dist dense tensor type print format.
    os << type.dialect().name();
    os << '.';
    if (auto tensor_type = type.dyn_cast<pir::DenseTensorType>()) {
      os << "tensor<";
      for (auto d : common::vectorize(tensor_type.dims())) {
        os << d;
        os << "x";
      }
      tensor_type.dtype().Print(os);
      os << ", ";
      PrintAttribute(dist_dense_tensor_type.tensor_dist_attr(), os);
      os << ">";
    }
  } else {
    os << "error_type!";
  }
}

using PrintFunction = std::function<void(std::ostream &)>;

void PrintIfNotDisabled(const std::string &attr_name,
                        const PrintFunction &print_fn,
                        std::ostream &os,
                        const std::vector<std::string> &attrs_to_disable) {
  if (std::find(attrs_to_disable.begin(), attrs_to_disable.end(), attr_name) ==
      attrs_to_disable.end()) {
    print_fn(os);
  }
}

void DistDialect::PrintAttribute(pir::Attribute attr, std::ostream &os) const {
  std::vector<std::string> attrs_to_disable;
  if (!FLAGS_disable_logging_tensor_dist_attr_list.empty()) {
    std::istringstream iss(FLAGS_disable_logging_value_attr_list);
    std::string attr;
    while (std::getline(iss, attr, ';')) {
      attrs_to_disable.push_back(attr);
    }
  }

  if (auto process_mesh_attr = attr.dyn_cast<ProcessMeshAttribute>()) {
    os << "mesh_shape:[" +
              phi::distributed::auto_parallel::str_join(
                  process_mesh_attr.shape()) +
              "]";
    os << ",process_ids:[" +
              phi::distributed::auto_parallel::str_join(
                  process_mesh_attr.process_ids()) +
              "]";
  } else if (auto tensor_dist_attr = attr.dyn_cast<TensorDistAttribute>()) {
    PrintIfNotDisabled(
        "mesh_shape",
        [&](std::ostream &os) {
          os << "mesh_shape:[" +
                    phi::distributed::auto_parallel::str_join(
                        tensor_dist_attr.process_mesh_attr().shape()) +
                    "]";
        },
        os,
        attrs_to_disable);

    PrintIfNotDisabled(
        "process_ids",
        [&](std::ostream &os) {
          os << ",process_ids:[" +
                    phi::distributed::auto_parallel::str_join(
                        tensor_dist_attr.process_mesh_attr().process_ids()) +
                    "]";
        },
        os,
        attrs_to_disable);

    PrintIfNotDisabled(
        "dims_mappings",
        [&](std::ostream &os) {
          os << ",dims_mappings:[" +
                    phi::distributed::auto_parallel::str_join(
                        tensor_dist_attr.dims_mapping()) +
                    "]";
        },
        os,
        attrs_to_disable);
  } else if (auto op_dist_attr = attr.dyn_cast<OperationDistAttribute>()) {
    os << "{mesh:{shape:[" +
              phi::distributed::auto_parallel::str_join(
                  op_dist_attr.process_mesh_attr().shape()) +
              "]";
    os << ",process_ids:[" +
              phi::distributed::auto_parallel::str_join(
                  op_dist_attr.process_mesh_attr().process_ids()) +
              "]}";
    for (uint32_t i = 0; i < op_dist_attr.num_operands(); ++i) {
      os << ",operand(" + std::to_string(i) + "):{" << op_dist_attr.operand(i)
         << "}";
    }
    for (uint32_t i = 0; i < op_dist_attr.num_results(); ++i) {
      os << ",result(" + std::to_string(i) + "):{" << op_dist_attr.result(i)
         << "}";
    }
    os << ",chunk_id:" << op_dist_attr.chunk_id();
    os << "}";
  } else {
    os << "error_attribute_type";
  }
}

pir::OpPrintFn DistDialect::PrintOperation(const pir::Operation &op) const {
  return nullptr;
}

}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DistDialect)
