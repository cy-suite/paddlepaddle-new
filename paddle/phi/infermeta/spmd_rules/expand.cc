/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/expand.h"

#include "glog/logging.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"

namespace phi::distributed {

SpmdInfo ExpandInferSpmd(const DistMetaTensor& x, const IntArray& shape) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  auto expand_shape = shape.GetData();
  std::vector<int64_t> out_dims_mapping(shape.size());
  int diff = expand_shape.size() - x_shape.size();
  for (int i = expand_shape.size() - 1; i >= diff; --i) {
    if (expand_shape[i] != -1 && expand_shape[i] != x_shape[i - diff]) {
      out_dims_mapping[i] = -1;

    } else {
      out_dims_mapping[i] = x_dims_mapping_src[i - diff];
    }
  }
  for (int i = 0; i < diff; i++) {
    out_dims_mapping[i] = -1;
  }
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);
  return {{x_dist_attr_src}, {out_dist_attr}};
}

SpmdInfo ExpandGradInferSpmd(const DistMetaTensor& x,
                             const DistMetaTensor& out_grad,
                             const IntArray& shape) {
  // TODO(lizhenxing): Support remove redundant dim
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  return {{x_dist_attr_src, x_dist_attr_src}, {x_dist_attr_src}};
}

}  // namespace phi::distributed
