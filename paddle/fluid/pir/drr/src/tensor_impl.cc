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

#include "paddle/common/enforce.h"

#include "paddle/fluid/pir/drr/include/drr_pattern_context.h"
#include "paddle/fluid/pir/drr/src/pattern_graph.h"

namespace paddle {
namespace drr{

    class ResultPatternGraph;

    const char TensorImpl::SOURCE_INPUT_NONE_TENSOR_NAME[] =  // NOLINT
    "__@source_input_none_tensor@__";
    const char TensorImpl::SOURCE_OUTPUT_NONE_TENSOR_NAME[] =  // NOLINT
        "__@source_output_none_tensor@__";
    const char TensorImpl::RESULT_INPUT_NONE_TENSOR_NAME[] =  // NOLINT
        "__@result_input_none_tensor@__";
    const char TensorImpl::RESULT_OUTPUT_NONE_TENSOR_NAME[] =  // NOLINT
        "__@result_output_none_tensor@__";


    void TensorImpl::Assign(const Tensor& other) {
        dynamic_cast<ResultPatternGraph*>(pattern_graph_)->AssignTensor(*tensor_, other);
    }

    void TensorImpl::operator=(const Tensor& other) const {
        // The two tensor must be in the same pattern graph.
        PADDLE_ENFORCE_EQ(
            this->pattern_graph_,
            other.pattern_graph(),
            common::errors::InvalidArgument("Matching failed."
                                            "Two Tensors must be in the same pattern "
                                            "graph to make the '=' judgment."));
        if (other.name().find(Op::prefix()) == 0 &&
            name_.find(Op::prefix()) == std::string::npos) {
            other.pattern_graph()->UpdateTmpTensor(other.name(), this->name_);
        }
    }

}
}