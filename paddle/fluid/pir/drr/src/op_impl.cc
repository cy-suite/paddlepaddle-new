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

#include "paddle/fluid/pir/drr/include/drr_pattern_context.h"
#include "paddle/fluid/pir/drr/src/pattern_graph.h"

namespace paddle {
namespace drr {

class Tensor;
class OpCall;
class PatternGraph;

thread_local int64_t OpImpl::count = 0;
const char* OpImpl::prefix_ = "@drr_temp@_";

void OpImpl::operator()(const Tensor& arg, const Tensor* out) const {
    std::vector<const Tensor*> inputs{&arg};
    std::vector<const Tensor*> outputs{out};
    pattern_graph_->AddOpCall(std::make_shared<OpCall>(op_, inputs, outputs));
}

void OpImpl::operator()(const std::vector<const Tensor*>& args,
                        const std::vector<const Tensor*>& outputs) const {
    pattern_graph_->AddOpCall(std::make_shared<OpCall>(op_, args, outputs));
}

Tensor& OpImpl::operator()(const Tensor& arg) const {
    std::vector<const Tensor*> inputs{&arg};
    auto& out = pattern_graph_->AddTmpTensor(std::shared_ptr<Tensor>(new Tensor(
        prefix_ + op_type_name_ + "_" + std::to_string(count++), pattern_graph_)));
    std::vector<const Tensor*> outputs{&out};
    pattern_graph_->AddOpCall(std::make_shared<OpCall>(op_, inputs, outputs));
    return out;
}

Tensor& OpImpl::operator()(const Tensor& arg1, const Tensor& arg2) const {
    std::vector<const Tensor*> inputs{&arg1, &arg2};
    auto& out = pattern_graph_->AddTmpTensor(std::shared_ptr<Tensor>(new Tensor(
        prefix_ + op_type_name_ + "_" + std::to_string(count++), pattern_graph_)));
    std::vector<const Tensor*> outputs{&out};
    pattern_graph_->AddOpCall(std::make_shared<OpCall>(op_, inputs, outputs));
    return out;
}

Tensor& OpImpl::operator()(const Tensor& arg0,
                       const Tensor& arg1,
                       const Tensor& arg2) const {
    std::vector<const Tensor*> inputs{&arg0, &arg1, &arg2};
    auto& out = pattern_graph_->AddTmpTensor(std::shared_ptr<Tensor>(new Tensor(
        prefix_ + op_type_name_ + "_" + std::to_string(count++), pattern_graph_)));
    std::vector<const Tensor*> outputs{&out};
    pattern_graph_->AddOpCall(std::make_shared<OpCall>(op_, inputs, outputs));
    return out;
}

Tensor& OpImpl::operator()() const {
    std::vector<const Tensor*> inputs{};
    auto& out = pattern_graph_->AddTmpTensor(std::shared_ptr<Tensor>(new Tensor(
        prefix_ + op_type_name_ + "_" + std::to_string(count++), pattern_graph_)));
    std::vector<const Tensor*> outputs{&out};
    pattern_graph_->AddOpCall(std::make_shared<OpCall>(op_, inputs, outputs));
    return out;
}

}
}
