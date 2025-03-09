// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/nvrtc/header_generator.h"
#include <fstream>
#include "glog/logging.h"
#include "jitify.hpp"  // NOLINT
#include "paddle/common/enforce.h"
namespace cinn {
namespace backends {
namespace nvrtc {

HeaderGeneratorBase& JitSafeHeaderGenerator::GetInstance() {
  static JitSafeHeaderGenerator instance;
  return instance;
}

const size_t JitSafeHeaderGenerator::size() const {
  PADDLE_ENFORCE_EQ(include_names_.size(),
                    headers_.size(),
                    ::common::errors::InvalidArgument(
                        "Internal error in size of header files."));
  return include_names_.size();
}

std::string read_file_as_string(const std::string& file_path) {
  std::ifstream file(file_path);

  if (!file) {
    throw std::runtime_error("Failed to open file");
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

void JitSafeHeaderGenerator::JitSafeHeaderGenerator() {
  const auto& headers_map = ::jitify::detail::get_jitsafe_headers_map();
  for (auto& pair : headers_map) {
    include_names_.emplace_back(pair.first.data());
    headers_.emplace_back(pair.second.data());
  }

  std::string cinn_float16_header =
      read_file_as_string("../../common/float16.h");
  std::string cinn_bfloat16_header =
      read_file_as_string("../../common/bfloat16.h");
  std::string cinn_with_cuda_header =
      "\n#define CINN_WITH_CUDA\n" std::string cinn_cuda_runtime_source_header =
          read_file_as_string("../../runtime/cuda/cinn_with_cuda.cuh");
  AddJitSafeHeader("float16_h", cinn_float16_header);
  AddJitSafeHeader("bfloat16_h", cinn_bfloat16_header);
  AddJitSafeHeader("cinn_with_cuda_h", cinn_with_cuda_header);
  AddJitSafeHeader("cinn_cuda_runtime_source_h",
                   cinn_cuda_runtime_source_header);
}

void JitSafeHeaderGenerator::AddJitSafeHeader(
    const std::string& header_name, const std::string& header_content) {
  include_names_.emplace_back(header_name.data());
  headers_.emplace_back(header_content.data());
}

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn
