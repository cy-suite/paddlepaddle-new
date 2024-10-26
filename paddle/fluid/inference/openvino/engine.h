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

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/common/flags.h"
// #include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"

#include "paddle/fluid/inference/openvino/helper.h"

namespace paddle {
namespace inference {
namespace openvino {

/*
 * OpenVINO Engine.
 *
 * There are two alternative ways to use it, one is to build from a paddle
 * protobuf model, another way is to manually construct the network.
 */
class OpenVINOEngine {
 public:
  /*
   * Construction parameters of OpenVINOEngine.
   */
  struct ConstructionParams {
    std::string model_program_path;
    std::string model_params_path;
    std::string model_opt_cache_dir;
  };

  explicit OpenVINOEngine(const ConstructionParams& params) : params_(params) {}

  ~OpenVINOEngine() {
    ovInferRequestFree(infer_request_);
    ovCompileModelFree(compiled_model_);
    ovModelFree(model_);
    ovCoreFree(core_);
  }

  void BuildEngine();

 private:
  //
  // Construction parameters.
  //
  ConstructionParams params_;

  ov_include::ov_core_t* core_{nullptr};
  ov_include::ov_model_t* model_{nullptr};
  ov_include::ov_compiled_model_t* compiled_model_{nullptr};
  ov_include::ov_infer_request_t* infer_request_{nullptr};

  // std::unordered_map<std::string /*name*/, nvinfer1::ITensor* /*ITensor*/>
  //     itensor_map_;

  std::mutex mutex_;

 public:
  thread_local static int predictor_id_per_thread;
};  // class OpenVINOEngine

class OVEngineManager {
 public:
  OVEngineManager() {}

  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engines_.empty();
  }

  bool Has(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (engines_.count(name) == 0) return false;
    return engines_.at(name).get() != nullptr;
  }

  OpenVINOEngine* Get(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engines_.at(name).get();
  }

  OpenVINOEngine* Create(const std::string& name,
                         const OpenVINOEngine::ConstructionParams& params) {
    auto engine = std::make_unique<OpenVINOEngine>(params);
    std::lock_guard<std::mutex> lock(mutex_);
    engines_[name].reset(engine.release());
    return engines_[name].get();
  }

  void DeleteAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : engines_) {
      item.second.reset(nullptr);
    }
    engines_.clear();
  }

  void DeleteKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = engines_.find(key);
    if (iter != engines_.end()) {
      iter->second.reset(nullptr);
      engines_.erase(iter);
    }
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<OpenVINOEngine>> engines_;
};

}  // namespace openvino
}  // namespace inference
}  // namespace paddle
