// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include "paddle/phi/api/include/tensor.h"
#include "paddle/utils/small_vector.h"

namespace forward_ad {

#define EXPECTED_MAX_LEVEL 2

struct ForwardGrad;
struct ForwardADLevel;

class ForwardADLevelManager {
 public:
  static ForwardADLevelManager& instance() {
    static ForwardADLevelManager instance;
    return instance;
  }

  std::vector<std::shared_ptr<ForwardADLevel>>& get_all_forward_levels() {
    return all_forward_levels_;
  }

  paddle::Tensor& get_singleton_undefined_tensor() {
    return singleton_undefined_tensor;
  }

 private:
  ForwardADLevelManager() = default;
  ~ForwardADLevelManager() = default;
  ForwardADLevelManager(const ForwardADLevelManager&) = delete;
  ForwardADLevelManager& operator=(const ForwardADLevelManager&) = delete;

  std::vector<std::shared_ptr<ForwardADLevel>> all_forward_levels_;
  paddle::Tensor singleton_undefined_tensor;
};

struct TEST_API ForwardADLevel {
  explicit ForwardADLevel(uint64_t idx) : idx_(idx) {}
  ~ForwardADLevel();

  static uint64_t get_next_idx() {
    auto& all_forward_levels_ =
        ForwardADLevelManager::instance().get_all_forward_levels();
    auto next_idx = all_forward_levels_.size();
    PD_CHECK(next_idx == 0,
             "Nested forward mode AD is not supported at the moment");
    all_forward_levels_.push_back(std::make_shared<ForwardADLevel>(next_idx));
    return next_idx;
  }

  static void release_idx(uint64_t idx) {
    auto& all_forward_levels_ =
        ForwardADLevelManager::instance().get_all_forward_levels();
    PD_CHECK(idx + 1 == all_forward_levels_.size(),
             "Exiting a forward AD level that is not the "
             "last that was created is not support. Ensure they are released "
             "in the reverse "
             "order they were created.");
    PD_CHECK(!all_forward_levels_.empty(),
             "all_forward_levels_ can not be empty.");
    all_forward_levels_.pop_back();
  }

  static std::shared_ptr<ForwardADLevel> get_by_idx(uint64_t idx) {
    auto& all_forward_levels_ =
        ForwardADLevelManager::instance().get_all_forward_levels();
    PD_CHECK(idx < all_forward_levels_.size(),
             "Trying to access a forward AD level with an invalid index. "
             "This index was either not created or is already deleted.");
    return all_forward_levels_[idx];
  }

  static std::shared_ptr<ForwardADLevel> try_get_by_idx(uint64_t idx) {
    auto& all_forward_levels_ =
        ForwardADLevelManager::instance().get_all_forward_levels();
    if (idx < all_forward_levels_.size()) {
      return all_forward_levels_[idx];
    } else {
      return nullptr;
    }
  }

  void erase(const std::shared_ptr<ForwardGrad>& grad) { grads_.erase(grad); }

  void insert(const std::shared_ptr<ForwardGrad>& grad) { grads_.insert(grad); }

 private:
  std::unordered_set<std::shared_ptr<ForwardGrad>> grads_;
  uint64_t idx_;
};

struct TEST_API ForwardGrad : std::enable_shared_from_this<ForwardGrad> {
  ForwardGrad() = default;

  void clear() {
    paddle::small_vector<uint64_t, EXPECTED_MAX_LEVEL> levels_idx;

    {
      for (auto& c : lvl_to_grad) {
        levels_idx.push_back(c.first);
      }
    }

    for (auto l_idx : levels_idx) {
      auto level = ForwardADLevel::try_get_by_idx(l_idx);
      if (level) {
        level->erase(shared_from_this());
      }
    }
  }

  void set_value(const paddle::Tensor& value, uint64_t level) {
    auto forward_level = ForwardADLevel::get_by_idx(level);
    forward_level->insert(shared_from_this());

    lvl_to_grad.insert({level, value});
  }

  // This function removes the tangent for a given level from this ForwardGrad
  // Use the update_level flag to disable notifying the level about this reset
  // This flag is most notably used by the ForwardADLevel destructor.
  void reset(uint64_t level, bool update_level = true) {
    if (update_level) {
      ForwardADLevel::get_by_idx(level)->erase(shared_from_this());
    }

    const auto& it = lvl_to_grad.find(level);
    PD_CHECK(it != lvl_to_grad.end(), "Resetting a non-existent level.");
    lvl_to_grad.erase(level);
  }

  const paddle::Tensor& value(uint64_t level) const {
    const auto& it = lvl_to_grad.find(level);
    return it == lvl_to_grad.end() ? ForwardADLevelManager::instance()
                                         .get_singleton_undefined_tensor()
                                   : (*it).second;
  }

  bool contains(uint64_t level) { return lvl_to_grad.count(level) > 0; }

  bool empty() const { return lvl_to_grad.empty(); }

  static const paddle::Tensor& undef_grad() {
    return ForwardADLevelManager::instance().get_singleton_undefined_tensor();
  }

 private:
  std::unordered_map<uint64_t, paddle::Tensor> lvl_to_grad;
};

}  // namespace forward_ad
