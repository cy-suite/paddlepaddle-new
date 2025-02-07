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

#include "paddle/fluid/eager/fwd/forward_grad.h"
// #include
// "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"

// class paddle::Tensor;

namespace forward_ad {

// namespace {
// // See discussion in forward_grad.h for why these are global variables and
// not
// // thread local

// std::mutex all_forward_levels_mutex_;
// std::vector<std::shared_ptr<ForwardADLevel>> all_forward_levels_;

// const static paddle::Tensor singleton_undefined_tensor;
// } // namespace

// TEST_API uint64_t ForwardADLevel::get_next_idx() {
//   std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
//   auto next_idx = all_forward_levels_.size();
//   PD_CHECK(
//       next_idx == 0, "Nested forward mode AD is not supported at the moment"
//   );
//   all_forward_levels_.push_back(std::make_shared<ForwardADLevel>(next_idx));
//   return next_idx;
// }

// TEST_API void ForwardADLevel::release_idx(uint64_t idx) {
//   std::unique_lock<std::mutex> lock(all_forward_levels_mutex_);
//   PD_CHECK(
//       idx + 1 == all_forward_levels_.size(),
//       "Exiting a forward AD level that is not the "
//       "last that was created is not support. Ensure they are released in the
//       reverse " "order they were created.");
//   PD_CHECK(!all_forward_levels_.empty(),
//     "all_forward_levels_ can not be empty."
//   );
//   // Keep the level alive until we have released the lock
//   auto lvl = std::move(all_forward_levels_.back());
//   all_forward_levels_.pop_back();
//   lock.unlock();
// }

// TEST_API std::shared_ptr<ForwardADLevel> ForwardADLevel::get_by_idx(uint64_t
// idx) {
//   std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
//   PD_CHECK(
//       idx < all_forward_levels_.size(),
//       "Trying to access a forward AD level with an invalid index. "
//       "This index was either not created or is already deleted.");
//   return all_forward_levels_[idx];
// }

// TEST_API std::shared_ptr<ForwardADLevel>
// ForwardADLevel::try_get_by_idx(uint64_t idx) {
//   std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
//   if (idx < all_forward_levels_.size()) {
//     return all_forward_levels_[idx];
//   } else {
//     return nullptr;
//   }
// }

ForwardADLevel::~ForwardADLevel() {
  // std::lock_guard<std::mutex> lock(mutex_);
  auto it = grads_.begin();
  while (it != grads_.end()) {
    // Warning this will lock *it mutex
    // This is ok as this function is the *only* one to call back into another
    // class's method.
    (*it)->reset(idx_, /* update_level */ false);
    it = grads_.erase(it);
  }
}

// TEST_API const paddle::Tensor& ForwardGrad::value(uint64_t level) const {
//   std::lock_guard<std::mutex> lock(mutex_);
//   const auto& it = content_.find(level);
//   return it == content_.end() ? singleton_undefined_tensor : (*it).second;
// }

// TEST_API const paddle::Tensor& ForwardGrad::undef_grad() {
//   return singleton_undefined_tensor;
// }

// Tensor _make_dual(
//     c10::DispatchKeySet ks,
//     const Tensor& primal,
//     const Tensor& tangent,
//     int64_t level) {
//   TORCH_CHECK(
//       !primal._fw_grad(level).defined(),
//       "Making a dual Tensor based on a Tensor that "
//       "already has a forward gradient at the same level ",
//       level,
//       " is not supported.");
//   auto& primal_ = unpack(primal, "primal", 0);
//   auto& tangent_ = unpack(tangent, "tangent", 0);
//   std::shared_ptr<ViewBackward0> grad_fn;
//   if (compute_requires_grad(primal_)) {
//     grad_fn = std::make_shared<ViewBackward0>();
//     grad_fn->self_sym_sizes = primal_.sym_sizes().vec();
//     grad_fn->set_next_edges(collect_next_edges(primal_));
//   }

//   auto result = ([&]() {
//     at::AutoDispatchBelowAutograd guard;
//     return at::redispatch::_make_dual(
//         ks & c10::after_autograd_keyset, primal_, tangent_, level);
//   })();

//   if (grad_fn) {
//     set_history(flatten_tensor_args(result), grad_fn);
//   }

//   TORCH_CHECK(level == 0, "Invalid level given to _make_dual");
//   result._set_fw_grad(tangent_, level, /* is_inplace_op */ false);
//   return result;
// }

}  // namespace forward_ad
