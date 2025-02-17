// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/forward_grad/forward_grad.h"
#include "paddle/fluid/eager/grad_node_info.h"
namespace egr {

using AbstractAutogradMeta = paddle::AbstractAutogradMeta;
using ForwardGrad = forward_ad::ForwardGrad;
/**
 *
 * AutogradMeta is what record the backward info for tensor. When we run
 * computation graph eagerly, we can not build a static paddle program like
 * static graph mode do, so we need a new method to record forward info to trace
 * backward when we finish all forward computation. This require our
 * AutogradMeta class record following main members
 *
 * 1. grad_op:
 * Grad_op indicate the grad operation of the forward op
 *
 * 2. grad:
 * Grad is the gradient of forward Tensor, which should be compute after
 * backward computation
 *
 * NOTE: grad should only be available when current tensor is a leaf tensor, and
 * for non-leaf tensor grad is only available while user set `retain_grad`
 * option as `true`.
 *
 * TODO(jiabin) : support hooks
 * 3. hooks:
 * Hooks are some computation logic which only attached with backward operation,
 * it registered by user and run before accumulator.
 *
 * 4. overridden_stop_gradient_
 * This member is used to finish some auto-prune related work, which indicate
 * user set stop_gradient should override the result indicated by framework.
 * All non-parameter tensor's stop_gradient properties should be true. We will
 * pass stop_gradient when we find one who need it.
 *
 * NOTE: AutogradMeta is inherited from AbstractAutogradMeta which is defined
 * in tensor's deps, we did this to avoid additional dependency on Autograd.
 * In eager execution, we will cast AbstractAutogradMeta as AutogradMeta to use
 * it.
 *
 * **/

// No other AutogradMeta class should be derived from AbstractAutogradMeta.
// It's only used by
class AutogradMeta : public AbstractAutogradMeta {
 public:
  explicit AutogradMeta(const Edge& edge = Edge()) {
    out_slot_id_ = edge.GetEdgeRankInfo().first;
    out_rank_ = edge.GetEdgeRankInfo().second;
    grad_node_ = edge.GetMutableGradNode();
  }

  ~AutogradMeta() override = default;

  const paddle::Tensor& Grad() const {
    PADDLE_ENFORCE_NOT_NULL(
        grad_.get(),
        common::errors::InvalidArgument(
            "Should Not get NULL from Grad pointer, since "
            "we should have default Tensor once we init AutoGradMeta. "
            "if you got this error may indicates framework error in "
            "PaddlePaddle"));
    return *(grad_.get());
  }

  paddle::Tensor* MutableGrad() { return grad_.get(); }

  std::weak_ptr<paddle::Tensor> WeakGrad() { return grad_; }

  void SetGradNode(const std::shared_ptr<GradNodeBase>& grad_node) {
    PADDLE_ENFORCE_NOT_NULL(
        grad_node.get(),
        common::errors::InvalidArgument(
            "Should Not set NULL as GradNode pointer, since "
            "our default Edge and autogradMeta has nullptr for "
            "grad node. Set Nullptr will lead error."));

    grad_node_ = grad_node;
  }

  std::shared_ptr<GradNodeBase> GetMutableGradNode() const {
    return grad_node_;
  }

  GradNodeBase* GradNode() const { return grad_node_.get(); }

  void ResetGradNode() { grad_node_.reset(); }

  void SetSingleOutRankWithSlot(size_t slot_id, size_t rank) {
    out_slot_id_ = slot_id;
    out_rank_ = rank;
  }

  std::pair</* slot id */ size_t, /* rank in slot */ size_t> OutRankInfo()
      const {
    return std::make_pair(out_slot_id_, out_rank_);
  }

  bool IsInitialized() const { return grad_node_.get(); }

  // TODO(jiabin): This may cause error, since -1 still can indication true;
  bool StopGradient() const { return stop_gradient_ != 0; }

  int NumericStopGradient() const { return stop_gradient_; }

  void SetStopGradient(bool stop_gradient) {
    stop_gradient_ = static_cast<int>(stop_gradient);
  }

  bool Persistable() const { return persistable_; }

  void SetPersistable(bool persistable) { persistable_ = persistable; }

  bool RetainGrads() const { return retain_grads_; }

  void SetRetainGrads(bool value) { retain_grads_ = value; }

  const paddle::Tensor& fw_grad(uint64_t level,
                                const paddle::Tensor& self) const {
    const auto& direct_fw_grad =
        fw_grad_ ? fw_grad_->value(level) : ForwardGrad::undef_grad();

    return direct_fw_grad;
  }

  const paddle::Tensor& fw_grad(uint64_t level) const {
    const auto& direct_fw_grad =
        fw_grad_ ? fw_grad_->value(level) : ForwardGrad::undef_grad();

    return direct_fw_grad;
  }

 private:
  // TODO(jiabin) :Should we use pointer instead of object?
  std::shared_ptr<paddle::Tensor> grad_ = std::make_shared<paddle::Tensor>();

  // GradNodeBase is base class of all grad op which is a
  // wrapper for grad op. This class will make grad op easy
  // to be traced.
  std::shared_ptr<GradNodeBase> grad_node_ = nullptr;

  /**
   * Why we need slot id here?
   * Because in paddle most of operators, inputs and outputs
   * are assemble in form of {"slot name", vector<tensor>}.
   * So its better for us to set a slot id to fit this format. **/
  size_t out_slot_id_;

  // output rank of forward op, this is a vital num, since
  // we are now trying to make our forward output is as same
  // sequence as backward input. In case of tracing backward
  // sequence we need to record output rank in slot here.
  size_t out_rank_;

  // TODO(jiabin) :Support hooks here and store it in AutogradMeta

  // Stop gradient flag to indicate should we compute backward
  int stop_gradient_{-1};

  bool persistable_{false};

  bool retain_grads_{false};

  // TODO(jiabin) :Support Quantum here and add cache mechanism as
  // VarCache defined in VarBase

  // This field is used to store all the forward AD gradients
  // associated with this AutogradMeta (and the Tensor it corresponds to)
  // There is a semantic 1:1 correspondence between AutogradMeta and
  // ForwardGrad but:
  //   - This field is lazily populated.
  //   - This field is a shared_ptr but it must never be
  //     shared by multiple Tensors. See Note [ Using ForwardGrad ]
  mutable std::shared_ptr<ForwardGrad> fw_grad_;

  /**
   * NOTE: This function is will ensure that the fw_grad_ is properly a view of
   * the base for inplace ops on Tensors that do not have forward grad
   * originally.
   */
  void set_fw_grad(const paddle::Tensor& new_grad_base,
                   const paddle::Tensor& self_base,
                   uint64_t level,
                   bool is_inplace_op) {
    PD_CHECK(
        !new_grad_base._fw_grad(level).defined(),
        "Setting a forward grad that "
        "itself has a forward gradient at the same level %d is not supported.",
        level);

    // Lazy initialization
    {
      if (!fw_grad_) {
        fw_grad_ = std::make_shared<ForwardGrad>();
      }
    }
    if (fw_grad_->contains(level)) {
      // Setting the forward grad again is only allowed if it is a no-op.
      // We do allow this case to simplify writing codegen for inplace ops.
      PD_CHECK(new_grad_base.defined(),
               "Cannot set a forward grad that is an undefined Tensor. Use "
               "_fw_primal(level) to get a new Tensor with this forward grad "
               "unset.");

      PD_CHECK(is_inplace_op,
               "Only inplace operations can re-set the forward grad of a "
               "Tensor that "
               "already has one.");

      PD_CHECK(fw_grad_->value(level).impl() == new_grad_base.impl(),
               "Cannot set a value of a forward grad if it "
               "already exists. Inplace operations should modify it inplace.");
    } else {
      paddle::Tensor self_ref(new_grad_base);
      const paddle::Tensor& self = self_ref;

      fw_grad_->set_value(self, level);
    }
  }
};
}  // namespace egr
