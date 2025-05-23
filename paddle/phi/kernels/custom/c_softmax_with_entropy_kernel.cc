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

#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {

template <typename T, typename Context>
void CSoftmaxWithEntropyKernel(const Context& dev_ctx,
                               const DenseTensor& logits_in,
                               const DenseTensor& label_in,
                               int64_t ignore_index,
                               int rank,
                               int nranks,
                               DenseTensor* softmax,
                               DenseTensor* loss) {
  auto comm = reinterpret_cast<phi::distributed::XCCLCommContext*>(
      dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(comm,
                    nullptr,
                    common::errors::Unavailable(
                        "XCCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));

  const phi::DenseTensor* logits = &logits_in;
  const phi::DenseTensor* labels = &label_in;
  auto softmax_dims = softmax->dims();
  auto loss_dims = loss->dims();

  const int axis = logits->dims().size() - 1;
  const int N = phi::funcs::SizeToAxis(axis, logits->dims());
  const int D = phi::funcs::SizeFromAxis(axis, logits->dims());

  auto logits_2d = std::make_shared<phi::DenseTensor>();
  auto labels_1d = std::make_shared<phi::DenseTensor>();
  logits_2d->ShareDataWith(*logits).Resize({N, D});
  labels_1d->ShareDataWith(*labels).Resize({N});
  paddle::Tensor logits_2d_tensor(logits_2d), labels_1d_tensor(labels_1d);

  // step 1, obtain logit_max
  auto logits_2d_max_tensor = logits_2d_tensor.max({1}, true);
  auto logits_2d_max =
      reinterpret_cast<phi::DenseTensor*>(logits_2d_max_tensor.impl().get());
  auto& stream = *dev_ctx.GetStream();
  phi::DeviceManager::CCLAllReduce(dev_ctx.GetPlace().GetDeviceType(),
                                   logits_2d_max->data<float>(),
                                   logits_2d_max->data<float>(),
                                   logits_2d_max->numel(),
                                   logits_2d_max->dtype(),
                                   phi::ccl::CCLReduceOp::MAX,
                                   comm->GetXcclComm(),
                                   stream);

  // step 2, obtain logit - logit_max
  auto logits_2d_sub_max = paddle::experimental::clip(
      logits_2d_tensor - logits_2d_max_tensor, -64., 0.);

  // step 3, obtain predict target
  const int start_index = rank * D;
  auto start_index_tensor =
      paddle::experimental::full_like(labels_1d_tensor,
                                      start_index,
                                      labels_1d_tensor.dtype(),
                                      labels_1d_tensor.place());
  auto end_index_tensor =
      paddle::experimental::full_like(labels_1d_tensor,
                                      start_index + D,
                                      labels_1d_tensor.dtype(),
                                      labels_1d_tensor.place());
  auto labels_1d_mask = paddle::experimental::logical_and(
      labels_1d_tensor.greater_equal(start_index_tensor),
      labels_1d_tensor.less_than(end_index_tensor));
  auto real_label_tensor = (labels_1d_tensor - start_index_tensor)
                               .multiply(paddle::experimental::cast(
                                   labels_1d_mask, labels_1d_tensor.dtype()));

  auto predicted_logits_tensor =
      logits_2d_sub_max
          .multiply(paddle::experimental::cast(
              paddle::experimental::one_hot(real_label_tensor, D),
              logits_2d_sub_max.dtype()))
          .sum({1}, logits_2d_sub_max.dtype(), false)
          .multiply(paddle::experimental::cast(labels_1d_mask,
                                               logits_2d_sub_max.dtype()));

  auto predicted_logits =
      reinterpret_cast<phi::DenseTensor*>(predicted_logits_tensor.impl().get());
  phi::DeviceManager::CCLAllReduce(dev_ctx.GetPlace().GetDeviceType(),
                                   predicted_logits->data<float>(),
                                   predicted_logits->data<float>(),
                                   predicted_logits->numel(),
                                   predicted_logits->dtype(),
                                   phi::ccl::CCLReduceOp::SUM,
                                   comm->GetXcclComm(),
                                   stream);

  // step 4, obtain exp(logit)
  auto softmax_2d_tensor = logits_2d_sub_max.exp();

  // step 5, obtain sum_exp_logits
  auto sum_exp_logits_tensor =
      softmax_2d_tensor.sum({1}, softmax_2d_tensor.dtype(), false);

  auto sum_exp_logits =
      reinterpret_cast<phi::DenseTensor*>(sum_exp_logits_tensor.impl().get());
  phi::DeviceManager::CCLAllReduce(dev_ctx.GetPlace().GetDeviceType(),
                                   sum_exp_logits->data<float>(),
                                   sum_exp_logits->data<float>(),
                                   sum_exp_logits->numel(),
                                   sum_exp_logits->dtype(),
                                   phi::ccl::CCLReduceOp::SUM,
                                   comm->GetXcclComm(),
                                   stream);

  auto softmax_out = softmax_2d_tensor.divide(
      paddle::experimental::reshape(sum_exp_logits_tensor, {N, 1}));
  auto labels_1d_not_equal_ignore = labels_1d_tensor.not_equal(
      paddle::experimental::full_like(labels_1d_tensor,
                                      ignore_index,
                                      labels_1d_tensor.dtype(),
                                      labels_1d_tensor.place()));
  auto loss_out =
      (sum_exp_logits_tensor.log() - predicted_logits_tensor)
          .multiply(paddle::experimental::cast(labels_1d_not_equal_ignore,
                                               sum_exp_logits_tensor.dtype()));
  softmax
      ->ShareDataWith(
          *reinterpret_cast<phi::DenseTensor*>(softmax_out.impl().get()))
      .Resize(softmax_dims);
  loss->ShareDataWith(
          *reinterpret_cast<phi::DenseTensor*>(loss_out.impl().get()))
      .Resize(loss_dims);
}
}  // namespace phi

PD_REGISTER_KERNEL(c_softmax_with_cross_entropy,
                   Custom,
                   ALL_LAYOUT,
                   phi::CSoftmaxWithEntropyKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
