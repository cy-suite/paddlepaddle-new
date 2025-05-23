/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/send_v2_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/common/flags.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/platform/collective_helper.h"
#endif
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/phi/api/include/tensor.h"

namespace paddle {
namespace operators {

#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
void send_shape_info(const phi::DenseTensor& x,
                     const phi::Place& place,
                     const gpuStream_t& stream,
                     platform::NCCLComm* comm,
                     phi::distributed::NCCLCommContext* comm_ctx,
                     const int& peer,
                     distributed::ProcessGroup* group) {
  if (!group) {
    PADDLE_ENFORCE_EQ(
        ((stream != nullptr && comm != nullptr) || comm_ctx != nullptr),
        true,
        common::errors::InvalidArgument(
            "NCCLComm and Stream should be provided if use NCCL "
            "to send the shape info."));
  }
  phi::DataType shape_dtype = phi::DataType::INT32;
  auto dims = x.dims();
  int shape_size = dims.size();

  // step1: send the shape size
  phi::DenseTensor cpu_shape_size_tensor(shape_dtype);
  cpu_shape_size_tensor.Resize({1});
  cpu_shape_size_tensor.mutable_data(phi::CPUPlace(), shape_dtype);
  auto* cpu_data = cpu_shape_size_tensor.data<int>();
  cpu_data[0] = shape_size;

  if (group) {
    std::vector<phi::DenseTensor> shape_size_tensor;
    shape_size_tensor.template emplace_back(cpu_shape_size_tensor);
    auto shape_size_task = group->Send(shape_size_tensor, peer);
  } else {
    // copy the shape size tensor to gpu and send
    phi::DenseTensor* gpu_shape_size_tensor = new phi::DenseTensor(shape_dtype);
    gpu_shape_size_tensor->Resize({1});
    gpu_shape_size_tensor->mutable_data(place, shape_dtype);
    framework::TensorCopySync(
        cpu_shape_size_tensor, place, gpu_shape_size_tensor);
    comm_ctx->Send(*gpu_shape_size_tensor, 1, peer, stream);
  }
  VLOG(3) << "send the shape size: " << shape_size << " to peer";

  // step2: send the shape
  phi::DenseTensor cpu_shape_tensor(shape_dtype);
  cpu_shape_tensor.Resize({shape_size});
  cpu_shape_tensor.mutable_data(phi::CPUPlace(), shape_dtype);
  auto* cpu_shape_data = cpu_shape_tensor.data<int>();
  for (int i = 0; i < shape_size; ++i) {
    cpu_shape_data[i] = dims[i];
  }

  if (group) {
    std::vector<phi::DenseTensor> shape_tensor;
    shape_tensor.template emplace_back(cpu_shape_tensor);
    auto shape_task = group->Send(shape_tensor, peer);
  } else {
    // copy the shape tensor to gpu and send
    phi::DenseTensor* gpu_shape_tensor = new phi::DenseTensor(shape_dtype);
    gpu_shape_tensor->Resize({shape_size});
    gpu_shape_tensor->mutable_data(place, shape_dtype);
    framework::TensorCopySync(cpu_shape_tensor, place, gpu_shape_tensor);
    comm_ctx->Send(*gpu_shape_tensor, shape_size, peer, stream);
  }
  VLOG(3) << "send the shape: (" << dims << ") to peer";
}
#endif

template <typename T, typename DeviceContext>
class SendOpV2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
    int rid = ctx.Attr<int>("ring_id");
    bool dynamic_shape = ctx.Attr<bool>("dynamic_shape");
    PADDLE_ENFORCE_GE(
        rid,
        0,
        common::errors::InvalidArgument(
            "The ring_id (%d) for send_v2 op must be non-negative.", rid));

    int peer = ctx.Attr<int>("peer");
    PADDLE_ENFORCE_GE(
        peer,
        0,
        common::errors::InvalidArgument(
            "The peer (%d) for send_v2 op must be non-negative.", peer));
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup* pg = map->get(rid);
      auto x = ctx.Input<phi::DenseTensor>("X");

      if (dynamic_shape) {
        // dynamic shape for switch send/recv
        VLOG(3) << "send_v2 will use dynamic shape with recv_v2 for switch";
        send_shape_info(*x,
                        ctx.GetPlace(),
                        /* gpuStream_t */ nullptr,
                        /* NCCLComm* */ nullptr,
                        /* NCCLCommContext * */ nullptr,
                        peer,
                        pg);
      }

      std::vector<phi::DenseTensor> in_tensor;
      in_tensor.push_back(*x);
      auto task = pg->Send(in_tensor, peer);
      return;
    }
    gpuStream_t stream = nullptr;
    auto place = ctx.GetPlace();
    platform::NCCLComm* comm = nullptr;
    phi::distributed::NCCLCommContext* comm_ctx = nullptr;

    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();

    PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(rid)),
                      true,
                      common::errors::InvalidArgument(
                          "You choose to use new communication library. "
                          "But ring_id(%d) is "
                          "not found in comm_context_manager.",
                          std::to_string(rid)));
    comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
        comm_context_manager.Get(std::to_string(rid)));
    PADDLE_ENFORCE_NE(comm_ctx,
                      nullptr,
                      common::errors::Unavailable(
                          "NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
    stream = comm_ctx->GetStream();
    VLOG(3) << "new comm_context_manager has rid " << rid;

    if (ctx.Attr<bool>("use_calc_stream")) {
      // should ExecutionContext for calc stream.
      stream = ctx.cuda_device_context().stream();
    }

    auto x = ctx.Input<phi::DenseTensor>("X");
    int numel = x->numel();

    if (dynamic_shape) {
      VLOG(3) << "send_v2 will use dynamic shape with recv_v2";
      send_shape_info(*x,
                      place,
                      stream,
                      comm,
                      comm_ctx,
                      peer,
                      /* ProcessGroup* */ nullptr);
    }

    comm_ctx->Send(*x, numel, peer, stream);
#else
    PADDLE_THROW(
        common::errors::Unavailable("PaddlePaddle should be compiled with NCCL "
                                    "and NCCL version >= 2.7.3 is needed."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(send_v2,
                          GPU,
                          ALL_LAYOUT,
                          ops::SendOpV2CUDAKernel,
                          float,
                          double,
#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
                          phi::dtype::bfloat16,
#endif
                          int,
                          int64_t,
                          int8_t,
                          phi::dtype::float16) {
}
