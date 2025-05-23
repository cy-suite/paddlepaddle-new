// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/process_group_bkcl.h"

#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/distributed/collective/bkcl_tools.h"
#include "paddle/fluid/distributed/collective/common.h"
#include "paddle/fluid/distributed/collective/process_group_kernel_utils.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/distributed/check/bkcl_dynamic_check.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/core/platform/device/xpu/bkcl_helper.h"
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#include "paddle/utils/string/string_helper.h"

COMMON_DECLARE_bool(enable_bkcl_dynamic_check);

namespace paddle {
namespace distributed {

using phi::distributed::CheckSizeOnEachRank;

ProcessGroupBKCL::BKCLTask::BKCLTask(const Place& place,
                                     int rank,
                                     CommType comm_type,
                                     bool sync_op,
                                     bool use_calc_stream)
    : TaskStream(rank, comm_type, sync_op, use_calc_stream), place_(place) {
  comm_event_ = std::make_shared<XPUEventManager>();
}

ProcessGroupBKCL::BKCLTask::~BKCLTask() {}

bool ProcessGroupBKCL::BKCLTask::IsCompleted() {
  LOG_FIRST_N(WARNING, 1) << "XPU do not support event query now.";
  return true;
}

// TODO(sheniang03): Add timeout for wait, now timeout unused
bool ProcessGroupBKCL::BKCLTask::Wait(std::chrono::milliseconds timeout) {
  const auto* calc_ctx =
      static_cast<XPUContext*>(phi::DeviceContextPool::Instance().Get(place_));
  if (barrier_) {
    // If we use the work to do barrier, we should block cpu

    // TODO(zhangxiaoci) There is no such function that can sync entire device
    // for xpu (for now), so all we can do is sync whatever stream that we know
    // and hope for the best. Note that for correctness the communication stream
    // needs to be in sync mode.
    phi::backends::xpu::XPUDeviceGuard guard(place_.GetDeviceId());
    xpu_wait();
    calc_ctx->Wait();
  }
  // Warning here when use calc stream but also invoke waiting explicitly.
  if (UseCalcStream()) {
    VLOG(3) << "Warning: The communication is on calc stream, wait here is "
               "useless.";
    return true;
  }

  comm_event_->Block(*calc_ctx);

  return true;
}

// Same as Wait
void ProcessGroupBKCL::BKCLTask::Synchronize() { Wait(kWaitTimeout); }

ProcessGroupBKCL::ProcessGroupBKCL(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid)
    : ProcessGroupWithStream(rank, size, gid), store_(store) {}

void ProcessGroupBKCL::GroupStart() {
  PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_group_start());
}

void ProcessGroupBKCL::GroupEnd() {
  PADDLE_ENFORCE_BKCL_SUCCESS(bkcl_group_end());
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Recv(
    phi::DenseTensor* tensor,
    int src_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  // numel > 0 indicates the tensor need to be sliced
  phi::DenseTensor partial_tensor;
  if (numel > 0) {
    partial_tensor = GetPartialTensor(*tensor, offset, numel);
    tensor = &partial_tensor;
  }

  return Point2Point(
      [&](phi::distributed::BKCLCommContext* comm_context,
          XPUStream stream,
          int rank_in_group) {
        VLOG(3) << "bkcl_recv "
                << "recvbuff: " << tensor->data()
                << ", count: " << tensor->numel() << ", datatype: "
                << BKCLDTypeToString(phi::ToBKCLDataType(tensor->dtype()))
                << ", src_in_group: " << src_rank
                << ", bkcl_comm: " << comm_context->GetBKCLComm()
                << ", stream: " << stream
                << ", rank_in_group: " << rank_in_group << ", nranks: " << size_
                << ", offset: " << offset << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        comm_context->Recv(tensor, tensor->numel(), rank_in_group, stream);
      },
      src_rank,
      *tensor,
      CommType::RECV,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Send(
    const phi::DenseTensor& tensor,
    int dst_rank,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(tensor);
  // numel > 0 indicates the tensor need to be sliced
  const phi::DenseTensor& tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(tensor, offset, numel) : tensor;

  return Point2Point(
      [&](phi::distributed::BKCLCommContext* comm_context,
          XPUStream stream,
          int rank_in_group) {
        VLOG(3) << "bkcl_send "
                << "sendbuff: " << tensor_maybe_partial.data()
                << ", count: " << tensor_maybe_partial.numel() << ", datatype: "
                << BKCLDTypeToString(
                       phi::ToBKCLDataType(tensor_maybe_partial.dtype()))
                << ", dst_in_group: " << dst_rank
                << ", bkcl_comm: " << comm_context->GetBKCLComm()
                << ", stream: " << stream
                << ", rank_in_group: " << rank_in_group << ", nranks: " << size_
                << ", offset: " << offset << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        comm_context->Send(tensor_maybe_partial,
                           tensor_maybe_partial.numel(),
                           rank_in_group,
                           stream);
      },
      dst_rank,
      tensor_maybe_partial,
      CommType::SEND,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroupBKCL::BKCLTask> ProcessGroupBKCL::CreateTask(
    const Place& place,
    int rank,
    CommType comm_type,
    bool is_sync,
    bool use_calc_stream) {
  return std::make_shared<ProcessGroupBKCL::BKCLTask>(
      place, rank, comm_type, is_sync, use_calc_stream);
}

void ProcessGroupBKCL::BroadcastUniqueBKCLID(BKCLUniqueId* bkcl_id) {
  auto key = "ProcessGroupBKCL/bkcl_ids/" + std::to_string(gid_) + "/0";
  if (rank_ == 0) {
    auto id = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(bkcl_id),
        reinterpret_cast<uint8_t*>(bkcl_id) + BKCL_UNIQUE_ID_BYTES);
    store_->set(key, id);
  } else {
    const auto& ret = store_->get(key);
    std::memcpy(bkcl_id, ret.data(), ret.size());
  }
}

void ProcessGroupBKCL::CreateBKCLEnvCache(const Place& place,
                                          const std::string& place_key) {
  phi::backends::xpu::XPUDeviceGuard guard(place.GetDeviceId());

  VLOG(3) << "init bkcl rank: " << rank_ << ", nranks: " << size_
          << ", place: " << place_key;

  phi::distributed::CommContextManager::CreateBKCLCommContext(
      store_, std::to_string(gid_), rank_, size_);

  calc_event_ = std::make_shared<XPUEventManager>();
  auto* calc_ctx = static_cast<phi::XPUContext*>(
      phi::DeviceContextPool::Instance().Get(place));
  // must use phi::XPUContext here to make sure XPUContext::Init() is called
  auto comm_ctx = std::make_unique<phi::XPUContext>(place, true);
  // comm_ctx does not require a pre-allocated GM buffer
  comm_ctx->x_context()->set_option("XPUAPI_DEFAULT_SIZE", "1");
  auto bkcl_comm_ctx = this->GetCommContext();
  comm_ctx->SetBkclContext(bkcl_comm_ctx->GetBKCLComm());

  // set allocator
  comm_ctx->SetAllocator(memory::allocation::AllocatorFacade::Instance()
                             .GetAllocator(place)
                             .get());
  // Note(lijin23): XPU use calc stream for communication now, so we disable the
  // creation of comm stream to reduce the total number of streams used.
  // comm_ctx->CreateStream();

  place_to_calc_ctx_[place_key] = calc_ctx;
  place_to_comm_ctx_[place_key] = std::move(comm_ctx);
}

void ProcessGroupBKCL::SyncCalcStream(const Place& place) {
  const std::string& key = GetKeyFromPlace(place);
  const auto* calc_ctx = place_to_calc_ctx_[key];
  const auto* comm_ctx = place_to_comm_ctx_[key].get();
  calc_event_->Record(*calc_ctx);
  calc_event_->Block(*comm_ctx);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Collective(
    std::function<void(phi::distributed::BKCLCommContext*, XPUStream)> fn,
    const std::vector<phi::DenseTensor>& tensors,
    CommType op_type,
    bool sync_op,
    bool use_calc_stream) {
  if (!use_calc_stream) {
    VLOG(3) << "For XPU, Communication on non-calc stream has minor effect on "
               "performance and might be conflict with streams in calc_ctx, so "
               "we disable it currently.";
    use_calc_stream = true;
  }
  CheckTensorContiguous(tensors);

  PADDLE_ENFORCE_GT(
      tensors.size(),
      0,
      common::errors::InvalidArgument("Num of tensors must be greater than 0"));
  const auto& place = tensors[0].place();
  const auto& key = GetKeyFromPlace(place);

  phi::backends::xpu::XPUDeviceGuard xpu_guard(place);

  if (!calc_event_ ||
      (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end())) {
    CreateBKCLEnvCache(place, key);
  }

  if (!use_calc_stream) {
    SyncCalcStream(place);
  }

  auto task = CreateTask(place, rank_, op_type, sync_op, use_calc_stream);

  const auto* calc_ctx = place_to_calc_ctx_.at(key);
  const auto& comm_ctx = place_to_comm_ctx_.at(key);
  auto bkcl_stream = use_calc_stream ? calc_ctx->stream() : comm_ctx->stream();

  auto bkcl_comm_ctx = this->GetCommContext();

  fn(bkcl_comm_ctx, bkcl_stream);

  if (!use_calc_stream) {
    PADDLE_ENFORCE_NOT_NULL(comm_ctx.get(),
                            common::errors::Fatal("comm context is nullptr."));
    if (!is_coalescing_) {
      task->comm_event_->Record(*comm_ctx.get());
    } else {
      coalescing_place_keys_.push_back(key);
    }
  }

  if (sync_op) {
    task->Wait();
  }

  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Collective(
    std::function<void(phi::distributed::BKCLCommContext*, XPUStream)> fn,
    const phi::DenseTensor& tensor,
    CommType op_type,
    bool sync_op,
    bool use_calc_stream) {
  const std::vector<phi::DenseTensor> tensors = {tensor};
  return Collective(fn, tensors, op_type, sync_op, use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Point2Point(
    std::function<void(phi::distributed::BKCLCommContext*, XPUStream, int)> fn,
    int peer,
    const phi::DenseTensor& tensor,
    CommType comm_type,
    bool sync_op,
    bool use_calc_stream) {
  if (!use_calc_stream) {
    VLOG(3) << "For XPU, Communication on non-calc stream has minor effect on "
               "performance and might be conflict with streams in calc_ctx, so "
               "we disable it currently.";
    use_calc_stream = true;
  }
  CheckTensorContiguous(tensor);
  const auto& place = tensor.place();

  int p2p_target_rank = peer;
  std::string key = GetKeyFromPlace(place);

  phi::backends::xpu::XPUDeviceGuard xpu_guard(place);

  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateBKCLEnvCache(place, key);
  }

  if (!use_calc_stream) {
    SyncCalcStream(place);
  }

  auto task = CreateTask(place, rank_, comm_type, sync_op, use_calc_stream);
  const auto* calc_ctx = place_to_calc_ctx_.at(key);
  const auto& comm_ctx = place_to_comm_ctx_.at(key);
  auto bkcl_stream = use_calc_stream ? calc_ctx->stream() : comm_ctx->stream();

  auto bkcl_comm_ctx = this->GetCommContext();
  fn(bkcl_comm_ctx, bkcl_stream, p2p_target_rank);

  if (!use_calc_stream) {
    PADDLE_ENFORCE_NOT_NULL(comm_ctx.get(),
                            common::errors::Fatal("comm context is nullptr."));
    if (!is_coalescing_) {
      task->comm_event_->Record(*comm_ctx.get());
    } else {
      coalescing_place_keys_.push_back(key);
    }
  }

  if (sync_op) {
    task->Wait();
  }

  return task;
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllReduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const AllreduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);

  return Collective(
      [&](phi::distributed::BKCLCommContext* comm_context, XPUStream stream) {
        VLOG(3) << "bkcl_all_reduce"
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << BKCLDTypeToString(phi::ToBKCLDataType(in_tensor.dtype()))
                << ", redop: " << ToBKCLRedType(opts.reduce_op)
                << ", bkcl_comm: " << comm_context->GetBKCLComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;

        comm_context->AllReduce(
            out_tensor, in_tensor, ToBKCLRedType(opts.reduce_op), stream);
      },
      in_tensor,
      CommType::ALLREDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllToAll(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const std::vector<int64_t>& out_size_each_rank,
    const std::vector<int64_t>& in_size_each_rank,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  std::vector<int64_t> out_split_sizes;
  std::vector<int64_t> in_split_sizes;
  bool is_equal_split = false;
  if (out_size_each_rank.empty() && in_size_each_rank.empty()) {
    out_split_sizes =
        std::vector<int64_t>(size_, out_tensor->dims()[0] / size_);
    in_split_sizes = std::vector<int64_t>(size_, in_tensor.dims()[0] / size_);
    is_equal_split = true;
  } else {
    out_split_sizes = out_size_each_rank;
    in_split_sizes = in_size_each_rank;
  }

  const phi::DDim& out_dim = out_tensor->dims();
  const phi::DDim& in_dim = in_tensor.dims();
  CheckSizeOnEachRank(out_dim, out_split_sizes, size_);
  CheckSizeOnEachRank(in_dim, in_split_sizes, size_);

  // AllToAllUnequalSplit requires allocating temporary memory and must use
  // calc_stream to ensure the correct lifecycle management of the temporary
  // tensor.
  if (!use_calc_stream) {
    VLOG(3) << "For XPU, Communication on non-calc stream has minor effect on "
               "performance and might be conflict with streams in calc_ctx, so "
               "we disable it currently.";
    use_calc_stream = true;
  }
  return Collective(
      [&](phi::distributed::BKCLCommContext* comm_context, XPUStream stream) {
        if (FLAGS_enable_bkcl_dynamic_check) {
          phi::distributed::BKCLDynamicCheck::CheckShape(
              *out_tensor,
              in_tensor,
              in_split_sizes,
              rank_,
              size_,
              comm_context->GetBKCLComm());
        }

        VLOG(3) << "[bkcl_all_to_all] "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << BKCLDTypeToString(phi::ToBKCLDataType(in_tensor.dtype()))
                << ", bkcl_comm: " << comm_context->GetBKCLComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", out_split_sizes: "
                << string::join_strings(out_split_sizes, ',')
                << ", in_split_sizes: "
                << string::join_strings(in_split_sizes, ',')
                << ", is_equal_split: " << is_equal_split
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;

        if (is_equal_split) {
          comm_context->AllToAll(out_tensor, in_tensor, stream);
        } else {
          int64_t in_row_size =
              in_dim[0] == 0 ? 0 : in_tensor.numel() / in_dim[0];
          int64_t out_row_size =
              out_dim[0] == 0 ? 0 : out_tensor->numel() / out_dim[0];

          int64_t nranks = size_;

          std::vector<int64_t> in_numel_vec(nranks);
          std::vector<int64_t> in_offset_vec(nranks);
          std::vector<int64_t> out_numel_vec(nranks);
          std::vector<int64_t> out_offset_vec(nranks);

          int64_t in_offset = 0;
          int64_t out_offset = 0;
          for (int64_t i = 0; i < nranks; i++) {
            int64_t in_numel = in_split_sizes[i] * in_row_size;
            int64_t out_numel = out_split_sizes[i] * out_row_size;

            in_numel_vec[i] = in_numel;
            in_offset_vec[i] = in_offset;
            in_offset += in_numel;

            out_numel_vec[i] = out_numel;
            out_offset_vec[i] = out_offset;
            out_offset += out_numel;
          }

          PADDLE_ENFORCE_GE(
              in_tensor.place().GetDeviceId(),
              0,
              common::errors::PreconditionNotMet(
                  "The all_to_all device id must greater or equal than 0."));
          phi::XPUPlace place = in_tensor.place();
          auto allocator = std::unique_ptr<phi::Allocator>(
              new paddle::experimental::DefaultAllocator(place));
          phi::DenseTensorMeta meta(phi::DataType::INT64, phi::DDim{nranks});

          phi::DenseTensor in_size_tensor = {allocator.get(), meta};
          phi::DenseTensor in_offset_tensor = {allocator.get(), meta};
          phi::DenseTensor out_size_tensor = {allocator.get(), meta};
          phi::DenseTensor out_offset_tensor = {allocator.get(), meta};

          memory::Copy(place,
                       in_size_tensor.data(),
                       phi::CPUPlace(),
                       in_numel_vec.data(),
                       in_size_tensor.numel() * sizeof(int64_t));

          memory::Copy(place,
                       in_offset_tensor.data(),
                       phi::CPUPlace(),
                       in_offset_vec.data(),
                       in_offset_tensor.numel() * sizeof(int64_t));

          memory::Copy(place,
                       out_size_tensor.data(),
                       phi::CPUPlace(),
                       out_numel_vec.data(),
                       out_size_tensor.numel() * sizeof(int64_t));

          memory::Copy(place,
                       out_offset_tensor.data(),
                       phi::CPUPlace(),
                       out_offset_vec.data(),
                       out_offset_tensor.numel() * sizeof(int64_t));

          comm_context->AllToAllUnequalSplit(out_tensor,
                                             in_tensor,
                                             out_size_tensor,
                                             out_offset_tensor,
                                             in_size_tensor,
                                             in_offset_tensor,
                                             stream);
        }
      },
      in_tensor,
      CommType::ALLTOALL,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllToAll(
    std::vector<phi::DenseTensor>* out_tensors,
    const std::vector<phi::DenseTensor>& in_tensors,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensors);
  CheckTensorContiguous(*out_tensors);
  CheckTensorSamePlace(in_tensors);
  CheckTensorSamePlace(*out_tensors);
  phi::distributed::CommStaticCheck::CheckDataType(*out_tensors, in_tensors);

  PADDLE_ENFORCE_EQ(
      out_tensors->size(),
      size_,
      common::errors::InvalidArgument(
          "Number of out tensors[%d] do not match the world size[%d].",
          out_tensors->size(),
          size_));
  PADDLE_ENFORCE_EQ(
      in_tensors.size(),
      size_,
      common::errors::InvalidArgument(
          "Number of in tensors[%d] do not match the world size[%d].",
          in_tensors.size(),
          size_));

  // AllToAllUnequalSplit requires allocating temporary memory and must use
  // calc_stream to ensure the correct lifecycle management of the temporary
  // tensor.
  if (!use_calc_stream) {
    VLOG(3) << "For XPU, Communication on non-calc stream has minor effect on "
               "performance and might be conflict with streams in calc_ctx, so "
               "we disable it currently.";
    use_calc_stream = true;
  }
  return Collective(
      [&](phi::distributed::BKCLCommContext* comm_context, XPUStream stream) {
        if (FLAGS_enable_bkcl_dynamic_check) {
          phi::distributed::BKCLDynamicCheck::CheckAlltoAllShape(
              *out_tensors,
              in_tensors,
              rank_,
              size_,
              comm_context->GetBKCLComm());
        }

        VLOG(3) << "[AllToAll] "
                << "sendbuff: "
                << string::join_strings(GetTensorPtrs(in_tensors), ',')
                << ", recvbuff: "
                << string::join_strings(GetTensorPtrs(*out_tensors), ',')
                << ", datatype: "
                << BKCLDTypeToString(phi::ToBKCLDataType(in_tensors[0].dtype()))
                << ", bkcl_comm: " << comm_context->GetBKCLComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", out_split_sizes: "
                << string::join_strings(GetAllToAllSplitSizes(*out_tensors),
                                        ',')
                << ", in_split_sizes: "
                << string::join_strings(GetAllToAllSplitSizes(in_tensors), ',')
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;

        int64_t nranks = size_;
        int64_t in_numel_sum = 0;
        int64_t out_numel_sum = 0;

        for (int64_t i = 0; i < nranks; i++) {
          in_numel_sum += in_tensors[i].numel();
          out_numel_sum += (*out_tensors)[i].numel();
        }

        std::vector<int64_t> in_numel_vec(nranks);
        std::vector<int64_t> in_offset_vec(nranks);
        std::vector<int64_t> out_numel_vec(nranks);
        std::vector<int64_t> out_offset_vec(nranks);

        int64_t in_offset = 0;
        int64_t out_offset = 0;
        for (int64_t i = 0; i < nranks; i++) {
          int64_t in_numel = in_tensors[i].numel();
          int64_t out_numel = (*out_tensors)[i].numel();

          in_numel_vec[i] = in_numel;
          in_offset_vec[i] = in_offset;
          in_offset += in_numel;

          out_numel_vec[i] = out_numel;
          out_offset_vec[i] = out_offset;
          out_offset += out_numel;
        }

        PADDLE_ENFORCE_GE(
            in_tensors[0].place().GetDeviceId(),
            0,
            common::errors::PreconditionNotMet(
                "The all_to_all device id must greater or equal than 0."));
        phi::XPUPlace place = in_tensors[0].place();
        auto allocator = std::unique_ptr<phi::Allocator>(
            new paddle::experimental::DefaultAllocator(place));

        phi::DenseTensorMeta concated_in_tensor_meta(in_tensors[0].dtype(),
                                                     phi::DDim{in_numel_sum});
        phi::DenseTensorMeta concated_out_tensor_meta((*out_tensors)[0].dtype(),
                                                      phi::DDim{out_numel_sum});
        phi::DenseTensorMeta split_meta(phi::DataType::INT64,
                                        phi::DDim{nranks});

        phi::DenseTensor concated_in_tensor = {allocator.get(),
                                               concated_in_tensor_meta};
        phi::DenseTensor concated_out_tensor = {allocator.get(),
                                                concated_out_tensor_meta};
        phi::DenseTensor in_size_tensor = {allocator.get(), split_meta};
        phi::DenseTensor in_offset_tensor = {allocator.get(), split_meta};
        phi::DenseTensor out_size_tensor = {allocator.get(), split_meta};
        phi::DenseTensor out_offset_tensor = {allocator.get(), split_meta};

        if (in_numel_sum > 0) {
          ConcatTensorByNumel(*GetDeviceContext(place, use_calc_stream),
                              in_tensors,
                              &concated_in_tensor);
        }

        memory::Copy(place,
                     in_size_tensor.data(),
                     phi::CPUPlace(),
                     in_numel_vec.data(),
                     in_size_tensor.numel() * sizeof(int64_t));

        memory::Copy(place,
                     in_offset_tensor.data(),
                     phi::CPUPlace(),
                     in_offset_vec.data(),
                     in_offset_tensor.numel() * sizeof(int64_t));

        memory::Copy(place,
                     out_size_tensor.data(),
                     phi::CPUPlace(),
                     out_numel_vec.data(),
                     out_size_tensor.numel() * sizeof(int64_t));

        memory::Copy(place,
                     out_offset_tensor.data(),
                     phi::CPUPlace(),
                     out_offset_vec.data(),
                     out_offset_tensor.numel() * sizeof(int64_t));

        comm_context->AllToAllUnequalSplit(&concated_out_tensor,
                                           concated_in_tensor,
                                           out_size_tensor,
                                           out_offset_tensor,
                                           in_size_tensor,
                                           in_offset_tensor,
                                           stream);

        if (out_numel_sum > 0) {
          SplitTensorByNumel(*GetDeviceContext(place, use_calc_stream),
                             concated_out_tensor,
                             out_tensors);
        }
      },
      in_tensors,
      CommType::ALLTOALL,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Broadcast(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const BroadcastOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return Collective(
      [&](phi::distributed::BKCLCommContext* comm_context, XPUStream stream) {
        int root = opts.source_rank + opts.source_root;

        VLOG(3) << "bkcl_broadcast "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << BKCLDTypeToString(phi::ToBKCLDataType(in_tensor.dtype()))
                << ", root: " << root
                << ", bkcl_comm: " << comm_context->GetBKCLComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        comm_context->Broadcast(out_tensor, in_tensor, root, stream);
      },
      in_tensor,
      CommType::BROADCAST,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::AllGather(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    int64_t offset,
    int64_t numel,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);

  const phi::DenseTensor& in_tensor_maybe_partial =
      numel > 0 ? GetPartialTensor(in_tensor, offset, numel) : in_tensor;
  phi::distributed::CommStaticCheck::GatherLikeShape(*out_tensor,
                                                     in_tensor_maybe_partial,
                                                     /*dst_rank*/ rank_,
                                                     /*cur_rank*/ rank_,
                                                     size_,
                                                     phi::AllocationType::XPU);
  return Collective(
      [&](phi::distributed::BKCLCommContext* comm_context, XPUStream stream) {
        VLOG(3) << "bkcl_all_gather "
                << "sendbuff: " << in_tensor_maybe_partial.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor_maybe_partial.numel()
                << ", datatype: "
                << BKCLDTypeToString(phi::ToBKCLDataType(in_tensor.dtype()))
                << ", bkcl_comm: " << comm_context->GetBKCLComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", offset: " << offset
                << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;

        comm_context->AllGather(out_tensor, in_tensor_maybe_partial, stream);
      },
      in_tensor_maybe_partial,
      CommType::ALLGATHER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Reduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return Collective(
      [&](phi::distributed::BKCLCommContext* comm_context, XPUStream stream) {
        VLOG(3) << "bkcl_reduce "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << BKCLDTypeToString(phi::ToBKCLDataType(in_tensor.dtype()))
                << ", redop: "
                << BKCLRedTypeToString(ToBKCLRedType(opts.reduce_op))
                << ", root: " << opts.root_rank
                << ", bkcl_comm: " << comm_context->GetBKCLComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        comm_context->Reduce(out_tensor,
                             in_tensor,
                             ToBKCLRedType(opts.reduce_op),
                             opts.root_rank,
                             stream);
      },
      in_tensor,
      CommType::REDUCE,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::ReduceScatter(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    const ReduceScatterOptions& opts,
    bool sync_op,
    bool use_calc_stream) {
  CheckTensorContiguous(in_tensor);
  CheckTensorContiguous(*out_tensor);

  return Collective(
      [&](phi::distributed::BKCLCommContext* comm_context, XPUStream stream) {
        VLOG(3) << "bkcl_reduce_scatter "
                << "sendbuff: " << in_tensor.data()
                << ", recvbuff: " << out_tensor->data()
                << ", count: " << in_tensor.numel() << ", datatype: "
                << BKCLDTypeToString(phi::ToBKCLDataType(in_tensor.dtype()))
                << ", redop: "
                << BKCLRedTypeToString(ToBKCLRedType(opts.reduce_op))
                << ", bkcl_comm: " << comm_context->GetBKCLComm()
                << ", stream: " << stream << ", rank_in_group: " << rank_
                << ", nranks: " << size_ << ", sync_op: " << sync_op
                << ", use_calc_stream: " << use_calc_stream;
        comm_context->ReduceScatter(
            out_tensor, in_tensor, ToBKCLRedType(opts.reduce_op), stream);
      },
      in_tensor,
      CommType::REDUCE_SCATTER,
      sync_op,
      use_calc_stream);
}

std::shared_ptr<ProcessGroup::Task> ProcessGroupBKCL::Barrier(
    const BarrierOptions& opts) {
  PADDLE_ENFORCE_GE(opts.device_id,
                    0,
                    common::errors::PreconditionNotMet(
                        "The barrier device id must greater or equal than 0."));
  phi::XPUPlace place(opts.device_id);
  auto allocator = std::unique_ptr<phi::Allocator>(
      new paddle::experimental::DefaultAllocator(place));
  phi::DenseTensorMeta meta(phi::DataType::FLOAT32, phi::DDim{1});
  phi::DenseTensor barrier_tensor{allocator.get(), meta};

  auto task = AllReduce(&barrier_tensor,
                        barrier_tensor,
                        {},
                        /*sync_op*/ true,
                        /*use_calc_stream*/ false);
  auto bkcl_task = dynamic_cast<BKCLTask*>(task.get());
  bkcl_task->barrier_ = true;
  return task;
}

phi::DeviceContext* ProcessGroupBKCL::GetDeviceContext(
    const Place& place) const {
  return GetDeviceContext(place, /*use_calc_stream*/ false);
}

phi::DeviceContext* ProcessGroupBKCL::GetDeviceContext(
    const Place& place, bool use_calc_stream) const {
  if (!use_calc_stream) {
    VLOG(3) << "For XPU, Communication on non-calc stream has minor effect on "
               "performance and might be conflict with streams in calc_ctx, so "
               "we disable it currently.";
    use_calc_stream = true;
  }
  const std::string& key = GetKeyFromPlace(place);
  if (use_calc_stream) {
    const auto& iter = place_to_calc_ctx_.find(key);
    return iter->second;
  } else {
    const auto& iter = place_to_comm_ctx_.find(key);
    PADDLE_ENFORCE_NE(iter,
                      place_to_comm_ctx_.end(),
                      common::errors::InvalidArgument(
                          "Cannot find device context in process group."));
    return iter->second.get();
  }
}

std::shared_ptr<ProcessGroupBKCL> ProcessGroupBKCL::CreateProcessGroupBKCL(
    const std::shared_ptr<phi::distributed::Store>& store,
    int rank,
    int size,
    int gid) {
  auto process_group =
      std::make_shared<ProcessGroupBKCL>(store, rank, size, gid);
  ProcessGroupIdMap::GetInstance().emplace(gid, process_group);
  return process_group;
}

phi::distributed::BKCLCommContext* ProcessGroupBKCL::GetOrCreateCommContext(
    const Place& place, CommType comm_type) {
  const auto& key = GetKeyFromPlace(place);
  if (place_to_comm_ctx_.find(key) == place_to_comm_ctx_.end()) {
    CreateBKCLEnvCache(place, key);
  }
  return GetCommContext();
}

phi::distributed::BKCLCommContext* ProcessGroupBKCL::GetCommContext() {
  const auto& comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  auto comm_context = static_cast<phi::distributed::BKCLCommContext*>(
      comm_context_manager.Get(std::to_string(this->gid_)));
  PADDLE_ENFORCE_NE(comm_context,
                    nullptr,
                    common::errors::Unavailable("BKCLCommContext is nullptr"));
  return comm_context;
}

void ProcessGroupBKCL::StartCoalescing() {
  PADDLE_ENFORCE_EQ(is_coalescing_,
                    false,
                    common::errors::PreconditionNotMet(
                        "Coalescing is on, please call EndCoalesce."));
  is_coalescing_ = true;
  GroupStart();
}

void ProcessGroupBKCL::EndCoalescing(
    std::optional<std::vector<std::shared_ptr<ProcessGroup::Task>>> tasks_opt) {
  GroupEnd();

  // NOTE(shenliang03): If using calculate stream, no need to record stream and
  // update task.
  if (!tasks_opt.has_value() | coalescing_place_keys_.empty()) {
    is_coalescing_ = false;
    return;
  }

  auto& tasks = tasks_opt.value();

  PADDLE_ENFORCE_EQ(
      tasks.size(),
      coalescing_place_keys_.size(),
      common::errors::PreconditionNotMet(
          "Number of tasks[%d] do not match number of collectives[%d].",
          tasks.size(),
          coalescing_place_keys_.size()));

  for (size_t i = 0; i < tasks.size(); ++i) {
    auto* task = static_cast<ProcessGroupBKCL::BKCLTask*>(tasks[i].get());
    const auto& key = coalescing_place_keys_[i];
    const auto& comm_ctx = place_to_comm_ctx_.at(key);
    task->comm_event_->Record(*comm_ctx.get());
  }

  is_coalescing_ = false;
  coalescing_place_keys_.clear();
}

}  //  namespace distributed
}  //  namespace paddle
