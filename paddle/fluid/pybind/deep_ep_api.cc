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

#include <Python.h>
#include "pybind11/stl.h"

#ifdef PADDLE_WITH_DEEP_EP
#include "paddle/fluid/distributed/collective/deep_ep/deep_ep.hpp"
#endif
#include "paddle/fluid/pybind/deep_ep_api.h"
#include "paddle/utils/pybind.h"

namespace py = pybind11;

#ifdef PADDLE_WITH_DEEP_EP
namespace deep_ep {
deep_ep::detail::Tensor ConvertPyObjectToDetailTensor(
    const py::handle& tensor) {
  return ConvertPaddleTensorToDetailTensor(
      paddle::pybind::CastPyArg2Tensor(tensor.ptr(), 0));
}

py::object ConvertDetailTensorToPyObject(
    const deep_ep::detail::Tensor& tensor) {
  PyObject* py_obj =
      paddle::pybind::ToPyObject(ConvertDetailTensorToPaddleTensor(tensor));
  return py::reinterpret_borrow<py::object>(py_obj);
}

std::optional<deep_ep::detail::Tensor> ConvertOptionalPyObjectToDetailTensor(
    const std::optional<py::handle>& tensor) {
  std::optional<deep_ep::detail::Tensor> res;
  if (tensor.has_value()) {
    res = ConvertPyObjectToDetailTensor(tensor.value());
  }
  return res;
}

std::optional<py::object> ConvertOptionalDetailTensorToPyObject(
    const std::optional<deep_ep::detail::Tensor>& tensor) {
  std::optional<py::object> res;
  if (tensor.has_value()) {
    res = ConvertDetailTensorToPyObject(tensor.value());
  }
  return res;
}

std::tuple<py::object,
           std::optional<py::object>,
           std::optional<py::object>,
           std::optional<py::object>,
           std::vector<int>,
           py::object,
           py::object,
           std::optional<py::object>,
           py::object,
           std::optional<py::object>,
           py::object,
           std::optional<py::object>,
           std::optional<py::object>,
           std::optional<py::object>,
           std::optional<EventHandle>>
internode_dispatch_api(
    deep_ep::Buffer& self,  // NOLINT
    const py::handle& x,
    const std::optional<py::handle>& x_scales,
    const std::optional<py::handle>& topk_idx,
    const std::optional<py::handle>& topk_weights,
    const std::optional<py::handle>& num_tokens_per_rank,
    const std::optional<py::handle>& num_tokens_per_rdma_rank,
    const py::handle& is_token_in_rank,
    const std::optional<py::handle>& num_tokens_per_expert,
    int cached_num_recv_tokens,
    int cached_num_rdma_recv_tokens,
    const std::optional<py::handle>& cached_rdma_channel_prefix_matrix,
    const std::optional<py::handle>& cached_recv_rdma_rank_prefix_sum,
    const std::optional<py::handle>& cached_gbl_channel_prefix_matrix,
    const std::optional<py::handle>& cached_recv_gbl_rank_prefix_sum,
    int expert_alignment,
    const Config& config,
    std::optional<EventHandle>& previous_event,  // NOLINT
    bool async,
    bool allocate_on_comm_stream) {
#ifdef PADDLE_WITH_NVSHMEM
  const auto& x_ = ConvertPyObjectToDetailTensor(x);
  std::optional<deep_ep::detail::Tensor> x_scales_ =
      ConvertOptionalPyObjectToDetailTensor(x_scales);

  std::optional<deep_ep::detail::Tensor> topk_idx_ =
      ConvertOptionalPyObjectToDetailTensor(topk_idx);
  std::optional<deep_ep::detail::Tensor> topk_weights_ =
      ConvertOptionalPyObjectToDetailTensor(topk_weights);
  std::optional<deep_ep::detail::Tensor> num_tokens_per_rank_ =
      ConvertOptionalPyObjectToDetailTensor(num_tokens_per_rank);
  std::optional<deep_ep::detail::Tensor> num_tokens_per_rdma_rank_ =
      ConvertOptionalPyObjectToDetailTensor(num_tokens_per_rdma_rank);

  const auto& is_token_in_rank_ =
      ConvertPyObjectToDetailTensor(is_token_in_rank);
  std::optional<deep_ep::detail::Tensor> num_tokens_per_expert_ =
      ConvertOptionalPyObjectToDetailTensor(num_tokens_per_expert);

  std::optional<deep_ep::detail::Tensor> cached_rdma_channel_prefix_matrix_ =
      ConvertOptionalPyObjectToDetailTensor(cached_rdma_channel_prefix_matrix);
  std::optional<deep_ep::detail::Tensor> cached_recv_rdma_rank_prefix_sum_ =
      ConvertOptionalPyObjectToDetailTensor(cached_recv_rdma_rank_prefix_sum);
  std::optional<deep_ep::detail::Tensor> cached_gbl_channel_prefix_matrix_ =
      ConvertOptionalPyObjectToDetailTensor(cached_gbl_channel_prefix_matrix);
  std::optional<deep_ep::detail::Tensor> cached_recv_gbl_rank_prefix_sum_ =
      ConvertOptionalPyObjectToDetailTensor(cached_recv_gbl_rank_prefix_sum);

  auto res = self.internode_dispatch(x_,
                                     x_scales_,
                                     topk_idx_,
                                     topk_weights_,
                                     num_tokens_per_rank_,
                                     num_tokens_per_rdma_rank_,
                                     is_token_in_rank_,
                                     num_tokens_per_expert_,
                                     cached_num_recv_tokens,
                                     cached_num_rdma_recv_tokens,
                                     cached_rdma_channel_prefix_matrix_,
                                     cached_recv_rdma_rank_prefix_sum_,
                                     cached_gbl_channel_prefix_matrix_,
                                     cached_recv_gbl_rank_prefix_sum_,
                                     expert_alignment,
                                     config,
                                     previous_event,
                                     async,
                                     allocate_on_comm_stream);

  auto recv_x_ = ConvertDetailTensorToPyObject(std::get<0>(res));
  std::optional<py::object> recv_x_scales_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<1>(res));

  std::optional<py::object> recv_topk_idx_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<2>(res));
  std::optional<py::object> recv_topk_weights_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<3>(res));

  const auto& num_recv_tokens_per_expert_list = std::get<4>(res);

  auto rdma_channel_prefix_matrix_ =
      ConvertDetailTensorToPyObject(std::get<5>(res));

  auto gbl_channel_prefix_matrix_ =
      ConvertDetailTensorToPyObject(std::get<6>(res));

  std::optional<py::object> recv_rdma_channel_prefix_matrix_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<7>(res));
  auto recv_rdma_rank_prefix_sum_ =
      ConvertDetailTensorToPyObject(std::get<8>(res));

  std::optional<py::object> recv_gbl_channel_prefix_matrix_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<9>(res));
  auto recv_gbl_rank_prefix_sum_ =
      ConvertDetailTensorToPyObject(std::get<10>(res));

  std::optional<py::object> recv_src_meta_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<11>(res));

  std::optional<py::object> send_rdma_head_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<12>(res));
  std::optional<py::object> send_nvl_head_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<13>(res));

  const auto& event = std::get<14>(res);

  return {recv_x_,
          recv_x_scales_,
          recv_topk_idx_,
          recv_topk_weights_,
          num_recv_tokens_per_expert_list,
          rdma_channel_prefix_matrix_,
          gbl_channel_prefix_matrix_,
          recv_rdma_channel_prefix_matrix_,
          recv_rdma_rank_prefix_sum_,
          recv_gbl_channel_prefix_matrix_,
          recv_gbl_rank_prefix_sum_,
          recv_src_meta_,
          send_rdma_head_,
          send_nvl_head_,
          event};
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
  return {};
#endif
}

std::tuple<py::object, std::optional<py::object>, std::optional<EventHandle>>
internode_combine_api(deep_ep::Buffer& self,  // NOLINT
                      const py::handle& x,
                      const std::optional<py::handle>& topk_weights,
                      const py::handle& src_meta,
                      const py::handle& is_combined_token_in_rank,
                      const py::handle& rdma_channel_prefix_matrix,
                      const py::handle& rdma_rank_prefix_sum,
                      const py::handle& gbl_channel_prefix_matrix,
                      const py::handle& combined_rdma_head,
                      const py::handle& combined_nvl_head,
                      const Config& config,
                      std::optional<EventHandle>& previous_event,  // NOLINT
                      bool async,
                      bool allocate_on_comm_stream) {
#ifdef PADDLE_WITH_NVSHMEM
  const auto& x_ = ConvertPyObjectToDetailTensor(x);

  std::optional<deep_ep::detail::Tensor> topk_weights_ =
      ConvertOptionalPyObjectToDetailTensor(topk_weights);

  const auto& src_meta_ = ConvertPyObjectToDetailTensor(src_meta);
  const auto& is_combined_token_in_rank_ =
      ConvertPyObjectToDetailTensor(is_combined_token_in_rank);

  const auto& rdma_channel_prefix_matrix_ =
      ConvertPyObjectToDetailTensor(rdma_channel_prefix_matrix);
  const auto& rdma_rank_prefix_sum_ =
      ConvertPyObjectToDetailTensor(rdma_rank_prefix_sum);
  const auto& gbl_channel_prefix_matrix_ =
      ConvertPyObjectToDetailTensor(gbl_channel_prefix_matrix);

  const auto& combined_rdma_head_ =
      ConvertPyObjectToDetailTensor(combined_rdma_head);
  const auto& combined_nvl_head_ =
      ConvertPyObjectToDetailTensor(combined_nvl_head);

  auto res = self.internode_combine(x_,
                                    topk_weights_,
                                    src_meta_,
                                    is_combined_token_in_rank_,
                                    rdma_channel_prefix_matrix_,
                                    rdma_rank_prefix_sum_,
                                    gbl_channel_prefix_matrix_,
                                    combined_rdma_head_,
                                    combined_nvl_head_,
                                    config,
                                    previous_event,
                                    async,
                                    allocate_on_comm_stream);

  auto combined_x_ = ConvertDetailTensorToPyObject(std::get<0>(res));
  std::optional<py::object> combined_topk_weights_ =
      ConvertOptionalDetailTensorToPaddleTensor(std::get<1>(res));

  const auto& event = std::get<2>(res);

  return {combined_x_, combined_topk_weights_, event};
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
  return {};
#endif
}

std::tuple<py::object,
           py::object,
           py::object,
           py::object,
           py::object,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
low_latency_dispatch_api(deep_ep::Buffer& self,  // NOLINT
                         const py::handle& x,
                         const py::handle& topk_idx,
                         int num_max_dispatch_tokens_per_rank,
                         int num_experts,
                         bool async,
                         bool return_recv_hook) {
#ifdef PADDLE_WITH_NVSHMEM
  const auto& x_ = ConvertPyObjectToDetailTensor(x);
  const auto& topk_idx_ = ConvertPyObjectToDetailTensor(topk_idx);

  auto res = self.low_latency_dispatch(x_,
                                       topk_idx_,
                                       num_max_dispatch_tokens_per_rank,
                                       num_experts,
                                       async,
                                       return_recv_hook);

  auto packed_recv_x_ = ConvertDetailTensorToPyObject(std::get<0>(res));
  auto packed_recv_x_scales_ = ConvertDetailTensorToPyObject(std::get<1>(res));
  auto packed_recv_count_ = ConvertDetailTensorToPyObject(std::get<2>(res));
  auto packed_recv_src_info_ = ConvertDetailTensorToPyObject(std::get<3>(res));
  auto packed_recv_layout_range_ =
      ConvertDetailTensorToPyObject(std::get<4>(res));

  const auto& event = std::get<5>(res);
  auto recv_hook = std::get<6>(res);

  return {packed_recv_x_,
          packed_recv_x_scales_,
          packed_recv_count_,
          packed_recv_src_info_,
          packed_recv_layout_range_,
          event,
          recv_hook};
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
  return {};
#endif
}

std::tuple<py::object,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
low_latency_combine_api(deep_ep::Buffer& self,  // NOLINT
                        const py::handle& x,
                        const py::handle& topk_idx,
                        const py::handle& topk_weights,
                        const py::handle& src_info,
                        const py::handle& layout_range,
                        int num_max_dispatch_tokens_per_rank,
                        int num_experts,
                        bool async,
                        bool return_recv_hook) {
#ifdef PADDLE_WITH_NVSHMEM
  const auto& x_ = ConvertPyObjectToDetailTensor(x);
  const auto& topk_idx_ = ConvertPyObjectToDetailTensor(topk_idx);
  const auto& topk_weights_ = ConvertPyObjectToDetailTensor(topk_weights);
  const auto& src_info_ = ConvertPyObjectToDetailTensor(src_info);
  const auto& layout_range_ = ConvertPyObjectToDetailTensor(layout_range);

  auto res = self.low_latency_combine(x_,
                                      topk_idx_,
                                      topk_weights_,
                                      src_info_,
                                      layout_range_,
                                      num_max_dispatch_tokens_per_rank,
                                      num_experts,
                                      async,
                                      return_recv_hook);

  auto combined_x_ = ConvertDetailTensorToPyObject(std::get<0>(res));
  const auto& event = std::get<1>(res);
  auto recv_hook = std::get<2>(res);

  return {combined_x_, event, recv_hook};
#else
  LOG(ERROR) << "NVSHMEM is not enabled. You can enable it by setting cmake "
                "option WITH_NVSHMEM=ON.";
  return {};
#endif
}

std::tuple<py::object,
           std::optional<py::object>,
           py::object,
           py::object,
           std::optional<EventHandle>>
get_dispatch_layout_api(deep_ep::Buffer& self,  // NOLINT
                        const py::handle& topk_idx,
                        int num_experts,
                        std::optional<EventHandle>& previous_event,  // NOLINT
                        bool async,
                        bool allocate_on_comm_stream) {
  const auto& topk_idx_ = ConvertPyObjectToDetailTensor(topk_idx);
  auto res = self.get_dispatch_layout(
      topk_idx_, num_experts, previous_event, async, allocate_on_comm_stream);
  const auto& num_tokens_per_rank = std::get<0>(res);
  const auto& num_tokens_per_rdma_rank = std::get<1>(res);
  const auto& num_tokens_per_expert = std::get<2>(res);
  const auto& is_token_in_rank = std::get<3>(res);
  const auto& event = std::get<4>(res);
  auto num_tokens_per_rank_ =
      ConvertDetailTensorToPyObject(num_tokens_per_rank);
  std::optional<py::object> num_tokens_per_rdma_rank_ = std::nullopt;
  if (num_tokens_per_rdma_rank.has_value()) {
    num_tokens_per_rdma_rank_ =
        ConvertDetailTensorToPyObject(num_tokens_per_rdma_rank.value());
  }
  auto num_tokens_per_expert_ =
      ConvertDetailTensorToPyObject(num_tokens_per_expert);
  auto is_token_in_rank_ = ConvertDetailTensorToPyObject(is_token_in_rank);
  return {num_tokens_per_rank_,
          num_tokens_per_rdma_rank_,
          num_tokens_per_expert_,
          is_token_in_rank_,
          event};
}

std::tuple<py::object,
           std::optional<py::object>,
           std::optional<py::object>,
           std::optional<py::object>,
           std::vector<int>,
           py::object,
           py::object,
           py::object,
           py::object,
           py::object,
           std::optional<EventHandle>>
intranode_dispatch_api(
    deep_ep::Buffer& self,  // NOLINT
    const py::handle& x,
    const std::optional<py::handle>& x_scales,
    const std::optional<py::handle>& topk_idx,
    const std::optional<py::handle>& topk_weights,
    const std::optional<py::handle>& num_tokens_per_rank,
    const py::handle& is_token_in_rank,
    const std::optional<py::handle>& num_tokens_per_expert,
    int cached_num_recv_tokens,
    const std::optional<py::handle>& cached_rank_prefix_matrix,
    const std::optional<py::handle>& cached_channel_prefix_matrix,
    int expert_alignment,
    const Config& config,
    std::optional<EventHandle>& previous_event,  // NOLINT
    bool async,
    bool allocate_on_comm_stream) {
  const auto& x_ = ConvertPyObjectToDetailTensor(x);
  std::optional<deep_ep::detail::Tensor> x_scales_;
  if (x_scales.has_value()) {
    x_scales_ = ConvertPyObjectToDetailTensor(x_scales.value());
  }
  std::optional<deep_ep::detail::Tensor> topk_idx_;
  if (topk_idx.has_value()) {
    topk_idx_ = ConvertPyObjectToDetailTensor(topk_idx.value());
  }
  std::optional<deep_ep::detail::Tensor> topk_weights_;
  if (topk_weights.has_value()) {
    topk_weights_ = ConvertPyObjectToDetailTensor(topk_weights.value());
  }
  std::optional<deep_ep::detail::Tensor> num_tokens_per_rank_;
  if (num_tokens_per_rank.has_value()) {
    num_tokens_per_rank_ =
        ConvertPyObjectToDetailTensor(num_tokens_per_rank.value());
  }
  const auto& is_token_in_rank_ =
      ConvertPyObjectToDetailTensor(is_token_in_rank);
  std::optional<deep_ep::detail::Tensor> num_tokens_per_expert_;
  if (num_tokens_per_expert.has_value()) {
    num_tokens_per_expert_ =
        ConvertPyObjectToDetailTensor(num_tokens_per_expert.value());
  }
  std::optional<deep_ep::detail::Tensor> cached_rank_prefix_matrix_;
  if (cached_rank_prefix_matrix.has_value()) {
    cached_rank_prefix_matrix_ =
        ConvertPyObjectToDetailTensor(cached_rank_prefix_matrix.value());
  }
  std::optional<deep_ep::detail::Tensor> cached_channel_prefix_matrix_;
  if (cached_channel_prefix_matrix.has_value()) {
    cached_channel_prefix_matrix_ =
        ConvertPyObjectToDetailTensor(cached_channel_prefix_matrix.value());
  }

  auto res = self.intranode_dispatch(x_,
                                     x_scales_,
                                     topk_idx_,
                                     topk_weights_,
                                     num_tokens_per_rank_,
                                     is_token_in_rank_,
                                     num_tokens_per_expert_,
                                     cached_num_recv_tokens,
                                     cached_rank_prefix_matrix_,
                                     cached_channel_prefix_matrix_,
                                     expert_alignment,
                                     config,
                                     previous_event,
                                     async,
                                     allocate_on_comm_stream);

  const auto& recv_x = std::get<0>(res);
  const auto& recv_x_scales = std::get<1>(res);
  const auto& recv_topk_idx = std::get<2>(res);
  const auto& recv_topk_weights = std::get<3>(res);
  const auto& num_recv_tokens_per_expert_list = std::get<4>(res);
  const auto& rank_prefix_matrix = std::get<5>(res);
  const auto& channel_prefix_matrix = std::get<6>(res);
  const auto& recv_channel_prefix_matrix = std::get<7>(res);
  const auto& recv_src_idx = std::get<8>(res);
  const auto& send_head = std::get<9>(res);
  const auto& event = std::get<10>(res);

  auto recv_x_ = ConvertDetailTensorToPyObject(recv_x);
  std::optional<py::object> recv_x_scales_;
  if (recv_x_scales.has_value()) {
    recv_x_scales_ = ConvertDetailTensorToPyObject(recv_x_scales.value());
  }
  std::optional<py::object> recv_topk_idx_;
  if (recv_topk_idx.has_value()) {
    recv_topk_idx_ = ConvertDetailTensorToPyObject(recv_topk_idx.value());
  }
  std::optional<py::object> recv_topk_weights_;
  if (recv_topk_weights.has_value()) {
    recv_topk_weights_ =
        ConvertDetailTensorToPyObject(recv_topk_weights.value());
  }
  auto rank_prefix_matrix_ = ConvertDetailTensorToPyObject(rank_prefix_matrix);
  auto channel_prefix_matrix_ =
      ConvertDetailTensorToPyObject(channel_prefix_matrix);
  auto recv_channel_prefix_matrix_ =
      ConvertDetailTensorToPyObject(recv_channel_prefix_matrix);
  auto recv_src_idx_ = ConvertDetailTensorToPyObject(recv_src_idx);
  auto send_head_ = ConvertDetailTensorToPyObject(send_head);
  return {recv_x_,
          recv_x_scales_,
          recv_topk_idx_,
          recv_topk_weights_,
          num_recv_tokens_per_expert_list,
          rank_prefix_matrix_,
          channel_prefix_matrix_,
          recv_channel_prefix_matrix_,
          recv_src_idx_,
          send_head_,
          event};
}

std::tuple<py::object, std::optional<py::object>, std::optional<EventHandle>>
intranode_combine_api(deep_ep::Buffer& self,  // NOLINT
                      const py::handle& x,
                      const std::optional<py::handle>& topk_weights,
                      const py::handle& src_idx,
                      const py::handle& rank_prefix_matrix,
                      const py::handle& channel_prefix_matrix,
                      const py::handle& send_head,
                      const Config& config,
                      std::optional<EventHandle>& previous_event,  // NOLINT
                      bool async,
                      bool allocate_on_comm_stream) {
  const auto& x_ = ConvertPyObjectToDetailTensor(x);
  std::optional<deep_ep::detail::Tensor> topk_weights_;
  if (topk_weights.has_value()) {
    topk_weights_ = ConvertPyObjectToDetailTensor(topk_weights.value());
  }
  const auto& src_idx_ = ConvertPyObjectToDetailTensor(src_idx);
  const auto& rank_prefix_matrix_ =
      ConvertPyObjectToDetailTensor(rank_prefix_matrix);
  const auto& channel_prefix_matrix_ =
      ConvertPyObjectToDetailTensor(channel_prefix_matrix);
  const auto& send_head_ = ConvertPyObjectToDetailTensor(send_head);

  auto res = self.intranode_combine(x_,
                                    topk_weights_,
                                    src_idx_,
                                    rank_prefix_matrix_,
                                    channel_prefix_matrix_,
                                    send_head_,
                                    config,
                                    previous_event,
                                    async,
                                    allocate_on_comm_stream);

  const auto& recv_x = std::get<0>(res);
  const auto& recv_topk_weights = std::get<1>(res);
  const auto& event = std::get<2>(res);

  auto recv_x_ = ConvertDetailTensorToPyObject(recv_x);
  std::optional<py::object> recv_topk_weights_;
  if (recv_topk_weights.has_value()) {
    recv_topk_weights_ =
        ConvertDetailTensorToPyObject(recv_topk_weights.value());
  }
  auto event_ = event;
  return {recv_x_, recv_topk_weights_, event_};
}
}  // namespace deep_ep
#endif

namespace paddle::pybind {

void BindDeepEPApi(pybind11::module* m) {
#ifdef PADDLE_WITH_DEEP_EP
  pybind11::class_<deep_ep::Config>(*m, "Config")
      .def(pybind11::init<int, int, int, int, int>(),
           py::arg("num_sms") = 20,
           py::arg("num_max_nvl_chunked_send_tokens") = 6,
           py::arg("num_max_nvl_chunked_recv_tokens") = 256,
           py::arg("num_max_rdma_chunked_send_tokens") = 6,
           py::arg("num_max_rdma_chunked_recv_tokens") = 256)
      .def("get_nvl_buffer_size_hint",
           &deep_ep::Config::get_nvl_buffer_size_hint)
      .def("get_rdma_buffer_size_hint",
           &deep_ep::Config::get_rdma_buffer_size_hint);
  m->def("get_low_latency_rdma_size_hint",
         &deep_ep::get_low_latency_rdma_size_hint);

  pybind11::class_<deep_ep::EventHandle>(*m, "EventHandle")
      .def(pybind11::init<>())
      .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

  pybind11::class_<deep_ep::Buffer>(*m, "Buffer")
      .def(pybind11::init<int, int, int64_t, int64_t, bool, int>())
      .def("is_available", &deep_ep::Buffer::is_available)
      .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
      .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
      .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
      .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
      .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
      .def("get_local_nvshmem_unique_id",
           &deep_ep::Buffer::get_local_nvshmem_unique_id)
      .def("sync", &deep_ep::Buffer::sync)
      .def("get_dispatch_layout", &deep_ep::get_dispatch_layout_api)
      .def("intranode_dispatch", &deep_ep::intranode_dispatch_api)
      .def("intranode_combine", &deep_ep::intranode_combine_api)
      .def("internode_dispatch", &deep_ep::internode_dispatch_api)
      .def("internode_combine", &deep_ep::internode_combine_api)
      .def("clean_low_latency_buffer",
           &deep_ep::Buffer::clean_low_latency_buffer)
      .def("low_latency_dispatch", &deep_ep::low_latency_dispatch_api)
      .def("low_latency_combine", &deep_ep::low_latency_combine_api);
#endif
}

}  // namespace paddle::pybind
