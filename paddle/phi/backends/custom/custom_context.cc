/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/phi/backends/custom/custom_context.h"

#include "paddle/common/exception.h"
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/backends/stream.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/backends/device_manager.h"

namespace phi {

struct CustomContext::Impl {
  explicit Impl(const CustomPlace& place) : place_(place) {}

  ~Impl() {
    phi::DeviceGuard guard(place_);
    if (owned_) {
      DeviceManager::DestoryEigenDevice(place_, eigen_device_);
    }
    if (stream_owned_ && stream_) {
      stream_->Destroy();
    }
  }

  void Init() {
    owned_ = true;
    phi::DeviceGuard guard(place_);
    stream_.reset(new phi::stream::Stream());
    stream_->Init(place_);
    compute_capability_ = DeviceManager::GetComputeCapability(place_);
    runtime_version_ = DeviceManager::GetRuntimeVersion(place_);
    driver_version_ = DeviceManager::GetDriverVersion(place_);
    multi_process_ = DeviceManager::GetMultiProcessors(place_);
    max_threads_per_mp_ = DeviceManager::GetMaxThreadsPerMultiProcessor(place_);
    max_threads_per_block_ = DeviceManager::GetMaxThreadsPerBlock(place_);
    max_grid_dim_size_ = DeviceManager::GetMaxGridDimSize(place_);
    eigen_device_ = DeviceManager::InitEigenDevice(place_);
  }

  const Place& GetPlace() const { return place_; }

  phi::stream::stream_t stream() const {
    return reinterpret_cast<phi::stream::stream_t>(stream_->raw_stream());
  }

  std::shared_ptr<phi::stream::Stream> GetStream() const { return stream_; }

  void SetStream(std::shared_ptr<phi::stream::Stream> stream) {
    stream_owned_ = true;
    stream_ = stream;
  }

  void SetEigenDevice(Eigen::GpuDevice* device) { eigen_device_ = device; }

  void SetEigenDevice(std::function<Eigen::GpuDevice*()>&& creator) {
    eigen_device_creator_ = std::move(creator);
  }

  Eigen::GpuDevice* eigen_device() {
    std::call_once(flag_eigen_device_, [&]() {
      if (!eigen_device_) {
        if (!eigen_device_creator_) {
          // use default initial
          eigen_device_ = DeviceManager::InitEigenDevice(place_);
        } else {
          eigen_device_ = eigen_device_creator_();
        }
      }
    });
    PADDLE_ENFORCE_NOT_NULL(
        eigen_device_,
        common::errors::InvalidArgument(
            "The custom eigen_device is nullptr. It must not be null."));
    return eigen_device_;
  }

  void Wait() const { stream_->Wait(); }

  phi::ccl::CCLComm xccl_comm() const { return comm_; }

  void set_xccl_comm(phi::ccl::CCLComm comm) { comm_ = comm; }

  Place place_;

  std::shared_ptr<phi::stream::Stream> stream_;

  phi::ccl::CCLComm comm_;

  //////////////////////
  bool owned_{false};
  bool stream_owned_{false};
  int compute_capability_ = 0;
  int runtime_version_ = 0;
  int driver_version_ = 0;
  int multi_process_ = 0;
  int max_threads_per_mp_ = 0;
  int max_threads_per_block_ = 0;
  std::array<unsigned int, 3> max_grid_dim_size_;

  Eigen::GpuDevice* eigen_device_{nullptr};
  std::function<Eigen::GpuDevice*()> eigen_device_creator_{nullptr};
  std::once_flag flag_eigen_device_;

  // std::unique_ptr<void*> eigen_stream_{nullptr};
};

void CustomContext::Init() { impl_->Init(); }

const Place& CustomContext::GetPlace() const { return impl_->GetPlace(); }

phi::stream::stream_t CustomContext::stream() const { return impl_->stream(); }

std::shared_ptr<phi::stream::Stream> CustomContext::GetStream() const {
  return impl_->GetStream();
}

void CustomContext::SetStream(std::shared_ptr<phi::stream::Stream> stream) {
  impl_->SetStream(stream);
}

// void CustomContext::SetStream(gpuStream_t stream) { impl_->SetStream(stream); }
void CustomContext::Wait() const { return impl_->Wait(); }

CustomContext::CustomContext(const CustomPlace& place)
    : DeviceContext(), impl_(std::make_unique<Impl>(place)) {
  impl_->Init();
}

CustomContext::~CustomContext() { impl_.reset(); }

phi::ccl::CCLComm CustomContext::xccl_comm() const {
  return impl_->xccl_comm();
}

void CustomContext::set_xccl_comm(phi::ccl::CCLComm comm) {
  impl_->set_xccl_comm(comm);
}

////////////////////////for cuda///////////////////////////////
int CustomContext::GetComputeCapability() const {
  return impl_->compute_capability_;
}

int CustomContext::GetMaxThreadsPerBlock() const {
  return impl_->max_threads_per_block_;
}

int CustomContext::GetSMCount() const { return impl_->multi_process_; }

std::array<unsigned int, 3> CustomContext::GetCUDAMaxGridDimSize() const {
  return impl_->max_grid_dim_size_;
}

int CustomContext::GetMaxPhysicalThreadCount() const {
  return impl_->multi_process_ * impl_->max_threads_per_mp_;
}

void CustomContext::SetComputeCapability(int val) {
  impl_->compute_capability_ = val;
}

void CustomContext::SetMaxThreadsPerMultiProcessor(int val) {
  impl_->max_threads_per_mp_ = val;
}

void CustomContext::SetMultiProcessors(int val) { impl_->multi_process_ = val; }

void CustomContext::SetMaxThreadsPerBlock(int val) {
  impl_->max_threads_per_block_ = val;
}

void CustomContext::SetMaxGridDimSize(const std::array<unsigned int, 3>& val) {
  impl_->max_grid_dim_size_ = val;
}

void CustomContext::SetDriverVersion(int val) { impl_->driver_version_ = val; }

void CustomContext::SetRuntimeVersion(int val) { impl_->runtime_version_ = val; }
////////////////////////for cuda///////////////////////////////
}  // namespace phi
