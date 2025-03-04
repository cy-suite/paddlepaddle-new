// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/xpu_async_load.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/backends/xpu/xpu_context.h"  // For phi::XPUContext
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/common/place.h"

#include <iostream>  // For debug printing

namespace paddle {
namespace distributed {

//
// Implementation of XpuAsyncLoad::Task
//

XpuAsyncLoad::Task::Task(const Place& place)
    : task_place_(place),
      event_manager_(std::make_shared<XPUEventManager>()) {
  std::cout << "[XpuAsyncLoad::Task] Created task for place: " << task_place_ << std::endl;
}

XpuAsyncLoad::Task::~Task() {}

bool XpuAsyncLoad::Task::IsCompleted() {
  // XPU event query is not supported; assume the task is complete.
  return true;
}

void XpuAsyncLoad::Task::XpuSynchronize() {
  auto* ctx = static_cast<phi::XPUContext*>(
      phi::DeviceContextPool::Instance().Get(task_place_));
  ctx->Wait();
}

void XpuAsyncLoad::Task::CpuSynchronize() {
  auto* ctx = static_cast<phi::XPUContext*>(
      phi::DeviceContextPool::Instance().Get(task_place_));
  ctx->Wait();
}

void XpuAsyncLoad::Task::UpdateWaitChain(const phi::DeviceContext& ctx) {
  auto* xpu_ctx = dynamic_cast<const phi::XPUContext*>(&ctx);
  if (xpu_ctx) {
    std::cout << "[XpuAsyncLoad::Task] UpdateWaitChain: Recording event on XPU context" << std::endl;
    event_manager_->Record(*xpu_ctx);
  }
}

//
// Implementation of XpuAsyncLoad methods
//

std::shared_ptr<XpuAsyncLoad::Task> XpuAsyncLoad::CreateTask(const Place& place) {
  return std::make_shared<Task>(place);
}

void XpuAsyncLoad::SyncCalcStream(const Place& /*place*/,
                                  phi::XPUContext* ctx,
                                  XPUEventManager& event_manager) {
  std::cout << "[SyncCalcStream] Recording event and blocking on context." << std::endl;
  event_manager.Record(*ctx);
  event_manager.Block(*ctx);
}

std::shared_ptr<XpuAsyncLoad::Task> XpuAsyncLoad::Offload(
    phi::DenseTensor* dst, const phi::DenseTensor& src) {
  // XPU -> XPUPinned
  const auto& place = src.place();
  std::cout << "src place is: " << phi::AllocationTypeStr(src.place().GetType()) << std::endl;

  PADDLE_ENFORCE_EQ(
      phi::is_xpu_place(place),
      true,
      common::errors::InvalidArgument("XpuAsyncLoad::Offload only supports XPU -> XPUPinned now, src place is not correct"));
  

  std::cout << "[Offload] Offloading tensor from place: " << place << std::endl;
  dst->Resize(src.dims());
  auto size = src.numel() * phi::SizeOf(src.dtype());
  std::cout << "[Offload] Tensor size in bytes: " << size << std::endl;

  auto* dev_ctx = static_cast<phi::XPUContext*>(
      phi::DeviceContextPool::Instance().Get(place));
  // Allocate pinned memory (true)
  auto* dst_ptr = dev_ctx->Alloc(dst, src.dtype(), size, true);
  std::cout << "[Offload] Allocated destination pinned memory at: " << dst_ptr << std::endl;
  auto* src_ptr = src.data();
  std::cout << "[Offload] Source data pointer: " << src_ptr << std::endl;

  std::string key = "load";
  if (!is_initialized_) {
    is_initialized_ = true;
    xpu_place_ = place;
    std::cout << "[Offload] Initializing load environment on place: " << place << std::endl;
    // Create and store an XPUEventManager for this key.
    place_to_calc_event_.emplace(key, XPUEventManager());
    // Set allocators for dev_ctx using the AllocatorFacade.
    dev_ctx->SetAllocator(memory::allocation::AllocatorFacade::Instance().GetAllocator(place).get());
    dev_ctx->SetPinnedAllocator(memory::allocation::AllocatorFacade::Instance().GetAllocator(::phi::GetPinnedPlace(place)).get());
  }
  
  std::cout << "dst place is: " << phi::AllocationTypeStr(dst->place().GetType()) << std::endl;
  PADDLE_ENFORCE_EQ(
      phi::is_xpu_pinned_place(dst->place()),
      true,
      common::errors::InvalidArgument("XpuAsyncLoad::Offload only supports XPU -> XPUPinned now, dst place is not correct"));
      
  // Perform the synchronous memory copy.
  std::cout << "[Offload] Copying memory from src (" << src_ptr << ") to dst (" << dst_ptr << ") size " << size << std::endl;
  phi::memory_utils::Copy(dst->place(), dst_ptr, src.place(), src_ptr, size);
  dev_ctx->Wait();
  // SyncCalcStream(xpu_place_, dev_ctx, place_to_calc_event_.at(key));
  // auto stream = dev_ctx->stream();
  // phi::memory_utils::Copy(dst->place(), dst_ptr, src.place(), src_ptr, size, stream);
  std::cout << "[Offload] Copy complete; device context waited." << std::endl;

  auto task = CreateTask(place);
  task->UpdateWaitChain(*dev_ctx);
  return task;
}

std::shared_ptr<XpuAsyncLoad::Task> XpuAsyncLoad::OffloadWithOffset(
    phi::DenseTensor* dst,
    const phi::DenseTensor& src,
    size_t dst_offset,
    size_t src_offset,
    size_t offload_size) {
  // XPU -> XPUPinned
  const auto& place = src.place();
  std::cout << "src place is: " << phi::AllocationTypeStr(src.place().GetType()) << std::endl;
  
  PADDLE_ENFORCE_EQ(
      phi::is_xpu_place(place),
      true,
      common::errors::InvalidArgument("XpuAsyncLoad::OffloadWithOffset only supports XPU src now, src place is not correct"));
  PADDLE_ENFORCE_EQ(dst->initialized(), true,
      common::errors::PreconditionNotMet("XpuAsyncLoad::OffloadWithOffset requires initialized tensors for both dst and src."));
  PADDLE_ENFORCE_LE(
      src_offset + offload_size,
      src.numel(),
      common::errors::InvalidArgument("XpuAsyncLoad::OffloadWithOffset: src_offset + offload_size must be <= src tensor size."));
  PADDLE_ENFORCE_LE(
      dst_offset + offload_size,
      dst->numel(),
      common::errors::InvalidArgument("XpuAsyncLoad::OffloadWithOffset: dst_offset + offload_size must be <= dst tensor size."));

  std::cout << "[OffloadWithOffset] Offloading with offset; offload_size: " << offload_size << std::endl;
  auto size_in_bytes = offload_size * phi::SizeOf(src.dtype());
  auto src_offset_in_bytes = src_offset * phi::SizeOf(src.dtype());
  auto dst_offset_in_bytes = dst_offset * phi::SizeOf(src.dtype());

  auto* dst_ptr = dst->data();
  auto* src_ptr = src.data();
  std::cout << "[OffloadWithOffset] Original dst pointer: " << dst_ptr << ", src pointer: " << src_ptr << std::endl;
  auto* dst_ptr_tmp = static_cast<char*>(dst_ptr);
  auto* src_ptr_tmp = static_cast<const char*>(src_ptr);
  dst_ptr = static_cast<void*>(dst_ptr_tmp + dst_offset_in_bytes);
  src_ptr = static_cast<const void*>(src_ptr_tmp + src_offset_in_bytes);
  std::cout << "[OffloadWithOffset] Adjusted dst pointer: " << dst_ptr << ", src pointer: " << src_ptr << std::endl;

  auto* dev_ctx = static_cast<phi::XPUContext*>(
      phi::DeviceContextPool::Instance().Get(place));

  std::string key = "load";
  if (!is_initialized_) {
    is_initialized_ = true;
    xpu_place_ = place;
    std::cout << "[OffloadWithOffset] Initializing load environment with offset on place: " << place << std::endl;
    place_to_calc_event_.emplace(key, XPUEventManager());
    dev_ctx->SetAllocator(memory::allocation::AllocatorFacade::Instance().GetAllocator(place).get());
    dev_ctx->SetPinnedAllocator(memory::allocation::AllocatorFacade::Instance().GetAllocator(::phi::GetPinnedPlace(place)).get());
  }

  std::cout << "dst place is: " << phi::AllocationTypeStr(dst->place().GetType()) << std::endl;
  PADDLE_ENFORCE_EQ(
      phi::is_xpu_pinned_place(dst->place()),
      true,
      common::errors::InvalidArgument("XpuAsyncLoad::Offload only supports XPU -> XPUPinned now, dst place is not correct"));

  std::cout << "[OffloadWithOffset] Copying " << size_in_bytes << " bytes with offset." << std::endl;
  phi::memory_utils::Copy(dst->place(), dst_ptr, src.place(), src_ptr, size_in_bytes);
  dev_ctx->Wait();
  // SyncCalcStream(xpu_place_, dev_ctx, place_to_calc_event_.at(key));
  // auto stream = dev_ctx->stream();
  // phi::memory_utils::Copy(dst->place(), dst_ptr, src.place(), src_ptr, size_in_bytes, stream);
  std::cout << "[OffloadWithOffset] Copy complete; waited on dev_ctx." << std::endl;

  auto task = CreateTask(place);
  task->UpdateWaitChain(*dev_ctx);
  return task;
}

std::shared_ptr<XpuAsyncLoad::Task> XpuAsyncLoad::Reload(
    phi::DenseTensor* dst, const phi::DenseTensor& src) {
  // XPUPinned -> XPU
  const auto& place = src.place();
  std::cout << "src place is: " << phi::AllocationTypeStr(src.place().GetType()) << std::endl;
  
  PADDLE_ENFORCE_EQ(
      phi::is_xpu_pinned_place(place),
      true,
      common::errors::InvalidArgument("XpuAsyncLoad::Reload only supports XPUPinned -> XPU now, src place is not correct"));
  
  PADDLE_ENFORCE_EQ(is_initialized_, true,
      common::errors::PreconditionNotMet("You must call Offload before Reload."));

  std::cout << "[Reload] Reloading tensor from XPUPinned to XPU." << std::endl;
  auto* dev_ctx = static_cast<phi::XPUContext*>(
      phi::DeviceContextPool::Instance().Get(xpu_place_));

  dst->Resize(src.dims());
  auto size = src.numel() * phi::SizeOf(src.dtype());
  std::cout << "[Reload] Allocating destination XPU memory; size: " << size << std::endl;
  auto* dst_ptr = dev_ctx->Alloc(dst, src.dtype(), size, /*pinned=*/false);
  auto* src_ptr = src.data();

  std::cout << "dst place is: " << phi::AllocationTypeStr(dst->place().GetType()) << std::endl;
  PADDLE_ENFORCE_EQ(
      phi::is_xpu_place(dst->place()),
      true,
      common::errors::InvalidArgument("XpuAsyncLoad::Offload only supports XPU -> XPUPinned now, dst place is not correct"));

  std::cout << "[Reload] Copying data from pinned src (" << src_ptr << ") to XPU dst (" << dst_ptr << ")" << std::endl;
  phi::memory_utils::Copy(dst->place(), dst_ptr, src.place(), src_ptr, size);
  dev_ctx->Wait();
  // std::string key = "load";
  // SyncCalcStream(xpu_place_, dev_ctx, place_to_calc_event_.at(key));
  // auto stream = dev_ctx->stream();
  // phi::memory_utils::Copy(dst->place(), dst_ptr, src.place(), src_ptr, size, stream);
  std::cout << "[Reload] Reload complete; waited on dev_ctx." << std::endl;

  auto task = CreateTask(xpu_place_);
  task->UpdateWaitChain(*dev_ctx);
  return task;
}

void XpuAsyncLoad::PrepareLoadEnv(const std::string& key, const Place& place) {
  if (!is_initialized_) {
    is_initialized_ = true;
    xpu_place_ = place;
    std::cout << "[PrepareLoadEnv] Initializing environment with key: " << key
              << " for place: " << place << std::endl;
    place_to_calc_event_.emplace(key, XPUEventManager());
    // Optionally, one can initialize a load context here.
    // For example:
    // load_ctx_ = std::make_unique<phi::XPUContext>(place);
  }
}

}  // namespace distributed
}  // namespace paddle
