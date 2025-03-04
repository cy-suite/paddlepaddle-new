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
#include "paddle/fluid/platform/device/xpu/xpu_info.h"

#include <algorithm>
#include <cstdlib>
#include <string>

#include "gflags/gflags.h"
#include "paddle/fluid/platform/lock_guard_ptr.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/enforce_xpu.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/monitor.h"

//#include "paddle/phi/backends/xpu/xpu_context.h"


#include "paddle/fluid/platform/flags.h"
// TODO(wilber): The phi computing library requires a component to manage
// flags.
#include "paddle/phi/core/flags.h"


PHI_DECLARE_double(fraction_of_gpu_memory_to_use);
PHI_DECLARE_uint64(initial_gpu_memory_in_mb);
PHI_DECLARE_uint64(reallocate_gpu_memory_in_mb);
PHI_DECLARE_uint64(gpu_memory_limit_mb);

constexpr static float fraction_reserve_gpu_memory = 0.01f;
USE_XPU_MEM_STAT;
namespace paddle {
namespace platform {

/**************************** Version Management **************************/

//! Get the version of XPU Driver
int GetDriverVersion() { return phi::backends::xpu::GetDriverVersion(); }

//! Get the version of XPU Runtime
int GetRuntimeVersion() { return phi::backends::xpu::GetRuntimeVersion(); }

/**************************** Device Management **************************/

int GetXPUDeviceCount() { return phi::backends::xpu::GetXPUDeviceCount(); }

int GetXPUCurrentDeviceId() {
  return phi::backends::xpu::GetXPUCurrentDeviceId();
}

void SetXPUDeviceId(int id) { phi::backends::xpu::SetXPUDeviceId(id); }

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetXPUSelectedDevices() {
  // use user specified XPUs in single-node multi-process mode.
  return phi::backends::xpu::GetXPUSelectedDevices();
}

size_t XPUAvailableMemToAlloc() {
  size_t total = 0;
  size_t available = 0;
  XPUMemoryUsage(&available, &total);
  size_t reserving =
      static_cast<size_t>(fraction_reserve_gpu_memory * available);
  // If available size is less than minimum chunk size, no usable memory exists
  size_t available_to_alloc = available - reserving;
  size_t min_chunk_size = XPUMinChunkSize();
  if (available_to_alloc < min_chunk_size) {
    available_to_alloc = 0;
  }
  VLOG(1) << "XPU usage " << (available >> 20) << "M/"
           << (total >> 20) << "M, " << (available_to_alloc >> 20)
           << "M available to allocate";
  return available_to_alloc;
}

//! Get the memory usage of current XPU device.
void XPUMemoryUsage(size_t *available, size_t *total) {
  size_t actual_available, actual_total;
  RecordedXPUMemGetInfo(available,
                        total,
                        &actual_available,
                        &actual_total,
                        GetXPUCurrentDeviceId());
  //测试有值
  VLOG(1)<<"XPUMemoryUsage"<<*available<<"total"<<*total<<"actual_avail"<<actual_available<<"actual_total"<<actual_total;
}

//! Get the maximum allocation size of current XPU device.
size_t XPUMaxAllocSize() {
  return std::max(XPUInitAllocSize(), XPUReallocSize());
}
static size_t XPUAllocSize(bool realloc) {
  size_t available_to_alloc = XPUAvailableMemToAlloc();
  PADDLE_ENFORCE_GT(
      available_to_alloc,
      0,
      phi::errors::ResourceExhausted("Not enough available  memory."));
  // If FLAGS_initial_gpu_memory_in_mb is 0, then initial memory will be
  // allocated by fraction
  size_t flag_mb = realloc ? FLAGS_reallocate_gpu_memory_in_mb
                           : FLAGS_initial_gpu_memory_in_mb;
  size_t alloc_bytes =
      (flag_mb > 0ul
           ? flag_mb << 20
           : available_to_alloc * FLAGS_fraction_of_gpu_memory_to_use);
  //后期待修改

  PADDLE_ENFORCE_GE(
      available_to_alloc,
      alloc_bytes,
      phi::errors::ResourceExhausted("Not enough available MLU memory."));
  VLOG(10) << "Alloc size is " << (alloc_bytes >> 20)
           << " MiB, is it Re-alloc: " << realloc;
  return alloc_bytes;
}

//! Get the initial allocation size of current XPU device.
size_t XPUInitAllocSize() { return XPUAllocSize(/* realloc = */ false); }

//! Get the re-allocation size of current XPU device.
size_t XPUReallocSize() { return XPUAllocSize(/* realloc = */ true); }

//! Get the minimum chunk size for XPU buddy allocator.
size_t XPUMinChunkSize() {
  // Allow to allocate the minimum chunk size is 256 bytes.
   return phi::backends::xpu::XPUMinChunkSize();
}

//! Get the maximum chunk size for XPU buddy allocator.
size_t XPUMaxChunkSize() {
  size_t max_chunk_size = XPUMaxAllocSize();
  VLOG(10) << "Max chunk size " << (max_chunk_size >> 20) << "M";
  return max_chunk_size;
}

/**************************** Memory Management **************************/

void MemcpySyncH2D(void* dst,
                   const void* src,
                   size_t count,
                   const platform::XPUPlace& dst_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(dst_place);
  phi::backends::xpu::MemcpySyncH2D(dst, src, count, dst_place, *dev_ctx);
}

void MemcpySyncD2H(void* dst,
                   const void* src,
                   size_t count,
                   const platform::XPUPlace& src_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2H(dst, src, count, src_place, *dev_ctx);
}

// if src.device == dst.device and you need sync , after call this function,
// need to call dev_ctx.Wait()
void MemcpySyncD2D(void* dst,
                   const platform::XPUPlace& dst_place,
                   const void* src,
                   const platform::XPUPlace& src_place,
                   size_t count) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2D(
      dst, dst_place, src, src_place, count, *dev_ctx);
}

void XPUStreamSync(xpuStream stream) {
  PADDLE_ENFORCE_XDNN_SUCCESS(xpu_wait(stream), "xpu_wait");
}

static void RaiseNonOutOfMemoryError(int status) {
  if (status != XPU_SUCCESS) {
    status = XPU_SUCCESS;
  }
  PADDLE_ENFORCE_XPU_SUCCESS(status);
}

//返回值待修改
int xpuGetMemInfo(size_t* actual_avail,size_t* actual_total) {  
   

    FILE* pipe_total = popen("xpu_smi | awk 'NR==7' | awk '{{print $24}}'", "r");  
    if (!pipe_total) {  
        std::cerr << "popen() failed!" << std::endl;  
        return XPUERR_NOMEM;  
    }  
  
    char buffer_total[128];  
    if (fgets(buffer_total, sizeof(buffer_total), pipe_total) != NULL) {  
        std::string total_str(buffer_total); 
        size_t total=std::stoi(total_str);
        total=total<<20;
        *actual_total = total;   
    } else {  
        std::cerr << "fgets() failed or reached end of file." << std::endl;  
    } 
 
    // 关闭管道  
    pclose(pipe_total);

    FILE* pipe_avail = popen("xpu_smi | awk 'NR==7' | awk '{{print $22}}'", "r");  
    if (!pipe_avail) {  
        std::cerr << "popen() failed!" << std::endl;  
        return XPUERR_NOMEM;  
    }  
  
    char buffer_avail[128];  
    if (fgets(buffer_avail, sizeof(buffer_avail), pipe_avail) != NULL) {  
        std::string avail_str(buffer_avail); // 将C风格字符串转换为std::string  
        size_t avail = std::stoi(avail_str);
        avail = avail<<20;
        avail = *actual_total-avail;
        *actual_avail = avail; // 将std::string转换为int   
    } else {  
        std::cerr << "fgets() failed or reached end of file." << std::endl;  
    } 
  
    // 关闭管道  
    pclose(pipe_avail);  

    return XPU_SUCCESS;
    
} 


class RecordedXPUMallocHelper {
 private:
  explicit RecordedXPUMallocHelper(int dev_id, uint64_t limit_size = 0)
      : dev_id_(dev_id), limit_size_(limit_size) {
    if (NeedRecord()) {
      mtx_.reset(new std::mutex());
    }
  }

  DISABLE_COPY_AND_ASSIGN(RecordedXPUMallocHelper);

 public:
  static RecordedXPUMallocHelper *Instance(int dev_id) {
    std::call_once(once_flag_, [] {
      int dev_cnt = GetXPUDeviceCount();
      instances_.reserve(dev_cnt);
      for (int i = 0; i < dev_cnt; ++i) {
        // NOTE(zhiqiu): share the flags with gpu, avoid more flags.
        instances_.emplace_back(
            new RecordedXPUMallocHelper(i, FLAGS_gpu_memory_limit_mb << 20));
            
      }
    });

    PADDLE_ENFORCE_GE(
        dev_id,
        0,
        phi::errors::OutOfRange(
            "Device id must be not less than 0, but got %d.", dev_id));
    PADDLE_ENFORCE_LT(
        dev_id,
        instances_.size(),
        phi::errors::OutOfRange("Device id %d exceeds xpu card number %d.",
                                     dev_id,
                                     instances_.size()));
    return instances_[dev_id].get();
  }

  /**
   * Try to allocate `size` xpu memory. Only ACL_ERROR_BAD_ALLOC
   * or ACL_ERROR_NONE would be returned.
   */
  int Malloc(void **ptr, size_t size) {
    paddle::platform::LockGuardPtr<std::mutex> lock(mtx_);
    if (UNLIKELY(NeedRecord() && cur_size_ + size > limit_size_)) {
      return XPUERR_NOMEM;
    }

    XPUDeviceGuard guard(dev_id_);
    auto result = xpu_malloc(ptr, size);//xpu分配内存，待修改
    if (ptr != NULL) {
      if (NeedRecord()) {
        cur_size_ += size;
      }
      
      STAT_INT_ADD("STAT_xpu" + std::to_string(dev_id_) + "_mem_size", size);
      return result;
    } else {
      RaiseNonOutOfMemoryError(XPUERR_NOMEM);
      //PADDLE_ENFORCE_XPU_SUCCESS(result);

      // Non out of memory error would be raised inside
      // RaiseNonOutOfMemoryError. Therefore, we can
      // return cudaErrorMemoryAllocation directly here.
      return XPUERR_NOMEM;
      
    }

  }
  /**
   * Free gpu memory. Usually, free is not allowed to raise error.
   * If it does raise error, the process should be crashed.
   */
  void Free(void *ptr, size_t size) {
    XPUDeviceGuard guard(dev_id_);
    auto result = xpu_free(ptr);
    PADDLE_ENFORCE_XPU_SUCCESS(result);
    if (NeedRecord()) {
      std::lock_guard<std::mutex> guard(*mtx_);
      cur_size_ -= size;
    }
    STAT_INT_SUB("STAT_xpu" + std::to_string(dev_id_) + "_mem_size", size);
  }

  bool GetMemInfo(size_t *avail,
                  size_t *total,
                  size_t *actual_avail,
                  size_t *actual_total) {
    {
      XPUDeviceGuard guard(dev_id_);
      auto result = xpuGetMemInfo(actual_avail, actual_total);
      if (result != XPU_SUCCESS) {
        *actual_avail = 0;
      }
      RaiseNonOutOfMemoryError(result);
    }

    if (NeedRecord()) {
      std::lock_guard<std::mutex> guard(*mtx_);
      *avail = std::min(*actual_avail, limit_size_ - cur_size_);
      *total = std::min(*actual_total, limit_size_);
      return *total < *actual_total;
    } else {
      *avail = *actual_avail;
      *total = *actual_total;
      return false;
    }
  }

  inline bool NeedRecord() const { return limit_size_ != 0; }

  uint64_t RecordedSize() const {
    paddle::platform::LockGuardPtr<std::mutex> lock(mtx_);
    return NeedRecord() ? cur_size_ : 0;
  }

  uint64_t LimitSize() const { return limit_size_; }

 private:
  const int dev_id_;
  const uint64_t limit_size_;
  uint64_t cur_size_{0};

  mutable std::unique_ptr<std::mutex> mtx_;

  static std::once_flag once_flag_;
  static std::vector<std::unique_ptr<RecordedXPUMallocHelper>> instances_;
};

std::once_flag RecordedXPUMallocHelper::once_flag_;
std::vector<std::unique_ptr<RecordedXPUMallocHelper>>
    RecordedXPUMallocHelper::instances_;

bool RecordedXPUMemGetInfo(size_t *avail,
                           size_t *total,
                           size_t *actual_avail,
                           size_t *actual_total,
                           int dev_id) {
  return  RecordedXPUMallocHelper::Instance(dev_id)->GetMemInfo(
      avail, total, actual_avail, actual_total);
      //测试有值
      VLOG(1)<<"RecordedXPUMemGetInfo"<<*avail<<"total"<<*total<<"actual_avail"<<*actual_avail<<"actual_total"<<*actual_total;
    
}
int RecordedXPUMalloc(void **ptr, size_t size, int dev_id) {
  return RecordedXPUMallocHelper::Instance(dev_id)->Malloc(ptr, size);
}

void RecordedXPUFree(void *p, size_t size, int dev_id) {
  return RecordedXPUMallocHelper::Instance(dev_id)->Free(p, size);
}

/**************************** Others **************************/

phi::backends::xpu::XPUVersion get_xpu_version(int dev_id) {
  return phi::backends::xpu::get_xpu_version(dev_id);
}

}  // namespace platform
}  // namespace paddle
