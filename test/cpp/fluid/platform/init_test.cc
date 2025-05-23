/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/platform/init.h"

#include "gtest/gtest.h"
#include "paddle/phi/core/platform/device_context.h"

TEST(InitDevices, CPU) {
  using paddle::framework::InitDevices;
  using phi::DeviceContextPool;

#if !defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_XPU) && \
    !defined(PADDLE_WITH_HIP)
  InitDevices();
  DeviceContextPool& pool = DeviceContextPool::Instance();
  ASSERT_EQ(pool.Size(), 1U);
#endif
}

TEST(InitDevices, CUDA) {
  using paddle::framework::InitDevices;
  using phi::DeviceContextPool;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  int count = paddle::platform::GetGPUDeviceCount();
  InitDevices();
  DeviceContextPool& pool = DeviceContextPool::Instance();
  ASSERT_EQ(pool.Size(), 2U + static_cast<unsigned>(count));
#endif
}

TEST(InitDevices, XPU) {
  using paddle::framework::InitDevices;
  using phi::DeviceContextPool;

#ifdef PADDLE_WITH_XPU
  int count = paddle::platform::GetXPUDeviceCount();
  InitDevices();
  DeviceContextPool& pool = DeviceContextPool::Instance();
  ASSERT_EQ(pool.Size(), 2U + static_cast<unsigned>(count));
#endif
}

#ifndef _WIN32
TEST(SignalHandle, SignalHandle) {
  std::string msg = "Signal raises";
  paddle::framework::SignalHandle(msg.c_str(), static_cast<int>(msg.size()));
}
#endif
