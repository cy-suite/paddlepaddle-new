// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <chrono>
#include <ctime>
#include <fstream>
#include <ratio>
#include <string>
#include <thread>
#include <vector>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"

namespace cinn {
namespace common {

struct Event {
  std::thread::id thread_id;
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  std::string name;
};

class TEST_API RecordEvent {
 public:
  explicit RecordEvent(const std::string& name) {
    name_ = name;
    start_time_ = std::chrono::high_resolution_clock::now();
  }

  void End();

  ~RecordEvent() { End(); }

  static std::vector<Event> Events() { return RecordEvent::g_events_; }

 private:
  std::string name_;
  std::chrono::high_resolution_clock::time_point start_time_;
  bool recorded_{false};

  static std::vector<Event> g_events_;
  static paddle::memory::SpinLock g_spinlock_;
};

struct MergedEvent {
  size_t count{0};
  size_t used_time_second{0};
};

void DumpRecordEvent(std::string path);

}  // namespace common
}  // namespace cinn
