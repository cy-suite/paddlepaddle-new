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

#include "paddle/cinn/common/event_tracing.h"

namespace cinn {
namespace common {

std::vector<Event> RecordEvent::g_events_;
paddle::memory::SpinLock RecordEvent::g_spinlock_;

std::vector<Count> RecordCount::g_counts_;
paddle::memory::SpinLock RecordCount::g_spinlock_;

void RecordEvent::End() {
  if (recorded_) {
    return;
  }
  recorded_ = true;
  Event e;
  e.name = name_;
  e.start_time = start_time_;
  e.end_time = std::chrono::high_resolution_clock::now();
  e.thread_id = std::this_thread::get_id();
  std::lock_guard<paddle::memory::SpinLock> guard(g_spinlock_);
  RecordEvent::g_events_.emplace_back(e);
}

void DumpRecordEvent(std::string path) {
  std::unordered_map<std::string, MergedEvent> merged_event;
  std::unordered_map<std::string, uint64_t> merged_count;
  std::ofstream event_file(path + "/event.log");
  std::ofstream count_file(path + "/count.log");
  std::ofstream merged_file(path + "/merged.log");
  if (!event_file.is_open() || !count_file.is_open() ||
      !merged_file.is_open()) {
    PADDLE_THROW(
        phi::errors::InvalidArgument("Open file faild, path = %s", path));
  }
  for (const auto& e : RecordEvent::Events()) {
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(
                     e.start_time.time_since_epoch())
                     .count();
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
                   e.end_time.time_since_epoch())
                   .count();
    auto used = end - start;
    event_file << e.name << ", start = " << start << ", end = " << end
               << ", used(ms) = " << used << ", thread = " << e.thread_id
               << std::endl;
    merged_event[e.name].count++;
    merged_event[e.name].used_time += used;
  }

  for (const auto& c : RecordCount::Counts()) {
    count_file << c.name << ", count = " << c.count << std::endl;
    merged_count[c.name] += c.count;
  }

  for (auto& me : merged_event) {
    merged_file << me.first << ", call times = " << me.second.count
                << ", used(s) = " << me.second.used_time / 1000 << std::endl;
  }

  for (auto& mc : merged_count) {
    merged_file << mc.first << ", count = " << mc.second << std::endl;
  }

  merged_file << "SubGraph Count: " << merged_event["ApplyCinnPass"].count
              << std::endl;
  merged_file << "Group Count: "
              << merged_event["OpLowererImpl::BucketLower"].count << std::endl;
  merged_file << "BC Count: " << merged_count["BC"] << std::endl;
  merged_file << "Kernel Count: " << merged_event["Compiler::Build"].count
              << std::endl;

  merged_file << "ApplyCinnPass use time: "
              << merged_event["ApplyCinnPass"].used_time / 1000
              << ", call count: " << merged_event["ApplyCinnPass"].count
              << std::endl;
  merged_file << "OpLowererImpl::LowerX86 use time: "
              << merged_event["OpLowererImpl::LowerX86"].used_time / 1000
              << ", call count: "
              << merged_event["OpLowererImpl::LowerX86"].count << std::endl;
  merged_file << "OpLowererImpl::LowerOps use time: "
              << merged_event["OpLowererImpl::LowerOps"].used_time / 1000
              << ", call count: "
              << merged_event["OpLowererImpl::LowerOps"].count << std::endl;
  merged_file << "OperationFusion use time: "
              << merged_event["OperationFusion"].used_time / 1000
              << ", call count: " << merged_event["OperationFusion"].count
              << std::endl;
  merged_file
      << "DynamicShapeGroupScheduler::Schedule use time: "
      << merged_event["DynamicShapeGroupScheduler::Schedule"].used_time / 1000
      << ", call count: "
      << merged_event["DynamicShapeGroupScheduler::Schedule"].count
      << std::endl;
  merged_file << "OpLowererImpl::PostProcess use time: "
              << merged_event["OpLowererImpl::PostProcess"].used_time / 1000
              << ", call count: "
              << merged_event["OpLowererImpl::PostProcess"].count << std::endl;

  merged_file << "Compiler::Build use time: "
              << merged_event["Compiler::Build"].used_time / 1000
              << ", call count: " << merged_event["Compiler::Build"].count
              << std::endl;
  merged_file << "Compiler::AppendCX86 use time: "
              << merged_event["Compiler::AppendCX86"].used_time / 1000
              << ", call count: " << merged_event["Compiler::AppendCX86"].count
              << std::endl;
  merged_file
      << "Compiler::AppendBroadcastSwitchModule use time: "
      << merged_event["Compiler::AppendBroadcastSwitchModule"].used_time / 1000
      << ", call count: "
      << merged_event["Compiler::AppendBroadcastSwitchModule"].count
      << std::endl;
  merged_file << "Compiler::EndCompile use time: "
              << merged_event["Compiler::EndCompile"].used_time / 1000
              << ", call count: " << merged_event["Compiler::EndCompile"].count
              << std::endl;
}

}  // namespace common
}  // namespace cinn
