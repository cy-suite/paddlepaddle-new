#pragma once

#include "paddle/cinn/ir/group_schedule/tactic/schedule_tactic.h"

namespace cinn {
namespace ir {

std::unique_ptr<ScheduleTactic> CreateTileDiscreteTactic();

}  // namespace ir
}  // namespace cinn