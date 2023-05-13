// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <vector>

#include "core/hle/service/time/clock_types.h"

namespace Kernel {
class KEvent;
}

namespace Service::Time::Clock {

// Parts of this implementation were based on Ryujinx (https://github.com/Ryujinx/Ryujinx/pull/783).
// This code was released under public domain.

class SystemClockContextUpdateCallback {
public:
    SystemClockContextUpdateCallback();
    virtual ~SystemClockContextUpdateCallback();

    bool NeedUpdate(const SystemClockContext& value) const;

    void RegisterOperationEvent(std::shared_ptr<Kernel::KEvent>&& event);

    void BroadcastOperationEvent();

    Result Update(const SystemClockContext& value);

protected:
    virtual Result Update();

    SystemClockContext context{};

private:
    bool has_context{};
    std::vector<std::shared_ptr<Kernel::KEvent>> operation_event_list;
};

} // namespace Service::Time::Clock
