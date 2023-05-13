// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/hle/service/time/errors.h"
#include "core/hle/service/time/system_clock_context_update_callback.h"
#include "core/hle/service/time/time_sharedmemory.h"

namespace Service::Time::Clock {

class NetworkSystemClockContextWriter final : public SystemClockContextUpdateCallback {
public:
    explicit NetworkSystemClockContextWriter(SharedMemory& shared_memory_)
        : SystemClockContextUpdateCallback{}, shared_memory{shared_memory_} {}

protected:
    Result Update() override {
        shared_memory.UpdateNetworkSystemClockContext(context);
        return ResultSuccess;
    }

private:
    SharedMemory& shared_memory;
};

} // namespace Service::Time::Clock
