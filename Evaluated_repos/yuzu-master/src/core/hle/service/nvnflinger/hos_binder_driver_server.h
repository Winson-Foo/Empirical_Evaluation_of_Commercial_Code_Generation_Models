// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "common/common_types.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/nvnflinger/binder.h"

namespace Core {
class System;
}

namespace Service::Nvnflinger {

class HosBinderDriverServer final {
public:
    explicit HosBinderDriverServer(Core::System& system_);
    ~HosBinderDriverServer();

    u64 RegisterProducer(std::unique_ptr<android::IBinder>&& binder);

    android::IBinder* TryGetProducer(u64 id);

private:
    KernelHelpers::ServiceContext service_context;

    std::unordered_map<u64, std::unique_ptr<android::IBinder>> producers;
    std::mutex lock;
    u64 last_id{};
};

} // namespace Service::Nvnflinger
