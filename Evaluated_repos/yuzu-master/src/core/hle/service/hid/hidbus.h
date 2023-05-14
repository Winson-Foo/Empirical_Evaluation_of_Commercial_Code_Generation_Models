// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>

#include "core/hle/service/hid/hidbus/hidbus_base.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/service.h"

namespace Core::Timing {
struct EventType;
} // namespace Core::Timing

namespace Core {
class System;
} // namespace Core

namespace Service::HID {

class HidBus final : public ServiceFramework<HidBus> {
public:
    explicit HidBus(Core::System& system_);
    ~HidBus() override;

private:
    static const std::size_t max_number_of_handles = 0x13;

    enum class HidBusDeviceId : std::size_t {
        RingController = 0x20,
        FamicomRight = 0x21,
        Starlink = 0x28,
    };

    // This is nn::hidbus::detail::StatusManagerType
    enum class StatusManagerType : u32 {
        None,
        Type16,
        Type32,
    };

    // This is nn::hidbus::BusType
    enum class BusType : u32 {
        LeftJoyRail,
        RightJoyRail,
        InternalBus, // Lark microphone

        MaxBusType,
    };

    // This is nn::hidbus::BusHandle
    struct BusHandle {
        u32 abstracted_pad_id;
        u8 internal_index;
        u8 player_number;
        u8 bus_type_id;
        bool is_valid;
    };
    static_assert(sizeof(BusHandle) == 0x8, "BusHandle is an invalid size");

    // This is nn::hidbus::JoyPollingReceivedData
    struct JoyPollingReceivedData {
        std::array<u8, 0x30> data;
        u64 out_size;
        u64 sampling_number;
    };
    static_assert(sizeof(JoyPollingReceivedData) == 0x40,
                  "JoyPollingReceivedData is an invalid size");

    struct HidbusStatusManagerEntry {
        u8 is_connected{};
        INSERT_PADDING_BYTES(0x3);
        Result is_connected_result{0};
        u8 is_enabled{};
        u8 is_in_focus{};
        u8 is_polling_mode{};
        u8 reserved{};
        JoyPollingMode polling_mode{};
        INSERT_PADDING_BYTES(0x70); // Unknown
    };
    static_assert(sizeof(HidbusStatusManagerEntry) == 0x80,
                  "HidbusStatusManagerEntry is an invalid size");

    struct HidbusStatusManager {
        std::array<HidbusStatusManagerEntry, max_number_of_handles> entries{};
        INSERT_PADDING_BYTES(0x680); // Unused
    };
    static_assert(sizeof(HidbusStatusManager) <= 0x1000, "HidbusStatusManager is an invalid size");

    struct HidbusDevice {
        bool is_device_initializated{};
        BusHandle handle{};
        std::unique_ptr<HidbusBase> device{nullptr};
    };

    void GetBusHandle(HLERequestContext& ctx);
    void IsExternalDeviceConnected(HLERequestContext& ctx);
    void Initialize(HLERequestContext& ctx);
    void Finalize(HLERequestContext& ctx);
    void EnableExternalDevice(HLERequestContext& ctx);
    void GetExternalDeviceId(HLERequestContext& ctx);
    void SendCommandAsync(HLERequestContext& ctx);
    void GetSendCommandAsynceResult(HLERequestContext& ctx);
    void SetEventForSendCommandAsycResult(HLERequestContext& ctx);
    void GetSharedMemoryHandle(HLERequestContext& ctx);
    void EnableJoyPollingReceiveMode(HLERequestContext& ctx);
    void DisableJoyPollingReceiveMode(HLERequestContext& ctx);
    void SetStatusManagerType(HLERequestContext& ctx);

    void UpdateHidbus(std::uintptr_t user_data, std::chrono::nanoseconds ns_late);
    std::optional<std::size_t> GetDeviceIndexFromHandle(BusHandle handle) const;

    template <typename T>
    void MakeDevice(BusHandle handle) {
        const auto device_index = GetDeviceIndexFromHandle(handle);
        if (device_index) {
            devices[device_index.value()].device = std::make_unique<T>(system, service_context);
        }
    }

    bool is_hidbus_enabled{false};
    HidbusStatusManager hidbus_status{};
    std::array<HidbusDevice, max_number_of_handles> devices{};
    std::shared_ptr<Core::Timing::EventType> hidbus_update_event;
    KernelHelpers::ServiceContext service_context;
};

} // namespace Service::HID
