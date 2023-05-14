// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/hid/hid_types.h"
#include "core/hid/irs_types.h"
#include "core/hle/service/hid/irsensor/processor_base.h"
#include "core/hle/service/service.h"

namespace Core {
class System;
}

namespace Core::HID {
class EmulatedController;
} // namespace Core::HID

namespace Service::IRS {

class IRS final : public ServiceFramework<IRS> {
public:
    explicit IRS(Core::System& system_);
    ~IRS() override;

private:
    // This is nn::irsensor::detail::AruidFormat
    struct AruidFormat {
        u64 sensor_aruid;
        u64 sensor_aruid_status;
    };
    static_assert(sizeof(AruidFormat) == 0x10, "AruidFormat is an invalid size");

    // This is nn::irsensor::detail::StatusManager
    struct StatusManager {
        std::array<Core::IrSensor::DeviceFormat, 9> device;
        std::array<AruidFormat, 5> aruid;
    };
    static_assert(sizeof(StatusManager) == 0x8000, "StatusManager is an invalid size");

    void ActivateIrsensor(HLERequestContext& ctx);
    void DeactivateIrsensor(HLERequestContext& ctx);
    void GetIrsensorSharedMemoryHandle(HLERequestContext& ctx);
    void StopImageProcessor(HLERequestContext& ctx);
    void RunMomentProcessor(HLERequestContext& ctx);
    void RunClusteringProcessor(HLERequestContext& ctx);
    void RunImageTransferProcessor(HLERequestContext& ctx);
    void GetImageTransferProcessorState(HLERequestContext& ctx);
    void RunTeraPluginProcessor(HLERequestContext& ctx);
    void GetNpadIrCameraHandle(HLERequestContext& ctx);
    void RunPointingProcessor(HLERequestContext& ctx);
    void SuspendImageProcessor(HLERequestContext& ctx);
    void CheckFirmwareVersion(HLERequestContext& ctx);
    void SetFunctionLevel(HLERequestContext& ctx);
    void RunImageTransferExProcessor(HLERequestContext& ctx);
    void RunIrLedProcessor(HLERequestContext& ctx);
    void StopImageProcessorAsync(HLERequestContext& ctx);
    void ActivateIrsensorWithFunctionLevel(HLERequestContext& ctx);

    Result IsIrCameraHandleValid(const Core::IrSensor::IrCameraHandle& camera_handle) const;
    Core::IrSensor::DeviceFormat& GetIrCameraSharedMemoryDeviceEntry(
        const Core::IrSensor::IrCameraHandle& camera_handle);

    template <typename T>
    void MakeProcessor(const Core::IrSensor::IrCameraHandle& handle,
                       Core::IrSensor::DeviceFormat& device_state) {
        const auto index = static_cast<std::size_t>(handle.npad_id);
        if (index > sizeof(processors)) {
            LOG_CRITICAL(Service_IRS, "Invalid index {}", index);
            return;
        }
        processors[index] = std::make_unique<T>(device_state);
    }

    template <typename T>
    void MakeProcessorWithCoreContext(const Core::IrSensor::IrCameraHandle& handle,
                                      Core::IrSensor::DeviceFormat& device_state) {
        const auto index = static_cast<std::size_t>(handle.npad_id);
        if (index > sizeof(processors)) {
            LOG_CRITICAL(Service_IRS, "Invalid index {}", index);
            return;
        }

        if constexpr (std::is_constructible_v<T, Core::System&, Core::IrSensor::DeviceFormat&,
                                              std::size_t>) {
            processors[index] = std::make_unique<T>(system, device_state, index);
        } else {
            processors[index] = std::make_unique<T>(system.HIDCore(), device_state, index);
        }
    }

    template <typename T>
    T& GetProcessor(const Core::IrSensor::IrCameraHandle& handle) {
        const auto index = static_cast<std::size_t>(handle.npad_id);
        if (index > sizeof(processors)) {
            LOG_CRITICAL(Service_IRS, "Invalid index {}", index);
            return static_cast<T&>(*processors[0]);
        }
        return static_cast<T&>(*processors[index]);
    }

    template <typename T>
    const T& GetProcessor(const Core::IrSensor::IrCameraHandle& handle) const {
        const auto index = static_cast<std::size_t>(handle.npad_id);
        if (index > sizeof(processors)) {
            LOG_CRITICAL(Service_IRS, "Invalid index {}", index);
            return static_cast<T&>(*processors[0]);
        }
        return static_cast<T&>(*processors[index]);
    }

    Core::HID::EmulatedController* npad_device = nullptr;
    StatusManager* shared_memory = nullptr;
    std::array<std::unique_ptr<ProcessorBase>, 9> processors{};
};

class IRS_SYS final : public ServiceFramework<IRS_SYS> {
public:
    explicit IRS_SYS(Core::System& system);
    ~IRS_SYS() override;
};

} // namespace Service::IRS
