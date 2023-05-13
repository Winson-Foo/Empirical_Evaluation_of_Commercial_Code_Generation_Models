// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/hle/service/hid/irsensor/moment_processor.h"

namespace Service::IRS {
MomentProcessor::MomentProcessor(Core::IrSensor::DeviceFormat& device_format)
    : device(device_format) {
    device.mode = Core::IrSensor::IrSensorMode::MomentProcessor;
    device.camera_status = Core::IrSensor::IrCameraStatus::Unconnected;
    device.camera_internal_status = Core::IrSensor::IrCameraInternalStatus::Stopped;
}

MomentProcessor::~MomentProcessor() = default;

void MomentProcessor::StartProcessor() {}

void MomentProcessor::SuspendProcessor() {}

void MomentProcessor::StopProcessor() {}

void MomentProcessor::SetConfig(Core::IrSensor::PackedMomentProcessorConfig config) {
    current_config.camera_config.exposure_time = config.camera_config.exposure_time;
    current_config.camera_config.gain = config.camera_config.gain;
    current_config.camera_config.is_negative_used = config.camera_config.is_negative_used;
    current_config.camera_config.light_target =
        static_cast<Core::IrSensor::CameraLightTarget>(config.camera_config.light_target);
    current_config.window_of_interest = config.window_of_interest;
    current_config.preprocess =
        static_cast<Core::IrSensor::MomentProcessorPreprocess>(config.preprocess);
    current_config.preprocess_intensity_threshold = config.preprocess_intensity_threshold;
}

} // namespace Service::IRS
