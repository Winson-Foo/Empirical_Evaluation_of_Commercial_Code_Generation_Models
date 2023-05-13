// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "core/hle/service/audio/audctl.h"
#include "core/hle/service/ipc_helpers.h"

namespace Service::Audio {

AudCtl::AudCtl(Core::System& system_) : ServiceFramework{system_, "audctl"} {
    // clang-format off
    static const FunctionInfo functions[] = {
        {0, nullptr, "GetTargetVolume"},
        {1, nullptr, "SetTargetVolume"},
        {2, &AudCtl::GetTargetVolumeMin, "GetTargetVolumeMin"},
        {3, &AudCtl::GetTargetVolumeMax, "GetTargetVolumeMax"},
        {4, nullptr, "IsTargetMute"},
        {5, nullptr, "SetTargetMute"},
        {6, nullptr, "IsTargetConnected"},
        {7, nullptr, "SetDefaultTarget"},
        {8, nullptr, "GetDefaultTarget"},
        {9, nullptr, "GetAudioOutputMode"},
        {10, nullptr, "SetAudioOutputMode"},
        {11, nullptr, "SetForceMutePolicy"},
        {12, nullptr, "GetForceMutePolicy"},
        {13, nullptr, "GetOutputModeSetting"},
        {14, nullptr, "SetOutputModeSetting"},
        {15, nullptr, "SetOutputTarget"},
        {16, nullptr, "SetInputTargetForceEnabled"},
        {17, nullptr, "SetHeadphoneOutputLevelMode"},
        {18, nullptr, "GetHeadphoneOutputLevelMode"},
        {19, nullptr, "AcquireAudioVolumeUpdateEventForPlayReport"},
        {20, nullptr, "AcquireAudioOutputDeviceUpdateEventForPlayReport"},
        {21, nullptr, "GetAudioOutputTargetForPlayReport"},
        {22, nullptr, "NotifyHeadphoneVolumeWarningDisplayedEvent"},
        {23, nullptr, "SetSystemOutputMasterVolume"},
        {24, nullptr, "GetSystemOutputMasterVolume"},
        {25, nullptr, "GetAudioVolumeDataForPlayReport"},
        {26, nullptr, "UpdateHeadphoneSettings"},
        {27, nullptr, "SetVolumeMappingTableForDev"},
        {28, nullptr, "GetAudioOutputChannelCountForPlayReport"},
        {29, nullptr, "BindAudioOutputChannelCountUpdateEventForPlayReport"},
        {30, nullptr, "SetSpeakerAutoMuteEnabled"},
        {31, nullptr, "IsSpeakerAutoMuteEnabled"},
        {32, nullptr, "GetActiveOutputTarget"},
        {33, nullptr, "GetTargetDeviceInfo"},
        {34, nullptr, "AcquireTargetNotification"},
        {35, nullptr, "SetHearingProtectionSafeguardTimerRemainingTimeForDebug"},
        {36, nullptr, "GetHearingProtectionSafeguardTimerRemainingTimeForDebug"},
        {37, nullptr, "SetHearingProtectionSafeguardEnabled"},
        {38, nullptr, "IsHearingProtectionSafeguardEnabled"},
        {39, nullptr, "IsHearingProtectionSafeguardMonitoringOutputForDebug"},
        {40, nullptr, "GetSystemInformationForDebug"},
        {41, nullptr, "SetVolumeButtonLongPressTime"},
        {42, nullptr, "SetNativeVolumeForDebug"},
        {10000, nullptr, "NotifyAudioOutputTargetForPlayReport"},
        {10001, nullptr, "NotifyAudioOutputChannelCountForPlayReport"},
        {10002, nullptr, "NotifyUnsupportedUsbOutputDeviceAttachedForPlayReport"},
        {10100, nullptr, "GetAudioVolumeDataForPlayReport"},
        {10101, nullptr, "BindAudioVolumeUpdateEventForPlayReport"},
        {10102, nullptr, "BindAudioOutputTargetUpdateEventForPlayReport"},
        {10103, nullptr, "GetAudioOutputTargetForPlayReport"},
        {10104, nullptr, "GetAudioOutputChannelCountForPlayReport"},
        {10105, nullptr, "BindAudioOutputChannelCountUpdateEventForPlayReport"},
        {10106, nullptr, "GetDefaultAudioOutputTargetForPlayReport"},
        {50000, nullptr, "SetAnalogInputBoostGainForPrototyping"},
    };
    // clang-format on

    RegisterHandlers(functions);
}

AudCtl::~AudCtl() = default;

void AudCtl::GetTargetVolumeMin(HLERequestContext& ctx) {
    LOG_DEBUG(Audio, "called.");

    // This service function is currently hardcoded on the
    // actual console to this value (as of 8.0.0).
    constexpr s32 target_min_volume = 0;

    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(ResultSuccess);
    rb.Push(target_min_volume);
}

void AudCtl::GetTargetVolumeMax(HLERequestContext& ctx) {
    LOG_DEBUG(Audio, "called.");

    // This service function is currently hardcoded on the
    // actual console to this value (as of 8.0.0).
    constexpr s32 target_max_volume = 15;

    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(ResultSuccess);
    rb.Push(target_max_volume);
}

} // namespace Service::Audio
