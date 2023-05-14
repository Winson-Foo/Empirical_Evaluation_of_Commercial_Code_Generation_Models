// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>

#include "common/logging/filter.h"
#include "common/string_util.h"

namespace Common::Log {

namespace {

template <typename It>
Level GetLevelByName(const It begin, const It end) {
    static constexpr u8 levelCount = static_cast<u8>(Level::Count);
    for (u8 i = 0; i < levelCount; ++i) {
        const char* levelName = GetLevelName(static_cast<Level>(i));
        if (Common::ComparePartialString(begin, end, levelName)) {
            return static_cast<Level>(i);
        }
    }
    return Level::Count;
}

template <typename It>
Class GetClassByName(const It begin, const It end) {
    static constexpr u8 classCount = static_cast<u8>(Class::Count);
    for (u8 i = 0; i < classCount; ++i) {
        const char* className = GetLogClassName(static_cast<Class>(i));
        if (Common::ComparePartialString(begin, end, className)) {
            return static_cast<Class>(i);
        }
    }
    return Class::Count;
}

template <typename Iterator>
bool ParseFilterRule(Filter& instance, Iterator begin, Iterator end) {
    static constexpr char kLevelSeparator = ':';
    if (const auto levelSeparator = std::find(begin, end, kLevelSeparator); levelSeparator == end) {
        LOG_ERROR(Log, "Invalid log filter. Must specify a log level after `:`: {}", std::string(begin, end));
        return false;
    }

    const Level level = GetLevelByName(levelSeparator + 1, end);
    if (level == Level::Count) {
        LOG_ERROR(Log, "Unknown log level in filter: {}", std::string(begin, end));
        return false;
    }

    if (Common::ComparePartialString(begin, levelSeparator, "*")) {
        instance.ResetAll(level);
        return true;
    }

    const Class logClass = GetClassByName(begin, levelSeparator);
    if (logClass == Class::Count) {
        LOG_ERROR(Log, "Unknown log class in filter: {}", std::string(begin, end));
        return false;
    }

    instance.SetClassLevel(logClass, level);
    return true;
}

} // namespace

const char* GetLogClassName(Class logClass) {
    static constexpr std::array<std::pair<Class, const char*>, 94> kLogClasses = {{
        {Class::Log, "Log"},
        {Class::Common, "Common"},
        {Class::Common_Filesystem, "Common.Filesystem"},
        {Class::Common_Memory, "Common.Memory"},
        {Class::Core, "Core"},
        {Class::Core_ARM, "Core.ARM"},
        {Class::Core_Timing, "Core.Timing"},
        {Class::Config, "Config"},
        {Class::Debug, "Debug"},
        {Class::Debug_Emulated, "Debug.Emulated"},
        {Class::Debug_GPU, "Debug.GPU"},
        {Class::Debug_Breakpoint, "Debug.Breakpoint"},
        {Class::Debug_GDBStub, "Debug.GDBStub"},
        {Class::Kernel, "Kernel"},
        {Class::Kernel_SVC, "Kernel.SVC"},
        {Class::Service, "Service"},
        {Class::Service_ACC, "Service.ACC"},
        {Class::Service_Audio, "Service.Audio"},
        {Class::Service_AM, "Service.AM"},
        {Class::Service_AOC, "Service.AOC"},
        {Class::Service_APM, "Service.APM"},
        {Class::Service_ARP, "Service.ARP"},
        {Class::Service_BCAT, "Service.BCAT"},
        {Class::Service_BPC, "Service.BPC"},
        {Class::Service_BGTC, "Service.BGTC"},
        {Class::Service_BTDRV, "Service.BTDRV"},
        {Class::Service_BTM, "Service.BTM"},
        {Class::Service_Capture, "Service.Capture"},
        {Class::Service_ERPT, "Service.ERPT"},
        {Class::Service_ETicket, "Service.ETicket"},
        {Class::Service_EUPLD, "Service.EUPLD"},
        {Class::Service_Fatal, "Service.Fatal"},
        {Class::Service_FGM, "Service.FGM"},
        {Class::Service_Friend, "Service.Friend"},
        {Class::Service_FS, "Service.FS"},
        {Class::Service_GRC, "Service.GRC"},
        {Class::Service_HID, "Service.HID"},
        {Class::Service_IRS, "Service.IRS"},
        {Class::Service_JIT, "Service.JIT"},
        {Class::Service_LBL, "Service.LBL"},
        {Class::Service_LDN, "Service.LDN"},
        {Class::Service_LDR, "Service.LDR"},
        {Class::Service_LM, "Service.LM"},
        {Class::Service_Migration, "Service.Migration"},
        {Class::Service_Mii, "Service.Mii"},
        {Class::Service_MM, "Service.MM"},
        {Class::Service_MNPP, "Service.MNPP"},
        {Class::Service_NCM, "Service.NCM"},
        {Class::Service_NFC, "Service.NFC"},
        {Class::Service_NFP, "Service.NFP"},
        {Class::Service_NGCT, "Service.NGCT"},
        {Class::Service_NIFM, "Service.NIFM"},
        {Class::Service_NIM, "Service.NIM"},
        {Class::Service_NOTIF, "Service.NOTIF"},
        {Class::Service_NPNS, "Service.NPNS"},
        {Class::Service_NS, "Service.NS"},
        {Class::Service_NVDRV, "Service.NVDRV"},
        {Class::Service_Nvnflinger, "Service.Nvnflinger"},
        {Class::Service_OLSC, "Service.OLSC"},
        {Class::Service_PCIE, "Service.PCIE"},
        {Class::Service_PCTL, "Service.PCTL"},
        {Class::Service_PCV, "Service.PCV"},
        {Class::Service_PM, "Service.PM"},
        {Class::Service_PREPO, "Service.PREPO"},
        {Class::Service_PSC, "Service.PSC"},
        {Class::Service_PTM, "Service.PTM"},
        {Class::Service_SET, "Service.SET"},
        {Class::Service_SM, "Service.SM"},
        {Class::Service_SPL, "Service.SPL"},
        {Class::Service_SSL, "Service.SSL"},
        {Class::Service_TCAP, "Service.TCAP"},
        {Class::Service_Time, "Service.Time"},
        {Class::Service_USB, "Service.USB"},
        {Class::Service_VI, "Service.VI"},
        {Class::Service_WLAN, "Service.WLAN"},
        {Class::HW, "HW"},
        {Class::HW_Memory, "HW.Memory"},
        {Class::HW_LCD, "HW.LCD"},
        {Class::HW_GPU, "HW.GPU"},
        {Class::HW_AES, "HW.AES"},
        {Class::IPC, "IPC"},
        {Class::Frontend, "Frontend"},
        {Class::Render, "Render"},
        {Class::Render_Software, "Render.Software"},
        {Class::Render_OpenGL, "Render.OpenGL"},
        {Class::Render_Vulkan, "Render.Vulkan"},
        {Class::Shader, "Shader"},
        {Class::Shader_SPIRV, "Shader.SPIRV"},
        {Class::Shader_GLASM, "Shader.GLASM"},
        {Class::Shader_GLSL, "Shader.GLSL"},
        {Class::Audio, "Audio"},
        {Class::Audio_DSP, "Audio.DSP"},
        {Class::Audio_Sink, "Audio.Sink"},
        {Class::Input, "Input"},
        {Class::Network, "Network"},
        {Class::Loader, "Loader"},
        {Class::CheatEngine, "CheatEngine"},
        {Class::Crypto, "Crypto"},
    }};
    for (const auto& [log_cls, name] : kLogClasses) {
        if (log_cls == logClass) {
            return name;
        }
    }
    return "Invalid";
}

const char* GetLevelName(Level logLevel) {
    static constexpr std::array<const char*, 6> kLevelNames = {{
        "Trace", "Debug", "Info", "Warning", "Error", "Critical"
    }};
    static constexpr std::size_t kLevelCount = kLevelNames.size();
    if (static_cast<std::size_t>(logLevel) < kLevelCount) {
        return kLevelNames[static_cast<std::size_t>(logLevel)];
    }
    return "Invalid";
}

Filter::Filter(Level defaultLevel) {
    ResetAll(defaultLevel);
}

void Filter::ResetAll(Level level) {
    classLevels.fill(level);
}

void Filter::SetClassLevel(Class logClass, Level level) {
    classLevels[static_cast<std::size_t>(logClass)] = level;
}

void Filter::ParseFilterString(std::string_view filterView) {
    auto clauseBegin = filterView.cbegin();
    while (clauseBegin != filterView.cend()) {
        auto clauseEnd = std::find(clauseBegin, filterView.cend(), ' ');

        // If clause isn't empty
        if (clauseEnd != clauseBegin) {
            ParseFilterRule(*this, clauseBegin, clauseEnd);
        }

        if (clauseEnd != filterView.cend()) {
            // Skip over the whitespace
            ++clauseEnd;
        }
        clauseBegin = clauseEnd;
    }
}

bool Filter::CheckMessage(Class logClass, Level level) const {
    return static_cast<int>(level) >= static_cast<int>(classLevels[static_cast<std::size_t>(logClass)]);
}

bool Filter::IsDebug() const {
    static constexpr std

