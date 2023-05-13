// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include "common/logging/log.h"
#include "common/settings.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/set/set.h"

namespace Service::Set {
namespace {
constexpr std::array<LanguageCode, 18> available_language_codes = {{
    LanguageCode::JA,
    LanguageCode::EN_US,
    LanguageCode::FR,
    LanguageCode::DE,
    LanguageCode::IT,
    LanguageCode::ES,
    LanguageCode::ZH_CN,
    LanguageCode::KO,
    LanguageCode::NL,
    LanguageCode::PT,
    LanguageCode::RU,
    LanguageCode::ZH_TW,
    LanguageCode::EN_GB,
    LanguageCode::FR_CA,
    LanguageCode::ES_419,
    LanguageCode::ZH_HANS,
    LanguageCode::ZH_HANT,
    LanguageCode::PT_BR,
}};

enum class KeyboardLayout : u64 {
    Japanese = 0,
    EnglishUs = 1,
    EnglishUsInternational = 2,
    EnglishUk = 3,
    French = 4,
    FrenchCa = 5,
    Spanish = 6,
    SpanishLatin = 7,
    German = 8,
    Italian = 9,
    Portuguese = 10,
    Russian = 11,
    Korean = 12,
    ChineseSimplified = 13,
    ChineseTraditional = 14,
};

constexpr std::array<std::pair<LanguageCode, KeyboardLayout>, 18> language_to_layout{{
    {LanguageCode::JA, KeyboardLayout::Japanese},
    {LanguageCode::EN_US, KeyboardLayout::EnglishUs},
    {LanguageCode::FR, KeyboardLayout::French},
    {LanguageCode::DE, KeyboardLayout::German},
    {LanguageCode::IT, KeyboardLayout::Italian},
    {LanguageCode::ES, KeyboardLayout::Spanish},
    {LanguageCode::ZH_CN, KeyboardLayout::ChineseSimplified},
    {LanguageCode::KO, KeyboardLayout::Korean},
    {LanguageCode::NL, KeyboardLayout::EnglishUsInternational},
    {LanguageCode::PT, KeyboardLayout::Portuguese},
    {LanguageCode::RU, KeyboardLayout::Russian},
    {LanguageCode::ZH_TW, KeyboardLayout::ChineseTraditional},
    {LanguageCode::EN_GB, KeyboardLayout::EnglishUk},
    {LanguageCode::FR_CA, KeyboardLayout::FrenchCa},
    {LanguageCode::ES_419, KeyboardLayout::SpanishLatin},
    {LanguageCode::ZH_HANS, KeyboardLayout::ChineseSimplified},
    {LanguageCode::ZH_HANT, KeyboardLayout::ChineseTraditional},
    {LanguageCode::PT_BR, KeyboardLayout::Portuguese},
}};

constexpr std::size_t PRE_4_0_0_MAX_ENTRIES = 0xF;
constexpr std::size_t POST_4_0_0_MAX_ENTRIES = 0x40;

constexpr Result ResultInvalidLanguage{ErrorModule::Settings, 625};

void PushResponseLanguageCode(HLERequestContext& ctx, std::size_t num_language_codes) {
    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(ResultSuccess);
    rb.Push(static_cast<u32>(num_language_codes));
}

void GetAvailableLanguageCodesImpl(HLERequestContext& ctx, std::size_t max_entries) {
    const std::size_t requested_amount = ctx.GetWriteBufferNumElements<LanguageCode>();
    const std::size_t max_amount = std::min(requested_amount, max_entries);
    const std::size_t copy_amount = std::min(available_language_codes.size(), max_amount);
    const std::size_t copy_size = copy_amount * sizeof(LanguageCode);

    ctx.WriteBuffer(available_language_codes.data(), copy_size);
    PushResponseLanguageCode(ctx, copy_amount);
}

void GetKeyCodeMapImpl(HLERequestContext& ctx) {
    const auto language_code = available_language_codes[Settings::values.language_index.GetValue()];
    const auto key_code =
        std::find_if(language_to_layout.cbegin(), language_to_layout.cend(),
                     [=](const auto& element) { return element.first == language_code; });
    KeyboardLayout layout = KeyboardLayout::EnglishUs;
    if (key_code == language_to_layout.cend()) {
        LOG_ERROR(Service_SET,
                  "Could not find keyboard layout for language index {}, defaulting to English us",
                  Settings::values.language_index.GetValue());
    } else {
        layout = key_code->second;
    }

    ctx.WriteBuffer(layout);

    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(ResultSuccess);
}
} // Anonymous namespace

LanguageCode GetLanguageCodeFromIndex(std::size_t index) {
    return available_language_codes.at(index);
}

void SET::GetAvailableLanguageCodes(HLERequestContext& ctx) {
    LOG_DEBUG(Service_SET, "called");

    GetAvailableLanguageCodesImpl(ctx, PRE_4_0_0_MAX_ENTRIES);
}

void SET::MakeLanguageCode(HLERequestContext& ctx) {
    IPC::RequestParser rp{ctx};
    const auto index = rp.Pop<u32>();

    if (index >= available_language_codes.size()) {
        LOG_ERROR(Service_SET, "Invalid language code index! index={}", index);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(Set::ResultInvalidLanguage);
        return;
    }

    IPC::ResponseBuilder rb{ctx, 4};
    rb.Push(ResultSuccess);
    rb.PushEnum(available_language_codes[index]);
}

void SET::GetAvailableLanguageCodes2(HLERequestContext& ctx) {
    LOG_DEBUG(Service_SET, "called");

    GetAvailableLanguageCodesImpl(ctx, POST_4_0_0_MAX_ENTRIES);
}

void SET::GetAvailableLanguageCodeCount(HLERequestContext& ctx) {
    LOG_DEBUG(Service_SET, "called");

    PushResponseLanguageCode(ctx, PRE_4_0_0_MAX_ENTRIES);
}

void SET::GetAvailableLanguageCodeCount2(HLERequestContext& ctx) {
    LOG_DEBUG(Service_SET, "called");

    PushResponseLanguageCode(ctx, POST_4_0_0_MAX_ENTRIES);
}

void SET::GetQuestFlag(HLERequestContext& ctx) {
    LOG_DEBUG(Service_SET, "called");

    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(ResultSuccess);
    rb.Push(static_cast<u32>(Settings::values.quest_flag.GetValue()));
}

void SET::GetLanguageCode(HLERequestContext& ctx) {
    LOG_DEBUG(Service_SET, "called {}", Settings::values.language_index.GetValue());

    IPC::ResponseBuilder rb{ctx, 4};
    rb.Push(ResultSuccess);
    rb.PushEnum(available_language_codes[Settings::values.language_index.GetValue()]);
}

void SET::GetRegionCode(HLERequestContext& ctx) {
    LOG_DEBUG(Service_SET, "called");

    IPC::ResponseBuilder rb{ctx, 3};
    rb.Push(ResultSuccess);
    rb.Push(Settings::values.region_index.GetValue());
}

void SET::GetKeyCodeMap(HLERequestContext& ctx) {
    LOG_DEBUG(Service_SET, "Called {}", ctx.Description());
    GetKeyCodeMapImpl(ctx);
}

void SET::GetKeyCodeMap2(HLERequestContext& ctx) {
    LOG_DEBUG(Service_SET, "Called {}", ctx.Description());
    GetKeyCodeMapImpl(ctx);
}

void SET::GetDeviceNickName(HLERequestContext& ctx) {
    LOG_DEBUG(Service_SET, "called");
    IPC::ResponseBuilder rb{ctx, 2};
    rb.Push(ResultSuccess);
    ctx.WriteBuffer(Settings::values.device_name.GetValue());
}

SET::SET(Core::System& system_) : ServiceFramework{system_, "set"} {
    // clang-format off
    static const FunctionInfo functions[] = {
        {0, &SET::GetLanguageCode, "GetLanguageCode"},
        {1, &SET::GetAvailableLanguageCodes, "GetAvailableLanguageCodes"},
        {2, &SET::MakeLanguageCode, "MakeLanguageCode"},
        {3, &SET::GetAvailableLanguageCodeCount, "GetAvailableLanguageCodeCount"},
        {4, &SET::GetRegionCode, "GetRegionCode"},
        {5, &SET::GetAvailableLanguageCodes2, "GetAvailableLanguageCodes2"},
        {6, &SET::GetAvailableLanguageCodeCount2, "GetAvailableLanguageCodeCount2"},
        {7, &SET::GetKeyCodeMap, "GetKeyCodeMap"},
        {8, &SET::GetQuestFlag, "GetQuestFlag"},
        {9, &SET::GetKeyCodeMap2, "GetKeyCodeMap2"},
        {10, nullptr, "GetFirmwareVersionForDebug"},
        {11, &SET::GetDeviceNickName, "GetDeviceNickName"},
    };
    // clang-format on

    RegisterHandlers(functions);
}

SET::~SET() = default;

} // namespace Service::Set
