﻿// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <string>
#include <QMetaType>
#include <QVariant>
#include "common/settings.h"
#include "yuzu/uisettings.h"

class QSettings;

namespace Core {
class System;
}

class Config {
public:
    enum class ConfigType {
        GlobalConfig,
        PerGameConfig,
        InputProfile,
    };

    explicit Config(const std::string& config_name = "qt-config",
                    ConfigType config_type = ConfigType::GlobalConfig);
    ~Config();

    void Reload();
    void Save();

    void ReadControlPlayerValue(std::size_t player_index);
    void SaveControlPlayerValue(std::size_t player_index);
    void ClearControlPlayerValues();

    const std::string& GetConfigFilePath() const;

    static const std::array<int, Settings::NativeButton::NumButtons> default_buttons;
    static const std::array<int, Settings::NativeMotion::NumMotions> default_motions;
    static const std::array<std::array<int, 4>, Settings::NativeAnalog::NumAnalogs> default_analogs;
    static const std::array<int, 2> default_stick_mod;
    static const std::array<int, 2> default_ringcon_analogs;
    static const std::array<int, Settings::NativeMouseButton::NumMouseButtons>
        default_mouse_buttons;
    static const std::array<int, Settings::NativeKeyboard::NumKeyboardKeys> default_keyboard_keys;
    static const std::array<int, Settings::NativeKeyboard::NumKeyboardMods> default_keyboard_mods;
    static const std::array<UISettings::Shortcut, 22> default_hotkeys;

    static constexpr UISettings::Theme default_theme{
#ifdef _WIN32
        UISettings::Theme::DarkColorful
#else
        UISettings::Theme::DefaultColorful
#endif
    };

private:
    void Initialize(const std::string& config_name);
    bool IsCustomConfig();

    void ReadValues();
    void ReadPlayerValue(std::size_t player_index);
    void ReadDebugValues();
    void ReadKeyboardValues();
    void ReadMouseValues();
    void ReadTouchscreenValues();
    void ReadMotionTouchValues();
    void ReadHidbusValues();
    void ReadIrCameraValues();

    // Read functions bases off the respective config section names.
    void ReadAudioValues();
    void ReadControlValues();
    void ReadCoreValues();
    void ReadDataStorageValues();
    void ReadDebuggingValues();
    void ReadServiceValues();
    void ReadDisabledAddOnValues();
    void ReadMiscellaneousValues();
    void ReadPathValues();
    void ReadCpuValues();
    void ReadRendererValues();
    void ReadScreenshotValues();
    void ReadShortcutValues();
    void ReadSystemValues();
    void ReadUIValues();
    void ReadUIGamelistValues();
    void ReadUILayoutValues();
    void ReadWebServiceValues();
    void ReadMultiplayerValues();

    void SaveValues();
    void SavePlayerValue(std::size_t player_index);
    void SaveDebugValues();
    void SaveMouseValues();
    void SaveTouchscreenValues();
    void SaveMotionTouchValues();
    void SaveHidbusValues();
    void SaveIrCameraValues();

    // Save functions based off the respective config section names.
    void SaveAudioValues();
    void SaveControlValues();
    void SaveCoreValues();
    void SaveDataStorageValues();
    void SaveDebuggingValues();
    void SaveNetworkValues();
    void SaveDisabledAddOnValues();
    void SaveMiscellaneousValues();
    void SavePathValues();
    void SaveCpuValues();
    void SaveRendererValues();
    void SaveScreenshotValues();
    void SaveShortcutValues();
    void SaveSystemValues();
    void SaveUIValues();
    void SaveUIGamelistValues();
    void SaveUILayoutValues();
    void SaveWebServiceValues();
    void SaveMultiplayerValues();

    /**
     * Reads a setting from the qt_config.
     *
     * @param name The setting's identifier
     * @param default_value The value to use when the setting is not already present in the config
     */
    QVariant ReadSetting(const QString& name) const;
    QVariant ReadSetting(const QString& name, const QVariant& default_value) const;

    /**
     * Only reads a setting from the qt_config if the current config is a global config, or if the
     * current config is a custom config and the setting is overriding the global setting. Otherwise
     * it does nothing.
     *
     * @param setting The variable to be modified
     * @param name The setting's identifier
     * @param default_value The value to use when the setting is not already present in the config
     */
    template <typename Type>
    void ReadSettingGlobal(Type& setting, const QString& name, const QVariant& default_value) const;

    /**
     * Writes a setting to the qt_config.
     *
     * @param name The setting's idetentifier
     * @param value Value of the setting
     * @param default_value Default of the setting if not present in qt_config
     * @param use_global Specifies if the custom or global config should be in use, for custom
     * configs
     */
    void WriteSetting(const QString& name, const QVariant& value);
    void WriteSetting(const QString& name, const QVariant& value, const QVariant& default_value);
    void WriteSetting(const QString& name, const QVariant& value, const QVariant& default_value,
                      bool use_global);

    /**
     * Reads a value from the qt_config and applies it to the setting, using its label and default
     * value. If the config is a custom config, this will also read the global state of the setting
     * and apply that information to it.
     *
     * @param The setting
     */
    template <typename Type, bool ranged>
    void ReadGlobalSetting(Settings::SwitchableSetting<Type, ranged>& setting);

    /**
     * Sets a value to the qt_config using the setting's label and default value. If the config is a
     * custom config, it will apply the global state, and the custom value if needed.
     *
     * @param The setting
     */
    template <typename Type, bool ranged>
    void WriteGlobalSetting(const Settings::SwitchableSetting<Type, ranged>& setting);

    /**
     * Reads a value from the qt_config using the setting's label and default value and applies the
     * value to the setting.
     *
     * @param The setting
     */
    template <typename Type, bool ranged>
    void ReadBasicSetting(Settings::Setting<Type, ranged>& setting);

    /** Sets a value from the setting in the qt_config using the setting's label and default value.
     *
     * @param The setting
     */
    template <typename Type, bool ranged>
    void WriteBasicSetting(const Settings::Setting<Type, ranged>& setting);

    ConfigType type;
    std::unique_ptr<QSettings> qt_config;
    std::string qt_config_loc;
    bool global;
};

// These metatype declarations cannot be in common/settings.h because core is devoid of QT
Q_DECLARE_METATYPE(Settings::CPUAccuracy);
Q_DECLARE_METATYPE(Settings::GPUAccuracy);
Q_DECLARE_METATYPE(Settings::FullscreenMode);
Q_DECLARE_METATYPE(Settings::NvdecEmulation);
Q_DECLARE_METATYPE(Settings::ResolutionSetup);
Q_DECLARE_METATYPE(Settings::ScalingFilter);
Q_DECLARE_METATYPE(Settings::AntiAliasing);
Q_DECLARE_METATYPE(Settings::RendererBackend);
Q_DECLARE_METATYPE(Settings::ShaderBackend);
