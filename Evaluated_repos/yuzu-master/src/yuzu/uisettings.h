// SPDX-FileCopyrightText: 2016 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <vector>
#include <QByteArray>
#include <QMetaType>
#include <QString>
#include <QStringList>
#include <QVector>
#include "common/common_types.h"
#include "common/settings.h"

namespace UISettings {

bool IsDarkTheme();

struct ContextualShortcut {
    QString keyseq;
    QString controller_keyseq;
    int context;
    bool repeat;
};

struct Shortcut {
    QString name;
    QString group;
    ContextualShortcut shortcut;
};

enum class Theme {
    Default,
    DefaultColorful,
    Dark,
    DarkColorful,
    MidnightBlue,
    MidnightBlueColorful,
};

using Themes = std::array<std::pair<const char*, const char*>, 6>;
extern const Themes themes;

struct GameDir {
    QString path;
    bool deep_scan = false;
    bool expanded = false;
    bool operator==(const GameDir& rhs) const {
        return path == rhs.path;
    }
    bool operator!=(const GameDir& rhs) const {
        return !operator==(rhs);
    }
};

struct Values {
    QByteArray geometry;
    QByteArray state;

    QByteArray renderwindow_geometry;

    QByteArray gamelist_header_state;

    QByteArray microprofile_geometry;
    Settings::Setting<bool> microprofile_visible{false, "microProfileDialogVisible"};

    Settings::Setting<bool> single_window_mode{true, "singleWindowMode"};
    Settings::Setting<bool> fullscreen{false, "fullscreen"};
    Settings::Setting<bool> display_titlebar{true, "displayTitleBars"};
    Settings::Setting<bool> show_filter_bar{true, "showFilterBar"};
    Settings::Setting<bool> show_status_bar{true, "showStatusBar"};

    Settings::Setting<bool> confirm_before_closing{true, "confirmClose"};
    Settings::Setting<bool> first_start{true, "firstStart"};
    Settings::Setting<bool> pause_when_in_background{false, "pauseWhenInBackground"};
    Settings::Setting<bool> mute_when_in_background{false, "muteWhenInBackground"};
    Settings::Setting<bool> hide_mouse{true, "hideInactiveMouse"};
    // Set when Vulkan is known to crash the application
    bool has_broken_vulkan = false;

    Settings::Setting<bool> select_user_on_boot{false, "select_user_on_boot"};

    // Discord RPC
    Settings::Setting<bool> enable_discord_presence{true, "enable_discord_presence"};

    Settings::Setting<bool> enable_screenshot_save_as{true, "enable_screenshot_save_as"};

    QString roms_path;
    QString symbols_path;
    QString game_dir_deprecated;
    bool game_dir_deprecated_deepscan;
    QVector<UISettings::GameDir> game_dirs;
    QStringList recent_files;
    QString language;

    QString theme;

    // Shortcut name <Shortcut, context>
    std::vector<Shortcut> shortcuts;

    Settings::Setting<uint32_t> callout_flags{0, "calloutFlags"};

    // multiplayer settings
    Settings::Setting<QString> multiplayer_nickname{{}, "nickname"};
    Settings::Setting<QString> multiplayer_ip{{}, "ip"};
    Settings::SwitchableSetting<uint, true> multiplayer_port{24872, 0, UINT16_MAX, "port"};
    Settings::Setting<QString> multiplayer_room_nickname{{}, "room_nickname"};
    Settings::Setting<QString> multiplayer_room_name{{}, "room_name"};
    Settings::SwitchableSetting<uint, true> multiplayer_max_player{8, 0, 8, "max_player"};
    Settings::SwitchableSetting<uint, true> multiplayer_room_port{24872, 0, UINT16_MAX,
                                                                  "room_port"};
    Settings::SwitchableSetting<uint, true> multiplayer_host_type{0, 0, 1, "host_type"};
    Settings::Setting<qulonglong> multiplayer_game_id{{}, "game_id"};
    Settings::Setting<QString> multiplayer_room_description{{}, "room_description"};
    std::pair<std::vector<std::string>, std::vector<std::string>> multiplayer_ban_list;

    // logging
    Settings::Setting<bool> show_console{false, "showConsole"};

    // Game List
    Settings::Setting<bool> show_add_ons{true, "show_add_ons"};
    Settings::Setting<uint32_t> game_icon_size{64, "game_icon_size"};
    Settings::Setting<uint32_t> folder_icon_size{48, "folder_icon_size"};
    Settings::Setting<uint8_t> row_1_text_id{3, "row_1_text_id"};
    Settings::Setting<uint8_t> row_2_text_id{2, "row_2_text_id"};
    std::atomic_bool is_game_list_reload_pending{false};
    Settings::Setting<bool> cache_game_list{true, "cache_game_list"};
    Settings::Setting<bool> favorites_expanded{true, "favorites_expanded"};
    QVector<u64> favorited_ids;

    // Compatibility List
    Settings::Setting<bool> show_compat{false, "show_compat"};

    // Size & File Types Column
    Settings::Setting<bool> show_size{true, "show_size"};
    Settings::Setting<bool> show_types{true, "show_types"};

    bool configuration_applied;
    bool reset_to_defaults;
    bool shortcut_already_warned{false};
    Settings::Setting<bool> disable_web_applet{true, "disable_web_applet"};
};

extern Values values;

} // namespace UISettings

Q_DECLARE_METATYPE(UISettings::GameDir*);
