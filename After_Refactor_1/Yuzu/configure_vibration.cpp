#include "common/settings.h"
#include "core/hid/emulated_controller.h"
#include "core/hid/hid_core.h"
#include "core/hid/hid_types.h"
#include "ui_configure_vibration.h"
#include "yuzu/configuration/configure_vibration.h"

constexpr int NUM_PLAYERS = 8;
constexpr float LOW_FREQUENCY = 160.0f;
constexpr float HIGH_FREQUENCY = 320.0f;
constexpr float VIBRATION_AMPLITUDE = 1.0f;
constexpr Core::HID::VibrationValue DEFAULT_VIBRATION_VALUE = {
    .low_amplitude = 0.0f, .low_frequency = 0.0f, .high_amplitude = 0.0f, .high_frequency = 0.0f,
};

class PlayerVibrationSettings {
public:
    bool enabled = false;
    int strength = 0;
};

class ConfigureVibration : public QDialog {
    Q_OBJECT

public:
    ConfigureVibration(QWidget* parent, Core::HID::HIDCore& hid_core_);
    ~ConfigureVibration();

public slots:
    void ApplyConfiguration();

protected:
    void changeEvent(QEvent* event) override;

private:
    void RetranslateUI();

    void VibrateController(Core::HID::ControllerTriggerType type, std::size_t player_index);
    void StopVibrations();

    QWidget* parent_;
    std::unique_ptr<Ui::ConfigureVibration> ui;
    Core::HID::HIDCore& hid_core;
    std::array<QGroupBox*, NUM_PLAYERS> vibration_groupboxes;
    std::array<QSpinBox*, NUM_PLAYERS> vibration_spinboxes;
    std::array<std::size_t, NUM_PLAYERS> controller_callback_key;
    std::array<PlayerVibrationSettings, NUM_PLAYERS> player_vibration_settings{};
};

ConfigureVibration::ConfigureVibration(QWidget* parent, Core::HID::HIDCore& hid_core_)
    : QDialog(parent),
      parent_(parent),
      ui(std::make_unique<Ui::ConfigureVibration>()),
      hid_core(hid_core_) {
    ui->setupUi(this);

    vibration_groupboxes = {
        ui->vibrationGroupPlayer1, ui->vibrationGroupPlayer2, ui->vibrationGroupPlayer3,
        ui->vibrationGroupPlayer4, ui->vibrationGroupPlayer5, ui->vibrationGroupPlayer6,
        ui->vibrationGroupPlayer7, ui->vibrationGroupPlayer8,
    };

    vibration_spinboxes = {
        ui->vibrationSpinPlayer1, ui->vibrationSpinPlayer2, ui->vibrationSpinPlayer3,
        ui->vibrationSpinPlayer4, ui->vibrationSpinPlayer5, ui->vibrationSpinPlayer6,
        ui->vibrationSpinPlayer7, ui->vibrationSpinPlayer8,
    };

    const auto& players = Settings::values.players.GetValue();

    for (std::size_t i = 0; i < NUM_PLAYERS; ++i) {
        auto controller = hid_core.GetEmulatedControllerByIndex(i);
        Core::HID::ControllerUpdateCallback engine_callback{
            .on_change = [this, i](Core::HID::ControllerTriggerType type) {
                VibrateController(type, i);
            },
            .is_npad_service = false,
        };
        controller_callback_key[i] = controller->SetCallback(engine_callback);

        auto& player_settings = player_vibration_settings[i];
        player_settings.enabled = players[i].vibration_enabled;
        player_settings.strength = players[i].vibration_strength;

        vibration_groupboxes[i]->setChecked(player_settings.enabled);
        vibration_spinboxes[i]->setValue(player_settings.strength);
    }

    ui->checkBoxAccurateVibration->setChecked(Settings::values.enable_accurate_vibrations.GetValue());

    if (!Settings::IsConfiguringGlobal()) {
        ui->checkBoxAccurateVibration->setDisabled(true);
    }

    RetranslateUI();
}

ConfigureVibration::~ConfigureVibration() {
    StopVibrations();

    for (std::size_t i = 0; i < NUM_PLAYERS; ++i) {
        auto controller = hid_core.GetEmulatedControllerByIndex(i);
        controller->DeleteCallback(controller_callback_key[i]);
    }
};

void ConfigureVibration::ApplyConfiguration() {
    auto& players = Settings::values.players.GetValue();

    for (std::size_t i = 0; i < NUM_PLAYERS; ++i) {
        auto& player_settings = player_vibration_settings[i];
        players[i].vibration_enabled = player_settings.enabled;
        players[i].vibration_strength = player_settings.strength;
    }

    Settings::values.enable_accurate_vibrations.SetValue(
        ui->checkBoxAccurateVibration->isChecked());
}

void ConfigureVibration::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QDialog::changeEvent(event);
}

void ConfigureVibration::RetranslateUI() {
    ui->retranslateUi(this);
}

void ConfigureVibration::VibrateController(Core::HID::ControllerTriggerType type,
                                           std::size_t player_index) {
    if (type != Core::HID::ControllerTriggerType::Button) {
        return;
    }

    auto& player_settings = player_vibration_settings[player_index];
    if (!player_settings.enabled) {
        StopVibrations();
        return;
    }

    auto controller = hid_core.GetEmulatedControllerByIndex(player_index);
    const auto& buttons = controller->GetButtonsValues();

    bool button_is_pressed = false;
    for (const auto& button : buttons) {
        if (button.value) {
            button_is_pressed = true;
            break;
        }
    }

    if (!button_is_pressed) {
        StopVibrations();
        return;
    }

    const Core::HID::VibrationValue vibration{
        .low_amplitude = VIBRATION_AMPLITUDE,
        .low_frequency = LOW_FREQUENCY,
        .high_amplitude = VIBRATION_AMPLITUDE,
        .high_frequency = HIGH_FREQUENCY,
    };
    controller->SetVibration(0, vibration);
    controller->SetVibration(1, vibration);
}

void ConfigureVibration::StopVibrations() {
    for (std::size_t i = 0; i < NUM_PLAYERS; ++i) {
        auto controller = hid_core.GetEmulatedControllerByIndex(i);
        controller->SetVibration(0, DEFAULT_VIBRATION_VALUE);
        controller->SetVibration(1, DEFAULT_VIBRATION_VALUE);
    }
}