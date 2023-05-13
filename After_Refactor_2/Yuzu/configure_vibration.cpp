#include "common/settings.h"
#include "core/hid/emulated_controller.h"
#include "core/hid/hid_core.h"
#include "core/hid/hid_types.h"
#include "ui_configure_vibration.h"
#include "yuzu/configuration/configure_vibration.h"

namespace {

constexpr std::size_t kNumPlayers = 8;

}

ConfigureVibration::ConfigureVibration(QWidget* parent, Core::HID::HIDCore& hid_core)
    : QDialog(parent),
      ui(std::make_unique<Ui::ConfigureVibration>()),
      hid_core_(hid_core),
      vibration_groupboxes_(kNumPlayers),
      vibration_spinboxes_(kNumPlayers) {
  ui->setupUi(this);

  const auto& players = Settings::values.players.GetValue();

  for (std::size_t i = 0; i < kNumPlayers; ++i) {
    auto* controller = hid_core_.GetEmulatedControllerByIndex(i);
    controller_callback_keys_[i] = controller->SetCallback({
        .on_change = [this, i](Core::HID::ControllerTriggerType type) {
          if (type == Core::HID::ControllerTriggerType::Button) {
            VibrateController(i);
          }
        },
        .is_npad_service = false,
    });

    vibration_groupboxes_[i] = ui->findChild<QGroupBox*>(QString("vibrationGroupPlayer%1").arg(i+1));
    vibration_spinboxes_[i] = ui->findChild<QSpinBox*>(QString("vibrationSpinPlayer%1").arg(i+1));
    vibration_groupboxes_[i]->setChecked(players[i].vibration_enabled);
    vibration_spinboxes_[i]->setValue(players[i].vibration_strength);
  }

  ui->checkBoxAccurateVibration->setChecked(Settings::values.enable_accurate_vibrations.GetValue());

  if (!Settings::IsConfiguringGlobal()) {
    ui->checkBoxAccurateVibration->setDisabled(true);
  }

  RetranslateUI();
}

ConfigureVibration::~ConfigureVibration() {
  StopVibrations();

  for (auto& controller_callback_key : controller_callback_keys_) {
    auto* controller = hid_core_.GetEmulatedControllerByIndex(&controller_callback_key - &controller_callback_keys_[0]);
    controller->DeleteCallback(controller_callback_key);
  }
}

void ConfigureVibration::ApplyConfiguration() {
  auto& players = Settings::values.players.GetValue();

  for (std::size_t i = 0; i < kNumPlayers; ++i) {
    players[i].vibration_enabled = vibration_groupboxes_[i]->isChecked();
    players[i].vibration_strength = vibration_spinboxes_[i]->value();
  }

  Settings::values.enable_accurate_vibrations.SetValue(ui->checkBoxAccurateVibration->isChecked());
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

void ConfigureVibration::VibrateController(std::size_t player_index) {
  const auto& player = Settings::values.players.GetValue()[player_index];
  const int vibration_strength = vibration_spinboxes_[player_index]->value();
  const Core::HID::VibrationValue vibration{
      .low_amplitude = 1.0f,
      .low_frequency = 160.0f,
      .high_amplitude = 1.0f,
      .high_frequency = 320.0f,
  };

  auto* controller = hid_core_.GetEmulatedControllerByIndex(player_index);

  if (!player.vibration_enabled) {
    StopVibrations();
  } else {
    controller->SetVibration(0, vibration);
    controller->SetVibration(1, vibration);
  }
}

void ConfigureVibration::StopVibrations() {
  const Core::HID::VibrationValue default_vibration = Core::HID::DEFAULT_VIBRATION_VALUE;

  for (std::size_t i = 0; i < kNumPlayers; ++i) {
    auto* controller = hid_core_.GetEmulatedControllerByIndex(i);
    controller->SetVibration(0, default_vibration);
    controller->SetVibration(1, default_vibration);
  }
}