#include "common/settings.h"
#include "core/hid/emulated_controller.h"
#include "core/hid/hid_core.h"
#include "core/hid/hid_types.h"
#include "ui_configure_vibration.h"
#include "yuzu/configuration/configure_vibration.h"

const int kNumPlayers = 8;

class ConfigureVibration : public QDialog {
  Q_OBJECT

 public:
  ConfigureVibration(QWidget* parent, Core::HID::HIDCore& hid_core);
  ~ConfigureVibration();

 private slots:
  void ApplyConfiguration();
  void changeEvent(QEvent* event);

 private:
  void InitializeCallbacks();
  void RetranslateUI();

  void VibrateController(Core::HID::ControllerTriggerType type, std::size_t player_index);
  void StopVibrations();

  std::unique_ptr<Ui::ConfigureVibration> ui_;
  Core::HID::HIDCore& hid_core_;
  std::vector<QGroupBox*> vibration_groupboxes_;
  std::vector<QSpinBox*> vibration_spinboxes_;
  std::vector<Core::HID::CallbackKey> controller_callback_key_;
  bool is_configuring_global_;
};

namespace {
constexpr int kLowFrequency = 160;
constexpr int kHighFrequency = 320;
constexpr float kDefaultAmplitude = 1.0f;
constexpr int kDefaultVibrationStrength = 50;
const Core::HID::VibrationValue kDefaultVibrationValue{
    .low_amplitude = kDefaultAmplitude,
    .low_frequency = kLowFrequency,
    .high_amplitude = kDefaultAmplitude,
    .high_frequency = kHighFrequency,
};
}  // namespace

ConfigureVibration::ConfigureVibration(QWidget* parent, Core::HID::HIDCore& hid_core)
    : QDialog(parent), ui_(std::make_unique<Ui::ConfigureVibration>()), hid_core_(hid_core) {
  ui_->setupUi(this);

  vibration_groupboxes_ = {
      ui_->vibrationGroupPlayer1, ui_->vibrationGroupPlayer2, ui_->vibrationGroupPlayer3,
      ui_->vibrationGroupPlayer4, ui_->vibrationGroupPlayer5, ui_->vibrationGroupPlayer6,
      ui_->vibrationGroupPlayer7, ui_->vibrationGroupPlayer8,
  };

  vibration_spinboxes_ = {
      ui_->vibrationSpinPlayer1, ui_->vibrationSpinPlayer2, ui_->vibrationSpinPlayer3,
      ui_->vibrationSpinPlayer4, ui_->vibrationSpinPlayer5, ui_->vibrationSpinPlayer6,
      ui_->vibrationSpinPlayer7, ui_->vibrationSpinPlayer8,
  };

  is_configuring_global_ = Settings::IsConfiguringGlobal();

  InitializeCallbacks();
  RetranslateUI();
}

ConfigureVibration::~ConfigureVibration() {
  StopVibrations();
  for (std::size_t i = 0; i < kNumPlayers; ++i) {
    auto controller = hid_core_.GetEmulatedControllerByIndex(i);
    controller->DeleteCallback(controller_callback_key_[i]);
  }
};

void ConfigureVibration::ApplyConfiguration() {
  auto& players = Settings::values.players.GetValue();

  for (std::size_t i = 0; i < kNumPlayers; ++i) {
    players[i].vibration_enabled = vibration_groupboxes_[i]->isChecked();
    players[i].vibration_strength = vibration_spinboxes_[i]->value();
  }

  Settings::values.enable_accurate_vibrations.SetValue(ui_->checkBoxAccurateVibration->isChecked());
}

void ConfigureVibration::changeEvent(QEvent* event) {
  if (event->type() == QEvent::LanguageChange) {
    RetranslateUI();
  }

  QDialog::changeEvent(event);
}

void ConfigureVibration::InitializeCallbacks() {
  auto& players = Settings::values.players.GetValue();
  for (std::size_t i = 0; i < kNumPlayers; ++i) {
    auto controller = hid_core_.GetEmulatedControllerByIndex(i);
    Core::HID::ControllerUpdateCallback engine_callback{
        .on_change = [this, i](Core::HID::ControllerTriggerType type) { VibrateController(type, i); },
        .is_npad_service = false,
    };
    controller_callback_key_.push_back(controller->SetCallback(engine_callback));
    vibration_groupboxes_[i]->setChecked(players[i].vibration_enabled);
    vibration_spinboxes_[i]->setValue(players[i].vibration_strength);
  }
}

void ConfigureVibration::RetranslateUI() {
  ui_->retranslateUi(this);
  if (!is_configuring_global_) {
    ui_->checkBoxAccurateVibration->setDisabled(true);
  }
}

void ConfigureVibration::VibrateController(Core::HID::ControllerTriggerType type,
                                           std::size_t player_index) {
  if (type != Core::HID::ControllerTriggerType::Button) {
    return;
  }

  auto& player = Settings::values.players.GetValue()[player_index];
  auto controller = hid_core_.GetEmulatedControllerByIndex(player_index);
  const int vibration_strength = vibration_spinboxes_[player_index]->value();
  const auto& buttons = controller->GetButtonsValues();

  bool button_is_pressed = false;
  for (std::size_t i = 0; i < buttons.size(); ++i) {
    if (buttons[i].value) {
      button_is_pressed = true;
      break;
    }
  }

  if (!button_is_pressed) {
    StopVibrations();
    return;
  }

  const Core::HID::VibrationValue vibration{
      .low_amplitude = kDefaultAmplitude,
      .low_frequency = kLowFrequency + vibration_strength,
      .high_amplitude = kDefaultAmplitude,
      .high_frequency = kHighFrequency + vibration_strength,
  };
  controller->SetVibration(0, vibration);
  controller->SetVibration(1, vibration);
}

void ConfigureVibration::StopVibrations() {
  for (std::size_t i = 0; i < kNumPlayers; ++i) {
    auto controller = hid_core_.GetEmulatedControllerByIndex(i);
    controller->SetVibration(0, kDefaultVibrationValue);
    controller->SetVibration(1, kDefaultVibrationValue);
  }
}