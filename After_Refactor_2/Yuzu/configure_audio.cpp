#include <memory>
#include <string>

#include "audio_core/sink/sink.h"
#include "audio_core/sink/sink_details.h"
#include "common/settings.h"
#include "core/core.h"
#include "ui_configure_audio.h"
#include "yuzu/configuration/configuration_shared.h"
#include "yuzu/configuration/configure_audio.h"
#include "yuzu/uisettings.h"

ConfigureAudio::ConfigureAudio(const Core::System& system, QWidget* parent)
    : QWidget(parent), ui(std::make_unique<Ui::ConfigureAudio>()), system(system) {
  ui->setupUi(this);

  InitializeAudioSinkComboBox();

  connect(ui->volume_slider, &QSlider::valueChanged, this,
          &ConfigureAudio::SetVolumeIndicatorText);
  connect(
      ui->sink_combo_box, qOverload<int>(&QComboBox::currentIndexChanged), this,
      [this](int index) {
        UpdateAudioDevices(index);
        SetOutputSinkFromSinkID();
      });

  ui->volume_label->setVisible(Settings::IsConfiguringGlobal());
  ui->volume_combo_box->setVisible(!Settings::IsConfiguringGlobal());

  SetupPerGameUI();

  SetConfiguration();

  const bool is_powered_on = system.IsPoweredOn();
  ui->sink_combo_box->setEnabled(!is_powered_on);
  ui->output_combo_box->setEnabled(!is_powered_on);
  ui->input_combo_box->setEnabled(!is_powered_on);
}

void ConfigureAudio::SetConfiguration() {
  SetOutputSinkFromSinkID();

  UpdateAudioDevices(ui->sink_combo_box->currentIndex());

  SetAudioDevicesFromDeviceID();

  const auto volume_value = static_cast<int>(Settings::values.volume.GetValue());
  ui->volume_slider->setValue(volume_value);
  ui->toggle_background_mute->setChecked(UISettings::values.mute_when_in_background.GetValue());

  if (!Settings::IsConfiguringGlobal()) {
    if (Settings::values.volume.UsingGlobal()) {
      ui->volume_combo_box->setCurrentIndex(0);
      ui->volume_slider->setEnabled(false);
    } else {
      ui->volume_combo_box->setCurrentIndex(1);
      ui->volume_slider->setEnabled(true);
    }
    ConfigurationShared::SetPerGameSetting(ui->combo_sound, &Settings::values.sound_index);
    ConfigurationShared::SetHighlight(ui->mode_label, !Settings::values.sound_index.UsingGlobal());
    ConfigurationShared::SetHighlight(ui->volume_layout, !Settings::values.volume.UsingGlobal());
  } else {
    ui->combo_sound->setCurrentIndex(Settings::values.sound_index.GetValue());
  }
  SetVolumeIndicatorText(ui->volume_slider->sliderPosition());
}

void ConfigureAudio::SetOutputSinkFromSinkID() {
  const auto& sink_id = Settings::values.sink_id.GetValue();
  const auto index = ui->sink_combo_box->findText(QString::fromStdString(sink_id));
  ui->sink_combo_box->setCurrentIndex(index);
}

void ConfigureAudio::SetAudioDevicesFromDeviceID() {
  const auto& audio_output_device_id = Settings::values.audio_output_device_id.GetValue();
  const auto output_index = ui->output_combo_box->findText(QString::fromStdString(audio_output_device_id));
  ui->output_combo_box->setCurrentIndex(output_index);

  const auto& audio_input_device_id = Settings::values.audio_input_device_id.GetValue();
  const auto input_index = ui->input_combo_box->findText(QString::fromStdString(audio_input_device_id));
  ui->input_combo_box->setCurrentIndex(input_index);
}

void ConfigureAudio::SetVolumeIndicatorText(int percentage) {
  ui->volume_indicator->setText(tr("%1%", "Volume percentage (e.g. 50%)").arg(percentage));
}

void ConfigureAudio::ApplyConfiguration() {
  ConfigurationShared::ApplyPerGameSetting(&Settings::values.sound_index, ui->combo_sound);

  if (Settings::IsConfiguringGlobal()) {
    const auto& sink_id = ui->sink_combo_box->currentText().toStdString();
    Settings::values.sink_id = sink_id;
    const auto& audio_output_device_id = ui->output_combo_box->currentText().toStdString();
    Settings::values.audio_output_device_id.SetValue(audio_output_device_id);
    const auto& audio_input_device_id = ui->input_combo_box->currentText().toStdString();
    Settings::values.audio_input_device_id.SetValue(audio_input_device_id);
    UISettings::values.mute_when_in_background = ui->toggle_background_mute->isChecked();

    if (Settings::values.volume.UsingGlobal()) {
      const auto volume = static_cast<u8>(ui->volume_slider->value());
      Settings::values.volume.SetValue(volume);
    }
  } else {
    if (ui->volume_combo_box->currentIndex() == 0) {
      Settings::values.volume.SetGlobal(true);
    } else {
      Settings::values.volume.SetGlobal(false);
      const auto volume = static_cast<u8>(ui->volume_slider->value());
      Settings::values.volume.SetValue(volume);
    }
  }
}

void ConfigureAudio::changeEvent(QEvent* event) {
  if (event->type() == QEvent::LanguageChange) {
    RetranslateUI();
  }

  QWidget::changeEvent(event);
}

void ConfigureAudio::UpdateAudioDevices(int sink_index) {
  ui->output_combo_box->clear();
  ui->output_combo_box->addItem(QString::fromUtf8(AudioCore::Sink::auto_device_name));

  const auto& sink_id = ui->sink_combo_box->itemText(sink_index).toStdString();
  for (const auto& device : AudioCore::Sink::GetDeviceListForSink(sink_id, false)) {
    ui->output_combo_box->addItem(QString::fromStdString(device));
  }

  ui->input_combo_box->clear();
  ui->input_combo_box->addItem(QString::fromUtf8(AudioCore::Sink::auto_device_name));
  for (const auto& device : AudioCore::Sink::GetDeviceListForSink(sink_id, true)) {
    ui->input_combo_box->addItem(QString::fromStdString(device));
  }
}

void ConfigureAudio::InitializeAudioSinkComboBox() {
  ui->sink_combo_box->clear();
  ui->sink_combo_box->addItem(QString::fromUtf8(AudioCore::Sink::auto_device_name));

  for (const auto& id : AudioCore::Sink::GetSinkIDs()) {
    ui->sink_combo_box->addItem(QString::fromStdString(id));
  }
}

void ConfigureAudio::RetranslateUI() {
  ui->retranslateUi(this);
  SetVolumeIndicatorText(ui->volume_slider->sliderPosition());
}

void ConfigureAudio::SetupPerGameUI() {
  if (Settings::IsConfiguringGlobal()) {
    ui->combo_sound->setEnabled(Settings::values.sound_index.UsingGlobal());
    ui->volume_slider->setEnabled(Settings::values.volume.UsingGlobal());
    return;
  }

  ConfigurationShared::SetColoredComboBox(ui->combo_sound, ui->mode_label,
                                          Settings::values.sound_index.GetValue(true));

  connect(ui->volume_combo_box, qOverload<int>(&QComboBox::activated), this,
          [this](int index) {
            ui->volume_slider->setEnabled(index == 1);
            ConfigurationShared::SetHighlight(ui->volume_layout, index == 1);
          });

  ui->sink_combo_box->setVisible(false);
  ui->sink_label->setVisible(false);
  ui->output_combo_box->setVisible(false);
  ui->output_label->setVisible(false);
  ui->input_combo_box->setVisible(false);
  ui->input_label->setVisible(false);
}

