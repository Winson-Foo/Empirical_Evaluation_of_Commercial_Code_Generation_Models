#include <memory>
#include <string>

#include <QEvent>
#include <QSignalBlocker>
#include <QString>

#include "audio_core/sink/sink.h"
#include "audio_core/sink/sink_details.h"
#include "common/settings.h"
#include "core/core.h"
#include "ui_configure_audio.h"
#include "yuzu/configuration/configuration_shared.h"
#include "yuzu/configuration/configure_audio.h"
#include "yuzu/uisettings.h"

enum class ComboIndex {
  GLOBAL = 0,
  PER_GAME = 1
};

enum class LabelType {
  SOUND_MODE,
  VOLUME
};

constexpr int kDefaultVolume = 50;
constexpr int kMaxVolume = 100;
constexpr int kAudioDeviceRefreshDelay = 100;

ConfigureAudio::ConfigureAudio(const Core::System& system_, QWidget* parent)
    : QWidget(parent),
      ui(std::make_unique<Ui::ConfigureAudio>()),
      system_(system_)
{
    ui->setupUi(this);

    InitializeAudioSinkComboBox();

    connectSignals();

    ui->volume_label->setVisible(Settings::IsConfiguringGlobal());
    ui->volume_combo_box->setVisible(!Settings::IsConfiguringGlobal());

    setupPerGameUI();

    setConfiguration();

    const bool is_powered_on = system_.IsPoweredOn();
    ui->sink_combo_box->setEnabled(!is_powered_on);
    ui->output_combo_box->setEnabled(!is_powered_on);
    ui->input_combo_box->setEnabled(!is_powered_on);
}

ConfigureAudio::~ConfigureAudio() = default;

void ConfigureAudio::setConfiguration() {
    setOutputSinkFromSinkID();

    // The device list cannot be pre-populated (nor listed) until the output sink is known.
    updateAudioDevices(ui->sink_combo_box->currentIndex());

    setAudioDevicesFromDeviceID();

    const auto volume_value = static_cast<int>(Settings::values.volume.GetValue());
    ui->volume_slider->setValue(volume_value);
    ui->toggle_background_mute->setChecked(UISettings::values.mute_when_in_background.GetValue());

    if (!Settings::IsConfiguringGlobal()) {
        if (Settings::values.volume.UsingGlobal()) {
            ui->volume_combo_box->setCurrentIndex(static_cast<int>(ComboIndex::GLOBAL));
            ui->volume_slider->setEnabled(false);
        } else {
            ui->volume_combo_box->setCurrentIndex(static_cast<int>(ComboIndex::PER_GAME));
            ui->volume_slider->setEnabled(true);
        }
        ConfigurationShared::setPerGameSetting(ui->combo_sound, &Settings::values.sound_index);
        ConfigurationShared::setHighlight(ui->mode_label, !Settings::values.sound_index.UsingGlobal());
        ConfigurationShared::setHighlight(ui->volume_layout, !Settings::values.volume.UsingGlobal());
    } else {
        ui->combo_sound->setCurrentIndex(Settings::values.sound_index.GetValue());
    }
    setVolumeIndicatorText(ui->volume_slider->sliderPosition());
}

void ConfigureAudio::setOutputSinkFromSinkID() {
    const QSignalBlocker blocker(ui->sink_combo_box);

    int new_sink_index = 0;
    const QString sink_id = QString::fromStdString(Settings::values.sink_id.GetValue());
    for (int index = 0; index < ui->sink_combo_box->count(); index++) {
        if (ui->sink_combo_box->itemText(index) == sink_id) {
            new_sink_index = index;
            break;
        }
    }

    ui->sink_combo_box->setCurrentIndex(new_sink_index);
}

void ConfigureAudio::setAudioDevicesFromDeviceID() {
    const QString output_device_id =
        QString::fromStdString(Settings::values.audio_output_device_id.GetValue());
    int new_device_output_index = ui->output_combo_box->findText(output_device_id);
    ui->output_combo_box->setCurrentIndex(new_device_output_index);

    const QString input_device_id =
        QString::fromStdString(Settings::values.audio_input_device_id.GetValue());
    int new_device_input_index = ui->input_combo_box->findText(input_device_id);
    ui->input_combo_box->setCurrentIndex(new_device_input_index);
}

void ConfigureAudio::setVolumeIndicatorText(int percentage) {
    ui->volume_indicator->setText(tr("%1%", "Volume percentage (e.g. 50%)").arg(percentage));
}

void ConfigureAudio::applyConfiguration() {
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.sound_index, ui->combo_sound);

    if (Settings::IsConfiguringGlobal()) {
        Settings::values.sink_id =
            ui->sink_combo_box->itemText(ui->sink_combo_box->currentIndex()).toStdString();
        Settings::values.audio_output_device_id.SetValue(
            ui->output_combo_box->itemText(ui->output_combo_box->currentIndex()).toStdString());
        Settings::values.audio_input_device_id.SetValue(
            ui->input_combo_box->itemText(ui->input_combo_box->currentIndex()).toStdString());
        UISettings::values.mute_when_in_background = ui->toggle_background_mute->isChecked();

        // Guard if during game and set to game-specific value
        if (Settings::values.volume.UsingGlobal()) {
            const auto volume = static_cast<u8>(ui->volume_slider->value());
            Settings::values.volume.SetValue(volume);
        }
    } else {
        if (ui->volume_combo_box->currentIndex() == static_cast<int>(ComboIndex::GLOBAL)) {
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

void ConfigureAudio::updateAudioDevices(int sink_index) {
    const QSignalBlocker blocker(ui->output_combo_box);

    ui->output_combo_box->clear();
    ui->output_combo_box->addItem(QString::fromUtf8(AudioCore::Sink::auto_device_name));

    const std::string sink_id = ui->sink_combo_box->itemText(sink_index).toStdString();
    for (const auto& device : AudioCore::Sink::GetDeviceListForSink(sink_id, false)) {
        ui->output_combo_box->addItem(QString::fromStdString(device));
    }

    ui->input_combo_box->clear();
    ui->input_combo_box->addItem(QString::fromUtf8(AudioCore::Sink::auto_device_name));
    for (const auto& device : AudioCore::Sink::GetDeviceListForSink(sink_id, true)) {
        ui->input_combo_box->addItem(QString::fromStdString(device));
    }
}

void ConfigureAudio::initializeAudioSinkComboBox() {
    ui->sink_combo_box->clear();
    ui->sink_combo_box->addItem(QString::fromUtf8(AudioCore::Sink::auto_device_name));

    for (const auto& id : AudioCore::Sink::GetSinkIDs()) {
        ui->sink_combo_box->addItem(QString::fromUtf8(id.data(), static_cast<s32>(id.length())));
    }
}

void ConfigureAudio::connectSignals() {
    connect(ui->volume_combo_box, qOverload<int>(&QComboBox::activated), this, [this](int index) {
        ui->volume_slider->setEnabled(index == static_cast<int>(ComboIndex::PER_GAME));
        ConfigurationShared::setHighlight(ui->volume_layout,
                                          index == static_cast<int>(ComboIndex::PER_GAME));
    });

    connect(ui->volume_slider, &QSlider::valueChanged, this, &ConfigureAudio::setVolumeIndicatorText);

    connect(ui->sink_combo_box, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &ConfigureAudio::updateAudioDevices);
}

void ConfigureAudio::setupPerGameUI() {
    if (!Settings::IsConfiguringGlobal()) {
        ConfigurationShared::setColoredComboBox(ui->combo_sound, ui->mode_label,
                                                Settings::values.sound_index.GetValue(true));

        ui->combo_sound->setEnabled(Settings::values.sound_index.UsingGlobal());
        ui->volume_slider->setEnabled(Settings::values.volume.UsingGlobal());
        ui->volume_combo_box->setCurrentIndex(
            Settings::values.volume.UsingGlobal() ? static_cast<int>(ComboIndex::GLOBAL)
                                                  : static_cast<int>(ComboIndex::PER_GAME));
        ConfigurationShared::setHighlight(ui->mode_label, !Settings::values.sound_index.UsingGlobal());
        ConfigurationShared::setHighlight(ui->volume_layout, !Settings::values.volume.UsingGlobal());

        ui->sink_combo_box->setVisible(false);
        ui->sink_label->setVisible(false);
        ui->output_combo_box->setVisible(false);
        ui->output_label->setVisible(false);
        ui->input_combo_box->setVisible(false);
        ui->input_label->setVisible(false);
    }
}

void ConfigureAudio::RetranslateUI() {
    ui->retranslateUi(this);
    setVolumeIndicatorText(ui->volume_slider->sliderPosition());
}

