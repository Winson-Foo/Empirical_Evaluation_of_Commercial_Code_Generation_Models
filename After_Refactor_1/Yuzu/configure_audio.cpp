// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <QObject>
#include <QSharedPointer>

#include "audio_sink.h"
#include "configuration_settings.h"
#include "ui_configure_audio.h"

/**
 * @brief The AudioSettings class represents the audio settings configuration panel.
 */
class AudioSettings : public QWidget {
public:
    AudioSettings(const QSharedPointer<System>& system, QWidget* parent = nullptr);
    virtual ~AudioSettings() override = default;

    void applyConfiguration();

protected:
    void changeEvent(QEvent* event) override;

private:
    void initializeUI();
    void initializeAudioSinkComboBox();
    void updateAudioDevices(int sinkIndex);
    void setOutputSinkFromSinkID();
    void setAudioDevicesFromDeviceID();
    void setVolumeIndicatorText(int percentage);
    void setConfiguration();
    void setupPerGameUI();
    void retranslateUI();

private slots:
    void onVolumeSliderValueChanged(int value);
    void onSinkComboBoxCurrentIndexChanged(int index);
    void onVolumeComboBoxActivated(int index);

private:
    QSharedPointer<System> m_system;
    QSharedPointer<AudioSink> m_sink;
    QSharedPointer<Ui::AudioSettingsUI> m_ui;
    QSharedPointer<ConfigurationSettings> m_settings;
};

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "audio_sink.h"

/**
 * @brief The AudioCore class provides the audio functionality for the emulator.
 */
class AudioCore {
public:
    AudioCore();
    ~AudioCore() = default;

    QSharedPointer<AudioSink> createAudioSink();

private:
    // Implementation details can be hidden as private
};

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <vector>

/**
 * @brief The AudioSink class represents an audio sink device for the emulator.
 */
class AudioSink {
public:
    static const std::string autoDeviceName;

    AudioSink(const std::string& sinkID,
              const std::string& outputDeviceID = "",
              const std::string& inputDeviceID = "");
    virtual ~AudioSink() = default;

    std::string getSinkID() const;
    void setSinkID(const std::string& sinkID);

    std::vector<std::string> getOutputDeviceList() const;
    std::vector<std::string> getInputDeviceList() const;

    std::string getOutputDeviceID() const;
    void setOutputDeviceID(const std::string& outputDeviceID);

    std::string getInputDeviceID() const;
    void setInputDeviceID(const std::string& inputDeviceID);

    bool isMutedWhenInBackground() const;
    void setMutedWhenInBackground(bool muted);

    int getVolume() const;
    void setVolume(int volume);

private:
    // Implementation details can be hidden as private
};

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <QObject>
#include <QSettings>

/**
 * @brief The ConfigurationSettings class provides the configuration settings for the emulator.
 */
class ConfigurationSettings : public QObject {
    Q_OBJECT
public:
    ConfigurationSettings(QSettings& settings);
    virtual ~ConfigurationSettings() override = default;

    int getSoundIndex() const;
    void setSoundIndex(int index);

    int getVolume() const;
    void setVolume(int volume);

    bool isMuted() const;
    void setMuted(bool muted);

private:
    QSettings& m_settings;
    QSharedPointer<QObject> m_defaultSoundIndex;
    QSharedPointer<QObject> m_defaultVolume;
    QSharedPointer<QObject> m_defaultMuted;
};