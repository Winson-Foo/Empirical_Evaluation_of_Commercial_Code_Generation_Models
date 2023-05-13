// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>

#include "audio_core/audio_manager.h"
#include "audio_core/renderer/adsp/adsp.h"
#include "audio_core/sink/sink.h"

namespace Core {
class System;
}

namespace AudioCore {

class AudioManager;
/**
 * Main audio class, stored inside the core, and holding the audio manager, all sinks, and the ADSP.
 */
class AudioCore {
public:
    explicit AudioCore(Core::System& system);
    ~AudioCore();

    /**
     * Shutdown the audio core.
     */
    void Shutdown();

    /**
     * Get a reference to the audio manager.
     *
     * @return Ref to the audio manager.
     */
    AudioManager& GetAudioManager();

    /**
     * Get the audio output sink currently in use.
     *
     * @return Ref to the sink.
     */
    Sink::Sink& GetOutputSink();

    /**
     * Get the audio input sink currently in use.
     *
     * @return Ref to the sink.
     */
    Sink::Sink& GetInputSink();

    /**
     * Get the ADSP.
     *
     * @return Ref to the ADSP.
     */
    AudioRenderer::ADSP::ADSP& GetADSP();

    /**
     * Toggle NVDEC state, used to avoid stall in playback.
     *
     * @param active - Set true if nvdec is active, otherwise false.
     */
    void SetNVDECActive(bool active);

    /**
     * Get NVDEC state.
     */
    bool IsNVDECActive() const;

private:
    /**
     * Create the sinks on startup.
     */
    void CreateSinks();

    /// Main audio manager for audio in/out
    std::unique_ptr<AudioManager> audio_manager;
    /// Sink used for audio renderer and audio out
    std::unique_ptr<Sink::Sink> output_sink;
    /// Sink used for audio input
    std::unique_ptr<Sink::Sink> input_sink;
    /// The ADSP in the sysmodule
    std::unique_ptr<AudioRenderer::ADSP::ADSP> adsp;
    /// Is NVDec currently active?
    bool nvdec_active{false};
};

} // namespace AudioCore
