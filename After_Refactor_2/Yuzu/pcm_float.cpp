// PcmFloatDataSourceCommand.h

#ifndef PCM_FLOAT_DATA_SOURCE_COMMAND_H
#define PCM_FLOAT_DATA_SOURCE_COMMAND_H

#include "audio_core/renderer/adsp/command_list_processor.h"
#include "audio_core/renderer/command/data_source/decode.h"
#include <memory>

namespace AudioCore::AudioRenderer {

class PcmFloatDataSourceCommand {
public:
    virtual ~PcmFloatDataSourceCommand() = default;
    virtual void Dump(const ADSP::CommandListProcessor& processor, std::string& string) = 0;
    virtual void Process(const ADSP::CommandListProcessor& processor) = 0;
    virtual bool Verify(const ADSP::CommandListProcessor& processor) = 0;
protected:
    PcmFloatDataSourceCommand(uint8_t output_index, uint8_t channel_index, uint8_t channel_count,
                              uint32_t sample_rate, uint32_t src_quality, uint32_t pitch,
                              uint32_t flags, const void* voice_state,
                              const std::array<WaveBuffer, 8>& wave_buffers)
        : output_index(output_index)
        , channel_index(channel_index)
        , channel_count(channel_count)
        , sample_rate(sample_rate)
        , src_quality(src_quality)
        , pitch(pitch)
        , flags(flags)
        , voice_state(voice_state)
        , wave_buffers(wave_buffers)
    {
    }
    uint8_t output_index{};
    uint8_t channel_index{};
    uint8_t channel_count{};
    uint32_t sample_rate{};
    uint32_t src_quality{};
    uint32_t pitch{};
    uint32_t flags{};
    const void* voice_state{};
    std::array<WaveBuffer, 8> wave_buffers{};
};

class PcmFloatDataSourceVersion1Command final : public PcmFloatDataSourceCommand {
public:
    PcmFloatDataSourceVersion1Command(uint8_t output_index, uint8_t channel_index,
                                      uint8_t channel_count, uint32_t sample_rate,
                                      uint32_t src_quality, uint32_t pitch, uint32_t flags,
                                      const void* voice_state,
                                      const std::array<WaveBuffer, 8>& wave_buffers)
        : PcmFloatDataSourceCommand(output_index, channel_index, channel_count, sample_rate,
                                    src_quality, pitch, flags, voice_state, wave_buffers)
    {
    }
    void Dump(const ADSP::CommandListProcessor& processor, std::string& string) override;
    void Process(const ADSP::CommandListProcessor& processor) override;
    bool Verify(const ADSP::CommandListProcessor& processor) override;
};

class PcmFloatDataSourceVersion2Command final : public PcmFloatDataSourceCommand {
public:
    PcmFloatDataSourceVersion2Command(uint8_t output_index, uint8_t channel_index,
                                      uint8_t channel_count, uint32_t sample_rate,
                                      uint32_t src_quality, uint32_t pitch, uint32_t flags,
                                      const void* voice_state,
                                      const std::array<WaveBuffer, 8>& wave_buffers)
        : PcmFloatDataSourceCommand(output_index, channel_index, channel_count, sample_rate,
                                    src_quality, pitch, flags, voice_state, wave_buffers)
    {
    }
    void Dump(const ADSP::CommandListProcessor& processor, std::string& string) override;
    void Process(const ADSP::CommandListProcessor& processor) override;
    bool Verify(const ADSP::CommandListProcessor& processor) override;
};

} // namespace AudioCore::AudioRenderer

#endif // PCM_FLOAT_DATA_SOURCE_COMMAND_H

// PcmFloatDataSourceCommand.cpp

#include "audio_core/renderer/command/data_source/pcm_float.h"

namespace AudioCore::AudioRenderer {

void PcmFloatDataSourceVersion1Command::Dump(const ADSP::CommandListProcessor& processor,
                                             std::string& string) {
    string +=
        fmt::format("PcmFloatDataSourceVersion1Command\n\toutput_index {:02X} channel {} "
                    "channel count {} source sample rate {} target sample rate {} src quality {}\n",
                    output_index, channel_index, channel_count, sample_rate,
                    processor.target_sample_rate, src_quality);
}

void PcmFloatDataSourceVersion1Command::Process(const ADSP::CommandListProcessor& processor) {
    auto out_buffer = processor.mix_buffers.subspan(output_index * processor.sample_count,
                                                    processor.sample_count);

    DecodeFromWaveBuffersArgs args{
        .sample_format{SampleFormat::PcmFloat},
        .output{out_buffer},
        .voice_state{reinterpret_cast<VoiceState*>(voice_state)},
        .wave_buffers{wave_buffers},
        .channel{channel_index},
        .channel_count{channel_count},
        .src_quality{src_quality},
        .pitch{pitch},
        .source_sample_rate{sample_rate},
        .target_sample_rate{processor.target_sample_rate},
        .sample_count{processor.sample_count},
        .data_address{0},
        .data_size{0},
        .IsVoicePlayedSampleCountResetAtLoopPointSupported{(flags & 1) != 0},
        .IsVoicePitchAndSrcSkippedSupported{(flags & 2) != 0},
    };

    DecodeFromWaveBuffers(*processor.memory, args);
}

bool PcmFloatDataSourceVersion1Command::Verify(const ADSP::CommandListProcessor& processor) {
    return true;
}

void PcmFloatDataSourceVersion2Command::Dump(const ADSP::CommandListProcessor& processor,
                                             std::string& string) {
    string +=
        fmt::format("PcmFloatDataSourceVersion2Command\n\toutput_index {:02X} channel {} "
                    "channel count {} source sample rate {} target sample rate {} src quality {}\n",
                    output_index, channel_index, channel_count, sample_rate,
                    processor.target_sample_rate, src_quality);
}

void PcmFloatDataSourceVersion2Command::Process(const ADSP::CommandListProcessor& processor) {
    auto out_buffer = processor.mix_buffers.subspan(output_index * processor.sample_count,
                                                    processor.sample_count);

    DecodeFromWaveBuffersArgs args{
        .sample_format{SampleFormat::PcmFloat},
        .output{out_buffer},
        .voice_state{reinterpret_cast<VoiceState*>(voice_state)},
        .wave_buffers{wave_buffers},
        .channel{channel_index},
        .channel_count{channel_count},
        .src_quality{src_quality},
        .pitch{pitch},
        .source_sample_rate{sample_rate},
        .target_sample_rate{processor.target_sample_rate},
        .sample_count{processor.sample_count},
        .data_address{0},
        .data_size{0},
        .IsVoicePlayedSampleCountResetAtLoopPointSupported{(flags & 1) != 0},
        .IsVoicePitchAndSrcSkippedSupported{(flags & 2) != 0},
    };

    DecodeFromWaveBuffers(*processor.memory, args);
}

bool PcmFloatDataSourceVersion2Command::Verify(const ADSP::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::AudioRenderer