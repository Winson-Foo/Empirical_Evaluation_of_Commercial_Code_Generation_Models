#include "audio_core/renderer/adsp/command_list_processor.h"
#include "audio_core/renderer/command/data_source/decode.h"
#include "fmt/format.h"

namespace AudioCore::AudioRenderer {

class PcmFloatDataSourceCommand {
public:
    PcmFloatDataSourceCommand(uint8_t output_index, uint8_t channel_index, uint8_t channel_count,
                              uint32_t sample_rate, uint32_t src_quality, uint16_t pitch, uint8_t flags,
                              const std::array<ADSP::WaveBuffer, 8>& wave_buffers, void* voice_state)
        : output_index(output_index)
        , channel_index(channel_index)
        , channel_count(channel_count)
        , sample_rate(sample_rate)
        , src_quality(src_quality)
        , pitch(pitch)
        , flags(flags)
        , wave_buffers(wave_buffers)
        , voice_state(voice_state)
    {}

    virtual void Dump(const ADSP::CommandListProcessor& processor, std::string& string) = 0;
    virtual void Process(const ADSP::CommandListProcessor& processor) = 0;
    virtual bool Verify(const ADSP::CommandListProcessor& processor) = 0;

protected:
    const uint8_t output_index;
    const uint8_t channel_index;
    const uint8_t channel_count;
    const uint32_t sample_rate;
    const uint32_t src_quality;
    const uint16_t pitch;
    const uint8_t flags;
    const std::array<ADSP::WaveBuffer, 8> wave_buffers;
    void* const voice_state;

    DecodeFromWaveBuffersArgs CreateDecodeArgs(const ADSP::CommandListProcessor& processor) const {
        return {
            .sample_format{SampleFormat::PcmFloat},
            .output{processor.mix_buffers.subspan(output_index * processor.sample_count, processor.sample_count)},
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
    }
};

class PcmFloatDataSourceVersion1Command : public PcmFloatDataSourceCommand {
public:
    using PcmFloatDataSourceCommand::PcmFloatDataSourceCommand;

    void Dump(const ADSP::CommandListProcessor& processor, std::string& string) override {
        string += fmt::format(
            "PcmFloatDataSourceVersion1Command\n\toutput_index {:02X} channel {} channel count {} source sample rate {} "
            "target sample rate {} src quality {}\n",
            output_index, channel_index, channel_count, sample_rate, processor.target_sample_rate, src_quality);
    }

    void Process(const ADSP::CommandListProcessor& processor) override {
        DecodeFromWaveBuffersArgs args = CreateDecodeArgs(processor);
        DecodeFromWaveBuffers(*processor.memory, args);
    }

    bool Verify(const ADSP::CommandListProcessor& processor) override { return true; }
};

class PcmFloatDataSourceVersion2Command : public PcmFloatDataSourceCommand {
public:
    using PcmFloatDataSourceCommand::PcmFloatDataSourceCommand;

    void Dump(const ADSP::CommandListProcessor& processor, std::string& string) override {
        string += fmt::format(
            "PcmFloatDataSourceVersion2Command\n\toutput_index {:02X} channel {} channel count {} source sample rate {} "
            "target sample rate {} src quality {}\n",
            output_index, channel_index, channel_count, sample_rate, processor.target_sample_rate, src_quality);
    }

    void Process(const ADSP::CommandListProcessor& processor) override {
        DecodeFromWaveBuffersArgs args = CreateDecodeArgs(processor);
        DecodeFromWaveBuffers(*processor.memory, args);
    }

    bool Verify(const ADSP::CommandListProcessor& processor) override { return true; }
};

}  // namespace AudioCore::AudioRenderer

