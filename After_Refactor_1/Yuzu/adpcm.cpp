#include <span>

#include "audio_core/renderer/adsp/command_list_processor.h"
#include "audio_core/renderer/command/data_source/adpcm.h"
#include "audio_core/renderer/command/data_source/decode.h"

namespace AudioCore::AudioRenderer {

class AdpcmDataSourceCommandBase {
public:
    virtual void Dump(const ADSP::CommandListProcessor& processor, std::string& string) = 0;

    virtual void Process(const ADSP::CommandListProcessor& processor) {
        auto out_buffer{processor.mix_buffers.subspan(output_index_ * processor.sample_count, processor.sample_count)};

        DecodeFromWaveBuffersArgs args{
            .sample_format = SampleFormat::Adpcm,
            .output = out_buffer,
            .voice_state = reinterpret_cast<VoiceState*>(voice_state_),
            .wave_buffers = wave_buffers_,
            .channel = channel_,
            .channel_count = channel_count_,
            .src_quality = src_quality_,
            .pitch = pitch_,
            .source_sample_rate = sample_rate_,
            .target_sample_rate = processor.target_sample_rate,
            .sample_count = processor.sample_count,
            .data_address = data_address_,
            .data_size = data_size_,
            .IsVoicePlayedSampleCountResetAtLoopPointSupported = (flags_ & 1) != 0,
            .IsVoicePitchAndSrcSkippedSupported = (flags_ & 2) != 0,
        };

        DecodeFromWaveBuffers(*processor.memory, args);
    }

    virtual bool Verify(const ADSP::CommandListProcessor& processor) {
        return true;
    }

protected:
    uint8_t output_index_;
    uint32_t sample_rate_;
    uint32_t voice_state_;
    const uint8_t* wave_buffers_;
    uint8_t channel_;
    uint8_t channel_count_;
    uint8_t src_quality_;
    int16_t pitch_;
    uint32_t data_address_;
    uint32_t data_size_;
    uint8_t flags_;
};

class AdpcmDataSourceVersion1Command : public AdpcmDataSourceCommandBase {
public:
    AdpcmDataSourceVersion1Command(uint32_t word1, uint32_t word2, const uint8_t* wave_buffers)
        : output_index_{word1 & 0xff},
          sample_rate_{word2},
          voice_state_{(word1 >> 8) & 0xff},
          wave_buffers_{wave_buffers},
          channel_{(word1 >> 16) & 0xff},
          channel_count_{(word1 >> 24) & 0xff},
          src_quality_{word2 >> 16},
          pitch_{int16_t(word2 & 0xffff)},
          data_address_{0},
          data_size_{0},
          flags_{0} {}

    virtual void Dump(const ADSP::CommandListProcessor& processor, std::string& string) {
        string += fmt::format("AdpcmDataSourceVersion1Command\n\toutput_index {:02X} source sample "
                              "rate {} target sample rate {} src quality {}\n",
                              output_index_, sample_rate_, processor.target_sample_rate, src_quality_);
    }
};

class AdpcmDataSourceVersion2Command : public AdpcmDataSourceCommandBase {
public:
    AdpcmDataSourceVersion2Command(uint32_t word1, uint32_t word2, const uint8_t* wave_buffers)
        : output_index_{word1 & 0xff},
          sample_rate_{word2},
          voice_state_{(word1 >> 8) & 0xff},
          wave_buffers_{wave_buffers},
          channel_{(word1 >> 16) & 0xff},
          channel_count_{(word1 >> 24) & 0xff},
          src_quality_{word2 >> 16},
          pitch_{int16_t(word2 & 0xffff)},
          data_address_{0},
          data_size_{0},
          flags_{0} {}

    virtual void Dump(const ADSP::CommandListProcessor& processor, std::string& string) {
        string += fmt::format("AdpcmDataSourceVersion2Command\n\toutput_index {:02X} source sample "
                              "rate {} target sample rate {} src quality {}\n",
                              output_index_, sample_rate_, processor.target_sample_rate, src_quality_);
    }
};

} // namespace AudioCore::AudioRenderer