// SPDX-License-Identifier: GPL-2.0-or-later

#include "audio_core/renderer/adsp/command_list_processor.h"
#include "audio_core/renderer/command/data_source/decode.h"
#include "audio_core/renderer/command/data_source/pcm_float.h"

namespace AudioCore::AudioRenderer {

class PcmFloatDataSourceCommandBase {
public:
  PcmFloatDataSourceCommandBase(uint8_t output_index, uint8_t channel_index,
                                uint8_t channel_count, uint32_t sample_rate,
                                uint16_t src_quality, uint32_t flags)
      : output_index(output_index), channel_index(channel_index),
        channel_count(channel_count), sample_rate(sample_rate),
        src_quality(src_quality), flags(flags) {}

  virtual ~PcmFloatDataSourceCommandBase() = default;

  virtual void Dump(const ADSP::CommandListProcessor& processor, std::string& string) = 0;

  virtual void Process(const ADSP::CommandListProcessor& processor) {
    auto out_buffer =
        processor.mix_buffers.subspan(output_index * processor.sample_count, processor.sample_count);

    DecodeFromWaveBuffersArgs args = {
        .sample_format = SampleFormat::PcmFloat,
        .output = out_buffer,
        .voice_state = reinterpret_cast<VoiceState*>(voice_state),
        .wave_buffers = wave_buffers,
        .channel = channel_index,
        .channel_count = channel_count,
        .src_quality = src_quality,
        .pitch = pitch,
        .source_sample_rate = sample_rate,
        .target_sample_rate = processor.target_sample_rate,
        .sample_count = processor.sample_count,
        .data_address = 0,
        .data_size = 0,
        .IsVoicePlayedSampleCountResetAtLoopPointSupported = (flags & 1) != 0,
        .IsVoicePitchAndSrcSkippedSupported = (flags & 2) != 0,
    };

    DecodeFromWaveBuffers(*processor.memory, args);
  }

  virtual bool Verify(const ADSP::CommandListProcessor& processor) = 0;

protected:
  uint8_t output_index;
  uint8_t channel_index;
  uint8_t channel_count;
  uint32_t sample_rate;
  uint16_t src_quality;
  uint32_t flags;
  int16_t* voice_state{nullptr};
  std::span<const WaveBufData> wave_buffers;
  float pitch{1.0f};
};

class PcmFloatDataSourceVersion1Command : public PcmFloatDataSourceCommandBase {
public:
  PcmFloatDataSourceVersion1Command(uint8_t output_index, uint8_t channel_index,
                                    uint8_t channel_count, uint32_t sample_rate,
                                    uint16_t src_quality, uint32_t flags)
      : PcmFloatDataSourceCommandBase(output_index, channel_index, channel_count, sample_rate,
                                      src_quality, flags) {}

  void Dump(const ADSP::CommandListProcessor& processor, std::string& string) override {
    string +=
        fmt::format("PcmFloatDataSourceVersion1Command\n\toutput_index {:02X} channel {} "
                    "channel count {} source sample rate {} target sample rate {} src quality {}\n",
                    output_index, channel_index, channel_count, sample_rate,
                    processor.target_sample_rate, src_quality);
  }

  bool Verify(const ADSP::CommandListProcessor& processor) override { return true; }
};

class PcmFloatDataSourceVersion2Command : public PcmFloatDataSourceCommandBase {
public:
  PcmFloatDataSourceVersion2Command(uint8_t output_index, uint8_t channel_index,
                                    uint8_t channel_count, uint32_t sample_rate,
                                    uint16_t src_quality, uint32_t flags)
      : PcmFloatDataSourceCommandBase(output_index, channel_index, channel_count, sample_rate,
                                      src_quality, flags) {}

  void Dump(const ADSP::CommandListProcessor& processor, std::string& string) override {
    string +=
        fmt::format("PcmFloatDataSourceVersion2Command\n\toutput_index {:02X} channel {} "
                    "channel count {} source sample rate {} target sample rate {} src quality {}\n",
                    output_index, channel_index, channel_count, sample_rate,
                    processor.target_sample_rate, src_quality);
  }

  bool Verify(const ADSP::CommandListProcessor& processor) override { return true; }
};

void DecodePcmFloatDataSourceCommand(const ADSP::CommandListProcessor& processor,
                                     const std::shared_ptr<AudioRendererCommand>& command,
                                     std::string& string) {
  const auto pcm_float_command =
      std::static_pointer_cast<PcmFloatDataSourceCommandBase>(command);
  pcm_float_command->Dump(processor, string);
  pcm_float_command->Process(processor);
  pcm_float_command->Verify(processor);
}

} // namespace AudioCore::AudioRenderer

