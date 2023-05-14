// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright 2022 yuzu Emulator Project

#include <span>

#include "audio_core/renderer/adsp/command_list_processor.h"
#include "audio_core/renderer/command/data_source/adpcm.h"
#include "audio_core/renderer/command/data_source/decode.h"

namespace AudioCore::AudioRenderer {

class AdpcmDataSourceCommand {
 public:
  virtual ~AdpcmDataSourceCommand() = default;

  virtual void Dump(const ADSP::CommandListProcessor& processor, std::string& string) = 0;

  virtual void Process(const ADSP::CommandListProcessor& processor) {
    auto out_buffer =
        processor.mix_buffers.subspan(output_index * processor.sample_count, processor.sample_count);

    DecodeFromWaveBuffersArgs args{
        .sample_format = SampleFormat::Adpcm,
        .output = out_buffer,
        .voice_state = reinterpret_cast<VoiceState*>(voice_state),
        .wave_buffers = wave_buffers,
        .channel = 0,
        .channel_count = 1,
        .src_quality = src_quality,
        .pitch = pitch,
        .source_sample_rate = sample_rate,
        .target_sample_rate = processor.target_sample_rate,
        .sample_count = processor.sample_count,
        .data_address = data_address,
        .data_size = data_size,
        .IsVoicePlayedSampleCountResetAtLoopPointSupported = (flags & 1) != 0,
        .IsVoicePitchAndSrcSkippedSupported = (flags & 2) != 0,
    };

    DecodeFromWaveBuffers(*processor.memory, args);
  }

  virtual bool Verify(const ADSP::CommandListProcessor& processor) { return true; }

 protected:
  uint8_t output_index;
  uint32_t sample_rate;
  uint32_t voice_state;
  std::span<const std::array<std::byte, 0x100000>> wave_buffers;
  AudioCore::AudioRenderer::ADSP::SrcQuality src_quality;
  int16_t pitch;
  uint32_t flags;
  uint32_t data_address;
  uint32_t data_size;
};

class AdpcmDataSourceVersion1Command : public AdpcmDataSourceCommand {
 public:
  void Dump(const ADSP::CommandListProcessor& processor, std::string& string) override {
    string += fmt::format(
        "AdpcmDataSourceVersion1Command\n\toutput_index {:02X} source sample rate {} target "
        "sample rate {} src quality {}\n",
        output_index, sample_rate, processor.target_sample_rate, src_quality);
  }

  bool Verify(const ADSP::CommandListProcessor& processor) override { return true; }
};

class AdpcmDataSourceVersion2Command : public AdpcmDataSourceCommand {
 public:
  void Dump(const ADSP::CommandListProcessor& processor, std::string& string) override {
    string += fmt::format(
        "AdpcmDataSourceVersion2Command\n\toutput_index {:02X} source sample rate {} target "
        "sample rate {} src quality {}\n",
        output_index, sample_rate, processor.target_sample_rate, src_quality);
  }

  bool Verify(const ADSP::CommandListProcessor& processor) override { return true; }
};

}  // namespace AudioCore::AudioRenderer