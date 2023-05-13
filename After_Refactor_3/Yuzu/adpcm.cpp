void ProcessAdpcmDataSourceCommand(const ADSP::CommandListProcessor& processor,
                                   std::span<i16> out_buffer,
                                   SampleFormat sample_format, void* voice_state,
                                   const WaveBuffers& wave_buffers, u8 src_quality, f32 pitch,
                                   u32 sample_rate, u32 target_sample_rate, u32 sample_count,
                                   u64 data_address, u32 data_size, bool reset_at_loop_point_supported,
                                   bool pitch_and_src_skipped_supported) {
    DecodeFromWaveBuffersArgs args{
        .sample_format{sample_format},
        .output{out_buffer},
        .voice_state{reinterpret_cast<VoiceState*>(voice_state)},
        .wave_buffers{wave_buffers},
        .channel{0},
        .channel_count{1},
        .src_quality{src_quality},
        .pitch{pitch},
        .source_sample_rate{sample_rate},
        .target_sample_rate{target_sample_rate},
        .sample_count{sample_count},
        .data_address{data_address},
        .data_size{data_size},
        .IsVoicePlayedSampleCountResetAtLoopPointSupported{reset_at_loop_point_supported},
        .IsVoicePitchAndSrcSkippedSupported{pitch_and_src_skipped_supported},
    };

    DecodeFromWaveBuffers(*processor.memory, args);
}

void AdpcmDataSourceVersion1Command::Process(const ADSP::CommandListProcessor& processor) {
    auto out_buffer{processor.mix_buffers.subspan(output_index * processor.sample_count,
                                                  processor.sample_count)};
    ProcessAdpcmDataSourceCommand(processor, out_buffer, SampleFormat::Adpcm, voice_state,
                                  wave_buffers, src_quality, pitch, sample_rate,
                                  processor.target_sample_rate, processor.sample_count, data_address,
                                  data_size, (flags & 1) != 0, (flags & 2) != 0);
}

void AdpcmDataSourceVersion2Command::Process(const ADSP::CommandListProcessor& processor) {
    auto out_buffer{processor.mix_buffers.subspan(output_index * processor.sample_count,
                                                  processor.sample_count)};
    ProcessAdpcmDataSourceCommand(processor, out_buffer, SampleFormat::Adpcm, voice_state,
                                  wave_buffers, src_quality, pitch, sample_rate,
                                  processor.target_sample_rate, processor.sample_count, data_address,
                                  data_size, (flags & 1) != 0, (flags & 2) != 0);
}
```

2. Use const references for the function arguments where appropriate to avoid unnecessary copies.

```
void ProcessAdpcmDataSourceCommand(const ADSP::CommandListProcessor& processor,
                                   std::span<const i16> out_buffer,
                                   SampleFormat sample_format, const void* voice_state,
                                   const WaveBuffers& wave_buffers, u8 src_quality, f32 pitch,
                                   u32 sample_rate, u32 target_sample_rate, u32 sample_count,
                                   u64 data_address, u32 data_size, bool reset_at_loop_point_supported,
                                   bool pitch_and_src_skipped_supported) {
    DecodeFromWaveBuffersArgs args{
        .sample_format{sample_format},
        .output{out_buffer},
        .voice_state{reinterpret_cast<const VoiceState*>(voice_state)},
        .wave_buffers{wave_buffers},
        .channel{0},
        .channel_count{1},
        .src_quality{src_quality},
        .pitch{pitch},
        .source_sample_rate{sample_rate},
        .target_sample_rate{target_sample_rate},
        .sample_count{sample_count},
        .data_address{data_address},
        .data_size{data_size},
        .IsVoicePlayedSampleCountResetAtLoopPointSupported{reset_at_loop_point_supported},
        .IsVoicePitchAndSrcSkippedSupported{pitch_and_src_skipped_supported},
    };

    DecodeFromWaveBuffers(*processor.memory, args);
}

void AdpcmDataSourceVersion1Command::Process(const ADSP::CommandListProcessor& processor) {
    auto out_buffer{processor.mix_buffers.subspan(output_index * processor.sample_count,
                                                  processor.sample_count)};
    ProcessAdpcmDataSourceCommand(processor, out_buffer, SampleFormat::Adpcm, voice_state,
                                  wave_buffers, src_quality, pitch, sample_rate,
                                  processor.target_sample_rate, processor.sample_count, data_address,
                                  data_size, (flags & 1) != 0, (flags & 2) != 0);
}

void AdpcmDataSourceVersion2Command::Process(const ADSP::CommandListProcessor& processor) {
    auto out_buffer{processor.mix_buffers.subspan(output_index * processor.sample_count,
                                                  processor.sample_count)};
    ProcessAdpcmDataSourceCommand(processor, out_buffer, SampleFormat::Adpcm, voice_state,
                                  wave_buffers, src_quality, pitch, sample_rate,
                                  processor.target_sample_rate, processor.sample_count, data_address,
                                  data_size, (flags & 1) != 0, (flags & 2) != 0);
}
``` 

3. Use consistent naming conventions for function arguments and local variables to improve readability and maintainability.

```
void ProcessAdpcmDataSourceCommand(const ADSP::CommandListProcessor& processor,
                                   std::span<const i16> out_buffer,
                                   SampleFormat sample_format, const void* voice_state,
                                   const WaveBuffers& wave_buffers, u8 source_quality, f32 pitch,
                                   u32 source_sample_rate, u32 target_sample_rate, u32 sample_count,
                                   u64 data_address, u32 data_size, bool reset_at_loop_point_supported,
                                   bool pitch_and_src_skipped_supported) {
    DecodeFromWaveBuffersArgs args{
        .sample_format{sample_format},
        .output{out_buffer},
        .voice_state{reinterpret_cast<const VoiceState*>(voice_state)},
        .wave_buffers{wave_buffers},
        .channel{0},
        .channel_count{1},
        .src_quality{source_quality},
        .pitch{pitch},
        .source_sample_rate{source_sample_rate},
        .target_sample_rate{target_sample_rate},
        .sample_count{sample_count},
        .data_address{data_address},
        .data_size{data_size},
        .IsVoicePlayedSampleCountResetAtLoopPointSupported{reset_at_loop_point_supported},
        .IsVoicePitchAndSrcSkippedSupported{pitch_and_src_skipped_supported},
    };

    DecodeFromWaveBuffers(*processor.memory, args);
}

void AdpcmDataSourceVersion1Command::Process(const ADSP::CommandListProcessor& processor) {
    auto output_buffer = processor.mix_buffers.subspan(output_index * processor.sample_count,
                                                  processor.sample_count);
    ProcessAdpcmDataSourceCommand(processor, output_buffer, SampleFormat::Adpcm, voice_state,
                                  wave_buffers, src_quality, pitch, sample_rate,
                                  processor.target_sample_rate, processor.sample_count, data_address,
                                  data_size, (flags & 1) != 0, (flags & 2) != 0);
}

void AdpcmDataSourceVersion2Command::Process(const ADSP::CommandListProcessor& processor) {
    auto output_buffer = processor.mix_buffers.subspan(output_index * processor.sample_count,
                                                  processor.sample_count);
    ProcessAdpcmDataSourceCommand(processor, output_buffer, SampleFormat::Adpcm, voice_state,
                                  wave_buffers, src_quality, pitch, sample_rate,
                                  processor.target_sample_rate, processor.sample_count, data_address,
                                  data_size, (flags & 1) != 0, (flags & 2) != 0);
}

// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright 2022 yuzu Emulator Project

#include <span>

#include "audio_core/renderer/adsp/command_list_processor.h"
#include "audio_core/renderer/command/data_source/adpcm.h"
#include "audio_core/renderer/command/data_source/decode.h"

namespace AudioCore::AudioRenderer {

void AdpcmDataSourceVersion1Command::Dump(const ADSP::CommandListProcessor& processor,
                                          std::string& string) {
    string += fmt::format("AdpcmDataSourceVersion1Command\n\toutput_index {:02X} source sample "
                          "rate {} target sample rate {} src quality {}\n",
                          output_index, sample_rate, processor.target_sample_rate, src_quality);
}

void AdpcmDataSourceVersion2Command::Dump(const ADSP::CommandListProcessor& processor,
                                          std::string& string) {
    string += fmt::format("AdpcmDataSourceVersion2Command\n\toutput_index {:02X} source sample "
                          "rate {} target sample rate {} src quality {}\n",
                          output_index, sample_rate, processor.target_sample_rate, src_quality);
}

void ProcessAdpcmDataSourceCommand(const ADSP::CommandListProcessor& processor,
                                   std::span<const i16> out_buffer,
                                   SampleFormat sample_format, const void* voice_state,
                                   const WaveBuffers& wave_buffers, u8 source_quality, f32 pitch,
                                   u32 source_sample_rate, u32 target_sample_rate, u32 sample_count,
                                   u64 data_address, u32 data_size, bool reset_at_loop_point_supported,
                                   bool pitch_and_src_skipped_supported) {
    DecodeFromWaveBuffersArgs args{
        .sample_format{sample_format},
        .output{out_buffer},
        .voice_state{reinterpret_cast<const VoiceState*>(voice_state)},
        .wave_buffers{wave_buffers},
        .channel{0},
        .channel_count{1},
        .src_quality{source_quality},
        .pitch{pitch},
        .source_sample_rate{source_sample_rate},
        .target_sample_rate{target_sample_rate},
        .sample_count{sample_count},
        .data_address{data_address},
        .data_size{data_size},
        .IsVoicePlayedSampleCountResetAtLoopPointSupported{reset_at_loop_point_supported},
        .IsVoicePitchAndSrcSkippedSupported{pitch_and_src_skipped_supported},
    };

    DecodeFromWaveBuffers(*processor.memory, args);
}

void AdpcmDataSourceVersion1Command::Process(const ADSP::CommandListProcessor& processor) {
    auto output_buffer = processor.mix_buffers.subspan(output_index * processor.sample_count,
                                                  processor.sample_count);
    ProcessAdpcmDataSourceCommand(processor, output_buffer, SampleFormat::Adpcm, voice_state,
                                  wave_buffers, src_quality, pitch, sample_rate,
                                  processor.target_sample_rate, processor.sample_count, data_address,
                                  data_size, (flags & 1) != 0, (flags & 2) != 0);
}

void AdpcmDataSourceVersion2Command::Process(const ADSP::CommandListProcessor& processor) {
    auto output_buffer = processor.mix_buffers.subspan(output_index * processor.sample_count,
                                                  processor.sample_count);
    ProcessAdpcmDataSourceCommand(processor, output_buffer, SampleFormat::Adpcm, voice_state,
                                  wave_buffers, src_quality, pitch, sample_rate,
                                  processor.target_sample_rate, processor.sample_count, data_address,
                                  data_size, (flags & 1) != 0, (flags & 2) != 0);
}

bool AdpcmDataSourceVersion1Command::Verify(const ADSP::CommandListProcessor& processor) {
    return true;
}

bool AdpcmDataSourceVersion2Command::Verify(const ADSP::CommandListProcessor& processor) {
    return true;
}

}  // namespace AudioCore::AudioRenderer