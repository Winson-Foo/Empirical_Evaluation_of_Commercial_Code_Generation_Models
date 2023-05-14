#pragma once

#include <array>

#include "common/fixed_point.h"

namespace AudioCore::AudioRenderer {

constexpr int kMaxSampleCount = 1024;

template <size_t Q>
s32 ApplyMixRamp(std::array<s32, kMaxSampleCount>& output, 
                 const std::array<s32, kMaxSampleCount>& input, 
                 const f32 volume, const f32 ramp, const u32 sample_count) {
    Common::FixedPoint<64 - Q, Q> volume_fp{volume};
    Common::FixedPoint<64 - Q, Q> sample_fp{0};

    if (ramp == 0.0f) {
        for (u32 i = 0; i < sample_count; i++) {
            sample_fp = input[i] * volume_fp;
            output[i] += sample_fp.to_int();
        }
    } else {
        Common::FixedPoint<64 - Q, Q> ramp_fp{ramp};
        for (u32 i = 0; i < sample_count; i++) {
            sample_fp = input[i] * volume_fp;
            output[i] += sample_fp.to_int();
            volume_fp += ramp_fp;
        }
    }
    return sample_fp.to_int();
}

} // namespace AudioCore::AudioRenderer

// mix_ramp_command.h

#pragma once

#include <string>

#include "audio_core/renderer/adsp/command_list_processor.h"
#include "common/logging/log.h"

namespace AudioCore::AudioRenderer {

class MixRampCommand {
public:
    u8 input_index;
    u8 output_index;
    f32 volume;
    f32 prev_volume;
    f32 ramp;
    u8 precision;
    u8 previous_sample[4];

    void Dump(const ADSP::CommandListProcessor& processor, std::string& string) {
        const auto ramp{(volume - prev_volume) / static_cast<f32>(processor.sample_count)};
        string += "MixRampCommand";
        string += "\n\tinput: " + std::to_string(input_index);
        string += "\n\toutput: " + std::to_string(output_index);
        string += "\n\tvolume: " + std::to_string(volume);
        string += "\n\tprev_volume: " + std::to_string(prev_volume);
        string += "\n\tramp: " + std::to_string(ramp);
        string += "\n";
    }

    void Process(const ADSP::CommandListProcessor& processor) {
        std::array<s32, kMaxSampleCount> output{};
        std::array<s32, kMaxSampleCount> input{};
        auto mix_buffers_begin = processor.mix_buffers.begin();

        std::copy(mix_buffers_begin + output_index * processor.sample_count,
                  mix_buffers_begin + (output_index + 1) * processor.sample_count,
                  output.begin());
        std::copy(mix_buffers_begin + input_index * processor.sample_count,
                  mix_buffers_begin + (input_index + 1) * processor.sample_count,
                  input.begin());

        const auto ramp{(volume - prev_volume) / static_cast<f32>(processor.sample_count)};
        s32* prev_sample_ptr{reinterpret_cast<s32*>(previous_sample)};

        // If previous volume and ramp are both 0, skip processing.
        if (prev_volume == 0.0f && ramp == 0.0f) {
            *prev_sample_ptr = 0;
            return;
        }

        switch (precision) {
            case 15:
                *prev_sample_ptr = ApplyMixRamp<15>(output, input, prev_volume, ramp,
                                                     processor.sample_count);
                break;

            case 23:
                *prev_sample_ptr = ApplyMixRamp<23>(output, input, prev_volume, ramp,
                                                     processor.sample_count);
                break;

            default:
                LOG_ERROR(Service_Audio, "Invalid precision: {}", precision);
                break;
        }

        std::copy(output.begin(), output.end(), mix_buffers_begin + output_index * processor.sample_count);
    }

    bool Verify(const ADSP::CommandListProcessor& processor) {
        return true;
    }
};

} // namespace AudioCore::AudioRenderer

