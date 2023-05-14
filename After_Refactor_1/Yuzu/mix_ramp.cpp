#include "audio_core/renderer/adsp/command_list_processor.h"
#include "audio_core/renderer/command/mix/mix_ramp.h"
#include "common/fixed_point.h"
#include "common/logging/log.h"

namespace AudioCore::AudioRenderer {

template <size_t Q>
s32 ApplyMixRamp(std::span<s32> output, std::span<const s32> input, const f32 volume_,
                 const f32 ramp_, const u32 sample_count) {
    Common::FixedPoint<64 - Q, Q> volume{volume_};
    Common::FixedPoint<64 - Q, Q> sample{0};

    if (ramp_ == 0.0f) {
        for (u32 i = 0; i < sample_count; i++) {
            sample = input[i] * volume;
            output[i] = (output[i] + sample).to_int();
        }
    } else {
        Common::FixedPoint<64 - Q, Q> ramp{ramp_};
        for (u32 i = 0; i < sample_count; i++) {
            sample = input[i] * volume;
            output[i] = (output[i] + sample).to_int();
            volume += ramp;
        }
    }
    return sample.to_int();
}

s32 ApplyMixRamp(std::span<s32> output, std::span<const s32> input, const f32 volume,
                 const f32 ramp, const u32 sample_count, const u8 precision) {
    switch (precision) {
    case 15:
        return ApplyMixRamp<15>(output, input, volume, ramp, sample_count);
    case 23:
        return ApplyMixRamp<23>(output, input, volume, ramp, sample_count);
    default:
        LOG_ERROR(Service_Audio, "Invalid precision {}", precision);
        return 0;
    }
}

void MixRampCommand::Dump(const ADSP::CommandListProcessor& processor, std::string& string) {
    const auto ramp{(volume - prev_volume) / static_cast<f32>(processor.sample_count)};
    string += fmt::format("MixRampCommand");
    string += fmt::format("\n\tinput {:02X}", input_index);
    string += fmt::format("\n\toutput {:02X}", output_index);
    string += fmt::format("\n\tvolume {:.8f}", volume);
    string += fmt::format("\n\tprev_volume {:.8f}", prev_volume);
    string += fmt::format("\n\tramp {:.8f}", ramp);
    string += "\n";
}

void MixRampCommand::Process(const ADSP::CommandListProcessor& processor) {
    if (prev_volume == 0.0f && ramp == 0.0f) {
        previous_sample = 0;
        return;
    }

    auto output{processor.mix_buffers.subspan(output_index * processor.sample_count,
                                              processor.sample_count)};
    auto input{processor.mix_buffers.subspan(input_index * processor.sample_count,
                                             processor.sample_count)};
    const auto ramp_ {(volume - prev_volume) / static_cast<f32>(processor.sample_count)};
    previous_sample = ApplyMixRamp(output, input, prev_volume, ramp_, processor.sample_count, precision);
}

bool MixRampCommand::Verify(const ADSP::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::AudioRenderer

