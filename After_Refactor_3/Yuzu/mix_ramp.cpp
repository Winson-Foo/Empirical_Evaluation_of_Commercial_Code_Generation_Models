#include "audio_core/renderer/adsp/command_list_processor.h"
#include "audio_core/renderer/command/mix/mix_ramp.h"
#include "common/fixed_point.h"
#include "common/logging/log.h"

namespace AudioCore::AudioRenderer {

template <size_t precision_bits>
std::int32_t ApplyMixRamp(std::span<std::int32_t> output, std::span<const std::int32_t> input, const float volume_, const float ramp_, const std::uint32_t sample_count) {
    Common::FixedPoint<64 - precision_bits, precision_bits> volume{volume_};
    Common::FixedPoint<64 - precision_bits, precision_bits> sample{0};

    if (ramp_ == 0.0f) {
        for (std::uint32_t i = 0; i < sample_count; i++) {
            sample = input[i] * volume;
            output[i] = (output[i] + sample).to_int();
        }
    } else {
        Common::FixedPoint<64 - precision_bits, precision_bits> ramp{ramp_};
        for (std::uint32_t i = 0; i < sample_count; i++) {
            sample = input[i] * volume;
            output[i] = (output[i] + sample).to_int();
            volume += ramp;
        }
    }
    return static_cast<std::int32_t>(sample.to_int());
}

template std::int32_t ApplyMixRamp<15>(std::span<std::int32_t>, std::span<const std::int32_t>, float, float, std::uint32_t);
template std::int32_t ApplyMixRamp<23>(std::span<std::int32_t>, std::span<const std::int32_t>, float, float, std::uint32_t);

void MixRampCommand::Dump(const ADSP::CommandListProcessor& processor, std::string& string) {
    const auto ramp{(volume - prev_volume) / static_cast<float>(processor.sample_count)};
    string += "MixRampCommand";
    string += fmt::format("\n\tinput {:02X}", input_index);
    string += fmt::format("\n\toutput {:02X}", output_index);
    string += fmt::format("\n\tvolume {:.8f}", volume);
    string += fmt::format("\n\tprev_volume {:.8f}", prev_volume);
    string += fmt::format("\n\tramp {:.8f}", ramp);
    string += "\n";
}

void MixRampCommand::Process(const ADSP::CommandListProcessor& processor) {
    const auto sample_count = processor.sample_count;
    auto output{processor.mix_buffers.subspan(output_index * sample_count, sample_count)};
    auto input{processor.mix_buffers.subspan(input_index * sample_count, sample_count)};
    const auto ramp{(volume - prev_volume) / static_cast<float>(sample_count)};
    auto prev_sample_ptr{reinterpret_cast<std::int32_t*>(previous_sample)};

    // If previous volume and ramp are both 0, nothing will be added to the output, so just skip.
    if (prev_volume == 0.0f && ramp == 0.0f) {
        *prev_sample_ptr = 0;
        return;
    }

    switch (precision) {
    case 15:
        *prev_sample_ptr = ApplyMixRamp<15>(output, input, prev_volume, ramp, sample_count);
        break;

    case 23:
        *prev_sample_ptr = ApplyMixRamp<23>(output, input, prev_volume, ramp, sample_count);
        break;

    default:
        LOG_ERROR(Service_Audio, "Invalid precision {}", precision);
        // Handle the error here
        break;
    }
}

bool MixRampCommand::Verify(const ADSP::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::AudioRenderer

