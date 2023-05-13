#include "audio_core/renderer/adsp/command_list_processor.h"
#include "audio_core/renderer/command/effect/biquad_filter.h"
#include "audio_core/renderer/voice/voice_state.h"
#include "common/bit_cast.h"

namespace AudioCore::AudioRenderer {

template <int BITS_INT, int BITS_DEC>
inline double FixedPointToDouble(const s16 value) {
    return Common::FixedPoint<BITS_INT, BITS_DEC>::from_base(value).to_double();
}

template <typename T>
void ApplyBiquadFilter(std::span<T> output, std::span<const T> input,
                       std::array<s16, 3>& b, std::array<s16, 2>& a,
                       VoiceState::BiquadFilterState& state, const u32 sample_count) {
    constexpr double kMin = static_cast<double>(std::numeric_limits<T>::min());
    constexpr double kMax = static_cast<double>(std::numeric_limits<T>::max());

    double b0_d = FixedPointToDouble<50, 14>(b[0]);
    double b1_d = FixedPointToDouble<50, 14>(b[1]);
    double b2_d = FixedPointToDouble<50, 14>(b[2]);
    double a0_d = FixedPointToDouble<50, 14>(a[0]);
    double a1_d = FixedPointToDouble<50, 14>(a[1]);
    double s0_d, s1_d, s2_d, s3_d;
    std::tie(s0_d, s1_d, s2_d, s3_d) = *state_;

    for (u32 i = 0; i < sample_count; i++) {
        double input_sample_d = static_cast<double>(input[i]);
        double output_sample_d =
            input_sample_d * b0_d + s0_d * b1_d + s1_d * b2_d + s2_d * a0_d + s3_d * a1_d;

        T output_sample_s = static_cast<T>(std::clamp(output_sample_d, kMin, kMax));
        output[i] = output_sample_s;

        s1_d = s0_d;
        s0_d = input_sample_d;
        s3_d = s2_d;
        s2_d = output_sample_d;
    }

    state.s0 = Common::BitCast<s64>(s0_d);
    state.s1 = Common::BitCast<s64>(s1_d);
    state.s2 = Common::BitCast<s64>(s2_d);
    state.s3 = Common::BitCast<s64>(s3_d);
}

void BiquadFilterCommand::Dump([[maybe_unused]] const ADSP::CommandListProcessor& processor,
                               std::string& string) {
    string += fmt::format(
        "BiquadFilterCommand\n\tinput {:02X} output {:02X} needs_init {} use_float_processing {}\n",
        input, output, needs_init, use_float_processing);
}

void BiquadFilterCommand::Process(const ADSP::CommandListProcessor& processor) {
    auto state_ = reinterpret_cast<VoiceState::BiquadFilterState*>(state);
    if (needs_init) {
        *state_ = {};
    }

    auto input_buffer =
        processor.mix_buffers.subspan(input * processor.sample_count, processor.sample_count);
    auto output_buffer =
        processor.mix_buffers.subspan(output * processor.sample_count, processor.sample_count);

    if (use_float_processing) {
        ApplyBiquadFilter<float>(output_buffer, input_buffer, biquad.b, biquad.a, *state_,
                                 processor.sample_count);
    } else {
        ApplyBiquadFilter<s32>(output_buffer, input_buffer, biquad.b, biquad.a, *state_,
                               processor.sample_count);
    }
}

bool BiquadFilterCommand::Verify(const ADSP::CommandListProcessor& processor) {
    return true;
}

}  // namespace AudioCore::AudioRenderer