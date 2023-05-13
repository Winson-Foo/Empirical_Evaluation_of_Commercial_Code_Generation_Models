#include "audio_core/renderer/adsp/command_list_processor.h"
#include "audio_core/renderer/command/effect/biquad_filter.h"
#include "audio_core/renderer/voice/voice_state.h"
#include "common/bit_cast.h"

namespace AudioCore::AudioRenderer {

constexpr int kFixedPointBase = 50;
constexpr int kFixedPointFractionalBits = 14;

constexpr double kMinS32 = std::numeric_limits<s32>::min();
constexpr double kMaxS32 = std::numeric_limits<s32>::max();
constexpr s64 kMinS64 = std::numeric_limits<s32>::min();
constexpr s64 kMaxS64 = std::numeric_limits<s32>::max();

void ApplyBiquadFilterFloat(std::span<s32> output, std::span<const s32> input,
                            std::array<s16, 3>& feedforward_coeffs, std::array<s16, 2>& feedback_coeffs,
                            VoiceState::BiquadFilterState& state, const u32 sample_count) {
    std::array<double, 3> b{Common::FixedPoint<kFixedPointBase, kFixedPointFractionalBits>::from_base(feedforward_coeffs[0]).to_double(),
                         Common::FixedPoint<kFixedPointBase, kFixedPointFractionalBits>::from_base(feedforward_coeffs[1]).to_double(),
                         Common::FixedPoint<kFixedPointBase, kFixedPointFractionalBits>::from_base(feedforward_coeffs[2]).to_double()};
    std::array<double, 2> a{Common::FixedPoint<kFixedPointBase, kFixedPointFractionalBits>::from_base(feedback_coeffs[0]).to_double(),
                         Common::FixedPoint<kFixedPointBase, kFixedPointFractionalBits>::from_base(feedback_coeffs[1]).to_double()};
    std::array<double, 4> s{Common::BitCast<double>(state.s0), Common::BitCast<double>(state.s1),
                         Common::BitCast<double>(state.s2), Common::BitCast<double>(state.s3)};

    for (u32 i = 0; i < sample_count; i++) {
        double in_sample{static_cast<double>(input[i])};
        double sample{in_sample * b[0] + s[0] * b[1] + s[1] * b[2] + s[2] * a[0] + s[3] * a[1]};

        output[i] = static_cast<s32>(std::clamp(sample, kMinS32, kMaxS32));

        s[1] = s[0];
        s[0] = in_sample;
        s[3] = s[2];
        s[2] = sample;
    }

    state.s0 = Common::BitCast<s64>(s[0]);
    state.s1 = Common::BitCast<s64>(s[1]);
    state.s2 = Common::BitCast<s64>(s[2]);
    state.s3 = Common::BitCast<s64>(s[3]);
}

void ApplyBiquadFilterInt(std::span<s32> output, std::span<const s32> input,
                            std::array<s16, 3>& feedforward_coeffs, std::array<s16, 2>& feedback_coeffs,
                            VoiceState::BiquadFilterState& state, const u32 sample_count) {

    for (u32 i = 0; i < sample_count; i++) {
        const s64 in_sample{input[i]};
        const s64 sample{in_sample * feedforward_coeffs[0] + state.s0};
        const s64 out_sample{std::clamp<s64>((sample + (1 << 13)) >> 14, kMinS64, kMaxS64)};

        output[i] = static_cast<s32>(out_sample);

        state.s0 = state.s1 + feedforward_coeffs[1] * in_sample + feedback_coeffs[0] * out_sample;
        state.s1 = feedforward_coeffs[2] * in_sample + feedback_coeffs[1] * out_sample;
    }
}

void BiquadFilterCommand::Dump(const ADSP::CommandListProcessor& processor, std::string& string) {
    string += fmt::format(
        "BiquadFilterCommand\n\tinput {:02X} output {:02X} needs_init {} use_float_processing {}\n",
        input, output, needs_init, use_float_processing);
}

void BiquadFilterCommand::Process(const ADSP::CommandListProcessor& processor) {
    auto state_{reinterpret_cast<VoiceState::BiquadFilterState*>(state)};
    if (needs_init) {
        *state_ = {};
    }

    auto input_buffer{
        processor.mix_buffers.subspan(input * processor.sample_count, processor.sample_count)};
    auto output_buffer{
        processor.mix_buffers.subspan(output * processor.sample_count, processor.sample_count)};

    if (use_float_processing) {
        ApplyBiquadFilterFloat(output_buffer, input_buffer, biquad.b, biquad.a, *state_,
                               processor.sample_count);
    } else {
        ApplyBiquadFilterInt(output_buffer, input_buffer, biquad.b, biquad.a, *state_,
                             processor.sample_count);
    }
}

bool BiquadFilterCommand::Verify(const ADSP::CommandListProcessor& processor) {
    return true;
}

}  // namespace AudioCore::AudioRenderer