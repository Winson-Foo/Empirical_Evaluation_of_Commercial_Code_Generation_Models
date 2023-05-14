#include "audio_core/renderer/adsp/command_list_processor.h"
#include "audio_core/renderer/command/effect/biquad_filter.h"
#include "audio_core/renderer/voice/voice_state.h"
#include "common/bit_cast.h"
#include <array>
#include <cmath>
#include <limits>
#include <span>

namespace AudioCore::AudioRenderer {

constexpr int kFixedPointBits = 14;
constexpr int kFixedPointFrac = 50;
constexpr int kNumBiquadCoefficients = 3;
constexpr int kNumBiquadFeedbackCoefficients = 2;

void ApplyBiquadFilterFloat(std::span<s32> output, std::span<const s32> input,
                            const std::array<s16, kNumBiquadCoefficients>& feedforward_coeffs,
                            const std::array<s16, kNumBiquadFeedbackCoefficients>& feedback_coeffs,
                            VoiceState::BiquadFilterState& state, const u32 sample_count) {
    constexpr double kMinSample = static_cast<double>(std::numeric_limits<s32>::min());
    constexpr double kMaxSample = static_cast<double>(std::numeric_limits<s32>::max());

    std::array<double, kNumBiquadCoefficients> b{
        Common::FixedPoint<kFixedPointFrac, kFixedPointBits>::from_base(feedforward_coeffs[0]).to_double(),
        Common::FixedPoint<kFixedPointFrac, kFixedPointBits>::from_base(feedforward_coeffs[1]).to_double(),
        Common::FixedPoint<kFixedPointFrac, kFixedPointBits>::from_base(feedforward_coeffs[2]).to_double()};
    std::array<double, kNumBiquadFeedbackCoefficients> a{
        Common::FixedPoint<kFixedPointFrac, kFixedPointBits>::from_base(feedback_coeffs[0]).to_double(),
        Common::FixedPoint<kFixedPointFrac, kFixedPointBits>::from_base(feedback_coeffs[1]).to_double()};
    std::array<double, 4> s{Common::BitCast<double>(state.s0), Common::BitCast<double>(state.s1),
                            Common::BitCast<double>(state.s2), Common::BitCast<double>(state.s3)};

    for (u32 i = 0; i < sample_count; i++) {
        const double in_sample = static_cast<double>(input[i]);
        double filtered_sample = in_sample * b[0] + s[0] * b[1] + s[1] * b[2] + s[2] * a[0] + s[3] * a[1];
        filtered_sample = std::clamp(filtered_sample, kMinSample, kMaxSample);
        output[i] = static_cast<s32>(filtered_sample);

        s[1] = s[0];
        s[0] = in_sample;
        s[3] = s[2];
        s[2] = filtered_sample;
    }

    state.s0 = Common::BitCast<s64>(s[0]);
    state.s1 = Common::BitCast<s64>(s[1]);
    state.s2 = Common::BitCast<s64>(s[2]);
    state.s3 = Common::BitCast<s64>(s[3]);
}

void ApplyBiquadFilterInt(std::span<s32> output, std::span<const s32> input,
                          const std::array<s16, kNumBiquadCoefficients>& feedforward_coeffs,
                          const std::array<s16, kNumBiquadFeedbackCoefficients>& feedback_coeffs,
                          VoiceState::BiquadFilterState& state, const u32 sample_count) {
    constexpr s64 kMinSample = std::numeric_limits<s32>::min();
    constexpr s64 kMaxSample = std::numeric_limits<s32>::max();

    for (u32 i = 0; i < sample_count; i++) {
        const s64 in_sample{input[i]};
        const s64 sample{in_sample * feedforward_coeffs[0] + state.s0};
        const s64 out_sample{std::clamp<s64>((sample + (1 << (kFixedPointBits - 1))) >> kFixedPointBits,
                                             kMinSample, kMaxSample)};
        output[i] = static_cast<s32>(out_sample);

        state.s0 = state.s1 + feedforward_coeffs[1] * in_sample + feedback_coeffs[0] * out_sample;
        state.s1 = feedforward_coeffs[2] * in_sample + feedback_coeffs[1] * out_sample;
    }
}

void BiquadFilterCommand::Dump(const ADSP::CommandListProcessor& processor, std::string& string) {
    string += fmt::format("BiquadFilterCommand\n\tinput {:02X} output {:02X} needs_init {} use_float_processing {}\n",
                          input, output, needs_init, use_float_processing);
}

void BiquadFilterCommand::Process(const ADSP::CommandListProcessor& processor) {
    auto state = reinterpret_cast<VoiceState::BiquadFilterState*>(state);
    if (needs_init) {
        *state = {};
    }

    auto input_buffer{processor.mix_buffers.subspan(input * processor.sample_count, processor.sample_count)};
    auto output_buffer{processor.mix_buffers.subspan(output * processor.sample_count, processor.sample_count)};

    if (use_float_processing) {
        ApplyBiquadFilterFloat(output_buffer, input_buffer, biquad.b, biquad.a, *state, processor.sample_count);
    } else {
        ApplyBiquadFilterInt(output_buffer, input_buffer, biquad.b, biquad.a, *state, processor.sample_count);
    }
}

bool BiquadFilterCommand::Verify(const ADSP::CommandListProcessor& processor) {
    return true;
}

} // namespace AudioCore::AudioRenderer