// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <chrono>
#include <thread>

#include "common/atomic_ops.h"
#include "common/steady_clock.h"
#include "common/uint128.h"
#include "common/x64/native_clock.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace Common {

#ifdef _MSC_VER
__forceinline static uint64_t FencedRDTSC() {
    _mm_lfence();
    _ReadWriteBarrier();
    const uint64_t result = __rdtsc();
    _mm_lfence();
    _ReadWriteBarrier();
    return result;
}
#else
static uint64_t FencedRDTSC() {
    uint64_t eax;
    uint64_t edx;
    asm volatile("lfence\n\t"
                 "rdtsc\n\t"
                 "lfence\n\t"
                 : "=a"(eax), "=d"(edx));
    return (edx << 32) | eax;
}
#endif

// Named constants
const uint64_t ONE_SECOND = 1000000000ULL;
const uint64_t ONE_MILLISECOND = 1000;
const uint64_t ONE_MICROSECOND = 1000 * ONE_MILLISECOND;
const uint64_t ONE_NANOSECOND = 1000 * ONE_MICROSECOND;

// Typedefs
typedef uint64_t u64;
typedef int64_t s64;

// Auxiliar functions
template <u64 Nearest>
static u64 RoundToNearest(u64 value) {
    const auto mod = value % Nearest;
    return mod >= (Nearest / 2) ? (value - mod + Nearest) : (value - mod);
}

static u64 GetTimeDifference(Common::TimePoint start_time, Common::TimePoint end_time) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
}

// Estimating RDTSC frequency
static u64 MeasureRDTSCFrequency() {
    const auto start_time = Common::RealTimeClock::Now();
    const u64 tsc_start = FencedRDTSC();
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    const u64 tsc_end = FencedRDTSC();
    const auto end_time = Common::RealTimeClock::Now();

    const u64 timer_diff = GetTimeDifference(start_time, end_time);
    const u64 tsc_diff = tsc_end - tsc_start;
    const u64 tsc_freq = MultiplyAndDivide64(tsc_diff, ONE_SECOND, timer_diff);
    return RoundToNearest<ONE_MILLISECOND>(tsc_freq);
}

u64 EstimateRDTSCFrequency() {
    // Discard the first result measuring the rdtsc.
    FencedRDTSC();
    // Get the RDTSC frequency
    const u64 rtsc_frequency = MeasureRDTSCFrequency();
    return rtsc_frequency;
}

namespace X64 {
NativeClock::NativeClock(u64 emulated_cpu_frequency, u64 emulated_clock_frequency, u64 rtsc_frequency)
    : WallClock(emulated_cpu_frequency, emulated_clock_frequency, true), rtsc_frequency(rtsc_frequency) {
    time_sync_thread = std::jthread{[this](std::stop_token token) {
        const auto start_time = Common::RealTimeClock::Now();
        const u64 tsc_start = FencedRDTSC();

        // Wait for 10 seconds.
        if (!Common::StoppableTimedWait(token, std::chrono::seconds{10})) {
            return;
        }

        const auto end_time = Common::RealTimeClock::Now();
        const u64 tsc_end = FencedRDTSC();

        const u64 timer_diff = GetTimeDifference(start_time, end_time);
        const u64 tsc_diff = tsc_end - tsc_start;
        const u64 new_rtsc_freq = MultiplyAndDivide64(tsc_diff, ONE_SECOND, timer_diff);

        rtsc_frequency = new_rtsc_freq;
        CalculateAndSetFactors();
    }};

    time_point.inner.last_measure = FencedRDTSC();
    time_point.inner.accumulated_ticks = 0U;
    CalculateAndSetFactors();
}

u64 NativeClock::GetRTSC() {
    TimePoint new_time_point{};
    TimePoint current_time_point{};

    current_time_point.pack = Common::AtomicLoad128(time_point.pack.data());
    do {
        const u64 current_measure = FencedRDTSC();
        u64 diff = current_measure - current_time_point.inner.last_measure;
        diff = diff & ~static_cast<u64>(static_cast<s64>(diff) >> 63); // max(diff, 0)
        new_time_point.inner.last_measure = current_measure > current_time_point.inner.last_measure
                                                ? current_measure
                                                : current_time_point.inner.last_measure;
        new_time_point.inner.accumulated_ticks = current_time_point.inner.accumulated_ticks + diff;
    } while (!Common::AtomicCompareAndSwap(time_point.pack.data(), new_time_point.pack,
                                           current_time_point.pack, current_time_point.pack));
    return new_time_point.inner.accumulated_ticks;
}

void NativeClock::Pause(bool is_paused) {
    if (!is_paused) {
        TimePoint current_time_point{};
        TimePoint new_time_point{};

        current_time_point.pack = Common::AtomicLoad128(time_point.pack.data());
        do {
            new_time_point.pack = current_time_point.pack;
            new_time_point.inner.last_measure = FencedRDTSC();
        } while (!Common::AtomicCompareAndSwap(time_point.pack.data(), new_time_point.pack,
                                               current_time_point.pack, current_time_point.pack));
    }
}

std::chrono::nanoseconds NativeClock::GetTimeNS() {
    const u64 rtsc_value = GetRTSC();
    return std::chrono::nanoseconds{MultiplyHigh(rtsc_value, ns_rtsc_factor)};
}

std::chrono::microseconds NativeClock::GetTimeUS() {
    const u64 rtsc_value = GetRTSC();
    return std::chrono::microseconds{MultiplyHigh(rtsc_value, us_rtsc_factor)};
}

std::chrono::milliseconds NativeClock::GetTimeMS() {
    const u64 rtsc_value = GetRTSC();
    return std::chrono::milliseconds{MultiplyHigh(rtsc_value, ms_rtsc_factor)};
}

u64 NativeClock::GetClockCycles() {
    const u64 rtsc_value = GetRTSC();
    return MultiplyHigh(rtsc_value, clock_rtsc_factor);
}

u64 NativeClock::GetCPUCycles() {
    const u64 rtsc_value = GetRTSC();
    return MultiplyHigh(rtsc_value, cpu_rtsc_factor);
}

void NativeClock::CalculateAndSetFactors() {
    ns_rtsc_factor = GetFixedPoint64Factor(ONE_NANOSECOND, rtsc_frequency);
    us_rtsc_factor = GetFixedPoint64Factor(ONE_MICROSECOND, rtsc_frequency);
    ms_rtsc_factor = GetFixedPoint64Factor(ONE_MILLISECOND, rtsc_frequency);
    clock_rtsc_factor = GetFixedPoint64Factor(emulated_clock_frequency, rtsc_frequency);
    cpu_rtsc_factor = GetFixedPoint64Factor(emulated_cpu_frequency, rtsc_frequency);
}

} // namespace X64

} // namespace Common

