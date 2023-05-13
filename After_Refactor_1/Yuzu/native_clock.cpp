// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <chrono>
#include <thread>

#include "common/atomic_ops.h"
#include "common/steady_clock.h"
#include "common/uint128.h"
#include "common/x64/native_clock.h"

namespace Common {

// Constants
constexpr u64 kMeasureIntervalMS = 250;
constexpr u64 kTimeSyncIntervalS = 10;

#ifdef _MSC_VER
// Fenced read of the timestamp counter (TSC) on x86 with MSVC
__forceinline static u64 FencedRDTSC() {
    _mm_lfence();
    _ReadWriteBarrier();
    const u64 result = __rdtsc();
    _mm_lfence();
    _ReadWriteBarrier();
    return result;
}
#else
// Fenced read of the TSC on x86 with GCC/Clang
static u64 FencedRDTSC() {
    u64 eax;
    u64 edx;
    asm volatile("lfence\n\t"
                 "rdtsc\n\t"
                 "lfence\n\t"
                 : "=a"(eax), "=d"(edx));
    return (edx << 32) | eax;
}
#endif

// Rounds a value to the nearest multiple of a given number
template <u64 Nearest>
static u64 RoundToNearest(u64 value) {
    const auto mod = value % Nearest;
    return mod >= (Nearest / 2) ? (value - mod + Nearest) : (value - mod);
}

// Estimates the frequency of the TSC
u64 EstimateRDTSCFrequency() {
    // Discard the first result measuring the TSC
    FencedRDTSC();
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    FencedRDTSC();

    // Measure the TSC and wall clock time during a fixed interval
    const auto start_time = Common::RealTimeClock::Now();
    const u64 tsc_start = FencedRDTSC();
    std::this_thread::sleep_for(std::chrono::milliseconds{kMeasureIntervalMS});
    const auto end_time = Common::RealTimeClock::Now();
    const u64 tsc_end = FencedRDTSC();

    // Calculate the TSC frequency
    const u64 timer_diff = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    const u64 tsc_diff = tsc_end - tsc_start;
    const u64 tsc_freq = MultiplyAndDivide64(tsc_diff, 1000000000ULL, timer_diff);

    return RoundToNearest<1000>(tsc_freq);
}

namespace X64 {

// Constructs a native clock that uses the TSC as the underlying clock
NativeClock::NativeClock(u64 emulated_cpu_frequency_, u64 emulated_clock_frequency_,
                         u64 rtsc_frequency_)
    : WallClock(emulated_cpu_frequency_, emulated_clock_frequency_, true), rtsc_frequency{
                                                                               rtsc_frequency_} {

    // Continuously adjusts the TSC frequency to account for drift
    time_sync_thread = std::jthread{[this](std::stop_token token) {
        while (!token.stop_requested()) {
            // Measure the TSC and wall clock time during a fixed interval
            const auto start_time = Common::RealTimeClock::Now();
            const u64 tsc_start = FencedRDTSC();
            if (!Common::StoppableTimedWait(token, std::chrono::seconds{kTimeSyncIntervalS})) {
                break;
            }
            const auto end_time = Common::RealTimeClock::Now();
            const u64 tsc_end = FencedRDTSC();

            // Calculate the TSC frequency and update the clock factors
            const u64 timer_diff = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
            const u64 tsc_diff = tsc_end - tsc_start;
            const u64 tsc_freq = MultiplyAndDivide64(tsc_diff, 1000000000ULL, timer_diff);
            rtsc_frequency = tsc_freq;
            UpdateClockFactors();
        }
    }};

    time_point.inner.last_measure = FencedRDTSC();
    time_point.inner.accumulated_ticks = 0U;
    UpdateClockFactors();
}

// Returns the current value of the TSC
u64 NativeClock::GetRTSC() {
    TimePoint current_time_point{};
    TimePoint new_time_point{};

    // Atomic read-modify-write of the time point
    do {
        current_time_point.pack = Common::AtomicLoad128(time_point.pack.data());
        new_time_point.inner.last_measure = FencedRDTSC();
        new_time_point.inner.accumulated_ticks = current_time_point.inner.accumulated_ticks +
            (new_time_point.inner.last_measure - current_time_point.inner.last_measure);
    } while (!Common::AtomicCompareAndSwap(time_point.pack.data(), new_time_point.pack,
                                           current_time_point.pack, current_time_point.pack));
    return new_time_point.inner.accumulated_ticks;
}

// Pauses or resumes the clock
void NativeClock::Pause(bool is_paused) {
    if (!is_paused) {
        // Atomic update of the last measurement
        TimePoint current_time_point{};
        TimePoint new_time_point{};
        current_time_point.pack = Common::AtomicLoad128(time_point.pack.data());
        new_time_point.pack = current_time_point.pack;
        new_time_point.inner.last_measure = FencedRDTSC();
        (void)Common::AtomicCompareAndSwap(time_point.pack.data(), new_time_point.pack,
                                           current_time_point.pack, current_time_point.pack);
    }
}

// Returns the current time using the TSC as the underlying clock
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

// Returns the current clock cycle count using the TSC as the underlying clock
u64 NativeClock::GetClockCycles() {
    const u64 rtsc_value = GetRTSC();
    return MultiplyHigh(rtsc_value, clock_rtsc_factor);
}

// Returns the current CPU cycle count using the TSC as the underlying clock
u64 NativeClock::GetCPUCycles() {
    const u64 rtsc_value = GetRTSC();
    return MultiplyHigh(rtsc_value, cpu_rtsc_factor);
}

// Updates the clock factors based on the TSC frequency
void NativeClock::UpdateClockFactors() {
    ns_rtsc_factor = GetFixedPoint64Factor(NS_RATIO, rtsc_frequency);
    us_rtsc_factor = GetFixedPoint64Factor(US_RATIO, rtsc_frequency);
    ms_rtsc_factor = GetFixedPoint64Factor(MS_RATIO, rtsc_frequency);
    clock_rtsc_factor = GetFixedPoint64Factor(emulated_clock_frequency, rtsc_frequency);
    cpu_rtsc_factor = GetFixedPoint64Factor(emulated_cpu_frequency, rtsc_frequency);
}

} // namespace X64

} // namespace Common