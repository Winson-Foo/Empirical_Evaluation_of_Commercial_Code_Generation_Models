// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/atomic_ops.h"
#include "common/steady_clock.h"
#include "common/uint128.h"

#include "common/x64/native_clock.h"

namespace Common {

    constexpr double NS_RATIO = 1000000000.0;
    constexpr double US_RATIO = 1000000.0;
    constexpr double MS_RATIO = 1000.0;

    static u64 FencedRDTSC() {
#ifdef _MSC_VER
        _mm_lfence();
        _ReadWriteBarrier();
        const u64 result = __rdtsc();
        _mm_lfence();
        _ReadWriteBarrier();
        return result;
#else
        u64 eax;
        u64 edx;
        asm volatile("lfence\n\t"
                     "rdtsc\n\t"
                     "lfence\n\t"
                     : "=a"(eax), "=d"(edx));
        return (edx << 32) | eax;
#endif
    }

    template <u64 Nearest>
    static u64 RoundToNearest(u64 value) {
        const auto mod = value % Nearest;
        return mod >= (Nearest / 2) ? (value - mod + Nearest) : (value - mod);
    }

    u64 EstimateRDTSCFrequency() {
        // Discard the first result measuring the rdtsc.
        FencedRDTSC();
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
        FencedRDTSC();

        // Get the current time.
        const auto start_time = RealTimeClock::Now();
        const u64 tsc_start = FencedRDTSC();
        // Wait for 250 milliseconds.
        std::this_thread::sleep_for(std::chrono::milliseconds{250});
        const auto end_time = RealTimeClock::Now();
        const u64 tsc_end = FencedRDTSC();
        // Calculate differences.
        const u64 timer_diff = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        const u64 tsc_diff = tsc_end - tsc_start;
        return RoundToNearest<1000>(MultiplyAndDivide64(tsc_diff, 1000000000ULL, timer_diff));
    }

    namespace X64 {

        NativeClock::NativeClock(u64 emulated_cpu_frequency_, u64 emulated_clock_frequency_,
                                 u64 rtsc_frequency_)
                : WallClock(emulated_cpu_frequency_, emulated_clock_frequency_, true),
                  rtsc_frequency{rtsc_frequency_} {
            // Thread to re-adjust the RDTSC frequency after 10 seconds has elapsed.
            time_sync_thread = std::jthread{[this](std::stop_token token) {
                adjustRTSCFrequency(token);
            }};

            time_point.inner.last_measure = FencedRDTSC();
            time_point.inner.accumulated_ticks = 0U;
            CalculateAndSetFactors();
        }

        void NativeClock::adjustRTSCFrequency(const std::stop_token& token) {
            // Get the current time.
            const auto start_time = RealTimeClock::Now();
            const u64 tsc_start = FencedRDTSC();

            // Wait for 10 seconds.
            if (!StoppableTimedWait(token, std::chrono::seconds{10})) {
                return;
            }

            // Calculate differences.
            const auto end_time = RealTimeClock::Now();
            const u64 tsc_end = FencedRDTSC();
            const u64 timer_diff = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
            const u64 tsc_diff = tsc_end - tsc_start;

            // Apply new frequency.
            rtsc_frequency = MultiplyAndDivide64(tsc_diff, 1000000000ULL, timer_diff);
            CalculateAndSetFactors();
        }

        void NativeClock::CalculateAndSetFactors() {
            ns_rtsc_factor = GetFixedPoint64Factor(NS_RATIO, rtsc_frequency);
            us_rtsc_factor = GetFixedPoint64Factor(US_RATIO, rtsc_frequency);
            ms_rtsc_factor = GetFixedPoint64Factor(MS_RATIO, rtsc_frequency);
            clock_rtsc_factor = GetFixedPoint64Factor(emulated_clock_frequency, rtsc_frequency);
            cpu_rtsc_factor = GetFixedPoint64Factor(emulated_cpu_frequency, rtsc_frequency);
        }

        u64 NativeClock::GetRTSC() {
            TimePoint new_time_point{};
            TimePoint current_time_point{};

            current_time_point.pack = AtomicLoad128(time_point.pack.data());
            do {
                const u64 current_measure = FencedRDTSC();
                u64 diff = current_measure - current_time_point.inner.last_measure;
                diff = diff & ~static_cast<u64>(static_cast<s64>(diff) >> 63); // max(diff, 0)
                new_time_point.inner.last_measure =
                        current_measure > current_time_point.inner.last_measure
                        ? current_measure
                        : current_time_point.inner.last_measure;
                new_time_point.inner.accumulated_ticks =
                        current_time_point.inner.accumulated_ticks + diff;
            } while (!AtomicCompareAndSwap(time_point.pack.data(), new_time_point.pack,
                                           current_time_point.pack, current_time_point.pack));
            return new_time_point.inner.accumulated_ticks;
        }

        void NativeClock::Pause(bool is_paused) {
            if (!is_paused) {
                TimePoint current_time_point{};
                TimePoint new_time_point{};

                current_time_point.pack = AtomicLoad128(time_point.pack.data());
                do {
                    new_time_point.pack = current_time_point.pack;
                    new_time_point.inner.last_measure = FencedRDTSC();
                } while (!AtomicCompareAndSwap(time_point.pack.data(), new_time_point.pack,
                    current_time_point.pack, current_time_point.pack));
            }
        }

        std::chrono::nanoseconds NativeClock::GetTimeNS() {
            const auto rtsc_value = GetRTSC();
            return std::chrono::nanoseconds{MultiplyHigh(rtsc_value, ns_rtsc_factor)};
        }

        std::chrono::microseconds NativeClock::GetTimeUS() {
            const auto rtsc_value = GetRTSC();
            return std::chrono::microseconds{MultiplyHigh(rtsc_value, us_rtsc_factor)};
        }

        std::chrono::milliseconds NativeClock::GetTimeMS() {
            const auto rtsc_value = GetRTSC();
            return std::chrono::milliseconds{MultiplyHigh(rtsc_value, ms_rtsc_factor)};
        }

        u64 NativeClock::GetClockCycles() {
            const auto rtsc_value = GetRTSC();
            return MultiplyHigh(rtsc_value, clock_rtsc_factor);
        }

        u64 NativeClock::GetCPUCycles() {
            const auto rtsc_value = GetRTSC();
            return MultiplyHigh(rtsc_value, cpu_rtsc_factor);
        }
    } // namespace X64

} // namespace Common

