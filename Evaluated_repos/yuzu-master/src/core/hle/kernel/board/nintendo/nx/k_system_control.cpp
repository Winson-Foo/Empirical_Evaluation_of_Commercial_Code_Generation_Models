// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <random>

#include "common/literals.h"
#include "common/settings.h"

#include "core/hle/kernel/board/nintendo/nx/k_system_control.h"
#include "core/hle/kernel/board/nintendo/nx/secure_monitor.h"
#include "core/hle/kernel/k_trace.h"

namespace Kernel::Board::Nintendo::Nx {

namespace impl {

using namespace Common::Literals;

constexpr const std::size_t RequiredNonSecureSystemMemorySizeVi = 0x2280 * 4_KiB;
constexpr const std::size_t RequiredNonSecureSystemMemorySizeViFatal = 0x200 * 4_KiB;
constexpr const std::size_t RequiredNonSecureSystemMemorySizeNvservices = 0x704 * 4_KiB;
constexpr const std::size_t RequiredNonSecureSystemMemorySizeMisc = 0x80 * 4_KiB;

} // namespace impl

constexpr const std::size_t RequiredNonSecureSystemMemorySize =
    impl::RequiredNonSecureSystemMemorySizeVi + impl::RequiredNonSecureSystemMemorySizeNvservices +
    impl::RequiredNonSecureSystemMemorySizeMisc;

constexpr const std::size_t RequiredNonSecureSystemMemorySizeWithFatal =
    RequiredNonSecureSystemMemorySize + impl::RequiredNonSecureSystemMemorySizeViFatal;

namespace {

using namespace Common::Literals;

u32 GetMemorySizeForInit() {
    return Settings::values.use_unsafe_extended_memory_layout ? Smc::MemorySize_8GB
                                                              : Smc::MemorySize_4GB;
}

Smc::MemoryArrangement GetMemoryArrangeForInit() {
    return Settings::values.use_unsafe_extended_memory_layout ? Smc::MemoryArrangement_8GB
                                                              : Smc::MemoryArrangement_4GB;
}
} // namespace

size_t KSystemControl::Init::GetRealMemorySize() {
    return GetIntendedMemorySize();
}

// Initialization.
size_t KSystemControl::Init::GetIntendedMemorySize() {
    switch (GetMemorySizeForInit()) {
    case Smc::MemorySize_4GB:
    default: // All invalid modes should go to 4GB.
        return 4_GiB;
    case Smc::MemorySize_6GB:
        return 6_GiB;
    case Smc::MemorySize_8GB:
        return 8_GiB;
    }
}

KPhysicalAddress KSystemControl::Init::GetKernelPhysicalBaseAddress(KPhysicalAddress base_address) {
    const size_t real_dram_size = KSystemControl::Init::GetRealMemorySize();
    const size_t intended_dram_size = KSystemControl::Init::GetIntendedMemorySize();
    if (intended_dram_size * 2 < real_dram_size) {
        return base_address;
    } else {
        return base_address + ((real_dram_size - intended_dram_size) / 2);
    }
}

bool KSystemControl::Init::ShouldIncreaseThreadResourceLimit() {
    return true;
}

std::size_t KSystemControl::Init::GetApplicationPoolSize() {
    // Get the base pool size.
    const size_t base_pool_size = []() -> size_t {
        switch (GetMemoryArrangeForInit()) {
        case Smc::MemoryArrangement_4GB:
        default:
            return 3285_MiB;
        case Smc::MemoryArrangement_4GBForAppletDev:
            return 2048_MiB;
        case Smc::MemoryArrangement_4GBForSystemDev:
            return 3285_MiB;
        case Smc::MemoryArrangement_6GB:
            return 4916_MiB;
        case Smc::MemoryArrangement_6GBForAppletDev:
            return 3285_MiB;
        case Smc::MemoryArrangement_8GB:
            // Real kernel sets this to 4916_MiB. We are not debugging applets.
            return 6547_MiB;
        }
    }();

    // Return (possibly) adjusted size.
    return base_pool_size;
}

size_t KSystemControl::Init::GetAppletPoolSize() {
    // Get the base pool size.
    const size_t base_pool_size = []() -> size_t {
        switch (GetMemoryArrangeForInit()) {
        case Smc::MemoryArrangement_4GB:
        default:
            return 507_MiB;
        case Smc::MemoryArrangement_4GBForAppletDev:
            return 1554_MiB;
        case Smc::MemoryArrangement_4GBForSystemDev:
            return 448_MiB;
        case Smc::MemoryArrangement_6GB:
            return 562_MiB;
        case Smc::MemoryArrangement_6GBForAppletDev:
            return 2193_MiB;
        case Smc::MemoryArrangement_8GB:
            //! Real kernel sets this to 2193_MiB. We are not debugging applets.
            return 562_MiB;
        }
    }();

    // Return (possibly) adjusted size.
    constexpr size_t ExtraSystemMemoryForAtmosphere = 33_MiB;
    return base_pool_size - ExtraSystemMemoryForAtmosphere - KTraceBufferSize;
}

size_t KSystemControl::Init::GetMinimumNonSecureSystemPoolSize() {
    // Verify that our minimum is at least as large as Nintendo's.
    constexpr size_t MinimumSizeWithFatal = RequiredNonSecureSystemMemorySizeWithFatal;
    static_assert(MinimumSizeWithFatal >= 0x2C04000);

    constexpr size_t MinimumSizeWithoutFatal = RequiredNonSecureSystemMemorySize;
    static_assert(MinimumSizeWithoutFatal >= 0x2A00000);

    return MinimumSizeWithFatal;
}

namespace {
template <typename F>
u64 GenerateUniformRange(u64 min, u64 max, F f) {
    // Handle the case where the difference is too large to represent.
    if (max == std::numeric_limits<u64>::max() && min == std::numeric_limits<u64>::min()) {
        return f();
    }

    // Iterate until we get a value in range.
    const u64 range_size = ((max + 1) - min);
    const u64 effective_max = (std::numeric_limits<u64>::max() / range_size) * range_size;
    while (true) {
        if (const u64 rnd = f(); rnd < effective_max) {
            return min + (rnd % range_size);
        }
    }
}

} // Anonymous namespace

u64 KSystemControl::GenerateRandomU64() {
    std::random_device device;
    std::mt19937 gen(device());
    std::uniform_int_distribution<u64> distribution(1, std::numeric_limits<u64>::max());
    return distribution(gen);
}

u64 KSystemControl::GenerateRandomRange(u64 min, u64 max) {
    return GenerateUniformRange(min, max, GenerateRandomU64);
}

} // namespace Kernel::Board::Nintendo::Nx
