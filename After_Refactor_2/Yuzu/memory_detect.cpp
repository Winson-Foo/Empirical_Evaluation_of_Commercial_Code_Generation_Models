#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#if defined(_WIN32)
#include "windows_detect.h"
#elif defined(__APPLE__)
#include "apple_detect.h"
#elif defined(__linux__)
#include "linux_detect.h"
#else
#include "unsupported_detect.h"
#endif

namespace Common {

struct MemoryInfo {
    std::uint64_t TotalPhysicalMemory = 0;
    std::uint64_t TotalSwapMemory = 0;
};

const MemoryInfo& GetMemInfo();

} // namespace Common

// memory_detect.cpp

namespace Common {

namespace {

const MemoryInfo& Detect() {
    static MemoryInfo mem_info = [] {
#if defined(_WIN32)
        return WindowsDetect();
#elif defined(__APPLE__)
        return AppleDetect();
#elif defined(__linux__)
        return LinuxDetect();
#else
        return UnsupportedDetect();
#endif
    }();

    return mem_info;
}

} // namespace

const MemoryInfo& GetMemInfo() {
    static const MemoryInfo& mem_info = Detect();
    return mem_info;
}

} // namespace Common

// windows_detect.h

#include <windows.h>
#include <sysinfoapi.h>

namespace Common {

MemoryInfo WindowsDetect();

} // namespace Common

// windows_detect.cpp

namespace Common {

MemoryInfo WindowsDetect() {
    MemoryInfo mem_info{};
    MEMORYSTATUSEX memorystatus;
    memorystatus.dwLength = sizeof(memorystatus);
    if (!GlobalMemoryStatusEx(&memorystatus)) {
        throw std::runtime_error{ "Error getting memory info on Windows" };
    }
    mem_info.TotalPhysicalMemory = memorystatus.ullTotalPhys;
    mem_info.TotalSwapMemory = memorystatus.ullTotalPageFile - mem_info.TotalPhysicalMemory;
    return mem_info;
}

} // namespace Common

// apple_detect.h

#include <sys/types.h>
#include <sys/sysctl.h>

namespace Common {

MemoryInfo AppleDetect();

} // namespace Common

// apple_detect.cpp

namespace Common {

MemoryInfo AppleDetect() {
    MemoryInfo mem_info{};
    u64 ramsize;
    std::size_t sizeof_ramsize = sizeof(ramsize);
    std::size_t sizeof_vmusage = sizeof(struct xsw_usage);
    struct xsw_usage vmusage;
    if (sysctlbyname("hw.memsize", &ramsize, &sizeof_ramsize, nullptr, 0) == -1) {
        throw std::runtime_error{ "Error getting memory info on Apple" };
    }
    if (sysctlbyname("vm.swapusage", &vmusage, &sizeof_vmusage, nullptr, 0) == -1) {
        throw std::runtime_error{ "Error getting swap info on Apple" };
    }
    mem_info.TotalPhysicalMemory = ramsize;
    mem_info.TotalSwapMemory = vmusage.xsu_total;
    return mem_info;
}

} // namespace Common

// linux_detect.h

#include <sys/sysinfo.h>

namespace Common {

MemoryInfo LinuxDetect();

} // namespace Common

// linux_detect.cpp

namespace Common {

MemoryInfo LinuxDetect() {
    MemoryInfo mem_info{};
    struct sysinfo meminfo;
    if (sysinfo(&meminfo) == -1) {
        throw std::runtime_error{ "Error getting memory info on Linux" };
    }
    mem_info.TotalPhysicalMemory = meminfo.totalram;
    mem_info.TotalSwapMemory = meminfo.totalswap;
    return mem_info;
}

} // namespace Common

// unsupported_detect.h

namespace Common {

MemoryInfo UnsupportedDetect(); // throws std::runtime_error

} // namespace Common

// unsupported_detect.cpp

namespace Common {

MemoryInfo UnsupportedDetect() {
    throw std::runtime_error{ "This platform is not supported." };
}

} // namespace Common

