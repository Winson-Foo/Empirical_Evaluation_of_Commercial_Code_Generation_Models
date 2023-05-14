#include "common/memory_detect.h"

#ifdef _WIN32
#include "common/windows/memory_info.h"
#elif defined(__APPLE__)
#include "common/apple/memory_info.h"
#elif defined(__FreeBSD__)
#include "common/freebsd/memory_info.h"
#elif defined(__linux__)
#include "common/linux/memory_info.h"
#else
#include "common/generic/memory_info.h"
#endif

namespace Common {

const MemoryInfo& GetMemInfo() {
    static MemoryInfo mem_info = platform_specific::get_mem_info();
    return mem_info;
}

} // namespace Common

// platform_specific files:

// windows/memory_info.h

#pragma once

#include <windows.h>
#include <sysinfoapi.h>
#include "../../memory_info.h"

namespace Common::platform_specific {

inline MemoryInfo get_mem_info() {
    MemoryInfo mem_info{};
    MEMORYSTATUSEX memorystatus;
    memorystatus.dwLength = sizeof(memorystatus);
    GlobalMemoryStatusEx(&memorystatus);
    mem_info.TotalPhysicalMemory = memorystatus.ullTotalPhys;
    mem_info.TotalSwapMemory = memorystatus.ullTotalPageFile - mem_info.TotalPhysicalMemory;
    return mem_info;
}

} // namespace Common::platform_specific

// apple/memory_info.h

#pragma once

#include <sys/types.h>
#include <sys/sysctl.h>
#include "../../memory_info.h"

namespace Common::platform_specific {

inline MemoryInfo get_mem_info() {
    MemoryInfo mem_info{};
    u64 ramsize;
    struct xsw_usage vmusage;
    std::size_t sizeof_ramsize = sizeof(ramsize);
    std::size_t sizeof_vmusage = sizeof(vmusage);
    sysctlbyname("hw.memsize", &ramsize, &sizeof_ramsize, nullptr, 0);
    sysctlbyname("vm.swapusage", &vmusage, &sizeof_vmusage, nullptr, 0);
    mem_info.TotalPhysicalMemory = ramsize;
    mem_info.TotalSwapMemory = vmusage.xsu_total;
    return mem_info;
}

} // namespace Common::platform_specific

// freebsd/memory_info.h

#pragma once

#include <sys/types.h>
#include <sys/sysctl.h>
#include "../../memory_info.h"

namespace Common::platform_specific {

inline MemoryInfo get_mem_info() {
    MemoryInfo mem_info{};
    u_long physmem, swap_total;
    std::size_t sizeof_u_long = sizeof(u_long);
    sysctlbyname("hw.physmem", &physmem, &sizeof_u_long, nullptr, 0);
    sysctlbyname("vm.swap_total", &swap_total, &sizeof_u_long, nullptr, 0);
    mem_info.TotalPhysicalMemory = physmem;
    mem_info.TotalSwapMemory = swap_total;
    return mem_info;
}

} // namespace Common::platform_specific

// linux/memory_info.h

#pragma once

#include <sys/sysinfo.h>
#include "../../memory_info.h"

namespace Common::platform_specific {

inline MemoryInfo get_mem_info() {
    MemoryInfo mem_info{};
    struct sysinfo meminfo;
    sysinfo(&meminfo);
    mem_info.TotalPhysicalMemory = meminfo.totalram;
    mem_info.TotalSwapMemory = meminfo.totalswap;
    return mem_info;
}

} // namespace Common::platform_specific

// generic/memory_info.h

#pragma once

#include <unistd.h>
#include "../../memory_info.h"

namespace Common::platform_specific {

inline MemoryInfo get_mem_info() {
    MemoryInfo mem_info{};
    mem_info.TotalPhysicalMemory = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
    mem_info.TotalSwapMemory = 0;
    return mem_info;
}

} // namespace Common::platform_specific

// memory_info.h

#pragma once

#include <cstdint>

namespace Common {

struct MemoryInfo {
    std::uint64_t TotalPhysicalMemory;
    std::uint64_t TotalSwapMemory;
};

const MemoryInfo& GetMemInfo();

} // namespace Common

