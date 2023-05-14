// common/memory_detect.h

#pragma once

#include <cstdint>

namespace Common {

struct MemoryInfo {
    std::uint64_t TotalPhysicalMemory = 0;
    std::uint64_t TotalSwapMemory = 0;
};

const MemoryInfo& GetMemoryInfo();

} // namespace Common


// common/memory_detect.cpp

#ifdef _WIN32
#include <windows.h>
#include <sysinfoapi.h>
#else
#include <sys/types.h>
#if defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#else
#include <unistd.h>
#endif
#endif

#include "common/memory_detect.h"

namespace Common {

static MemoryInfo DetectMemory() {
    MemoryInfo mem_info{0};

#ifdef _WIN32
    MEMORYSTATUSEX memory_status;
    memory_status.dwLength = sizeof(memory_status);
    GlobalMemoryStatusEx(&memory_status);
    mem_info.TotalPhysicalMemory = memory_status.ullTotalPhys;
    mem_info.TotalSwapMemory = memory_status.ullTotalPageFile - mem_info.TotalPhysicalMemory;
#elif defined(__APPLE__)
    u64 ram_size = 0;
    struct xsw_usage vm_usage{0};
    std::size_t ram_size_size = sizeof(ram_size);
    std::size_t vm_usage_size = sizeof(vm_usage);
    sysctlbyname("hw.memsize", &ram_size, &ram_size_size, nullptr, 0);
    sysctlbyname("vm.swapusage", &vm_usage, &vm_usage_size, nullptr, 0);
    mem_info.TotalPhysicalMemory = ram_size;
    mem_info.TotalSwapMemory = vm_usage.xsu_total;
#elif defined(__FreeBSD__)
    u_long phys_mem = 0;
    u_long swap_total = 0;
    std::size_t phys_mem_size = sizeof(phys_mem);
    std::size_t swap_total_size = sizeof(swap_total);
    sysctlbyname("hw.physmem", &phys_mem, &phys_mem_size, nullptr, 0);
    sysctlbyname("vm.swap_total", &swap_total, &swap_total_size, nullptr, 0);
    mem_info.TotalPhysicalMemory = phys_mem;
    mem_info.TotalSwapMemory = swap_total;
#elif defined(__linux__)
    struct sysinfo mem_info_struct{0};
    sysinfo(&mem_info_struct);
    mem_info.TotalPhysicalMemory = mem_info_struct.totalram;
    mem_info.TotalSwapMemory = mem_info_struct.totalswap;
#else
    mem_info.TotalPhysicalMemory = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
    mem_info.TotalSwapMemory = 0;
#endif

    return mem_info;
}

const MemoryInfo& GetMemoryInfo() {
    static const MemoryInfo mem_info = DetectMemory();
    return mem_info;
}

} // namespace Common

