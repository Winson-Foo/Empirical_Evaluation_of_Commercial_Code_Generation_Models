// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/memory.h"
#include "core/tools/freezer.h"

namespace yuzu {
namespace tools {
namespace freezer {

using namespace std::chrono_literals;

namespace {

constexpr auto kMemoryFreezeInterval = 16ms;

// Returns the value of the memory read with the specified data width.
u64 ReadMemory(Core::Memory::Memory const& memory, u32 width, VAddr address) {
    switch (width) {
    case 1:
        return memory.Read8(address);
    case 2:
        return memory.Read16(address);
    case 4:
        return memory.Read32(address);
    case 8:
        return memory.Read64(address);
    default:
        assert(false && "Invalid memory read width");
        return 0;
    }
}

// Writes the value to the memory with the specified data width.
void WriteMemory(Core::Memory::Memory& memory, u32 width, VAddr address, u64 value) {
    switch (width) {
    case 1:
        memory.Write8(address, static_cast<u8>(value));
        break;
    case 2:
        memory.Write16(address, static_cast<u16>(value));
        break;
    case 4:
        memory.Write32(address, static_cast<u32>(value));
        break;
    case 8:
        memory.Write64(address, value);
        break;
    default:
        assert(false && "Invalid memory write width");
    }
}

} // namespace

Freezer::Freezer(Core::Timing::CoreTiming& core_timing, Core::Memory::Memory& memory)
    : core_timing_{core_timing}, memory_{memory} {
    event_ = Core::Timing::CreateEvent("Freezer::FrameCallback",
        [this](std::uintptr_t user_data, s64 time, std::chrono::nanoseconds late_ns) -> std::optional<std::chrono::nanoseconds> {
            FrameCallback(user_data, late_ns);
            return std::nullopt;
        });
    core_timing_.ScheduleEvent(kMemoryFreezeInterval, event_);
}

Freezer::~Freezer() {
    core_timing_.UnscheduleEvent(event_, 0);
}

void Freezer::Activate(bool is_active) {
    if (!active_.exchange(is_active)) {
        FillEntriesReads();
        core_timing_.ScheduleEvent(kMemoryFreezeInterval, event_);
        LOG_DEBUG("Freezer", "Activated");
    } else {
        LOG_DEBUG("Freezer", "Deactivated");
    }
}

bool Freezer::IsActive() const {
    return active_.load(std::memory_order_relaxed);
}

void Freezer::ClearEntries() {
    std::scoped_lock lock{entries_mutex_};

    LOG_DEBUG("Freezer", "Clearing entries");

    entries_.clear();
}

u64 Freezer::Freeze(VAddr address, u32 width) {
    std::scoped_lock lock{entries_mutex_};

    const auto value = ReadMemory(memory_, width, address);
    entries_.push_back({address, width, value});

    LOG_DEBUG("Freezer", "Freezing memory: address={:016X}, width={:02X}, value={:016X}", address, width, value);

    return value;
}

void Freezer::Unfreeze(VAddr address) {
    std::scoped_lock lock{entries_mutex_};

    LOG_DEBUG("Freezer", "Unfreezing memory: address={:016X}", address);

    const auto iter = FindEntry(address);
    if (iter != entries_.end()) {
        entries_.erase(iter);
    }
}

bool Freezer::IsFrozen(VAddr address) const {
    std::scoped_lock lock{entries_mutex_};

    return FindEntry(address) != entries_.end();
}

void Freezer::SetFrozenValue(VAddr address, u64 value) {
    std::scoped_lock lock{entries_mutex_};

    const auto iter = FindEntry(address);
    if (iter == entries_.end()) {
        LOG_ERROR("Freezer", "Tried to set freeze value for unfrozen address: address={:016X}", address);
        return;
    }

    LOG_DEBUG("Freezer", "Overridden freeze value: address={:016X}, width={:02X}, value={:016X}", address, iter->width, value);
    iter->value = value;
}

std::optional<Freezer::Entry> Freezer::GetEntry(VAddr address) const {
    std::scoped_lock lock{entries_mutex_};

    const auto iter = FindEntry(address);
    if (iter == entries_.end()) {
        return std::nullopt;
    }

    return *iter;
}

std::vector<Freezer::Entry> Freezer::GetEntries() const {
    std::scoped_lock lock{entries_mutex_};

    return entries_;
}

Freezer::Entries::iterator Freezer::FindEntry(VAddr address) {
    return std::find_if(entries_.begin(), entries_.end(),
        [address](const Entry& entry) { return entry.address == address; });
}

Freezer::Entries::const_iterator Freezer::FindEntry(VAddr address) const {
    return std::find_if(entries_.begin(), entries_.end(),
        [address](const Entry& entry) { return entry.address == address; });
}

void Freezer::FrameCallback(std::uintptr_t, std::chrono::nanoseconds late_ns) {
    if (!IsActive()) {
        LOG_DEBUG("Freezer", "Deactivated, skipping frame callback");
        return;
    }

    std::scoped_lock lock{entries_mutex_};

    for (const auto& entry : entries_) {
        LOG_DEBUG("Freezer", "Enforcing freeze: address={:016X}, width={:02X}, value={:016X}", entry.address, entry.width, entry.value);
        WriteMemory(memory_, entry.width, entry.address, entry.value);
    }

    core_timing_.ScheduleEvent(kMemoryFreezeInterval - late_ns, event_);
}

void Freezer::FillEntriesReads() {
    std::scoped_lock lock{entries_mutex_};

    LOG_DEBUG("Freezer", "Filling memory freeze entries");

    for (auto& entry : entries_) {
        entry.value = ReadMemory(memory_, entry.width, entry.address);
    }
}

} // namespace freezer
} // namespace tools
} // namespace yuzu

