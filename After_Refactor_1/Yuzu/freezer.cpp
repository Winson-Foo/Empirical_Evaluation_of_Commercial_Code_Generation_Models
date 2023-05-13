#include <chrono>
#include <mutex>
#include <optional>
#include <vector>
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "core/memory.h"
#include "core/tools/freezer.h"

namespace Tools {

namespace {

constexpr auto kMemoryFreezerInterval = std::chrono::nanoseconds{1000000000 / 60};

u64 ReadMemoryValue(Core::Memory::Memory& memory, u32 width, VAddr address) {
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
      UNREACHABLE();
  }
}

void WriteMemoryValue(Core::Memory::Memory& memory, u32 width, VAddr address, u64 value) {
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
      UNREACHABLE();
  }
}

}  // namespace

Freezer::Freezer(Core::Timing::CoreTiming& core_timing, Core::Memory::Memory& memory)
    : core_timing_(core_timing),
      memory_(memory),
      event_(Core::Timing::CreateEvent(
          "MemoryFreezer::FrameCallback",
          [this](std::uintptr_t user_data, s64 time, std::chrono::nanoseconds ns_late)
              -> std::optional<std::chrono::nanoseconds> {
            FrameCallback(user_data, ns_late);
            return std::nullopt;
          })) {
  core_timing_.ScheduleEvent(kMemoryFreezerInterval, event_);
}

Freezer::~Freezer() {
  core_timing_.UnscheduleEvent(event_, 0);
}

void Freezer::SetActive(bool is_active) {
  if (active_.exchange(is_active)) {
    LOG_DEBUG(Common_Memory, "Memory freezer deactivated!");
    return;
  }

  FillEntryValues();

  core_timing_.ScheduleEvent(kMemoryFreezerInterval, event_);

  LOG_DEBUG(Common_Memory, "Memory freezer activated!");
}

bool Freezer::IsActive() const noexcept {
  return active_.load(std::memory_order_relaxed);
}

void Freezer::Clear() noexcept {
  std::scoped_lock lock(entries_mutex_);
  entries_.clear();
  LOG_DEBUG(Common_Memory, "Cleared all frozen memory values.");
}

u64 Freezer::Freeze(VAddr address, u32 width) {
  std::scoped_lock lock(entries_mutex_);

  const auto value = ReadMemoryValue(memory_, width, address);
  entries_.push_back({address, width, value});

  LOG_DEBUG(Common_Memory,
            "Frozen memory for address={:016X}, width={:02X}, current_value={:016X}", address, width,
            value);

  return value;
}

void Freezer::Unfreeze(VAddr address) {
  std::scoped_lock lock(entries_mutex_);

  LOG_DEBUG(Common_Memory, "Unfroze memory for address={:016X}", address);

  std::erase_if(entries_, [address](const Entry& entry) { return entry.address == address; });
}

bool Freezer::IsFrozen(VAddr address) const noexcept {
  std::scoped_lock lock(entries_mutex_);
  return FindEntry(address) != entries_.cend();
}

void Freezer::SetFrozenValue(VAddr address, u64 value) {
  std::scoped_lock lock(entries_mutex_);

  const auto iter = FindEntry(address);

  if (iter == entries_.cend()) {
    LOG_ERROR(Common_Memory,
              "Tried to set freeze value for address={:016X} that is not frozen!", address);
    return;
  }

  LOG_DEBUG(Common_Memory, "Manually set freeze value for address={:016X}, width={:02X} to value={:016X}",
            iter->address, iter->width, value);

  iter->value = value;
}

std::optional<Freezer::Entry> Freezer::GetEntry(VAddr address) const noexcept {
  std::scoped_lock lock(entries_mutex_);

  const auto iter = FindEntry(address);

  if (iter == entries_.cend()) {
    return std::nullopt;
  }

  return *iter;
}

std::vector<Freezer::Entry> Freezer::GetEntries() const noexcept {
  std::scoped_lock lock(entries_mutex_);
  return entries_;
}

Freezer::Entries::iterator Freezer::FindEntry(VAddr address) {
  return std::find_if(entries_.begin(), entries_.end(),
                      [address](const Entry& entry) { return entry.address == address; });
}

Freezer::Entries::const_iterator Freezer::FindEntry(VAddr address) const {
  return std::find_if(entries_.cbegin(), entries_.cend(),
                      [address](const Entry& entry) { return entry.address == address; });
}

void Freezer::FrameCallback(std::uintptr_t, std::chrono::nanoseconds ns_late) noexcept {
  if (!IsActive()) {
    LOG_DEBUG(Common_Memory, "Memory freezer has been deactivated, ending callback events.");
    return;
  }

  std::scoped_lock lock(entries_mutex_);

  for (const auto& entry : entries_) {
    LOG_DEBUG(Common_Memory,
              "Enforcing memory freeze at address={:016X}, value={:016X}, width={:02X}",
              entry.address, entry.value, entry.width);
    WriteMemoryValue(memory_, entry.width, entry.address, entry.value);
  }

  core_timing_.ScheduleEvent(kMemoryFreezerInterval - ns_late, event_);
}

void Freezer::FillEntryValues() {
  std::scoped_lock lock(entries_mutex_);

  LOG_DEBUG(Common_Memory, "Updating memory freeze entries to current values.");

  for (auto& entry : entries_) {
    entry.value = ReadMemoryValue(memory_, entry.width, entry.address);
  }
}

}  // namespace Tools