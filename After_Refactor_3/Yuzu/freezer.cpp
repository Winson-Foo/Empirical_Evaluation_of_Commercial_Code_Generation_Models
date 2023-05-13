#include <chrono>
#include <vector>
#include <algorithm>
#include <atomic>
#include <optional>

#include "assert.h"
#include "logging/log.h"
#include "core.h"
#include "core_timing.h"
#include "memory.h"

namespace Tools::MemoryFreezer {

    namespace {
        constexpr auto kMemoryFreezeIntervalNs = std::chrono::nanoseconds{1000000000 / 60};
    }

    struct FreezeEntry {
        Core::Memory::VAddr address;
        u32 width;
        u64 value;
    };

    class Freezer {
    public:
        explicit Freezer(Core::Timing::CoreTiming& core_timing, Core::Memory::Memory& memory)
            : m_coreTiming{ core_timing }, m_memory{ memory }
        {
            m_callbackEvent = Core::Timing::CreateEvent("MemoryFreezer::FrameCallback",
                [this](std::uintptr_t user_data, s64 time, std::chrono::nanoseconds ns_late) -> std::optional<std::chrono::nanoseconds> {
                    FrameCallback(user_data, ns_late);
                    return std::nullopt;
                });

            m_coreTiming.ScheduleEvent(kMemoryFreezeIntervalNs, m_callbackEvent);
        }

        ~Freezer()
        {
            m_coreTiming.UnscheduleEvent(m_callbackEvent, 0);
        }

        void SetActive(bool active)
        {
            if (m_active.exchange(active)) {
                m_coreTiming.UnscheduleEvent(m_callbackEvent, 0);
                LOG_DEBUG(kMemoryFreezer, "Memory freezer deactivated!");
            } else {
                m_entriesMutex.lock();
                FillEntries();
                m_coreTiming.ScheduleEvent(kMemoryFreezeIntervalNs, m_callbackEvent);
                m_entriesMutex.unlock();
                LOG_DEBUG(kMemoryFreezer, "Memory freezer activated!");
            }
        }

        bool IsActive() const {
            return m_active.load(std::memory_order_relaxed);
        }

        void Clear() {
            std::lock_guard lock{ m_entriesMutex };
            LOG_DEBUG(kMemoryFreezer, "Clearing all frozen memory values.");
            m_entries.clear();
        }

        u64 Freeze(Core::Memory::VAddr address, u32 width) {
            std::lock_guard lock{ m_entriesMutex };

            const auto current_value = ReadMemoryWidth(width, address);
            m_entries.push_back({ address, width, current_value });

            LOG_DEBUG(kMemoryFreezer,
                "Freezing memory for address={:016X}, width={:02X}, current_value={:016X}", address,
                width, current_value);

            return current_value;
        }

        void Unfreeze(Core::Memory::VAddr address) {
            std::lock_guard lock{ m_entriesMutex };

            LOG_DEBUG(kMemoryFreezer, "Unfreezing memory for address={:016X}", address);

            std::erase_if(m_entries, [address](const FreezeEntry& entry) { return entry.address == address; });
        }

        bool IsFrozen(Core::Memory::VAddr address) const {
            std::lock_guard lock{ m_entriesMutex };
            return FindEntry(address) != m_entries.cend();
        }

        void SetFrozenValue(Core::Memory::VAddr address, u64 value) {
            std::lock_guard lock{ m_entriesMutex };

            const auto iter = FindEntry(address);
            if (iter == m_entries.cend()) {
                LOG_ERROR(kMemoryFreezer,
                    "Tried to set freeze value for address={:016X} that is not frozen!", address);
                return;
            }
            iter->value = value;
            LOG_DEBUG(kMemoryFreezer,
                "Manually overridden freeze value for address={:016X}, width={:02X} to value={:016X}",
                iter->address, iter->width, value);
        }

        std::optional<FreezeEntry> GetEntry(Core::Memory::VAddr address) const {
            std::lock_guard lock{ m_entriesMutex };

            const auto iter = FindEntry(address);
            if (iter == m_entries.cend()) {
                return std::nullopt;
            }
            return *iter;
        }

        std::vector<FreezeEntry> GetEntries() const {
            std::lock_guard lock{ m_entriesMutex };
            return m_entries;
        }

    private:
        void FrameCallback(std::uintptr_t, std::chrono::nanoseconds ns_late) {
            if (!IsActive()) {
                LOG_DEBUG(kMemoryFreezer, "Memory freezer has been deactivated, ending callback events.");
                return;
            }

            std::lock_guard lock{ m_entriesMutex };

            for (auto& entry : m_entries) {
                LOG_DEBUG(kMemoryFreezer,
                    "Enforcing memory freeze at address={:016X}, value={:016X}, width={:02X}",
                    entry.address, entry.value, entry.width);
                WriteMemoryWidth(entry.width, entry.address, entry.value);
            }

            m_coreTiming.ScheduleEvent(kMemoryFreezeIntervalNs - ns_late, m_callbackEvent);
        }

        void FillEntries() {
            for (auto& entry : m_entries) {
                entry.value = ReadMemoryWidth(entry.width, entry.address);
            }
        }

        std::vector<FreezeEntry>::iterator FindEntry(Core::Memory::VAddr address) {
            return std::find_if(m_entries.begin(), m_entries.end(),
                [address](const FreezeEntry& entry) { return entry.address == address; });
        }

        std::vector<FreezeEntry>::const_iterator FindEntry(Core::Memory::VAddr address) const {
            return std::find_if(m_entries.cbegin(), m_entries.cend(),
                [address](const FreezeEntry& entry) { return entry.address == address; });
        }

        u64 ReadMemoryWidth(u32 width, Core::Memory::VAddr address) {
            switch (width) {
            case 1:
                return m_memory.Read8(address);
            case 2:
                return m_memory.Read16(address);
            case 4:
                return m_memory.Read32(address);
            case 8:
                return m_memory.Read64(address);
            default:
                UNREACHABLE();
            }
        }

        void WriteMemoryWidth(u32 width, Core::Memory::VAddr address, u64 value) {
            switch (width) {
            case 1:
                m_memory.Write8(address, static_cast<u8>(value));
                break;
            case 2:
                m_memory.Write16(address, static_cast<u16>(value));
                break;
            case 4:
                m_memory.Write32(address, static_cast<u32>(value));
                break;
            case 8:
                m_memory.Write64(address, value);
                break;
            default:
                UNREACHABLE();
            }
        }

        std::atomic_bool m_active { false };
        std::vector<FreezeEntry> m_entries;
        Core::Timing::CoreTiming& m_coreTiming;
        Core::Memory::Memory& m_memory;
        Core::Timing::EventHandle m_callbackEvent;
        std::mutex m_entriesMutex;
    };

} // namespace Tools::MemoryFreezer