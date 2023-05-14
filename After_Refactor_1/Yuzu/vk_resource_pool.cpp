// SPDX-License-Identifier: GPL-2.0-or-later

#include <optional>

#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

namespace Vulkan {

ResourcePool::ResourcePool(MasterSemaphore& masterSemaphore, size_t growStep)
    : m_masterSemaphore{&masterSemaphore}, m_growStep{growStep} {}

size_t ResourcePool::commitResource() {
    refreshSemaphore();
    const u64 gpuTick = m_masterSemaphore->knownGpuTick();
    std::optional<size_t> found = findFreeResource(gpuTick);
    if (!found) {
        found = searchFromBeginning(gpuTick);
        if (!found) {
            found = manageOverflow(gpuTick);
        }
    }
    hintIterator = (*found + 1) % m_ticks.size();
    return *found;
}

void ResourcePool::refreshSemaphore() {
    m_masterSemaphore->refresh();
}

std::optional<size_t> ResourcePool::findFreeResource(const u64 gpuTick) {
    const auto search = [this, gpuTick](size_t begin, size_t end) -> std::optional<size_t> {
        for (size_t iterator = begin; iterator < end; ++iterator) {
            if (gpuTick >= m_ticks[iterator]) {
                m_ticks[iterator] = m_masterSemaphore->currentTick();
                return iterator;
            }
        }
        return std::nullopt;
    };

    return search(hintIterator, m_ticks.size());
}

std::optional<size_t> ResourcePool::searchFromBeginning(const u64 gpuTick) {
    return findFreeResource(0, hintIterator, gpuTick);
}

std::optional<size_t> ResourcePool::findFreeResource(const size_t begin, const size_t end, const u64 gpuTick) {
    const auto search = [this, gpuTick](size_t begin, size_t end) -> std::optional<size_t> {
        for (size_t iterator = begin; iterator < end; ++iterator) {
            if (gpuTick >= m_ticks[iterator]) {
                m_ticks[iterator] = m_masterSemaphore->currentTick();
                return iterator;
            }
        }
        return std::nullopt;
    };

    return search(begin, end);
}

std::optional<size_t> ResourcePool::manageOverflow(const u64 gpuTick) {
    const size_t freeResource = grow();
    m_ticks[freeResource] = m_masterSemaphore->currentTick();
    return freeResource;
}

size_t ResourcePool::grow() {
    const size_t oldCapacity = m_ticks.size();
    m_ticks.resize(oldCapacity + m_growStep);
    allocate(oldCapacity, oldCapacity + m_growStep);
    // The last entry is guaranteed to be free, since it's the first element of the freshly
    // allocated resources.
    return oldCapacity;
}

void ResourcePool::allocate(const size_t begin, const size_t end) {
    // TODO: Implement this function.
}

} // namespace Vulkan

