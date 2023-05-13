#include <optional>

#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

namespace Vulkan {

ResourcePool::ResourcePool(const MasterSemaphore& master_semaphore, const size_t grow_step)
    : master_semaphore{ master_semaphore },
      grow_step{ grow_step },
      ticks{} {
}

size_t ResourcePool::CommitResource() {
  const u64 gpu_tick = GetGpuTick();

  // Try to find a free resource from the hinted position to the end.
  std::optional<size_t> found = FindFreeResource(next_free_index, ticks.size(), gpu_tick);
  if (!found) {
    // Search from beginning to the hinted position.
    found = FindFreeResource(0, next_free_index, gpu_tick);
    if (!found) {
      // Both searches failed, the pool is full; handle it.
      const size_t free_resource = ManageOverflow();
      ticks[free_resource] = GetGpuTick();
      found = free_resource;
    }
  }
  // Free index is hinted to the resource after the one that's been committed.
  next_free_index = (*found + 1) % ticks.size();
  return *found;
}

size_t ResourcePool::ManageOverflow() {
  const size_t old_capacity = ticks.size();
  Grow();
  // The last entry is guaranteed to be free, since it's the first element of the freshly
  // allocated resources.
  return old_capacity;
}

void ResourcePool::Grow() {
  const size_t old_capacity = ticks.size();
  const size_t new_capacity = old_capacity + grow_step;
  ticks.reserve(new_capacity);
  Allocate(old_capacity, new_capacity);
  ticks.emplace_back(GetGpuTick());
}

void ResourcePool::Allocate(const size_t begin, const size_t end) {
  ticks.resize(end);
  std::fill(ticks.begin() + begin, ticks.begin() + end, 0);
}

std::optional<size_t> ResourcePool::FindFreeResource(const size_t begin, const size_t end, const u64 gpu_tick) {
  const auto search = [this, gpu_tick](const size_t index) -> std::optional<size_t> {
    if (gpu_tick >= ticks[index]) {
      ticks[index] = GetGpuTick();
      return index;
    }
    return std::nullopt;
  };
  if (begin < end) {
    const auto found = std::find_if(ticks.begin() + begin, ticks.begin() + end, search);
    if (found != ticks.begin() + end) {
      return std::distance(ticks.begin(), found);
    }
  } else {
    const auto found = std::find_if(ticks.begin() + begin, ticks.end(), search);
    if (found != ticks.end()) {
      return std::distance(ticks.begin(), found);
    }
    const auto found2 = std::find_if(ticks.begin(), ticks.begin() + end, search);
    if (found2 != ticks.begin() + end) {
      return std::distance(ticks.begin(), found2);
    }
  }
  return std::nullopt;
}

size_t ResourcePool::FreeIteratorAfter(const size_t index) const {
  return (index + 1) % ticks.size();
}

u64 ResourcePool::GetGpuTick() const {
  master_semaphore.Refresh();
  return master_semaphore.KnownGpuTick();
}

} // namespace Vulkan