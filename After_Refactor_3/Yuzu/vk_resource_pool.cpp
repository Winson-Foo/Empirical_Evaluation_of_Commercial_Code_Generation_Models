namespace Vulkan {

ResourcePool::ResourcePool(const MasterSemaphore& master_semaphore, const size_t grow_step)
    : master_semaphore_{&master_semaphore}, grow_step_{grow_step} {}

void ResourcePool::Grow() {
    const size_t old_capacity{ticks_.size()};
    ticks_.resize(old_capacity + grow_step_);
    Allocate(old_capacity, old_capacity + grow_step_);
}

std::optional<size_t> ResourcePool::FindFreeResource(const u64 gpu_tick) {
    // Search for a free resource from the hinted position to the end.
    auto search = [this, gpu_tick](const size_t begin, const size_t end) -> std::optional<size_t> {
        for (size_t i{begin}; i < end; ++i) {
            if (gpu_tick >= ticks_[i]) {
                ticks_[i] = master_semaphore_->CurrentTick();
                return i;
            }
        }
        return std::nullopt;
    };
    // Try to find a free resource from the hinted position to the end.
    if (const auto found{search(hint_iterator_, ticks_.size())}) {
        return found;
    }
    // Search from beginning to the hinted position.
    if (const auto found{search(0, hint_iterator_)}) {
        return found;
    }
    // Both searches failed, the pool is full; handle it.
    const size_t free_resource{ManageOverflow()};
    ticks_[free_resource] = master_semaphore_->CurrentTick();
    return free_resource;
}

size_t ResourcePool::CommitResource() {
    // Refresh semaphore to query updated results.
    master_semaphore_->Refresh();
    const u64 gpu_tick{master_semaphore_->KnownGpuTick()};
    // Find a free resource from the pool.
    const auto found{FindFreeResource(gpu_tick)};
    // If no free resource was found, return -1 to indicate an error.
    if (!found) {
        return -1;
    }
    // Free iterator is hinted to the resource after the one that's been committed.
    hint_iterator_ = (*found + 1) % ticks_.size();
    return *found;
}

size_t ResourcePool::ManageOverflow() {
    const size_t old_capacity{ticks_.size()};
    Grow();
    // The last entry is guaranteed to be free, since it's the first element of the freshly
    // allocated resources.
    return old_capacity;
}

} // namespace Vulkan