#pragma once

namespace Vulkan {
    class InnerFence;
    class FenceManager {
    public:
        FenceManager(
            VideoCore::RasterizerInterface& rasterizer,
            Tegra::GPU& gpu,
            TextureCache& texture_cache,
            BufferCache& buffer_cache,
            QueryCache& query_cache,
            const Device& device,
            Scheduler& scheduler
        );
        InnerFence CreateFence(bool is_stubbed);
        void QueueFence(InnerFence& fence);
        bool IsFenceSignaled(const InnerFence& fence) const;
        void WaitFence(InnerFence& fence);
    private:
        Scheduler& scheduler;
    };
}

#include "vk_fence_manager.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

    class InnerFence {
    public:
        InnerFence(Scheduler& scheduler, bool is_stubbed) :
            is_stubbed(is_stubbed),
            scheduler(scheduler){}

        void Queue() {
            if (is_stubbed) return;
            wait_tick = scheduler.CurrentTick();
            scheduler.Flush();
        }

        bool IsSignaled() const {
            return is_stubbed || scheduler.IsFree(wait_tick);
        }

        void Wait() {
            if (is_stubbed) return;
            scheduler.Wait(wait_tick);
        }

    private:
        bool is_stubbed;
        u32 wait_tick;
        Scheduler& scheduler;
    };

    FenceManager::FenceManager(
        VideoCore::RasterizerInterface& rasterizer,
        Tegra::GPU& gpu,
        TextureCache& texture_cache,
        BufferCache& buffer_cache,
        QueryCache& query_cache,
        const Device& device,
        Scheduler& scheduler
    ) : scheduler(scheduler), 
        GenericFenceManager(rasterizer, gpu, texture_cache, buffer_cache, query_cache) {}

    InnerFence FenceManager::CreateFence(bool is_stubbed) {
        return InnerFence(scheduler, is_stubbed);
    }

    void FenceManager::QueueFence(InnerFence& fence) {
        fence.Queue();
    }

    bool FenceManager::IsFenceSignaled(const InnerFence& fence) const {
        return fence.IsSignaled();
    }

    void FenceManager::WaitFence(InnerFence& fence) {
        fence.Wait();
    }

} // namespace Vulkan

