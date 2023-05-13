#pragma once

namespace Vulkan {

class Scheduler;

class InnerFence final {
public:
    explicit InnerFence(Scheduler& scheduler, bool isStubbed);
    ~InnerFence();

    void queue();
    bool isSignaled() const;
    void wait();

private:
    Scheduler& mScheduler;
    bool mIsStubbed;
    uint64_t mWaitTick;
};

} // namespace Vulkan

// InnerFence.cpp

#include "Scheduler.h"

namespace Vulkan {

InnerFence::InnerFence(Scheduler& scheduler, bool isStubbed)
    : mScheduler{scheduler}, mIsStubbed{isStubbed} {}

InnerFence::~InnerFence() = default;

void InnerFence::queue() {
    if (mIsStubbed) {
        return;
    }
    mWaitTick = mScheduler.currentTick();
    mScheduler.flush();
}

bool InnerFence::isSignaled() const {
    if (mIsStubbed) {
        return true;
    }
    return mScheduler.isFree(mWaitTick);
}

void InnerFence::wait() {
    if (mIsStubbed) {
        return;
    }
    mScheduler.wait(mWaitTick);
}

} // namespace Vulkan

// FenceManager.h

#pragma once

#include <memory>

#include "Fence.h"

namespace VideoCore {
class RasterizerInterface;
}

namespace Tegra {
class GPU;
}

namespace Vulkan {
class BufferCache;
class Device;
class QueryCache;
class Scheduler;
class TextureCache;

class FenceManager final {
public:
    explicit FenceManager(VideoCore::RasterizerInterface& rasterizer,
                          Tegra::GPU& gpu,
                          TextureCache& textureCache,
                          BufferCache& bufferCache,
                          QueryCache& queryCache,
                          const Device& device,
                          Scheduler& scheduler);
    ~FenceManager() = default;

    std::shared_ptr<Fence> createFence(bool isStubbed = false);
    void queueFence(const std::shared_ptr<Fence>& fence);
    bool isFenceSignaled(const std::shared_ptr<Fence>& fence) const;
    void waitFence(const std::shared_ptr<Fence>& fence);

private:
    Scheduler& mScheduler;
    VideoCore::RasterizerInterface& mRasterizer;
    Tegra::GPU& mGpu;
    TextureCache& mTextureCache;
    BufferCache& mBufferCache; 
    QueryCache& mQueryCache;
};

} // namespace Vulkan

// FenceManager.cpp

#include "FenceManager.h"
#include "InnerFence.h"
#include "Scheduler.h"
#include "VulkanDevice.h"

namespace Vulkan {

FenceManager::FenceManager(VideoCore::RasterizerInterface& rasterizer,
                           Tegra::GPU& gpu,
                           TextureCache& textureCache,
                           BufferCache& bufferCache,
                           QueryCache& queryCache,
                           const Device& device,
                           Scheduler& scheduler)
    : mRasterizer{rasterizer}, mGpu{gpu}, mTextureCache{textureCache},
      mBufferCache{bufferCache}, mQueryCache{queryCache}, mScheduler{scheduler} {}

std::shared_ptr<Fence> FenceManager::createFence(bool isStubbed) {
    return std::make_shared<InnerFence>(mScheduler, isStubbed);
}

void FenceManager::queueFence(const std::shared_ptr<Fence>& fence) {
    fence->queue();
}

bool FenceManager::isFenceSignaled(const std::shared_ptr<Fence>& fence) const {
    return fence->isSignaled();
}

void FenceManager::waitFence(const std::shared_ptr<Fence>& fence) {
    fence->wait();
}

} // namespace Vulkan

// Fence.h

#pragma once

#include <memory>

namespace Vulkan {

class InnerFence;

using Fence = std::shared_ptr<InnerFence>;

} // namespace Vulkan

// Scheduler.h

#pragma once

#include <cstdint>

namespace Vulkan {

class Scheduler final {
public:
    uint64_t currentTick() const;
    bool isFree(uint64_t tick) const;
    void wait(uint64_t tick);
    void flush();
};

} // namespace Vulkan

// Scheduler.cpp

#include "Scheduler.h"

namespace Vulkan {

uint64_t Scheduler::currentTick() const {
    // return current tick
    return 0;
}

bool Scheduler::isFree(uint64_t tick) const {
    // return true if the tick has passed
    return true;
}

void Scheduler::wait(uint64_t tick) {
    // wait until tick has passed
}

void Scheduler::flush() {
    // flush all tasks
}

} // namespace Vulkan

// BufferCache.h

#pragma once

namespace Vulkan {

class BufferCache final {};

} // namespace Vulkan

// QueryCache.h

#pragma once

namespace Vulkan {

class QueryCache final {};

} // namespace Vulkan

// TextureCache.h

#pragma once

namespace Vulkan {

class TextureCache final {};

} // namespace Vulkan

// VulkanDevice.h

#pragma once

namespace Vulkan {

class Device final {};

} // namespace Vulkan

// RasterizerInterface.h

#pragma once

namespace VideoCore {

class RasterizerInterface final {};

} // namespace VideoCore

// GPU.h

#pragma once

namespace Tegra {

class GPU final {};

} // namespace Tegra

