// Fence.hpp
namespace Vulkan {

class Fence {
public:
    virtual ~Fence() = default;

    virtual void Queue() = 0;
    virtual bool IsSignaled() const = 0;
    virtual void Wait() = 0;
};

}

// StubbedFence.hpp
namespace Vulkan {

class StubbedFence : public Fence {
public:
    explicit StubbedFence(bool isStubbed);
    ~StubbedFence() override = default;

    void Queue() override;
    bool IsSignaled() const override;
    void Wait() override;

private:
    bool isStubbed_;
};

}

// SchedulerFence.hpp
namespace Vulkan {

class SchedulerFence : public Fence {
public:
    explicit SchedulerFence(Scheduler& scheduler);
    ~SchedulerFence() override = default;

    void Queue() override;
    bool IsSignaled() const override;
    void Wait() override;

private:
    Scheduler& scheduler_;
    Tick waitTick_;
};

}

// FenceManager.hpp
namespace Vulkan {

class FenceManager : public GenericFenceManager {
public:
    FenceManager(VideoCore::RasterizerInterface& rasterizer, Tegra::GPU& gpu,
        TextureCache& textureCache, BufferCache& bufferCache, QueryCache& queryCache, const Device& device, Scheduler& scheduler);
    ~FenceManager() override = default;

    std::shared_ptr<Fence> CreateFence(bool isStubbed);
    void QueueFence(std::shared_ptr<Fence> fence);
    bool IsFenceSignaled(const std::shared_ptr<Fence>& fence) const;
    void WaitFence(std::shared_ptr<Fence> fence);

private:
    Scheduler& scheduler_;
};

}

// Fence.cpp
namespace Vulkan {

StubbedFence::StubbedFence(bool isStubbed) : isStubbed_(isStubbed) {}

void StubbedFence::Queue() {}

bool StubbedFence::IsSignaled() const {
    return true;
}

void StubbedFence::Wait() {}

SchedulerFence::SchedulerFence(Scheduler& scheduler) : scheduler_(scheduler) {}

void SchedulerFence::Queue() {
    waitTick_ = scheduler_.CurrentTick();
    scheduler_.Flush();
}

bool SchedulerFence::IsSignaled() const {
    return scheduler_.IsFree(waitTick_);
}

void SchedulerFence::Wait() {
    scheduler_.Wait(waitTick_);
}

FenceManager::FenceManager(VideoCore::RasterizerInterface& rasterizer, Tegra::GPU& gpu,
        TextureCache& textureCache, BufferCache& bufferCache, QueryCache& queryCache, const Device& device, Scheduler& scheduler)
    : GenericFenceManager(rasterizer, gpu, textureCache, bufferCache, queryCache), scheduler_(scheduler) {}

std::shared_ptr<Fence> FenceManager::CreateFence(bool isStubbed) {
    if (isStubbed) {
        return std::make_shared<StubbedFence>(isStubbed);
    } else {
        return std::make_shared<SchedulerFence>(scheduler_);
    }
}

void FenceManager::QueueFence(std::shared_ptr<Fence> fence) {
    fence->Queue();
}

bool FenceManager::IsFenceSignaled(const std::shared_ptr<Fence>& fence) const {
    return fence->IsSignaled();
}

void FenceManager::WaitFence(std::shared_ptr<Fence> fence) {
    fence->Wait();
}

} // namespace Vulkan

