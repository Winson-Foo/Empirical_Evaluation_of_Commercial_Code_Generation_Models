#pragma once

#include "k_thread_queue.h"
#include "kernel.h"

namespace Kernel {

class KLightLock {
public:
    KLightLock(KernelCore& kernel) : m_kernel(kernel), m_tag(0) {}

    void Lock();
    void Unlock();
    bool IsLockedByCurrentThread() const;

private:
    bool LockSlowPath(uintptr_t owner, uintptr_t cur_thread);
    void UnlockSlowPath(uintptr_t cur_thread);

private:
    KernelCore& m_kernel;
    std::atomic<uintptr_t> m_tag;
};

}

// k_light_lock.cpp
#include "k_light_lock.h"
#include "k_thread.h"

namespace Kernel {

namespace {

class ThreadQueueImplForKLightLock final : public KThreadQueue {
public:
    explicit ThreadQueueImplForKLightLock(KernelCore& kernel) : KThreadQueue(kernel) {}

    void CancelWait(KThread* waitingThread, Result waitResult, bool cancelTimerTask) override {
        // Remove the thread as a waiter from its owner.
        if (KThread* owner = waitingThread->GetLockOwner(); owner != nullptr) {
            owner->RemoveWaiter(waitingThread);
        }

        // Invoke the base CancelWait handler.
        KThreadQueue::CancelWait(waitingThread, waitResult, cancelTimerTask);
    }
};

}

void KLightLock::Lock() {
    const auto curThread = reinterpret_cast<uintptr_t>(GetCurrentThreadPointer(m_kernel));

    while (true) {
        auto oldTag = m_tag.load(std::memory_order_relaxed);

        while (!m_tag.compare_exchange_weak(oldTag, (oldTag == 0) ? curThread : (oldTag | 1),
                                            std::memory_order_acquire)) {
        }

        if (oldTag == 0 || LockSlowPath(oldTag | 1, curThread)) {
            break;
        }
    }
}

void KLightLock::Unlock() {
    const auto curThread = reinterpret_cast<uintptr_t>(GetCurrentThreadPointer(m_kernel));

    auto expected = curThread;
    if (!m_tag.compare_exchange_strong(expected, 0, std::memory_order_release)) {
        UnlockSlowPath(curThread);
    }
}

bool KLightLock::LockSlowPath(uintptr_t owner, uintptr_t curThread) {
    auto currentThread = reinterpret_cast<KThread*>(curThread);
    ThreadQueueImplForKLightLock waitQueue(m_kernel);

    // Pend the current thread waiting on the owner thread.
    {
        KScopedSchedulerLock lock{m_kernel};

        // Ensure we actually have locking to do.
        if (m_tag.load(std::memory_order_relaxed) != owner) {
            return false;
        }

        // Add the current thread as a waiter on the owner.
        auto ownerThread = reinterpret_cast<KThread*>(owner & ~1ULL);
        currentThread->SetKernelAddressKey(reinterpret_cast<uintptr_t>(std::addressof(m_tag)));
        ownerThread->AddWaiter(currentThread);

        // Begin waiting to hold the lock.
        currentThread->BeginWait(std::addressof(waitQueue));

        if (ownerThread->IsSuspended()) {
            ownerThread->ContinueIfHasKernelWaiters();
        }
    }

    return true;
}

void KLightLock::UnlockSlowPath(uintptr_t curThread) {
    auto ownerThread = reinterpret_cast<KThread*>(curThread);

    // Unlock.
    {
        KScopedSchedulerLock lock(m_kernel);

        // Get the next owner.
        bool hasWaiters;
        auto nextOwner = ownerThread->RemoveKernelWaiterByKey(
            std::addressof(hasWaiters), reinterpret_cast<uintptr_t>(std::addressof(m_tag)));

        // Pass the lock to the next owner.
        auto nextTag = 0;
        if (nextOwner != nullptr) {
            nextTag = reinterpret_cast<uintptr_t>(nextOwner) | static_cast<uintptr_t>(hasWaiters);

            nextOwner->EndWait(ResultSuccess);

            if (nextOwner->IsSuspended()) {
                nextOwner->ContinueIfHasKernelWaiters();
            }
        }

        // We may have unsuspended in the process of acquiring the lock, so we'll re-suspend now if so.
        if (ownerThread->IsSuspended()) {
            ownerThread->TrySuspend();
        }

        // Write the new tag value.
        m_tag.store(nextTag, std::memory_order_release);
    }
}

bool KLightLock::IsLockedByCurrentThread() const {
    const auto curThread = reinterpret_cast<uintptr_t>(GetCurrentThreadPointer(m_kernel));
    return (m_tag.load() | 1ULL) == (curThread | 1ULL);
}

} // namespace Kernel