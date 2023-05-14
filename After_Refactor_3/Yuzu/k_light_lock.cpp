// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright 2021 yuzu Emulator Project

#include "core/hle/kernel/k_light_lock.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/k_thread_queue.h"
#include "core/hle/kernel/kernel.h"

namespace Kernel {

namespace {

// Implements a thread queue for the k light lock.
class ThreadQueueImplForKLightLock final : public KThreadQueue {
public:
    explicit ThreadQueueImplForKLightLock(KernelCore& kernel) : KThreadQueue(kernel) {}

    // Cancels a wait and removes the thread as a waiter from its owner.
    void CancelWait(KThread* waiting_thread, Result wait_result, bool cancel_timer_task) override {
        if (KThread* owner = waiting_thread->GetLockOwner(); owner != nullptr) {
            owner->RemoveWaiter(waiting_thread);
        }

        KThreadQueue::CancelWait(waiting_thread, wait_result, cancel_timer_task);
    }
};

} // namespace

// Lock the k light lock.
void KLightLock::Lock() {
    const auto cur_thread = GetCurrentThreadPointer(m_kernel);

    while (true) {
        auto old_tag = m_tag.load(std::memory_order_relaxed);

        while (!m_tag.compare_exchange_weak(old_tag, (old_tag == 0) ? cur_thread : (old_tag | 1),
                                            std::memory_order_acquire)) {
        }

        if (old_tag == 0 || LockSlowPath(old_tag | 1, cur_thread)) {
            break;
        }
    }
}

// Unlock the k light lock.
void KLightLock::Unlock() {
    const auto cur_thread = GetCurrentThreadPointer(m_kernel);

    auto expected = cur_thread;
    if (!m_tag.compare_exchange_strong(expected, 0, std::memory_order_release)) {
        UnlockSlowPath(cur_thread);
    }
}

// Lock using slow path.
bool KLightLock::LockSlowPath(uintptr_t owner, KThread* cur_thread) {
    ThreadQueueImplForKLightLock wait_queue(m_kernel);

    {
        KScopedSchedulerLock sl{m_kernel};

        if (m_tag.load(std::memory_order_relaxed) != owner) {
            return false;
        }

        cur_thread->SetKernelAddressKey(reinterpret_cast<uintptr_t>(std::addressof(m_tag)));
        reinterpret_cast<KThread*>(owner & ~1ULL)->AddWaiter(cur_thread);

        cur_thread->BeginWait(std::addressof(wait_queue));

        if (auto owner_thread = reinterpret_cast<KThread*>(owner & ~1ULL);
            owner_thread->IsSuspended()) {
            owner_thread->ContinueIfHasKernelWaiters();
        }
    }

    return true;
}

// Unlock using slow path.
void KLightLock::UnlockSlowPath(uintptr_t cur_thread) {
    auto owner_thread = reinterpret_cast<KThread*>(cur_thread);

    {
        KScopedSchedulerLock sl(m_kernel);

        bool has_waiters;
        auto next_owner = owner_thread->RemoveKernelWaiterByKey(
            std::addressof(has_waiters), reinterpret_cast<uintptr_t>(std::addressof(m_tag)));

        uintptr_t next_tag = 0;
        if (next_owner != nullptr) {
            next_tag =
                reinterpret_cast<uintptr_t>(next_owner) | static_cast<uintptr_t>(has_waiters);

            next_owner->EndWait(ResultSuccess);

            if (next_owner->IsSuspended()) {
                next_owner->ContinueIfHasKernelWaiters();
            }
        }

        if (owner_thread->IsSuspended()) {
            owner_thread->TrySuspend();
        }

        m_tag.store(next_tag, std::memory_order_release);
    }
}

// Check if the k light lock is currently locked by the current thread.
bool KLightLock::IsLockedByCurrentThread() const {
    return (m_tag.load() | 1ULL) ==
           (reinterpret_cast<uintptr_t>(GetCurrentThreadPointer(m_kernel)) | 1ULL);
}

} // namespace Kernel

