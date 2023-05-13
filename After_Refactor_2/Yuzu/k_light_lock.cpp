#include "core/hle/kernel/k_light_lock.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/k_thread_queue.h"
#include "core/hle/kernel/kernel.h"

namespace Kernel {

class KLightLockThreadQueue final : public KThreadQueue {
public:
    explicit KLightLockThreadQueue(KernelCore& kernel) : KThreadQueue(kernel) {}

    void CancelWait(KThread* waiting_thread, Result wait_result, bool cancel_timer_task) override {
        // Remove the thread as a waiter from its owner.
        if (KThread* owner = waiting_thread->GetLockOwner(); owner != nullptr) {
            owner->RemoveWaiter(waiting_thread);
        }

        // Invoke the base cancel wait handler.
        KThreadQueue::CancelWait(waiting_thread, wait_result, cancel_timer_task);
    }
};

KLightLock::KLightLock() : m_tag(0) {}

void KLightLock::Lock() {
    const auto current_thread = GetCurrentThreadPointer(m_kernel);
    const auto cur_thread_address = reinterpret_cast<uintptr_t>(current_thread);

    while (true) {
        const auto old_tag = m_tag.load(std::memory_order_relaxed);

        while (!m_tag.compare_exchange_weak(
                old_tag, (old_tag == 0) ? cur_thread_address : (old_tag | 1), std::memory_order_acquire)) {}

        if (old_tag == 0 || LockSlowPath(old_tag | 1, cur_thread_address)) {
            break;
        }
    }
}

void KLightLock::Unlock() {
    const auto current_thread = GetCurrentThreadPointer(m_kernel);
    const auto cur_thread_address = reinterpret_cast<uintptr_t>(current_thread);

    auto expected = cur_thread_address;
    if (!m_tag.compare_exchange_strong(expected, 0, std::memory_order_release)) {
        UnlockSlowPath(cur_thread_address);
    }
}

bool KLightLock::IsLockedByCurrentThread() const {
    const auto current_thread = GetCurrentThreadPointer(m_kernel);
    const auto cur_thread_address = reinterpret_cast<uintptr_t>(current_thread);

    return (m_tag.load() | 1ULL) == (cur_thread_address | 1ULL);
}

void KLightLock::UnlockSlowPath(uintptr_t cur_thread_address) {
    auto owner_thread = reinterpret_cast<KThread*>(cur_thread_address);

    // Unlock.
    {
        KScopedSchedulerLock sl(m_kernel);

        // Get the next owner.
        bool has_waiters;
        auto next_owner = owner_thread->RemoveKernelWaiterByKey(
            std::addressof(has_waiters), reinterpret_cast<uintptr_t>(std::addressof(m_tag)));

        // Pass the lock to the next owner.
        uintptr_t next_tag = 0;
        if (next_owner != nullptr) {
            next_tag = reinterpret_cast<uintptr_t>(next_owner) | static_cast<uintptr_t>(has_waiters);
            next_owner->EndWait(ResultSuccess);

            if (next_owner->IsSuspended()) {
                next_owner->ContinueIfHasKernelWaiters();
            }
        }

        // We may have unsuspended in the process of acquiring the lock, so we'll re-suspend now if
        // so.
        if (owner_thread->IsSuspended()) {
            owner_thread->TrySuspend();
        }

        // Write the new tag value.
        m_tag.store(next_tag, std::memory_order_release);
    }
}

bool KLightLock::LockSlowPath(uintptr_t owner_address, uintptr_t cur_thread_address) {
    auto cur_thread = reinterpret_cast<KThread*>(cur_thread_address);
    KLightLockThreadQueue waiting_queue(m_kernel);

    // Pend the current thread waiting on the owner thread.
    {
        KScopedSchedulerLock sl{m_kernel};

        // Ensure we actually have locking to do.
        if (m_tag.load(std::memory_order_relaxed) != owner_address) {
            return false;
        }

        // Add the current thread as a waiter on the owner.
        auto owner_thread = reinterpret_cast<KThread*>(owner_address & ~1ULL);
        cur_thread->SetKernelAddressKey(reinterpret_cast<uintptr_t>(std::addressof(m_tag)));
        owner_thread->AddWaiter(cur_thread);

        // Begin waiting to hold the lock.
        cur_thread->BeginWait(std::addressof(waiting_queue));

        if (owner_thread->IsSuspended()) {
            owner_thread->ContinueIfHasKernelWaiters();
        }
    }

    return true;
}

} // namespace Kernel