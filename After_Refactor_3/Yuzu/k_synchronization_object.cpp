#pragma once

#include <vector>
#include "common/assert.h"
#include "core/hle/kernel/k_auto_object.h"
#include "core/hle/kernel/k_base_object.h"
#include "core/hle/kernel/k_manual_object.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/k_thread_queue.h"
#include "core/hle/kernel/k_thread_queue_impl.h"
#include "core/hle/kernel/kernel.h"

namespace Kernel {

class KObjectWithList : public KBaseObject {
public:
    using ThreadListNode = KThreadQueueWithoutEndWait::ThreadListNode;

    virtual ~KObjectWithList() {}

    void LinkNode(ThreadListNode* node) {
        m_thread_list.LinkBack(node);
    }

    void UnlinkNode(ThreadListNode* node) {
        m_thread_list.Unlink(node);
    }

    void NotifyAvailable(Result result) {
        KScopedSchedulerLock lock(m_kernel);
        for (auto* cur_node = m_thread_list.GetHead(); cur_node != nullptr; cur_node = cur_node->next) {
            cur_node->thread->NotifyAvailable(this, result);
        }
    }

    std::vector<KThread*> GetWaitingThreads() const {
        std::vector<KThread*> threads;
        KScopedSchedulerLock lock(m_kernel);
        for (auto* cur_node = m_thread_list.GetHead(); cur_node != nullptr; cur_node = cur_node->next) {
            threads.emplace_back(cur_node->thread);
        }
        return threads;
    }

private:
    KThreadList m_thread_list;
};

class KSynchronizationObject : public KObjectWithList, public KAutoObjectWithList, public KManualObjectWithList {
public:
    KSynchronizationObject(KernelCore& kernel)
        : KObjectWithList(), KAutoObjectWithList(kernel), KManualObjectWithList(), m_signaled(false) {}

    virtual ~KSynchronizationObject() override {}

    void Finalize() override {
        this->OnFinalizeSynchronizationObject();
        KAutoObjectWithList::Finalize();
        KManualObjectWithList::Finalize();
    }

    Result WaitForMultipleObjects(KernelCore& kernel, s32* out_index, KSynchronizationObject** objects, const s32 num_objects, s64 timeout) {
        // Allocate space on stack for thread nodes.
        std::vector<ThreadListNode> thread_nodes(num_objects);

        // Prepare for wait.
        KThread* thread = GetCurrentThreadPointer(kernel);
        KHardwareTimer* timer{};
        KThreadQueueWithoutEndWait* wait_queues[num_objects];
        ThreadListNode* wait_nodes[num_objects];
        for (auto i = 0; i < num_objects; ++i) {
            wait_nodes[i] = std::addressof(thread_nodes[i]);
            wait_queues[i] = objects[i]->CreateWaitQueue(kernel, wait_nodes[i]);
        }

        {
            // Setup the scheduling lock and sleep.
            KScopedSchedulerLockAndSleep slp(kernel, std::addressof(timer), thread, timeout);

            // Check if the thread should terminate.
            if (thread->IsTerminationRequested()) {
                slp.CancelSleep();
                return ResultTerminationRequested;
            }

            // Check if any of the objects are already signaled.
            for (auto i = 0; i < num_objects; ++i) {
                ASSERT(objects[i] != nullptr);

                if (objects[i]->IsSignaled()) {
                    *out_index = i;
                    slp.CancelSleep();
                    return ResultSuccess;
                }
            }

            // Check if the timeout is zero.
            if (timeout == 0) {
                slp.CancelSleep();
                return ResultTimedOut;
            }

            // Check if waiting was canceled.
            if (thread->IsWaitCancelled()) {
                slp.CancelSleep();
                thread->ClearWaitCancelled();
                return ResultCancelled;
            }

            // Mark the thread as cancellable.
            thread->SetCancellable();

            // Clear the thread's synced index.
            thread->SetSyncedIndex(-1);

            // Wait for an object to be signaled.
            thread->BeginWaitForMultiple(wait_queues, num_objects);
        }

        // Set the output index.
        *out_index = thread->GetSyncedIndex();

        // Get the wait result.
        return thread->GetWaitResult();
    }

    Result Wait(KernelCore& kernel, s32* out_index, KSynchronizationObject** objects, const s32 num_objects, s64 timeout) {
        return WaitForMultipleObjects(kernel, out_index, objects, num_objects, timeout);
    }

    void Signal() {
        KScopedSchedulerLock lock(m_kernel);
        m_signaled = true;
    }

    bool IsSignaled() const {
        KScopedSchedulerLock lock(m_kernel);
        return m_signaled;
    }

    std::vector<KThread*> GetWaitingThreadsForDebugging() const {
        return GetWaitingThreads();
    }

protected:
    virtual void OnFinalizeSynchronizationObject() {}

private:
    bool m_signaled;
};

} // namespace Kernel

k_thread_queue_impl.h:

#pragma once

#include "core/hle/kernel/k_thread_queue.h"

namespace Kernel {

class KThreadQueueImplWithEndWait : public KThreadQueue {
public:
    void EndWait(KThread* waiting_thread, Result wait_result) override {
        // Set the waiting thread's sync index.
        waiting_thread->SetSyncedIndex(0);

        // Set the waiting thread as not cancellable.
        waiting_thread->ClearCancellable();

        // Invoke the base end wait handler.
        KThreadQueue::EndWait(waiting_thread, wait_result);
    }

    void CancelWait(KThread* waiting_thread, Result wait_result, bool cancel_timer_task) override {
        // Set the waiting thread as not cancellable.
        waiting_thread->ClearCancellable();

        // Invoke the base cancel wait handler.
        KThreadQueue::CancelWait(waiting_thread, wait_result, cancel_timer_task);
    }
};

class ThreadQueueImplForKSynchronizationObjectWait final : public KThreadQueueImplWithEndWait {
public:
    ThreadQueueImplForKSynchronizationObjectWait(KernelCore& kernel, KSynchronizationObject** o, KSynchronizationObject::ThreadListNode* n, s32 c)
        : KThreadQueueImplWithEndWait(), m_objects(o), m_nodes(n), m_count(c) {}

    void NotifyAvailable(KThread* waiting_thread, KSynchronizationObject* signaled_object, Result wait_result) override {
        // Determine the sync index, and unlink all nodes.
        s32 sync_index = -1;
        for (auto i = 0; i < m_count; ++i) {
            // Check if this is the signaled object.
            if (m_objects[i] == signaled_object && sync_index == -1) {
                sync_index = i;
            }

            // Unlink the current node from the current object.
            m_objects[i]->UnlinkNode(std::addressof(m_nodes[i]));
        }

        // Set the waiting thread's sync index.
        waiting_thread->SetSyncedIndex(sync_index);

        // Set the waiting thread as not cancellable.
        waiting_thread->ClearCancellable();

        // Invoke the base end wait handler.
        KThreadQueue::EndWait(waiting_thread, wait_result);
    }

private:
    KSynchronizationObject** m_objects;
    KSynchronizationObject::ThreadListNode* m_nodes;
    s32 m_count;
};

} // namespace Kernel

#pragma once

#include <atomic>
#include <cstdint>
#include "common/assert.h"
#include "common/common_types.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/k_thread_queue.h"

namespace Kernel {

enum class ThreadNotificationType {
    None,
    Wait,
    Transfer,
    Exception,
};

enum class ThreadWaitReasonForDebugging {
    None,
    Synchronization,
    Sleep,
    Wait,
};

class KThread {
public:
    struct WaitContext {
        KThreadQueue* queue = nullptr;
        bool safe_cancel = false;
    };

    KThread(KernelCore& kernel, u64 entrypoint, u64 stack_top, u32 ideal_core_index = 0);
    virtual ~KThread();

    virtual void Finalize();

    void Run();

    void SetContext(void* ctx);
    void* GetContext() const;

    void SetSyncedIndex(s32 index);
    s32 GetSyncedIndex() const;

    void SetCancellable();
    void ClearCancellable();
    bool IsCancellable() const;

    void BeginWait(KThreadQueue* queue);
    void EndWait(Result result);
    void CancelWait(Result result, bool cancel_timer_task = false);
    bool IsWaiting() const;

    void BeginWaitForMultiple(KThreadQueue** wait_queues, s32 num_wait_queues);
    void EndWaitForMultiple(Result result);
    bool IsWaitingForMultiple() const;

    void Transfer(KThread* dest);
    void SetTransferInfo(KThread* src);
    KThread* GetTransferInfo() const;

    void RaiseException(u32 type);
    void ClearException();
    bool HasException() const;
    u32 GetExceptionType() const;

    void RequestTermination();
    void ClearTermination();
    bool IsTerminationRequested() const;

    void RequestWaitCancellation();
    void ClearWaitCancelled();
    bool IsWaitCancelled() const;

    void NotifyAvailable(KSynchronizationObject* object, Result result);

    void SetWaitReasonForDebugging(ThreadWaitReasonForDebugging reason) {
        m_wait_reason_for_debugging = reason;
    }

    std::atomic<s32> m_status;
    std::atomic<s32> m_priority;
    std::atomic<bool> m_affinity_needed;
    std::atomic<u32> m_affinity_mask;
    std::atomic<bool> m_ideal_core_index_needed;
    std::atomic<u32> m_ideal_core_index;
    std::atomic<bool> m_terminate_request;
    std::atomic<bool> m_wait_cancel_request;
    std::atomic<bool> m_exception_active;
    std::atomic<u32> m_exception_type;

private:
    KernelCore& m_kernel;

    u64 m_entrypoint;
    u64 m_stack_top;
    void* m_context = nullptr;

    s32 m_sync_index = -1;

    std::atomic<ThreadNotificationType> m_notification_type;
    WaitContext m_wait_context;

    bool m_waiting_for

