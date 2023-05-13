#pragma once

#include "kthreadqueuewithoutendwait.h"
#include "kautoobjectwithlist.h"

namespace Kernel {

class KSynchronizationObject : public KAutoObjectWithList {
public:
    explicit KSynchronizationObject(KernelCore& kernel);
    virtual ~KSynchronizationObject();

    virtual bool IsSignaled() const = 0;
    virtual void LinkNode(KSynchronizationObject::ThreadListNode* node) = 0;
    virtual void UnlinkNode(KSynchronizationObject::ThreadListNode* node) = 0;

    void Finalize() override;

    Result Wait(KernelCore& kernel, s32& out_index, const std::vector<const KSynchronizationObject*>& objects, s64 timeout);

    void NotifyAvailable(Result result = ResultSuccess) override;
    std::vector<KThread*> GetWaitingThreadsForDebugging() const override;

    struct ThreadListNode {
        KThread* thread; // The thread that is waiting
        ThreadListNode* next; // The next node in the list
    };

protected:
    virtual void OnFinalizeSynchronizationObject() {}

private:
    KSynchronizationObject(const KSynchronizationObject&) = delete;
    KSynchronizationObject(KSynchronizationObject&&) = delete;
    KSynchronizationObject& operator=(const KSynchronizationObject&) = delete;
    KSynchronizationObject& operator=(KSynchronizationObject&&) = delete;
};

}

// File: ksynchronizationobject.cpp

#include "ksynchronizationobject.h"
#include "kscopedschedulerlock.h"
#include "kserviceresults.h"
#include "kthread.h"
#include <vector>

namespace Kernel {

namespace {

class ThreadQueueImplForKSynchronizationObjectWait final : public KThreadQueueWithoutEndWait {
public:
    explicit ThreadQueueImplForKSynchronizationObjectWait(KernelCore& kernel, const std::vector<const KSynchronizationObject*>& objects,
                                                           std::vector<KSynchronizationObject::ThreadListNode*>& nodes)
        : KThreadQueueWithoutEndWait(kernel), objects_(objects), nodes_(nodes) {}

    void NotifyAvailable(KThread* waiting_thread, KSynchronizationObject* signaled_object, Result wait_result) override {
        // Determine the sync index and notify the waiting thread of the available object.
        const auto it = std::find(objects_.cbegin(), objects_.cend(), signaled_object);
        const auto sync_index = static_cast<s32>(std::distance(objects_.cbegin(), it));
        waiting_thread->SetSyncedIndex(sync_index);
        waiting_thread->ClearCancellable();
        KThreadQueue::EndWait(waiting_thread, wait_result);
    }

    void CancelWait(KThread* waiting_thread, Result wait_result, bool cancel_timer_task) override {
        // Cancel the wait operation and notify the waiting thread of the cancellation.
        for (auto node : nodes_) {
            node->thread = nullptr;
            node->next = nullptr;
        }

        waiting_thread->ClearCancellable();
        KThreadQueue::CancelWait(waiting_thread, wait_result, cancel_timer_task);
    }

private:
    const std::vector<const KSynchronizationObject*>& objects_;
    std::vector<KSynchronizationObject::ThreadListNode*>& nodes_;
};

}

void KSynchronizationObject::Finalize()
{
    OnFinalizeSynchronizationObject();
    KAutoObject::Finalize();
}

Result KSynchronizationObject::Wait(KernelCore& kernel, s32& out_index, const std::vector<const KSynchronizationObject*>& objects, s64 timeout)
{
    std::vector<ThreadListNode> nodes(objects.size(), {});

    KThread* thread = GetCurrentThreadPointer(kernel);
    KHardwareTimer* timer{};
    ThreadQueueImplForKSynchronizationObjectWait wait_queue(kernel, objects, nodes);

    {
        KScopedSchedulerLockAndSleep slp(kernel, &timer, thread, timeout);
        if (thread->IsTerminationRequested()) {
            slp.CancelSleep();
            R_THROW(ResultTerminationRequested);
        }

        for (auto i = 0u; i < objects.size(); ++i) {
            if (objects[i]->IsSignaled()) {
                out_index = static_cast<s32>(i);
                slp.CancelSleep();
                R_THROW(ResultSuccess);
            }
        }

        if (timeout == 0) {
            slp.CancelSleep();
            R_THROW(ResultTimedOut);
        }

        if (thread->IsWaitCancelled()) {
            slp.CancelSleep();
            thread->ClearWaitCancelled();
            R_THROW(ResultCancelled);
        }

        for (auto i = 0u; i < objects.size(); ++i) {
            nodes[i].thread = thread;
            objects[i]->LinkNode(&nodes[i]);
        }

        thread->SetCancellable();
        thread->SetSyncedIndex(-1);

        wait_queue.SetHardwareTimer(timer);
        thread->BeginWait(&wait_queue);
        thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::Synchronization);
    }

    out_index = thread->GetSyncedIndex();
    R_RETURN(thread->GetWaitResult());
}

void KSynchronizationObject::NotifyAvailable(Result result) {
    KScopedSchedulerLock sl(m_kernel);

    if (!IsSignaled()) {
        return;
    }

    for (auto* cur_node = m_thread_list_head; cur_node != nullptr; cur_node = cur_node->next) {
        cur_node->thread->NotifyAvailable(this, result);
    }
}

std::vector<KThread*> KSynchronizationObject::GetWaitingThreadsForDebugging() const {
    std::vector<KThread*> threads;

    KScopedSchedulerLock lock(m_kernel);
    for (auto* cur_node = m_thread_list_head; cur_node != nullptr; cur_node = cur_node->next) {
        threads.emplace_back(cur_node->thread);
    }

    return threads;
}

}

