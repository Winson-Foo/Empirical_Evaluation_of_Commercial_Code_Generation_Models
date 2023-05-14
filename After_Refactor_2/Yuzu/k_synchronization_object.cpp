#include "common/assert.h"
#include "common/common_types.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_scoped_scheduler_lock_and_sleep.h"
#include "core/hle/kernel/k_synchronization_object.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/k_thread_queue.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc_results.h"

namespace Kernel {

namespace {

class ThreadQueueImplForKSynchronizationObjectWait final : public KThreadQueueWithoutEndWait {
public:
    ThreadQueueImplForKSynchronizationObjectWait(KernelCore& kernel, std::vector<std::unique_ptr<KSynchronizationObject>>& objects,
                                                 std::vector<KSynchronizationObject::ThreadListNode*>& nodes, s32 num_objects)
        : KThreadQueueWithoutEndWait(kernel), m_objects(objects), m_nodes(nodes), m_num_objects(num_objects) {}

    void NotifyAvailable(KThread* waiting_thread, KSynchronizationObject* signaled_object,
                         Result wait_result) override {
        unlinkNodes(signaled_object);

        waiting_thread->SetSyncedIndex(getSyncIndex(signaled_object));
        waiting_thread->ClearCancellable();

        KThreadQueue::EndWait(waiting_thread, wait_result);
    }

    void CancelWait(KThread* waiting_thread, Result wait_result, bool cancel_timer_task) override {
        for (auto i = 0; i < m_num_objects; ++i) {
            m_objects[i]->UnlinkNode(m_nodes[i]);
        }

        waiting_thread->ClearCancellable();

        KThreadQueue::CancelWait(waiting_thread, wait_result, cancel_timer_task);
    }

private:
    std::vector<std::unique_ptr<KSynchronizationObject>>& m_objects;
    std::vector<KSynchronizationObject::ThreadListNode*>& m_nodes;
    s32 m_num_objects;

    void unlinkNodes(KSynchronizationObject* signaled_object) {
        for (auto i = 0; i < m_num_objects; ++i) {
            m_objects[i]->UnlinkNode(m_nodes[i]);
        }
    }

    s32 getSyncIndex(KSynchronizationObject* signaled_object) {
        s32 sync_index = -1;

        for (auto i = 0; i < m_num_objects; ++i) {
            if (m_objects[i].get() == signaled_object && sync_index == -1) {
                sync_index = i;
            }
        }

        return sync_index;
    }
};

} // namespace

void KSynchronizationObject::Finalize() {
    this->OnFinalizeSynchronizationObject();
    KAutoObject::Finalize();
}

Result KSynchronizationObject::Wait(KernelCore& kernel, s32* out_index,
                                    std::vector<std::unique_ptr<KSynchronizationObject>>& objects,
                                    const s32 num_objects, s64 timeout) {
    std::vector<std::unique_ptr<KSynchronizationObject::ThreadListNode>> thread_nodes(num_objects);

    KThread* thread = GetCurrentThreadPointer(kernel);
    KHardwareTimer* timer{};
    ThreadQueueImplForKSynchronizationObjectWait wait_queue(kernel, objects, thread_nodes, num_objects);

    {
        KScopedSchedulerLockAndSleep slp(kernel, &timer, thread, timeout);

        if (thread->IsTerminationRequested()) {
            slp.CancelSleep();
            R_THROW(ResultTerminationRequested);
        }

        for (auto i = 0; i < num_objects; ++i) {
            ASSERT(objects[i] != nullptr);

            if (objects[i]->IsSignaled()) {
                *out_index = i;
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

        for (auto i = 0; i < num_objects; ++i) {
            thread_nodes[i] = std::make_unique<KSynchronizationObject::ThreadListNode>();
            thread_nodes[i]->thread = thread;
            thread_nodes[i]->next = nullptr;

            objects[i]->LinkNode(thread_nodes[i].get());
        }

        thread->SetCancellable();
        thread->SetSyncedIndex(-1);
        wait_queue.SetHardwareTimer(timer);
        thread->BeginWait(&wait_queue);
        thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::Synchronization);
    }

    *out_index = thread->GetSyncedIndex();
    R_RETURN(thread->GetWaitResult());
}

KSynchronizationObject::KSynchronizationObject(KernelCore& kernel) : KAutoObjectWithList{kernel} {}

KSynchronizationObject::~KSynchronizationObject() = default;

void KSynchronizationObject::NotifyAvailable(Result result) {
    KScopedSchedulerLock lock(m_kernel);

    if (!this->IsSignaled()) {
        return;
    }

    for (auto* cur_node = m_thread_list_head; cur_node != nullptr; cur_node = cur_node->next) {
        cur_node->thread->NotifyAvailable(this, result);
    }
}

std::vector<KThread*> KSynchronizationObject::GetWaitingThreadsForDebugging() const {
    std::vector<KThread*> threads;

    {
        KScopedSchedulerLock lock(m_kernel);

        for (auto* cur_node = m_thread_list_head; cur_node != nullptr; cur_node = cur_node->next) {
            threads.emplace_back(cur_node->thread);
        }
    }

    return threads;
}

} // namespace Kernel