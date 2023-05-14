#include "core/arm/dynarmic/arm_dynarmic_32.h"
#include "core/arm/dynarmic/arm_dynarmic_64.h"
#include "core/core.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/physical_core.h"

namespace Kernel {

PhysicalCore::PhysicalCore(std::size_t physical_core_index, const Core::System& system, const KScheduler& scheduler)
    : m_physical_core_index{physical_core_index}, m_system{system}, m_scheduler{scheduler} {
    m_arm_interface = Core::CPUArchitectureManager::GetInstance().CreateCPUArchitecture(system, physical_core_index);
}

void PhysicalCore::ExecuteInstructions() {
    m_arm_interface->Run();
    m_arm_interface->ClearExclusiveState();
}

void PhysicalCore::Idle() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_interrupt_condition_variable.wait(lock, [this] { return m_is_interrupted; });
}

bool PhysicalCore::IsInterrupted() const {
    return m_is_interrupted;
}

void PhysicalCore::SignalInterrupt() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_is_interrupted = true;
    m_arm_interface->SignalInterrupt();
    m_interrupt_condition_variable.notify_all();
}

void PhysicalCore::ClearInterrupt() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_is_interrupted = false;
    m_arm_interface->ClearInterrupt();
}

} // namespace Kernel

