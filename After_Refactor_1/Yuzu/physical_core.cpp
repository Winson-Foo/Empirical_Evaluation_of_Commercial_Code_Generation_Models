// SPDX-License-Identifier: GPL-2.0-or-later

#include "physical_core.h"
#include "../core.h"
#include "../hle/kernel/k_scheduler.h"
#include "../hle/kernel/kernel.h"

namespace Kernel {

PhysicalCore::PhysicalCore(std::size_t core_index, Core::System& system, KScheduler& scheduler)
    : m_core_index{core_index}, m_system{system}, m_scheduler{scheduler} {}

PhysicalCore::~PhysicalCore() = default;

void PhysicalCore::Initialize(bool is_64_bit) {
    auto& kernel = m_system.Kernel();
    if (is_64_bit) {
        m_arm_interface = std::make_unique<Core::ARM_Dynarmic_64>(
            m_system, kernel.IsMulticore(), kernel.GetExclusiveMonitor(), m_core_index);
    } else {
        m_arm_interface = std::make_unique<Core::ARM_Dynarmic_32>(
            m_system, kernel.IsMulticore(), kernel.GetExclusiveMonitor(), m_core_index);
    }
}

void PhysicalCore::Run() {
    m_arm_interface->Run();
    m_arm_interface->ClearExclusiveState();
}

void PhysicalCore::Idle() {
    std::unique_lock lk{m_guard};
    m_on_interrupt.wait(lk, [this] { return m_is_interrupted; });
}

bool PhysicalCore::IsInterrupted() const {
    return m_is_interrupted;
}

void PhysicalCore::Interrupt() {
    std::unique_lock lk{m_guard};
    m_is_interrupted = true;
    m_arm_interface->SignalInterrupt();
    m_on_interrupt.notify_all();
}

void PhysicalCore::ClearInterrupt() {
    std::unique_lock lk{m_guard};
    m_is_interrupted = false;
    m_arm_interface->ClearInterrupt();
}

} // namespace Kernel

