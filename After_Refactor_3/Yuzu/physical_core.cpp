// PhysicalCore.h (declarations)

#pragma once

#include <condition_variable>
#include <memory>
#include <cstddef>

namespace Core {
    class System;
    class ARMInterface;
}
class KScheduler;

namespace Kernel {

class InterruptManager;
class IdleManager;

class PhysicalCore {
public:
    PhysicalCore(std::size_t coreIndex, Core::System& system, KScheduler& scheduler);

    void Initialize(bool is64Bit);

    void Run();

    void ClearInterrupt();

private:
    std::unique_ptr<Core::ARMInterface> m_cpu;
    std::unique_ptr<InterruptManager> m_interruptManager;
    std::unique_ptr<IdleManager> m_idleManager;
};

} // namespace Kernel


// CPUManager.h (declaration)

#pragma once

#include <memory>

namespace Core {
    class System;
    class ARMInterface;
}

namespace Kernel {

class CPUManager {
public:
    static std::unique_ptr<Core::ARMInterface> CreateCPU(Core::System& system, std::size_t coreIndex);
};


// ARMInterface.h (declaration)

#pragma once

namespace Core {

class ARMInterface {
public:
    virtual ~ARMInterface() = default;

    virtual void Run() = 0;

    virtual void ClearInterrupt() = 0;

    virtual void SignalInterrupt() = 0;

    virtual void ClearExclusiveState() = 0;
};

} // namespace Core


// ARMInterface.cpp (implementation)

#include "core/arm/dynarmic/arm_dynarmic_32.h"
#include "core/arm/dynarmic/arm_dynarmic_64.h"
#include "core/core.h"
#include "CPUInterface.h"

namespace Core {

std::unique_ptr<ARMInterface> CreateARMInterface32(Core::System& system, bool isMulticore, bool& exclusiveMonitor, std::size_t coreIndex);
std::unique_ptr<ARMInterface> CreateARMInterface64(Core::System& system, bool isMulticore, bool& exclusiveMonitor, std::size_t coreIndex);

std::unique_ptr<ARMInterface> CreateARMInterface(Core::System& system, bool is64Bit, bool isMulticore, bool& exclusiveMonitor, std::size_t coreIndex) {
    if (is64Bit) {
        return CreateARMInterface64(system, isMulticore, exclusiveMonitor, coreIndex);
    } else {
        return CreateARMInterface32(system, isMulticore, exclusiveMonitor, coreIndex);
    }
}

} // namespace Core


// ARMInterface64.h (declaration)

#pragma once

#include "CPUInterface.h"
#include <vector>

namespace Core {

class ARMInterface64 final : public ARMInterface {
public:
    ARMInterface64(Core::System& system, bool isMulticore, bool& exclusiveMonitor, std::size_t coreIndex);

    ~ARMInterface64() override;

    void Run() override;

    void ClearInterrupt() override;

    void SignalInterrupt() override;

    void ClearExclusiveState() override;
};

} // namespace Core


// ARMInterface64.cpp (implementation)

#include "ARMInterface64.h"
#include "core/core.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/kernel.h"
#include "InterruptManager.h"
#include "IdleManager.h"

namespace Core {

ARMInterface64::ARMInterface64(Core::System& system, bool isMulticore, bool& exclusiveMonitor, std::size_t coreIndex)
    : m_system{system}, m_scheduler{system.KScheduler()}, m_interruptManager{std::make_unique<InterruptManager>()}, m_idleManager{std::make_unique<IdleManager>()} {
    auto& kernel = m_system.Kernel();
    m_cpu = std::make_unique<ARM_Dynarmic_64>(m_system, isMulticore, exclusiveMonitor, coreIndex);
}

ARMInterface64::~ARMInterface64() = default;

void ARMInterface64::Run() {
    m_cpu->Run();
}

void ARMInterface64::ClearInterrupt() {
    m_interruptManager->ClearInterrupt();
}

void ARMInterface64::SignalInterrupt() {
    m_interruptManager->SignalInterrupt();
}

void ARMInterface64::ClearExclusiveState() {
    m_cpu->ClearExclusiveState();
}


// ARMInterface32.h (declaration)

#pragma once

#include "CPUInterface.h"

namespace Core {

class ARMInterface32 final : public ARMInterface {
public:
    ARMInterface32(Core::System& system, bool isMulticore, bool& exclusiveMonitor, std::size_t coreIndex);

    ~ARMInterface32() override;

    void Run() override;

    void ClearInterrupt() override;

    void SignalInterrupt() override;

    void ClearExclusiveState() override;
};

} // namespace Core


// ARMInterface32.cpp (implementation)

#include "ARMInterface32.h"
#include "core/core.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/kernel.h"
#include "InterruptManager.h"
#include "IdleManager.h"

namespace Core {

ARMInterface32::ARMInterface32(Core::System& system, bool isMulticore, bool& exclusiveMonitor, std::size_t coreIndex)
    : m_system{system}, m_scheduler{system.KScheduler()}, m_interruptManager{std::make_unique<InterruptManager32>()}, m_idleManager{std::make_unique<IdleManager>()} {
    auto& kernel = m_system.Kernel();
    m_cpu = std::make_unique<ARM_Dynarmic_32>(m_system, isMulticore, exclusiveMonitor, coreIndex);
}

ARMInterface32::~ARMInterface32() = default;

void ARMInterface32::Run() {
    m_cpu->Run();
}

void ARMInterface32::ClearInterrupt() {
    m_interruptManager->ClearInterrupt();
}

void ARMInterface32::SignalInterrupt() {
    m_interruptManager->SignalInterrupt();
}

void ARMInterface32::ClearExclusiveState() {
    m_cpu->ClearExclusiveState();
}


// InterruptManager.h (declaration)

#pragma once

#include <condition_variable>

namespace Core {

class ARMInterface;

class InterruptManager {
public:
    virtual ~InterruptManager() = default;

    virtual void SignalInterrupt() = 0;

    virtual void ClearInterrupt() = 0;
};

class InterruptManager64 final : public InterruptManager {
public:
    void SignalInterrupt() override;

    void ClearInterrupt() override;

private:
    std::condition_variable m_onInterrupt;
    bool m_isInterrupted = false;
};

class InterruptManager32 final : public InterruptManager {
public:
    void SignalInterrupt() override;

    void ClearInterrupt() override;

private:
    std::condition_variable m_onInterrupt;
    bool m_isInterrupted = false;
};

} // namespace Core


// InterruptManager.cpp (implementation)

#include "InterruptManager.h"
#include "ARMInterface.h"

namespace Core {

void InterruptManager64::SignalInterrupt() {
    std::unique_lock lk{m_guard};
    m_isInterrupted = true;
    m_cpu->SignalInterrupt();
    m_on_interrupt.notify_all();
}

void InterruptManager64::ClearInterrupt() {
    std::unique_lock lk{m_guard};
    m_isInterrupted = false;
    m_cpu->ClearInterrupt();
}

void InterruptManager32::SignalInterrupt() {
    std::unique_lock lk{m_guard};
    m_isInterrupted = true;
    m_cpu->SignalInterrupt();
    m_on_interrupt.notify_all();
}

void InterruptManager32::ClearInterrupt() {
    std::unique_lock lk{m_guard};
    m_isInterrupted = false;
    m_cpu->ClearInterrupt();
}


// IdleManager.h (declaration)

#pragma once

#include <condition_variable>

namespace Core {

class ARMInterface;

class IdleManager {
public:
    void Idle();

    void WakeUp();

private:
    std::condition_variable m_onIdle;
    bool m_isIdle = false;
    std::mutex m_guard;
};

} // namespace Core


// IdleManager.cpp (implementation)

#include "IdleManager.h"

namespace Core {

void IdleManager::Idle() {
    std::unique_lock lk{m_guard};
    m_onIdle.wait(lk, [this] { return m_isIdle; });
}

void IdleManager::WakeUp() {
    std::unique_lock lk{m_guard};
    m_isIdle = false;
    m_onIdle.notify_all();
}

} // namespace Core


// CPUFactory.cpp (implementation)

#include "CPUManager.h"
#include "ARMInterface.h"
#include "ARMInterface32.h"
#include "ARMInterface64.h"
#include "core/core.h"

namespace Kernel {

std::unique_ptr<Core::ARMInterface> CPUManager::CreateCPU(Core::System& system, std::size_t coreIndex) {
#if defined(ARCHITECTURE_x86_64) || defined(ARCHITECTURE_arm64)
    auto& kernel = system.Kernel();
    bool is64Bit = true; // TODO: Detect platform architecture
    bool isMulticore = kernel.IsMulticore();
    bool& exclusiveMonitor = kernel.GetExclusiveMonitor();
    return Core::CreateARMInterface(system, is64Bit, isMulticore, exclusiveMonitor, coreIndex);
#else
#error Platform not supported yet.
#endif
}

} // namespace Kernel


// PhysicalCore.cpp (implementation)

#include "PhysicalCore.h"
#include "CPUManager.h"
#include "ARMInterface.h"
#include "InterruptManager.h"
#include "IdleManager.h"

namespace Kernel {

PhysicalCore::PhysicalCore(std::size_t coreIndex, Core::System& system, KScheduler& scheduler)
    : m_interruptManager{std::make_unique<InterruptManager>()}, m_idleManager{std::make_unique<IdleManager>()} {
    m_cpu = CPUManager::CreateCPU(system, coreIndex);
}

void PhysicalCore::Initialize(bool is64Bit) {
    // Nothing to do here (yet)
}

void PhysicalCore::Run() {
    m_cpu->Run();
}

void PhysicalCore::ClearInterrupt() {
    m_interruptManager->ClearInterrupt();
}

} // namespace Kernel

