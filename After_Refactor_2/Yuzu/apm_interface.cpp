// apm_service.h
#pragma once

#include <memory>
#include <string>

#include "ipc_helpers.h"
#include "logging/log.h"
#include "service_framework.h"
#include "core/hle/service/apm/apm_controller.h"

namespace Service::APM {

class ISession final : public ServiceFramework<ISession> {
public:
    explicit ISession(Core::System& system, Controller& controller)
        : ServiceFramework(system, "ISession"), controller_(controller) {
        static const FunctionInfo functions[] = {
            {0, &ISession::SetPerformanceConfiguration, "SetPerformanceConfiguration"},
            {1, &ISession::GetPerformanceConfiguration, "GetPerformanceConfiguration"},
            {2, &ISession::SetCpuOverclockEnabled, "SetCpuOverclockEnabled"},
        };
        RegisterHandlers(functions);
    }

private:
    void SetPerformanceConfiguration(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};

        const auto mode = rp.PopEnum<PerformanceMode>();
        const auto config = rp.PopEnum<PerformanceConfiguration>();
        LOG_DEBUG(Service_APM, "SetPerformanceConfiguration: mode={} config={}", mode, config);

        controller_.SetPerformanceConfiguration(mode, config);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void GetPerformanceConfiguration(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};

        const auto mode = rp.PopEnum<PerformanceMode>();
        LOG_DEBUG(Service_APM, "GetPerformanceConfiguration: mode={}", mode);

        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(ResultSuccess);
        rb.PushEnum(controller_.GetCurrentPerformanceConfiguration(mode));
    }

    void SetCpuOverclockEnabled(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};

        const auto cpu_overclock_enabled = rp.Pop<bool>();
        LOG_WARNING(Service_APM, "(STUBBED) SetCpuOverclockEnabled called, cpu_overclock_enabled={}",
                    cpu_overclock_enabled);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    Controller& controller_;
};

class APM final : public ServiceFramework<APM> {
public:
    APM(Core::System& system, std::shared_ptr<Module> apm, Controller& controller, const std::string& name)
        : ServiceFramework(system, name), apm_(std::move(apm)), controller_(controller) {
        static const FunctionInfo functions[] = {
            {0, &APM::OpenSession, "OpenSession"},
            {1, &APM::GetPerformanceMode, "GetPerformanceMode"},
            {6, &APM::IsCpuOverclockEnabled, "IsCpuOverclockEnabled"},
        };
        RegisterHandlers(functions);
    }

    ~APM() = default;

private:
    void OpenSession(HLERequestContext& ctx) {
        LOG_DEBUG(Service_APM, "OpenSession called");

        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<ISession>(system_, controller_);
    }

    void GetPerformanceMode(HLERequestContext& ctx) {
        LOG_DEBUG(Service_APM, "GetPerformanceMode called");

        IPC::ResponseBuilder rb{ctx, 2};
        rb.PushEnum(controller_.GetCurrentPerformanceMode());
    }

    void IsCpuOverclockEnabled(HLERequestContext& ctx) {
        LOG_WARNING(Service_APM, "(STUBBED) IsCpuOverclockEnabled called");

        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(ResultSuccess);
        rb.Push(false);
    }

    std::shared_ptr<Module> apm_;
    Controller& controller_;
};

class APM_Sys final : public ServiceFramework<APM_Sys> {
public:
    explicit APM_Sys(Core::System& system, Controller& controller)
        : ServiceFramework(system, "apm:sys"), controller_(controller) 
    {
        static const FunctionInfo functions[] = {
            {0, nullptr, "RequestPerformanceMode"},
            {1, &APM_Sys::GetPerformanceEvent, "GetPerformanceEvent"},
            {2, nullptr, "GetThrottlingState"},
            {3, nullptr, "GetLastThrottlingState"},
            {4, nullptr, "ClearLastThrottlingState"},
            {5, nullptr, "LoadAndApplySettings"},
            {6, &APM_Sys::SetCpuBoostMode, "SetCpuBoostMode"},
            {7, &APM_Sys::GetCurrentPerformanceConfiguration, "GetCurrentPerformanceConfiguration"},
        };
        RegisterHandlers(functions);
    }

    ~APM_Sys() = default;

private:
    void GetPerformanceEvent(HLERequestContext& ctx) {
        LOG_DEBUG(Service_APM, "GetPerformanceEvent called");

        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<ISession>(system_, controller_);
    }

    void SetCpuBoostMode(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto mode = rp.PopEnum<CpuBoostMode>();

        LOG_DEBUG(Service_APM, "SetCpuBoostMode called, mode={:08X}", mode);

        controller_.SetFromCpuBoostMode(mode);

        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void GetCurrentPerformanceConfiguration(HLERequestContext& ctx) {
        LOG_DEBUG(Service_APM, "GetCurrentPerformanceConfiguration called");

        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(ResultSuccess);
        rb.PushEnum(controller_.GetCurrentPerformanceConfiguration(controller_.GetCurrentPerformanceMode()));
    }

    Controller& controller_;
};
} // namespace Service::APM