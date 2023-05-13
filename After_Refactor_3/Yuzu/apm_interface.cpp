#include "common/logging/log.h"
#include "core/hle/service/apm/apm.h"
#include "core/hle/service/apm/apm_controller.h"
#include "core/hle/service/apm/apm_interface.h"
#include "core/hle/service/ipc_helpers.h"

namespace Service::APM {

class Session : public ServiceFramework<Session> {
public:
    explicit Session(Core::System& system, Controller& controller)
        : ServiceFramework{system, "APM::Session"}, controller_{controller} {
        static const FunctionInfo functions[] = {
            {0, &Session::SetPerformanceConfiguration, "SetPerformanceConfiguration"},
            {1, &Session::GetPerformanceConfiguration, "GetPerformanceConfiguration"},
            {2, &Session::SetCpuOverclockEnabled, "SetCpuOverclockEnabled"},
        };
        RegisterHandlers(functions);
    }

private:
    void SetPerformanceConfiguration(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto mode = rp.PopEnum<PerformanceMode>();
        const auto config = rp.PopEnum<PerformanceConfiguration>();
        LOG_DEBUG(Service_APM, "SetPerformanceConfiguration called mode={} config={}", mode, config);
        controller_.SetPerformanceConfiguration(mode, config);
        IPC::ResponseBuilder{ctx, 2}.Push(ResultSuccess);
    }

    void GetPerformanceConfiguration(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto mode = rp.PopEnum<PerformanceMode>();
        LOG_DEBUG(Service_APM, "GetPerformanceConfiguration called mode={}", mode);
        IPC::ResponseBuilder{ctx, 3}.Push(ResultSuccess).PushEnum(controller_.GetCurrentPerformanceConfiguration(mode));
    }

    void SetCpuOverclockEnabled(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto cpuOverclockEnabled = rp.Pop<bool>();
        LOG_WARNING(Service_APM, "(STUBBED) SetCpuOverclockEnabled called, cpuOverclockEnabled={}", cpuOverclockEnabled);
        IPC::ResponseBuilder{ctx, 2}.Push(ResultSuccess);
    }

    Controller& controller_;
};

APM::APM(Core::System& system, std::shared_ptr<Module> apm, Controller& controller, const char* name)
    : ServiceFramework{system, name}, apm_{std::move(apm)}, controller_{controller} {
    static const FunctionInfo functions[] = {
        {0, &APM::OpenSession, "OpenSession"},
        {1, &APM::GetPerformanceMode, "GetPerformanceMode"},
        {6, &APM::IsCpuOverclockEnabled, "IsCpuOverclockEnabled"},
    };
    RegisterHandlers(functions);
}

void APM::OpenSession(HLERequestContext& ctx) {
    LOG_DEBUG(Service_APM, "OpenSession called");
    IPC::ResponseBuilder{ctx, 2, 0, 1}.Push(ResultSuccess)
                                       .PushIpcInterface<Session>(system, controller_);
}

void APM::GetPerformanceMode(HLERequestContext& ctx) {
    LOG_DEBUG(Service_APM, "GetPerformanceMode called");
    IPC::ResponseBuilder{ctx, 2}.PushEnum(controller_.GetCurrentPerformanceMode());
}

void APM::IsCpuOverclockEnabled(HLERequestContext& ctx) {
    LOG_WARNING(Service_APM, "(STUBBED) IsCpuOverclockEnabled called");
    IPC::ResponseBuilder{ctx, 3}.Push(ResultSuccess).Push(false);
}

APM_Sys::APM_Sys(Core::System& system, Controller& controller)
    : ServiceFramework{system, "APM_Sys"}, controller_{controller} {
    static const FunctionInfo functions[] {
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

void APM_Sys::GetPerformanceEvent(HLERequestContext& ctx) {
    LOG_DEBUG(Service_APM, "GetPerformanceEvent called");
    IPC::ResponseBuilder{ctx, 2, 0, 1}.Push(ResultSuccess)
                                       .PushIpcInterface<Session>(system, controller_);
}

void APM_Sys::SetCpuBoostMode(HLERequestContext& ctx) {
    IPC::RequestParser rp{ctx};
    const auto mode = rp.PopEnum<CpuBoostMode>();
    LOG_DEBUG(Service_APM, "SetCpuBoostMode called, mode={:08X}", mode);
    controller_.SetFromCpuBoostMode(mode);
    IPC::ResponseBuilder{ctx, 2}.Push(ResultSuccess);
}

void APM_Sys::GetCurrentPerformanceConfiguration(HLERequestContext& ctx) {
    LOG_DEBUG(Service_APM, "GetCurrentPerformanceConfiguration called");
    IPC::ResponseBuilder{ctx, 3}.Push(ResultSuccess)
                                .PushEnum(controller_.GetCurrentPerformanceConfiguration(controller_.GetCurrentPerformanceMode()));
}

} // namespace Service::APM