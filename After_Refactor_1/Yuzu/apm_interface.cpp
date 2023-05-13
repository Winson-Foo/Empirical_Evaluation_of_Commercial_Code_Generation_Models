#include "common/logging/log.h"
#include "core/hle/service/apm/apm.h"
#include "core/hle/service/apm/apm_controller.h"
#include "core/hle/service/apm/apm_interface.h"
#include "core/hle/service/ipc_helpers.h"

namespace Service::APM {

class ISession final : public ServiceFramework<ISession> {
public:
    explicit ISession(Core::System& system, Controller& controller)
        : ServiceFramework{system, "ISession"}, controller{controller} {
        static const FunctionInfo functions[] = {
            {0, &ISession::setPerformanceConfiguration, "SetPerformanceConfiguration"},
            {1, &ISession::getPerformanceConfiguration, "GetPerformanceConfiguration"},
            {2, &ISession::setCpuOverclockEnabled, "SetCpuOverclockEnabled"},
        };
        RegisterHandlers(functions);
    }

private:
    void setPerformanceConfiguration(HLERequestContext& ctx) {
        IPC::RequestParser requestParser{ctx};

        const auto mode = requestParser.PopEnum<PerformanceMode>();
        const auto config = requestParser.PopEnum<PerformanceConfiguration>();
        LOG_DEBUG(Service_APM, "SetPerformanceConfiguration called mode={} config={}", mode, config);

        controller.SetPerformanceConfiguration(mode, config);

        IPC::ResponseBuilder responseBuilder{ctx, 2};
        responseBuilder.Push(ResultSuccess);
    }

    void getPerformanceConfiguration(HLERequestContext& ctx) {
        IPC::RequestParser requestParser{ctx};

        const auto mode = requestParser.PopEnum<PerformanceMode>();
        LOG_DEBUG(Service_APM, "GetPerformanceConfiguration called mode={}", mode);

        IPC::ResponseBuilder responseBuilder{ctx, 3};
        responseBuilder.Push(ResultSuccess);
        responseBuilder.PushEnum(controller.GetCurrentPerformanceConfiguration(mode));
    }

    void setCpuOverclockEnabled(HLERequestContext& ctx) {
        IPC::RequestParser requestParser{ctx};

        const auto cpuOverclockEnabled = requestParser.Pop<bool>();

        LOG_WARNING(Service_APM, "(STUBBED) SetCpuOverclockEnabled called, cpuOverclockEnabled={}",
                    cpuOverclockEnabled);

        IPC::ResponseBuilder responseBuilder{ctx, 2};
        responseBuilder.Push(ResultSuccess);
    }

    Controller& controller;
};

APM::APM(Core::System& system, std::shared_ptr<Module> apm, Controller& controller,
         const char* name)
    : ServiceFramework{system, name}, apm{std::move(apm)}, controller{controller} {
    static const FunctionInfo functions[] = {
        {0, &APM::openSession, "OpenSession"},
        {1, &APM::getPerformanceMode, "GetPerformanceMode"},
        {6, &APM::isCpuOverclockEnabled, "IsCpuOverclockEnabled"},
    };
    RegisterHandlers(functions);
}

APM::~APM() = default;

void APM::openSession(HLERequestContext& ctx) {
    LOG_DEBUG(Service_APM, "OpenSession called");

    IPC::ResponseBuilder responseBuilder{ctx, 2, 0, 1};
    responseBuilder.Push(ResultSuccess);
    responseBuilder.PushIpcInterface<ISession>(system, controller);
}

void APM::getPerformanceMode(HLERequestContext& ctx) {
    LOG_DEBUG(Service_APM, "GetPerformanceMode called");

    IPC::ResponseBuilder responseBuilder{ctx, 2};
    responseBuilder.PushEnum(controller.GetCurrentPerformanceMode());
}

void APM::isCpuOverclockEnabled(HLERequestContext& ctx) {
    LOG_WARNING(Service_APM, "(STUBBED) IsCpuOverclockEnabled called");

    IPC::ResponseBuilder responseBuilder{ctx, 3};
    responseBuilder.Push(ResultSuccess);
    responseBuilder.Push(false);
}

APM_Sys::APM_Sys(Core::System& system, Controller& controller)
    : ServiceFramework{system, "apm:sys"}, controller{controller} {
    static const FunctionInfo functions[] = {
        {0, nullptr, "RequestPerformanceMode"},
        {1, &APM_Sys::getPerformanceEvent, "GetPerformanceEvent"},
        {2, nullptr, "GetThrottlingState"},
        {3, nullptr, "GetLastThrottlingState"},
        {4, nullptr, "ClearLastThrottlingState"},
        {5, nullptr, "LoadAndApplySettings"},
        {6, &APM_Sys::setCpuBoostMode, "SetCpuBoostMode"},
        {7, &APM_Sys::getCurrentPerformanceConfiguration, "GetCurrentPerformanceConfiguration"},
    };

    RegisterHandlers(functions);
}
APM_Sys::~APM_Sys() = default;

void APM_Sys::getPerformanceEvent(HLERequestContext& ctx) {
    LOG_DEBUG(Service_APM, "GetPerformanceEvent called");

    IPC::ResponseBuilder responseBuilder{ctx, 2, 0, 1};
    responseBuilder.Push(ResultSuccess);
    responseBuilder.PushIpcInterface<ISession>(system, controller);
}

void APM_Sys::setCpuBoostMode(HLERequestContext& ctx) {
    IPC::RequestParser requestParser{ctx};
    const auto mode = requestParser.PopEnum<CpuBoostMode>();

    LOG_DEBUG(Service_APM, "SetCpuBoostMode called, mode={:08X}", mode);

    controller.SetFromCpuBoostMode(mode);

    IPC::ResponseBuilder responseBuilder{ctx, 2};
    responseBuilder.Push(ResultSuccess);
}

void APM_Sys::getCurrentPerformanceConfiguration(HLERequestContext& ctx) {
    LOG_DEBUG(Service_APM, "GetCurrentPerformanceConfiguration called");

    IPC::ResponseBuilder responseBuilder{ctx, 3};
    responseBuilder.Push(ResultSuccess);
    responseBuilder.PushEnum(
        controller.GetCurrentPerformanceConfiguration(controller.GetCurrentPerformanceMode()));
}

} // namespace Service::APM

