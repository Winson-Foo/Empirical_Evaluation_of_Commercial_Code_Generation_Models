#include <memory>
#include "common/logging/log.h"
#include "common/settings.h"
#include "core/core.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_null/renderer_null.h"
#include "video_core/renderer_opengl/renderer_opengl.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/video_core.h"

namespace VideoCore {

std::unique_ptr<RendererBase> CreateRenderer(
    const Settings::RendererBackend& backend,
    Core::TelemetrySession& telemetry_session,
    Core::Frontend::EmuWindow& emu_window,
    Core::ApplicationMemory& app_memory,
    Tegra::GPU& gpu,
    std::unique_ptr<Core::Frontend::GraphicsContext> context
) {
    switch (backend) {
        case Settings::RendererBackend::OpenGL:
            return std::make_unique<OpenGL::RendererOpenGL>(
                telemetry_session, emu_window, app_memory, gpu, std::move(context));
        case Settings::RendererBackend::Vulkan:
            return std::make_unique<Vulkan::RendererVulkan>(
                telemetry_session, emu_window, app_memory, gpu, std::move(context));
        case Settings::RendererBackend::Null:
            return std::make_unique<Null::RendererNull>(
                emu_window, app_memory, gpu, std::move(context));
        default:
            return nullptr;
    }
}

std::unique_ptr<Tegra::GPU> CreateGPU(Core::Frontend::EmuWindow& emu_window, Core::System& system) {
    Settings::UpdateRescalingInfo();

    const auto nvdec_value = Settings::values.nvdec_emulation.GetValue();
    const bool use_nvdec = nvdec_value != Settings::NvdecEmulation::Off;
    const bool use_async = Settings::values.use_asynchronous_gpu_emulation.GetValue();

    auto gpu = std::make_unique<Tegra::GPU>(system, use_async, use_nvdec);
    auto context = emu_window.CreateSharedContext();

    auto scope = context->Acquire();

    try {
        auto renderer = CreateRenderer(
            Settings::values.renderer_backend.GetValue(),
            system.TelemetrySession(),
            emu_window,
            system.ApplicationMemory(),
            *gpu,
            std::move(context)
        );

        gpu->BindRenderer(std::move(renderer));

        return gpu;
    } catch (const std::runtime_error& exception) {
        scope.Cancel();
        LOG_ERROR(HW_GPU, "Failed to initialize GPU: {}", exception.what());
        return nullptr;
    }
}

} // namespace VideoCore