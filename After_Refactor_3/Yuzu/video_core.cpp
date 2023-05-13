namespace {

std::unique_ptr<VideoCore::RendererBase> CreateOpenGLRenderer(Core::System& system, Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu, std::unique_ptr<Core::Frontend::GraphicsContext> context) {
    auto& telemetry_session = system.TelemetrySession();
    auto& cpu_memory = system.ApplicationMemory();
    return std::make_unique<OpenGL::RendererOpenGL>(telemetry_session, emu_window, cpu_memory, gpu, std::move(context));
}

std::unique_ptr<VideoCore::RendererBase> CreateVulkanRenderer(Core::System& system, Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu, std::unique_ptr<Core::Frontend::GraphicsContext> context) {
    auto& telemetry_session = system.TelemetrySession();
    auto& cpu_memory = system.ApplicationMemory();
    return std::make_unique<Vulkan::RendererVulkan>(telemetry_session, emu_window, cpu_memory, gpu, std::move(context));
}

std::unique_ptr<VideoCore::RendererBase> CreateNullRenderer(Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu, std::unique_ptr<Core::Frontend::GraphicsContext> context) {
    auto& cpu_memory = system.ApplicationMemory();
    return std::make_unique<Null::RendererNull>(emu_window, cpu_memory, gpu, std::move(context));
}

std::unique_ptr<VideoCore::RendererBase> CreateRenderer(Core::System& system, Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu, std::unique_ptr<Core::Frontend::GraphicsContext> context) {
    switch (Settings::values.renderer_backend.GetValue()) {
        case Settings::RendererBackend::OpenGL:
            return CreateOpenGLRenderer(system, emu_window, gpu, std::move(context));
        case Settings::RendererBackend::Vulkan:
            return CreateVulkanRenderer(system, emu_window, gpu, std::move(context));
        case Settings::RendererBackend::Null:
            return CreateNullRenderer(emu_window, gpu, std::move(context));
        default:
            return nullptr;
    }
}

} // Anonymous namespace

namespace VideoCore {

std::unique_ptr<Tegra::GPU> CreateGPU(Core::Frontend::EmuWindow& emu_window, Core::System& system) {
    Settings::UpdateRescalingInfo();

    const auto nvdec_value = Settings::values.nvdec_emulation.GetValue();
    const bool use_nvdec = nvdec_value != Settings::NvdecEmulation::Off;
    const bool use_async = Settings::values.use_asynchronous_gpu_emulation.GetValue();
    auto gpu = std::make_unique<Tegra::GPU>(system, use_async, use_nvdec);
    auto context = emu_window.CreateSharedContext();
    auto scope = context->Acquire();
    try {
        auto renderer = CreateRenderer(system, emu_window, *gpu, std::move(context));
        gpu->BindRenderer(std::move(renderer));
        return gpu;
    } catch (const std::runtime_error& exception) {
        scope.Cancel();
        LOG_ERROR(HW_GPU, "Failed to initialize GPU: {}", exception.what());
        return nullptr;
    }
}

} // namespace VideoCore