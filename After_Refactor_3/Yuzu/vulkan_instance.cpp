namespace Vulkan {

namespace {

// Constants for extension and layer names
constexpr const char* kWin32SurfaceExtensionName = VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
constexpr const char* kMacOSSurfaceExtensionName = VK_MVK_MACOS_SURFACE_EXTENSION_NAME;
constexpr const char* kAndroidSurfaceExtensionName = VK_KHR_ANDROID_SURFACE_EXTENSION_NAME;
constexpr const char* kXlibSurfaceExtensionName = VK_KHR_XLIB_SURFACE_EXTENSION_NAME;
constexpr const char* kWaylandSurfaceExtensionName = VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME;
constexpr const char* kPlainSurfaceExtensionName = VK_KHR_SURFACE_EXTENSION_NAME;
constexpr const char* kDebugUtilsExtensionName = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
constexpr const char* kPhysicalDevicePropertiesExtensionName = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;

constexpr const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";

// Returns a vector of required extensions based on the window type and validation flag
std::vector<const char*> RequiredExtensions(
    Core::Frontend::WindowSystemType window_type, bool enable_validation) {
    std::vector<const char*> extensions;
    extensions.reserve(6);
    switch (window_type) {
        case Core::Frontend::WindowSystemType::Headless:
            break;
#ifdef _WIN32
        case Core::Frontend::WindowSystemType::Windows:
            extensions.push_back(kWin32SurfaceExtensionName);
            break;
#elif defined(__APPLE__)
        case Core::Frontend::WindowSystemType::Cocoa:
            extensions.push_back(kMacOSSurfaceExtensionName);
            break;
#elif defined(__ANDROID__)
        case Core::Frontend::WindowSystemType::Android:
            extensions.push_back(kAndroidSurfaceExtensionName);
            break;
#else
        case Core::Frontend::WindowSystemType::X11:
            extensions.push_back(kXlibSurfaceExtensionName);
            break;
        case Core::Frontend::WindowSystemType::Wayland:
            extensions.push_back(kWaylandSurfaceExtensionName);
            break;
#endif
        default:
            LOG_ERROR(Render_Vulkan, "Presentation not supported on this platform");
            break;
    }
    if (window_type != Core::Frontend::WindowSystemType::Headless) {
        extensions.push_back(kPlainSurfaceExtensionName);
    }
    if (enable_validation) {
        extensions.push_back(kDebugUtilsExtensionName);
    }
    extensions.push_back(kPhysicalDevicePropertiesExtensionName);

#ifdef __APPLE__
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif
    return extensions;
}

// Checks if all required extensions are supported by the Vulkan instance
[[nodiscard]] bool AreExtensionsSupported(const vk::InstanceDispatch& dld, std::span<const char* const> extensions) {
    const std::optional properties = vk::EnumerateInstanceExtensionProperties(dld);
    if (!properties) {
        LOG_ERROR(Render_Vulkan, "Failed to query extension properties");
        return false;
    }
    for (const char* extension : extensions) {
        const auto it = std::find_if(properties->begin(), properties->end(), [&](const VkExtensionProperties& prop) {
            return std::strcmp(extension, prop.extensionName) == 0;
        });
        if (it == properties->end()) {
            LOG_ERROR(Render_Vulkan, "Required instance extension {} is not available", extension);
            return false;
        }
    }
    return true;
}

// Returns a vector of enabled layers based on the validation flag
[[nodiscard]] std::vector<const char*> Layers(bool enable_validation) {
    std::vector<const char*> layers;
    if (enable_validation) {
        layers.push_back(kValidationLayerName);
    }
    return layers;
}

// Removes any unavailable layers from the input vector
void RemoveUnavailableLayers(const vk::InstanceDispatch& dld, std::vector<const char*>& layers) {
    const std::optional layer_properties = vk::EnumerateInstanceLayerProperties(dld);
    if (!layer_properties) {
        LOG_ERROR(Render_Vulkan, "Failed to query layer properties, disabling layers");
        layers.clear();
    }
    std::erase_if(layers, [&](const char* layer) {
        const auto it = std::find_if(layer_properties->begin(), layer_properties->end(), [&](const VkLayerProperties& prop) {
            return std::strcmp(layer, prop.layerName) == 0;
        });
        if (it == layer_properties->end()) {
            LOG_ERROR(Render_Vulkan, "Layer {} not available, removing it", layer);
            return true;
        }
        return false;
    });
}

}

// Creates a Vulkan instance with the specified properties
vk::Instance CreateInstance(const Common::DynamicLibrary& library, vk::InstanceDispatch& dld,
    u32 required_version, Core::Frontend::WindowSystemType window_type, bool enable_validation) {

    // Ensure the Vulkan library is available
    if (!library.IsOpen()) {
        LOG_ERROR(Render_Vulkan, "Vulkan library not available");
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }

    // Load Vulkan function pointers
    if (!library.GetSymbol("vkGetInstanceProcAddr", &dld.vkGetInstanceProcAddr)) {
        LOG_ERROR(Render_Vulkan, "vkGetInstanceProcAddr not present in Vulkan");
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }
    if (!vk::Load(dld)) {
        LOG_ERROR(Render_Vulkan, "Failed to load Vulkan function pointers");
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }

    // Check for required extensions and layers
    const std::vector<const char*> extensions = RequiredExtensions(window_type, enable_validation);
    if (!AreExtensionsSupported(dld, extensions)) {
        throw vk::Exception(VK_ERROR_EXTENSION_NOT_PRESENT);
    }
    std::vector<const char*> layers = Layers(enable_validation);
    RemoveUnavailableLayers(dld, layers);

    // Check for required Vulkan version
    const u32 available_version = vk::AvailableVersion(dld);
    if (available_version < required_version) {
        LOG_ERROR(Render_Vulkan, "Vulkan {}.{} is not supported, {}.{} is required",
            VK_VERSION_MAJOR(available_version), VK_VERSION_MINOR(available_version),
            VK_VERSION_MAJOR(required_version), VK_VERSION_MINOR(required_version));
        throw vk::Exception(VK_ERROR_INCOMPATIBLE_DRIVER);
    }

    // Create and load the Vulkan instance
    vk::Instance instance =
        std::async([&] {
            return vk::Instance::Create(available_version, layers, extensions, dld);
        }).get();
    if (!vk::Load(*instance, dld)) {
        LOG_ERROR(Render_Vulkan, "Failed to load Vulkan instance function pointers");
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }
    return instance;
}

} // namespace Vulkan