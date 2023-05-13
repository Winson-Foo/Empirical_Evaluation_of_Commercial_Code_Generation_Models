namespace Vulkan {

namespace {

// Helper function to return the required extensions for the given window system type
// and whether validation layers are enabled
std::vector<const char*> RequiredExtensions(Core::Frontend::WindowSystemType window_type,
                                            bool enable_validation) {
    std::vector<const char*> extensions;
    extensions.reserve(6);

    switch (window_type) {
        case Core::Frontend::WindowSystemType::Headless:
            break;
        #ifdef _WIN32
            case Core::Frontend::WindowSystemType::Windows:
                extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
                break;
        #elif defined(__APPLE__)
            case Core::Frontend::WindowSystemType::Cocoa:
                extensions.push_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
                break;
        #elif defined(__ANDROID__)
            case Core::Frontend::WindowSystemType::Android:
                extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
                break;
        #else
            case Core::Frontend::WindowSystemType::X11:
                extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
                break;
            case Core::Frontend::WindowSystemType::Wayland:
                extensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
                break;
        #endif
        default:
            LOG_ERROR(Render_Vulkan, "Presentation not supported on this platform");
            break;
    }

    if (window_type != Core::Frontend::WindowSystemType::Headless) {
        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    }

    if (enable_validation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    #ifdef __APPLE__
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    #endif

    return extensions;
}

// Helper function to check if the required instance extensions are supported
bool AreExtensionsSupported(const vk::InstanceDispatch& dld,
                            std::span<const char* const> extensions) {
    const auto [properties, result] = vk::EnumerateInstanceExtensionProperties(dld);
    if (result != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "Failed to query extension properties");
        return false;
    }

    for (const char* extension : extensions) {
        const auto it = std::ranges::find_if(properties, [&](const VkExtensionProperties& prop) {
            return std::strcmp(extension, prop.extensionName) == 0;
        });

        if (it == properties.end()) {
            LOG_ERROR(Render_Vulkan, "Required instance extension {} is not available", extension);
            return false;
        }
    }

    return true;
}

// Helper function to return the validation layers to be enabled
std::vector<const char*> Layers(bool enable_validation) {
    std::vector<const char*> layers;

    if (enable_validation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    return layers;
}

// Helper function to remove layers that are not available
void RemoveUnavailableLayers(const vk::InstanceDispatch& dld, std::vector<const char*>& layers) {
    const auto [layer_properties, result] = vk::EnumerateInstanceLayerProperties(dld);
    if (result != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "Failed to query layer properties, disabling layers");
        layers.clear();
    }

    std::erase_if(layers, [&](const char* layer) {
        const auto it = std::ranges::find_if(layer_properties, [&](const VkLayerProperties& prop) {
            return std::strcmp(layer, prop.layerName) == 0;
        });

        if (it == layer_properties.end()) {
            LOG_ERROR(Render_Vulkan, "Layer {} not available, removing it", layer);
            return true;
        }

        return false;
    });
}

} // Anonymous namespace

// Create a Vulkan instance with the given library, instance dispatch, Vulkan version,
// window system type and whether validation is enabled
vk::Instance CreateInstance(const Common::DynamicLibrary& library, vk::InstanceDispatch& dld,
                           u32 required_version, Core::Frontend::WindowSystemType window_type,
                           bool enable_validation) {

    if (!library.IsOpen()) {
        LOG_ERROR(Render_Vulkan, "Vulkan library not available");
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }

    if (!library.GetSymbol("vkGetInstanceProcAddr", &dld.vkGetInstanceProcAddr)) {
        LOG_ERROR(Render_Vulkan, "vkGetInstanceProcAddr not present in Vulkan");
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }

    if (!vk::Load(dld)) {
        LOG_ERROR(Render_Vulkan, "Failed to load Vulkan function pointers");
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }

    const auto required_extensions = RequiredExtensions(window_type, enable_validation);

    if (!AreExtensionsSupported(dld, required_extensions)) {
        throw vk::Exception(VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    auto layers = Layers(enable_validation);
    RemoveUnavailableLayers(dld, layers);

    const auto available_version = vk::AvailableVersion(dld);

    if (available_version < required_version) {
        LOG_ERROR(Render_Vulkan, "Vulkan {}.{} is not supported, {}.{} is required",
            VK_VERSION_MAJOR(available_version), VK_VERSION_MINOR(available_version),
            VK_VERSION_MAJOR(required_version), VK_VERSION_MINOR(required_version));
        throw vk::Exception(VK_ERROR_INCOMPATIBLE_DRIVER);
    }

    auto instance = vk::Instance::Create(available_version, layers, required_extensions, dld);

    if (!instance) {
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }

    if (!vk::Load(*instance, dld)) {
        LOG_ERROR(Render_Vulkan, "Failed to load Vulkan instance function pointers");
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }

    return instance;
}

} // namespace Vulkan