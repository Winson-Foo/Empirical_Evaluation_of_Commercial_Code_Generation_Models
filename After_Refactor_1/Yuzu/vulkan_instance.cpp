namespace Vulkan {

namespace {

const std::vector<const char*> kDefaultExtensions = {VK_KHR_SURFACE_EXTENSION_NAME,
                                                     VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};

const std::vector<const char*> kDefaultValidationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef __APPLE__
const std::vector<const char*> kAppleExtensions = {VK_KHR_SURFACE_EXTENSION_NAME,
                                                   VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
                                                   VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
#else
const std::vector<const char*> kWindowsExtensions = {VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
const std::vector<const char*> kCocoaExtensions = {VK_MVK_MACOS_SURFACE_EXTENSION_NAME};
const std::vector<const char*> kAndroidExtensions = {VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
const std::vector<const char*> kLinuxX11Extensions = {VK_KHR_XLIB_SURFACE_EXTENSION_NAME};
const std::vector<const char*> kLinuxWaylandExtensions = {VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME};
#endif

std::vector<const char*> GetPlatformExtensions(Core::Frontend::WindowSystemType window_system) {
  switch (window_system) {
    case Core::Frontend::WindowSystemType::Headless:
      return {};
#ifdef _WIN32
    case Core::Frontend::WindowSystemType::Windows:
      return kDefaultExtensions + kWindowsExtensions;
#elif defined(__APPLE__)
    case Core::Frontend::WindowSystemType::Cocoa:
      return kDefaultExtensions + kCocoaExtensions;
#elif defined(__ANDROID__)
    case Core::Frontend::WindowSystemType::Android:
      return kDefaultExtensions + kAndroidExtensions;
#else
    case Core::Frontend::WindowSystemType::X11:
      return kDefaultExtensions + kLinuxX11Extensions;
    case Core::Frontend::WindowSystemType::Wayland:
      return kDefaultExtensions + kLinuxWaylandExtensions;
#endif
    default:
      LOG_ERROR(Render_Vulkan, "Presentation not supported on this platform");
      return {};
  }
}

std::vector<const char*> GetValidationLayers(bool enable_validation) {
  return enable_validation ? kDefaultValidationLayers : std::vector<const char*>{};
}

bool CheckRequiredExtensionsSupported(const std::vector<vk::ExtensionProperties>& available_extensions,
                                      std::span<const char* const> required_extensions) {
  for (const char* extension : required_extensions) {
    const auto it = std::ranges::find_if(available_extensions, [extension](const auto& prop) {
      return std::strcmp(extension, prop.extensionName) == 0;
    });
    if (it == available_extensions.end()) {
      LOG_ERROR(Render_Vulkan, "Required instance extension {} is not available", extension);
      return false;
    }
  }
  return true;
}

std::vector<const char*> FilterUnavailableLayers(const std::vector<vk::LayerProperties>& available_layers,
                                                 std::vector<const char*> layers) {
  std::erase_if(layers, [&available_layers](const char* layer_name) {
    const auto comp = [layer_name](const vk::LayerProperties& layer_prop) {
      return std::strcmp(layer_name, layer_prop.layerName) == 0;
    };
    return std::ranges::find_if(available_layers, comp) == available_layers.end();
  });
  return layers;
}
}  // namespace

vk::Instance CreateInstance(const Common::DynamicLibrary& library, vk::InstanceDispatch& dispatch,
                            u32 required_version, Core::Frontend::WindowSystemType window_system,
                            bool enable_validation) {
  if (!library.IsOpen()) {
    LOG_ERROR(Render_Vulkan, "Vulkan library not available");
    throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
  }

  if (!library.GetSymbol("vkGetInstanceProcAddr", &dispatch.vkGetInstanceProcAddr)) {
    LOG_ERROR(Render_Vulkan, "vkGetInstanceProcAddr not present in Vulkan");
    throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
  }

  if (!vk::Load(dispatch)) {
    LOG_ERROR(Render_Vulkan, "Failed to load Vulkan function pointers");
    throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
  }

  const auto required_extensions = GetPlatformExtensions(window_system);
  const auto required_layers = GetValidationLayers(enable_validation);

  if (!CheckRequiredExtensionsSupported(vk::EnumerateInstanceExtensionProperties(dispatch), required_extensions)) {
    throw vk::Exception(VK_ERROR_EXTENSION_NOT_PRESENT);
  }

  std::vector<vk::LayerProperties> available_layers;
  if (!vk::EnumerateInstanceLayerProperties(dispatch, &available_layers) ||
      available_layers.empty()) {
    LOG_ERROR(Render_Vulkan, "Failed to query layer properties, disabling layers");
    required_layers.clear();
  } else {
    required_layers = FilterUnavailableLayers(available_layers, std::move(required_layers));
  }

  const auto available_version = vk::AvailableVersion(dispatch);
  if (available_version < required_version) {
    LOG_ERROR(Render_Vulkan, "Vulkan {}.{} is not supported, {}.{} is required",
              VK_VERSION_MAJOR(available_version), VK_VERSION_MINOR(available_version),
              VK_VERSION_MAJOR(required_version), VK_VERSION_MINOR(required_version));
    throw vk::Exception(VK_ERROR_INCOMPATIBLE_DRIVER);
  }

  const vk::InstanceCreateInfo create_info{
      .enabledLayerNames = required_layers,
      .enabledExtensionNames = required_extensions,
      .apiVersion = required_version,
  };

  auto instance = vk::Instance::Create(create_info, nullptr, dispatch);
  if (!instance) {
    LOG_ERROR(Render_Vulkan, "Failed to create Vulkan instance");
    throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
  }

  if (!vk::Load(*instance, dispatch)) {
    LOG_ERROR(Render_Vulkan, "Failed to load Vulkan instance function pointers");
    throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
  }

  return instance;
}

}  // namespace Vulkan