#include <unordered_map>
#include <boost/container/static_vector.hpp>
#include <video_core/renderer_vulkan/maxwell_to_vk.h>
#include <video_core/renderer_vulkan/vk_render_pass_cache.h>
#include <video_core/surface.h>
#include <video_core/vulkan_common/vulkan_device.h>
#include <video_core/vulkan_common/vulkan_wrapper.h>

namespace Vulkan {

namespace {

using VideoCore::Surface::PixelFormat;

constexpr VkAttachmentDescription AttachmentDescription(const Device& device, PixelFormat format,
                                                         VkSampleCountFlagBits samples) {
    using MaxwellToVK::SurfaceFormat;
    return {
        .flags = VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT,
        .format = SurfaceFormat(device, FormatType::Optimal, true, format).format,
        .samples = samples,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
        .initialLayout = VK_IMAGE_LAYOUT_GENERAL,
        .finalLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
}

constexpr VkSubpassDescription CreateSubpassDescription(u32 num_attachments, const VkAttachmentReference* refs, 
                                                         bool has_depth, const VkAttachmentReference* depth_ref) {
    return {
        .flags = 0,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = num_attachments,
        .pColorAttachments = refs,
        .pResolveAttachments = nullptr,
        .pDepthStencilAttachment = has_depth ? depth_ref : nullptr,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = nullptr,
    };
}

void CreateAttachmentDescriptions(const Device& device, const RenderPassKey& key, 
                                  boost::container::static_vector<VkAttachmentDescription, 9>& descriptions, 
                                  std::array<VkAttachmentReference, 8>& refs, u32& num_attachments, u32& num_colors) {
    for (size_t index = 0; index < key.color_formats.size(); ++index) {
        const PixelFormat format{key.color_formats[index]};
        const bool is_valid{format != PixelFormat::Invalid};
        refs[index] = VkAttachmentReference{
            .attachment = is_valid ? num_colors : VK_ATTACHMENT_UNUSED,
            .layout = VK_IMAGE_LAYOUT_GENERAL,
        };
        if (is_valid) {
            descriptions.push_back(AttachmentDescription(device, format, key.samples));
            num_attachments = static_cast<u32>(index + 1);
            ++num_colors;
        }
    }
    if (key.depth_format != PixelFormat::Invalid) {
        refs[num_colors] = VkAttachmentReference{
            .attachment = num_colors,
            .layout = VK_IMAGE_LAYOUT_GENERAL,
        };
        descriptions.push_back(AttachmentDescription(device, key.depth_format, key.samples));
        ++num_attachments;
    }
}

} // end of anonymous namespace

RenderPassCache::RenderPassCache(const Device& device_) : device{&device_} {}

VkRenderPass RenderPassCache::Get(const RenderPassKey& key) {
    std::scoped_lock lock{mutex};
    const auto [pair, is_new] = cache.try_emplace(key);
    if (!is_new) {
        return *pair->second;
    }
    boost::container::static_vector<VkAttachmentDescription, 9> descriptions;
    std::array<VkAttachmentReference, 8> refs{};
    u32 num_attachments{};
    u32 num_colors{};
    CreateAttachmentDescriptions(*device, key, descriptions, refs, num_attachments, num_colors);
    const VkSubpassDescription subpass = CreateSubpassDescription(num_attachments, refs.data(), 
                                                                  key.depth_format != PixelFormat::Invalid, 
                                                                  &refs[num_colors]);
    pair->second = device->GetLogical().CreateRenderPass({
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .attachmentCount = static_cast<u32>(descriptions.size()),
        .pAttachments = descriptions.empty() ? nullptr : descriptions.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 0,
        .pDependencies = nullptr,
    });
    return *pair->second;
}

} // end of namespace Vulkan