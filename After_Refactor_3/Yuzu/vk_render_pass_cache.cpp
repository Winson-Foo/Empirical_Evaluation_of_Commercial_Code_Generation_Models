#include <unordered_map>
#include <boost/container/static_vector.hpp>

#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

using namespace Vulkan;
using namespace MaxwellToVK;
using VideoCore::Surface;
using VideoCore::VulkanCommon;

namespace {
    constexpr VkImageLayout kImageLayoutGeneral = VK_IMAGE_LAYOUT_GENERAL;
    constexpr VkAttachmentLoadOp kAttachmentLoadOpLoad = VK_ATTACHMENT_LOAD_OP_LOAD;
    constexpr VkAttachmentStoreOp kAttachmentStoreOpStore = VK_ATTACHMENT_STORE_OP_STORE;
    constexpr VkPipelineBindPoint kPipelineBindPointGraphics = VK_PIPELINE_BIND_POINT_GRAPHICS;
    constexpr u32 kMaxColorAttachments = 8;
}

RenderPassCache::RenderPassCache(const VulkanCommon::Device& device_)
    : device{&device_}
{}

VkRenderPass RenderPassCache::Get(const RenderPassKey& key) {
    std::scoped_lock lock{mutex};
    const auto [pair, is_new] = cache.try_emplace(key);
    if (!is_new) {
        return *pair->second;
    }
    boost::container::static_vector<VkAttachmentDescription, kMaxColorAttachments + 1> descriptions;
    std::array<VkAttachmentReference, kMaxColorAttachments> color_attachment_refs{};
    u32 num_attachments{};
    u32 num_colors{};
    for (size_t index = 0; index < key.color_formats.size(); ++index) {
        const Surface::PixelFormat format{key.color_formats[index]};
        const bool is_valid{format != Surface::PixelFormat::Invalid};
        color_attachment_refs[index] = VkAttachmentReference{
            .attachment = is_valid ? num_colors : VK_ATTACHMENT_UNUSED,
            .layout = kImageLayoutGeneral,
        };
        if (is_valid) {
            descriptions.push_back({
                .flags = VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT,
                .format = SurfaceFormat(*device, FormatType::Optimal, true, format).format,
                .samples = key.samples,
                .loadOp = kAttachmentLoadOpLoad,
                .storeOp = kAttachmentStoreOpStore,
                .stencilLoadOp = kAttachmentLoadOpLoad,
                .stencilStoreOp = kAttachmentStoreOpStore,
                .initialLayout = kImageLayoutGeneral,
                .finalLayout = kImageLayoutGeneral,
            });
            num_attachments = static_cast<u32>(index + 1);
            ++num_colors;
        }
    }
    const bool has_depth{key.depth_format != Surface::PixelFormat::Invalid};
    VkAttachmentReference depth_attachment_ref{};
    if (has_depth) {
        depth_attachment_ref = VkAttachmentReference{
            .attachment = num_colors,
            .layout = kImageLayoutGeneral,
        };
        descriptions.push_back({
            .flags = VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT,
            .format = SurfaceFormat(*device, FormatType::Optimal, true, key.depth_format).format,
            .samples = key.samples,
            .loadOp = kAttachmentLoadOpLoad,
            .storeOp = kAttachmentStoreOpStore,
            .stencilLoadOp = kAttachmentLoadOpLoad,
            .stencilStoreOp = kAttachmentStoreOpStore,
            .initialLayout = kImageLayoutGeneral,
            .finalLayout = kImageLayoutGeneral,
        });
    }
    const VkSubpassDescription subpass{
        .flags = 0,
        .pipelineBindPoint = kPipelineBindPointGraphics,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = num_attachments,
        .pColorAttachments = color_attachment_refs.data(),
        .pResolveAttachments = nullptr,
        .pDepthStencilAttachment = has_depth ? &depth_attachment_ref : nullptr,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = nullptr,
    };
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