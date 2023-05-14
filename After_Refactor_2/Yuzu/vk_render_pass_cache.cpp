#pragma once

#include <unordered_map>

#include <boost/container/static_vector.hpp>

#include "RenderPassKey.h"
#include "VulkanDevice.h"
#include "VulkanWrapper.h"

namespace Vulkan {

class RenderPassCache {
public:
    explicit RenderPassCache(const Device& device);

    VkRenderPass Get(const RenderPassKey& key);

private:
    const Device* device;
    std::unordered_map<RenderPassKey, std::unique_ptr<VkRenderPass>> cache;
    std::mutex mutex;
};

} // namespace Vulkan
```

RenderPassCache.cpp

```
#include "RenderPassCache.h"
#include "MaxwellToVK.h"
#include "Surface.h"

namespace Vulkan {

RenderPassCache::RenderPassCache(const Device& device) : device(&device) {}

VkAttachmentDescription AttachmentDescription(const Device& device, PixelFormat format,
                                              VkSampleCountFlagBits samples) {
    // Returns a VkAttachmentDescription for the given format and sample count.
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

VkRenderPass RenderPassCache::Get(const RenderPassKey& key) {
    std::lock_guard<std::mutex> lock(mutex);

    const auto [pair, is_new] = cache.try_emplace(key);
    if (!is_new) {
        return *pair->second;
    }

    // Create VkAttachmentDescriptions for color and depth attachments.
    boost::container::static_vector<VkAttachmentDescription, 9> descriptions;
    std::array<VkAttachmentReference, 8> color_references{};
    u32 num_attachments_{};
    u32 num_colors{};
    for (size_t index = 0; index < key.color_formats.size(); ++index) {
        const PixelFormat format{key.color_formats[index]};
        const bool is_valid{format != PixelFormat::Invalid};
        color_references[index] = VkAttachmentReference{
            .attachment = is_valid ? num_colors : VK_ATTACHMENT_UNUSED,
            .layout = VK_IMAGE_LAYOUT_GENERAL,
        };
        if (is_valid) {
            descriptions.push_back(AttachmentDescription(*device, format, key.samples));
            num_attachments_ = static_cast<u32>(index + 1);
            ++num_colors;
        }
    }
    const bool has_depth{key.depth_format != PixelFormat::Invalid};
    VkAttachmentReference depth_reference{};
    if (key.depth_format != PixelFormat::Invalid) {
        depth_reference = VkAttachmentReference{
            .attachment = num_colors,
            .layout = VK_IMAGE_LAYOUT_GENERAL,
        };
        descriptions.push_back(AttachmentDescription(*device, key.depth_format, key.samples));
    }

    // Create a VkSubpassDescription for the color and/or depth attachments.
    const VkSubpassDescription subpass{
        .flags = 0,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = num_attachments_,
        .pColorAttachments = color_references.data(),
        .pResolveAttachments = nullptr,
        .pDepthStencilAttachment = has_depth ? &depth_reference : nullptr,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = nullptr,
    };

    // Create a VkRenderPass using the created attachments and subpass.
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

} // namespace Vulkan