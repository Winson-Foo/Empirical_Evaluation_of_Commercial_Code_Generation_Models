// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <memory>
#include <optional>
#include <utility>

#include "common/assert.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "video_core/host_shaders/astc_decoder_comp_spv.h"
#include "video_core/host_shaders/vulkan_quad_indexed_comp_spv.h"
#include "video_core/host_shaders/vulkan_uint8_comp_spv.h"
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/texture_cache/accelerated_swizzle.h"
#include "video_core/texture_cache/types.h"
#include "video_core/textures/decoders.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

namespace {

constexpr u32 ASTC_BINDING_INPUT_BUFFER = 0;
constexpr u32 ASTC_BINDING_OUTPUT_IMAGE = 1;
constexpr size_t ASTC_NUM_BINDINGS = 2;

template <size_t size>
inline constexpr VkPushConstantRange COMPUTE_PUSH_CONSTANT_RANGE{
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    .offset = 0,
    .size = static_cast<u32>(size),
};

constexpr std::array<VkDescriptorSetLayoutBinding, 2> INPUT_OUTPUT_DESCRIPTOR_SET_BINDINGS{{
    {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    },
    {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    },
}};

constexpr DescriptorBankInfo INPUT_OUTPUT_BANK_INFO{
    .uniform_buffers = 0,
    .storage_buffers = 2,
    .texture_buffers = 0,
    .image_buffers = 0,
    .textures = 0,
    .images = 0,
    .score = 2,
};

constexpr std::array<VkDescriptorSetLayoutBinding, ASTC_NUM_BINDINGS> ASTC_DESCRIPTOR_SET_BINDINGS{{
    {
        .binding = ASTC_BINDING_INPUT_BUFFER,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    },
    {
        .binding = ASTC_BINDING_OUTPUT_IMAGE,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    },
}};

constexpr DescriptorBankInfo ASTC_BANK_INFO{
    .uniform_buffers = 0,
    .storage_buffers = 1,
    .texture_buffers = 0,
    .image_buffers = 0,
    .textures = 0,
    .images = 1,
    .score = 2,
};

constexpr VkDescriptorUpdateTemplateEntry INPUT_OUTPUT_DESCRIPTOR_UPDATE_TEMPLATE{
    .dstBinding = 0,
    .dstArrayElement = 0,
    .descriptorCount = 2,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .offset = 0,
    .stride = sizeof(DescriptorUpdateEntry),
};

constexpr std::array<VkDescriptorUpdateTemplateEntry, ASTC_NUM_BINDINGS>
    ASTC_PASS_DESCRIPTOR_UPDATE_TEMPLATE_ENTRY{{
        {
            .dstBinding = ASTC_BINDING_INPUT_BUFFER,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .offset = ASTC_BINDING_INPUT_BUFFER * sizeof(DescriptorUpdateEntry),
            .stride = sizeof(DescriptorUpdateEntry),
        },
        {
            .dstBinding = ASTC_BINDING_OUTPUT_IMAGE,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .offset = ASTC_BINDING_OUTPUT_IMAGE * sizeof(DescriptorUpdateEntry),
            .stride = sizeof(DescriptorUpdateEntry),
        },
    }};

struct AstcPushConstants {
    std::array<u32, 2> blocks_dims;
    u32 layer_stride;
    u32 block_size;
    u32 x_shift;
    u32 block_height;
    u32 block_height_mask;
};
} // Anonymous namespace

ComputePass::ComputePass(const Device& device_, DescriptorPool& descriptor_pool,
                         vk::Span<VkDescriptorSetLayoutBinding> bindings,
                         vk::Span<VkDescriptorUpdateTemplateEntry> templates,
                         const DescriptorBankInfo& bank_info,
                         vk::Span<VkPushConstantRange> push_constants, std::span<const u32> code)
    : device{device_} {
    descriptor_set_layout = device.GetLogical().CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = bindings.size(),
        .pBindings = bindings.data(),
    });
    layout = device.GetLogical().CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = descriptor_set_layout.address(),
        .pushConstantRangeCount = push_constants.size(),
        .pPushConstantRanges = push_constants.data(),
    });
    if (!templates.empty()) {
        descriptor_template = device.GetLogical().CreateDescriptorUpdateTemplate({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .descriptorUpdateEntryCount = templates.size(),
            .pDescriptorUpdateEntries = templates.data(),
            .templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET,
            .descriptorSetLayout = *descriptor_set_layout,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .pipelineLayout = *layout,
            .set = 0,
        });
        descriptor_allocator = descriptor_pool.Allocator(*descriptor_set_layout, bank_info);
    }
    module = device.GetLogical().CreateShaderModule({
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = static_cast<u32>(code.size_bytes()),
        .pCode = code.data(),
    });
    device.SaveShader(code);
    pipeline = device.GetLogical().CreateComputePipeline({
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *module,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        .layout = *layout,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = 0,
    });
}

ComputePass::~ComputePass() = default;

Uint8Pass::Uint8Pass(const Device& device_, Scheduler& scheduler_, DescriptorPool& descriptor_pool,
                     StagingBufferPool& staging_buffer_pool_,
                     ComputePassDescriptorQueue& compute_pass_descriptor_queue_)
    : ComputePass(device_, descriptor_pool, INPUT_OUTPUT_DESCRIPTOR_SET_BINDINGS,
                  INPUT_OUTPUT_DESCRIPTOR_UPDATE_TEMPLATE, INPUT_OUTPUT_BANK_INFO, {},
                  VULKAN_UINT8_COMP_SPV),
      scheduler{scheduler_}, staging_buffer_pool{staging_buffer_pool_},
      compute_pass_descriptor_queue{compute_pass_descriptor_queue_} {}

Uint8Pass::~Uint8Pass() = default;

std::pair<VkBuffer, VkDeviceSize> Uint8Pass::Assemble(u32 num_vertices, VkBuffer src_buffer,
                                                      u32 src_offset) {
    const u32 staging_size = static_cast<u32>(num_vertices * sizeof(u16));
    const auto staging = staging_buffer_pool.Request(staging_size, MemoryUsage::DeviceLocal);

    compute_pass_descriptor_queue.Acquire();
    compute_pass_descriptor_queue.AddBuffer(src_buffer, src_offset, num_vertices);
    compute_pass_descriptor_queue.AddBuffer(staging.buffer, staging.offset, staging_size);
    const void* const descriptor_data{compute_pass_descriptor_queue.UpdateData()};

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, descriptor_data, num_vertices](vk::CommandBuffer cmdbuf) {
        static constexpr u32 DISPATCH_SIZE = 1024;
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
        };
        const VkDescriptorSet set = descriptor_allocator.Commit();
        device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptor_data);
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
        cmdbuf.Dispatch(Common::DivCeil(num_vertices, DISPATCH_SIZE), 1, 1);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, WRITE_BARRIER);
    });
    return {staging.buffer, staging.offset};
}

QuadIndexedPass::QuadIndexedPass(const Device& device_, Scheduler& scheduler_,
                                 DescriptorPool& descriptor_pool_,
                                 StagingBufferPool& staging_buffer_pool_,
                                 ComputePassDescriptorQueue& compute_pass_descriptor_queue_)
    : ComputePass(device_, descriptor_pool_, INPUT_OUTPUT_DESCRIPTOR_SET_BINDINGS,
                  INPUT_OUTPUT_DESCRIPTOR_UPDATE_TEMPLATE, INPUT_OUTPUT_BANK_INFO,
                  COMPUTE_PUSH_CONSTANT_RANGE<sizeof(u32) * 3>, VULKAN_QUAD_INDEXED_COMP_SPV),
      scheduler{scheduler_}, staging_buffer_pool{staging_buffer_pool_},
      compute_pass_descriptor_queue{compute_pass_descriptor_queue_} {}

QuadIndexedPass::~QuadIndexedPass() = default;

std::pair<VkBuffer, VkDeviceSize> QuadIndexedPass::Assemble(
    Tegra::Engines::Maxwell3D::Regs::IndexFormat index_format, u32 num_vertices, u32 base_vertex,
    VkBuffer src_buffer, u32 src_offset, bool is_strip) {
    const u32 index_shift = [index_format] {
        switch (index_format) {
        case Tegra::Engines::Maxwell3D::Regs::IndexFormat::UnsignedByte:
            return 0;
        case Tegra::Engines::Maxwell3D::Regs::IndexFormat::UnsignedShort:
            return 1;
        case Tegra::Engines::Maxwell3D::Regs::IndexFormat::UnsignedInt:
            return 2;
        }
        ASSERT(false);
        return 2;
    }();
    const u32 input_size = num_vertices << index_shift;
    const u32 num_tri_vertices = (is_strip ? (num_vertices - 2) / 2 : num_vertices / 4) * 6;

    const std::size_t staging_size = num_tri_vertices * sizeof(u32);
    const auto staging = staging_buffer_pool.Request(staging_size, MemoryUsage::DeviceLocal);

    compute_pass_descriptor_queue.Acquire();
    compute_pass_descriptor_queue.AddBuffer(src_buffer, src_offset, input_size);
    compute_pass_descriptor_queue.AddBuffer(staging.buffer, staging.offset, staging_size);
    const void* const descriptor_data{compute_pass_descriptor_queue.UpdateData()};

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, descriptor_data, num_tri_vertices, base_vertex, index_shift,
                      is_strip](vk::CommandBuffer cmdbuf) {
        static constexpr u32 DISPATCH_SIZE = 1024;
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_INDEX_READ_BIT,
        };
        const std::array<u32, 3> push_constants{base_vertex, index_shift, is_strip ? 1u : 0u};
        const VkDescriptorSet set = descriptor_allocator.Commit();
        device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptor_data);
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
        cmdbuf.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants),
                             &push_constants);
        cmdbuf.Dispatch(Common::DivCeil(num_tri_vertices, DISPATCH_SIZE), 1, 1);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, WRITE_BARRIER);
    });
    return {staging.buffer, staging.offset};
}

ASTCDecoderPass::ASTCDecoderPass(const Device& device_, Scheduler& scheduler_,
                                 DescriptorPool& descriptor_pool_,
                                 StagingBufferPool& staging_buffer_pool_,
                                 ComputePassDescriptorQueue& compute_pass_descriptor_queue_,
                                 MemoryAllocator& memory_allocator_)
    : ComputePass(device_, descriptor_pool_, ASTC_DESCRIPTOR_SET_BINDINGS,
                  ASTC_PASS_DESCRIPTOR_UPDATE_TEMPLATE_ENTRY, ASTC_BANK_INFO,
                  COMPUTE_PUSH_CONSTANT_RANGE<sizeof(AstcPushConstants)>, ASTC_DECODER_COMP_SPV),
      scheduler{scheduler_}, staging_buffer_pool{staging_buffer_pool_},
      compute_pass_descriptor_queue{compute_pass_descriptor_queue_}, memory_allocator{
                                                                         memory_allocator_} {}

ASTCDecoderPass::~ASTCDecoderPass() = default;

void ASTCDecoderPass::Assemble(Image& image, const StagingBufferRef& map,
                               std::span<const VideoCommon::SwizzleParameters> swizzles) {
    using namespace VideoCommon::Accelerated;
    const std::array<u32, 2> block_dims{
        VideoCore::Surface::DefaultBlockWidth(image.info.format),
        VideoCore::Surface::DefaultBlockHeight(image.info.format),
    };
    scheduler.RequestOutsideRenderPassOperationContext();
    const VkPipeline vk_pipeline = *pipeline;
    const VkImageAspectFlags aspect_mask = image.AspectMask();
    const VkImage vk_image = image.Handle();
    const bool is_initialized = image.ExchangeInitialization();
    scheduler.Record([vk_pipeline, vk_image, aspect_mask,
                      is_initialized](vk::CommandBuffer cmdbuf) {
        const VkImageMemoryBarrier image_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = static_cast<VkAccessFlags>(is_initialized ? VK_ACCESS_SHADER_WRITE_BIT
                                                                       : VK_ACCESS_NONE),
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = is_initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = vk_image,
            .subresourceRange{
                .aspectMask = aspect_mask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        cmdbuf.PipelineBarrier(is_initialized ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
                                              : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, image_barrier);
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline);
    });
    for (const VideoCommon::SwizzleParameters& swizzle : swizzles) {
        const size_t input_offset = swizzle.buffer_offset + map.offset;
        const u32 num_dispatches_x = Common::DivCeil(swizzle.num_tiles.width, 8U);
        const u32 num_dispatches_y = Common::DivCeil(swizzle.num_tiles.height, 8U);
        const u32 num_dispatches_z = image.info.resources.layers;

        compute_pass_descriptor_queue.Acquire();
        compute_pass_descriptor_queue.AddBuffer(map.buffer, input_offset,
                                                image.guest_size_bytes - swizzle.buffer_offset);
        compute_pass_descriptor_queue.AddImage(image.StorageImageView(swizzle.level));
        const void* const descriptor_data{compute_pass_descriptor_queue.UpdateData()};

        // To unswizzle the ASTC data
        const auto params = MakeBlockLinearSwizzle2DParams(swizzle, image.info);
        ASSERT(params.origin == (std::array<u32, 3>{0, 0, 0}));
        ASSERT(params.destination == (std::array<s32, 3>{0, 0, 0}));
        ASSERT(params.bytes_per_block_log2 == 4);
        scheduler.Record([this, num_dispatches_x, num_dispatches_y, num_dispatches_z, block_dims,
                          params, descriptor_data](vk::CommandBuffer cmdbuf) {
            const AstcPushConstants uniforms{
                .blocks_dims = block_dims,
                .layer_stride = params.layer_stride,
                .block_size = params.block_size,
                .x_shift = params.x_shift,
                .block_height = params.block_height,
                .block_height_mask = params.block_height_mask,
            };
            const VkDescriptorSet set = descriptor_allocator.Commit();
            device.GetLogical().UpdateDescriptorSet(set, *descriptor_template, descriptor_data);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmdbuf.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, uniforms);
            cmdbuf.Dispatch(num_dispatches_x, num_dispatches_y, num_dispatches_z);
        });
    }
    scheduler.Record([vk_image, aspect_mask](vk::CommandBuffer cmdbuf) {
        const VkImageMemoryBarrier image_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = vk_image,
            .subresourceRange{
                .aspectMask = aspect_mask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, image_barrier);
    });
    scheduler.Finish();
}

} // namespace Vulkan
