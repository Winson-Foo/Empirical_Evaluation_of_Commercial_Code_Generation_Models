// constants.h

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "video_core/surface.h"

namespace VideoCommon::Accelerated {

namespace Constants {

    const u32 GOB_SIZE_SHIFT = Tegra::Texture::GOB_SIZE_SHIFT;
    const u32 GOB_SIZE_X = Tegra::Texture::GOB_SIZE_X;
    const u32 GOB_SIZE_X_SHIFT = Tegra::Texture::GOB_SIZE_X_SHIFT;
    
    inline u32 BytesPerBlock(const VideoCore::Surface::Format format) {
        return VideoCore::Surface::BytesPerBlock(format);
    }

    inline u32 CalculateLevelStrideAlignment(const ImageInfo& info, const u32 level) {
        return VideoCore::Texture::CalculateLevelStrideAlignment(info, level);
    }

    inline u32 DivCeilLog2(const u32 val, const u32 log2_denom) {
        return Common::DivCeilLog2(val, log2_denom);
    }

    inline u32 AlignUpLog2(const u32 val, const u32 log2_alignment) {
        return Common::AlignUpLog2(val, log2_alignment);
    }

} // namespace Constants

} // namespace VideoCommon::Accelerated

#endif // CONSTANTS_H


// swizzle_params.h

#ifndef SWIZZLE_PARAMS_H
#define SWIZZLE_PARAMS_H

namespace VideoCommon::Accelerated {

struct SwizzleParams {
    Extent3D block;
    Extent3D num_tiles;
    u32 level;
};

struct BlockLinearSwizzle2DParams {
    Extent3D origin;
    Extent3D destination;
    u32 bytes_per_block_log2;
    u32 layer_stride;
    u32 block_size;
    u32 x_shift;
    u32 block_height;
    u32 block_height_mask;
};

struct BlockLinearSwizzle3DParams {
    Extent3D origin;
    Extent3D destination;
    u32 bytes_per_block_log2;
    u32 slice_size;
    u32 block_size;
    u32 x_shift;
    u32 block_height;
    u32 block_height_mask;
    u32 block_depth;
    u32 block_depth_mask;
};

BlockLinearSwizzle2DParams makeBlockLinearSwizzleParams2D(const SwizzleParams& swizzle,
                                                          const ImageInfo& info) {
    const Extent3D block = swizzle.block;
    const Extent3D numTiles = swizzle.num_tiles;
    
    const u32 bytesPerBlock = Constants::BytesPerBlock(info.format);
    const u32 strideAlignment = Constants::CalculateLevelStrideAlignment(info, swizzle.level);
    const u32 stride = Constants::AlignUpLog2(numTiles.width, strideAlignment) * bytesPerBlock;
    const u32 gobsInX = Constants::DivCeilLog2(stride, Constants::GOB_SIZE_X_SHIFT);
    
    return {
        .origin = {0, 0, 0},
        .destination = {0, 0, 0},
        .bytes_per_block_log2 = static_cast<u32>(std::countr_zero(bytesPerBlock)),
        .layer_stride = info.layer_stride,
        .block_size = gobsInX << (Constants::GOB_SIZE_SHIFT + block.height + block.depth),
        .x_shift = Constants::GOB_SIZE_SHIFT + block.height + block.depth,
        .block_height = block.height,
        .block_height_mask = (1U << block.height) - 1,
    };
}

BlockLinearSwizzle3DParams makeBlockLinearSwizzleParams3D(const SwizzleParams& swizzle,
                                                          const ImageInfo& info) {
    const Extent3D block = swizzle.block;
    const Extent3D numTiles = swizzle.num_tiles;
    
    const u32 bytesPerBlock = Constants::BytesPerBlock(info.format);
    const u32 strideAlignment = Constants::CalculateLevelStrideAlignment(info, swizzle.level);
    const u32 stride = Constants::AlignUpLog2(numTiles.width, strideAlignment) * bytesPerBlock;
    
    const u32 gobsInX = (stride + Constants::GOB_SIZE_X - 1) >> Constants::GOB_SIZE_X_SHIFT;
    const u32 blockSize = gobsInX << (Constants::GOB_SIZE_SHIFT + block.height + block.depth);
    const u32 sliceSize = Constants::DivCeilLog2(numTiles.height, block.height + Constants::GOB_SIZE_Y_SHIFT) * blockSize;
    
    return {
        .origin = {0, 0, 0},
        .destination = {0, 0, 0},
        .bytes_per_block_log2 = static_cast<u32>(std::countr_zero(bytesPerBlock)),
        .slice_size = sliceSize,
        .block_size = blockSize,
        .x_shift = Constants::GOB_SIZE_SHIFT + block.height + block.depth,
        .block_height = block.height,
        .block_height_mask = (1U << block.height) - 1,
        .block_depth = block.depth,
        .block_depth_mask = (1U << block.depth) - 1,
    };
}

} // namespace VideoCommon::Accelerated

#endif // SWIZZLE_PARAMS_H

