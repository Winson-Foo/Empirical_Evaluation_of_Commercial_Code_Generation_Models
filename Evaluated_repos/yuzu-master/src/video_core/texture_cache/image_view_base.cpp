// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "common/assert.h"
#include "video_core/compatible_formats.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/image_view_info.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

ImageViewBase::ImageViewBase(const ImageViewInfo& info, const ImageInfo& image_info,
                             ImageId image_id_, GPUVAddr addr)
    : image_id{image_id_}, gpu_addr{addr}, format{info.format}, type{info.type}, range{info.range},
      size{
          .width = std::max(image_info.size.width >> range.base.level, 1u),
          .height = std::max(image_info.size.height >> range.base.level, 1u),
          .depth = std::max(image_info.size.depth >> range.base.level, 1u),
      } {
    ASSERT_MSG(VideoCore::Surface::IsViewCompatible(image_info.format, info.format, false, true),
               "Image view format {} is incompatible with image format {}", info.format,
               image_info.format);
    if (image_info.forced_flushed) {
        flags |= ImageViewFlagBits::PreemtiveDownload;
    }
    if (image_info.type == ImageType::e3D && info.type != ImageViewType::e3D) {
        flags |= ImageViewFlagBits::Slice;
    }
}

ImageViewBase::ImageViewBase(const ImageInfo& info, const ImageViewInfo& view_info, GPUVAddr addr)
    : image_id{NULL_IMAGE_ID}, gpu_addr{addr}, format{info.format}, type{ImageViewType::Buffer},
      size{
          .width = info.size.width,
          .height = 1,
          .depth = 1,
      } {
    ASSERT_MSG(view_info.type == ImageViewType::Buffer, "Expected texture buffer");
}

ImageViewBase::ImageViewBase(const NullImageViewParams&) : image_id{NULL_IMAGE_ID} {}

} // namespace VideoCommon
