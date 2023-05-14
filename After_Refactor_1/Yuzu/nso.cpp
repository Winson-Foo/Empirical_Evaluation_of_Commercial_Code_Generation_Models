// SPDX-License-Identifier: GPL-2.0-or-later

#include <cinttypes>
#include <cstring>
#include <vector>

#include "common/common_funcs.h"
#include "common/hex_util.h"
#include "common/logging/log.h"
#include "common/lz4_compression.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/loader/nso.h"
#include "core/memory.h"

namespace Loader {
    namespace {
        struct NSOSegment {
            u32_le offset;
            u32_le size;
            u32_le location;
        };

        struct NSOHeader {
            u32_le magic;
            u32_le flags;
            u32_le text_offset;
            u32_le text_size;
            u32_le rodata_offset;
            u32_le rodata_size;
            u32_le data_offset;
            u32_le data_size;
            u32_le bss_size;
            u32_le eh_frame_hdr_table_offset;
            u32_le eh_frame_hdr_table_size;
            u8 build_id[0x20];
            NSOSegment segments[3];
            u32_le segments_compressed_size[3];
        };
        static_assert(sizeof(NSOHeader) == 0x80, "NSOHeader has incorrect size.");

        struct NSOArgumentHeader {
            u32_le allocation_size;
            u32_le data_size;
            char padding[0x20 - sizeof(u32_le) - sizeof(u32_le)];
        };
        static_assert(sizeof(NSOArgumentHeader) == 0x20, "NSOArgumentHeader has incorrect size.");

        struct MODHeader {
            u32_le magic;
            u32_le dynamic_offset;
            u32_le bss_start_offset;
            u32_le bss_end_offset;
            u32_le eh_frame_hdr_start_offset;
            u32_le eh_frame_hdr_end_offset;
            u32_le module_offset;
        };
        static_assert(sizeof(MODHeader) == 0x1c, "MODHeader has incorrect size.");

        std::vector<u8> decompressSegment(const std::vector<u8>& compressedData, const NSOSegment& segment) {
            std::vector<u8> uncompressedData = Common::Compression::DecompressDataLZ4(compressedData, segment.size);
            ASSERT_MSG(uncompressedData.size() == segment.size, "{} != {}", segment.size, uncompressedData.size());
            return uncompressedData;
        }

        constexpr u32 pageAlignSize(u32 size) {
            return static_cast<u32>((size + Core::Memory::YUZU_PAGEMASK) & ~Core::Memory::YUZU_PAGEMASK);
        }
    }

    AppLoader_NSO::AppLoader_NSO(FileSys::VirtualFile file_) : AppLoader(std::move(file_)) {}

    FileType AppLoader_NSO::IdentifyType(const FileSys::VirtualFile& in_file) {
        u32 magic = 0;
        if (in_file->ReadObject(&magic) != sizeof(magic)) {
            return FileType::Error;
        }

        if (Common::MakeMagic('N', 'S', 'O', '0') != magic) {
            return FileType::Error;
        }

        return FileType::NSO;
    }

    std::optional<VAddr> AppLoader_NSO::LoadModule(Kernel::KProcess& process, Core::System& system, const FileSys::VfsFile& nsoFile, VAddr loadBase, bool shouldPassArguments, bool loadIntoProcess) {
        if (nsoFile.GetSize() < sizeof(NSOHeader)) {
            return std::nullopt;
        }

        NSOHeader nsoHeader{};
        if (sizeof(NSOHeader) != nsoFile.ReadObject(&nsoHeader)) {
            return std::nullopt;
        }

        if (nsoHeader.magic != Common::MakeMagic('N', 'S', 'O', '0')) {
            return std::nullopt;
        }

        // Build program image
        std::vector<u8> programImage;
        Kernel::CodeSet codeSet;

        for (std::size_t i = 0; i < 3; ++i) {
            NSOSegment& segment = nsoHeader.segments[i];

            std::vector<u8> data = nsoFile.ReadBytes(nsoHeader.segments_compressed_size[i], segment.offset);
            if (nsoHeader.flags & (1u << i)) {
                data = decompressSegment(data, segment);
            }

            const u32 location = segment.location;
            programImage.resize(location + static_cast<u32>(data.size()));
            std::memcpy(programImage.data() + location, data.data(), data.size());

            codeSet.segments[i].addr = location;
            codeSet.segments[i].offset = location;
            codeSet.segments[i].size = segment.size;
        }

        if (shouldPassArguments && !Settings::values.program_args.GetValue().empty()) {
            const auto argData = Settings::values.program_args.GetValue();

            const u32 argumentDataAllocationSize = NSO_ARGUMENT_DATA_ALLOCATION_SIZE;
            const u32 argumentDataSize = static_cast<u32>(argData.size());
            const u32 argumentDataOffset = programImage.size();
            programImage.resize(programImage.size() + argumentDataAllocationSize);
            std::memset(programImage.data() + argumentDataOffset, 0, argumentDataAllocationSize);
            std::memcpy(programImage.data() + argumentDataOffset, &NSOArgumentHeader{ argumentDataAllocationSize, argumentDataSize }, sizeof(NSOArgumentHeader));
            std::memcpy(programImage.data() + argumentDataOffset + sizeof(NSOArgumentHeader), argData.data(), argumentDataSize);

            codeSet.DataSegment().size += argumentDataAllocationSize;
        }

        const u32 bssSize = nsoHeader.segments[2].bss_size;
        codeSet.DataSegment().size += bssSize;
        const u32 imageSize = pageAlignSize(programImage.size() + bssSize);
        programImage.resize(imageSize);

        for (std::size_t i = 0; i < 3; ++i) {
            codeSet.segments[i].size = pageAlignSize(codeSet.segments[i].size);
        }

        // If we aren't actually loading (i.e. just computing the process code layout), we are done
        if (!loadIntoProcess) {
            return loadBase + imageSize;
        }

        // Load codeset for current process
        codeSet.memory = std::move(programImage);
        process.LoadModule(std::move(codeSet), loadBase);

        return loadBase + imageSize;
    }

    AppLoader_NSO::LoadResult AppLoader_NSO::Load(Kernel::KProcess& process, Core::System& system) {
        if (is_loaded) {
            return {ResultStatus::ErrorAlreadyLoaded, {}};
        }

        modules.clear();

        // Load module
        const VAddr baseAddress = GetInteger(process.PageTable().GetCodeRegionStart());
        if (!LoadModule(process, system, *file, baseAddress, true, true)) {
            return {ResultStatus::ErrorLoadingNSO, {}};
        }

        modules.emplace(baseAddress, file->GetName());
        LOG_DEBUG(Loader, "loaded module {} @ 0x{:X}", file->GetName(), baseAddress);

        is_loaded = true;
        return {ResultStatus::Success, LoadParameters{Kernel::KThread::DefaultThreadPriority, Core::Memory::DEFAULT_STACK_SIZE}};
    }

    ResultStatus AppLoader_NSO::ReadNSOModules(Modules& out_modules) {
        out_modules = this->modules;
        return ResultStatus::Success;
    }
}