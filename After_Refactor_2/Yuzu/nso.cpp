#pragma once
  
#include <vector>
#include <optional>
#include <memory>
  
#include "core/memory.h"
#include "core/system.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/file_sys/vfs.h"
#include "core/file_sys/file.h"
  
namespace Loader {
  
    namespace FileType {
        enum FileType {
            NSO,
            Error
        };
    }
  
    struct NSOSegmentHeader {
        u32_le location;
        u32_le offset;
        u32_le size;
    };
    static_assert(sizeof(NSOSegmentHeader) == 0xc, "NSOSegmentHeader has incorrect size.");
  
    struct NSOHeader {
        u32_le magic;
        u32_le flags;
        u32_le segment_count;
        NSOSegmentHeader segments[3];
        u32_le segments_compressed_size[3];
        u8    build_id[0x20];
    };
    static constexpr u32_le NSO_HEADER_MAGIC = 0x4F534E;
    static constexpr std::size_t NSO_HEADER_SIZE = 0x50;

    struct NSOArgumentHeader {
        u32_le size;
        u32_le argc;
        char   argv[1];
    };
    static constexpr std::size_t NSO_ARGUMENT_DATA_ALLOCATION_SIZE = 0x20000;
  
    using Modules = std::unordered_map<VAddr, std::string>;
  
    struct LoadParameters {
        u32 priority;
        u64 stack_size;
    };
  
    struct ResultStatus {
        enum ResultStatus {
            Success,
            ErrorLoadingNSO,
            ErrorAlreadyLoaded,
            ErrorUnknown
        };
  
        ResultStatus() : result(ErrorUnknown) {}
        ResultStatus(ResultStatus::ResultStatus result_) : result(result_) {}
        operator bool() const { return result == Success; }
        const ResultStatus::ResultStatus result;
    };
  
    class AppLoader {
    public:
        explicit AppLoader(FileSys::VirtualFile file_) : file(std::move(file_)) {}
        AppLoader(const AppLoader&) = delete;
        virtual ~AppLoader() = default;
  
        virtual ResultStatus Load(Kernel::KProcess& process,
                                  Core::System& system) = 0;
  
        virtual ResultStatus ReadNSOModules(Modules& out_modules) = 0;
  
        const FileSys::VirtualFile& GetFile() const { return file; }
  
    protected:
        std::unique_ptr<FileSys::VirtualFile> file;
        bool is_loaded{false};
    };
  
    class AppLoader_NSO : public AppLoader {
    public:
        explicit AppLoader_NSO(FileSys::VirtualFile file);
        ResultStatus Load(Kernel::KProcess& process,
                           Core::System& system) override;
        ResultStatus ReadNSOModules(Modules& out_modules) override;
  
    private:
        static FileType::FileType IdentifyType(
            const FileSys::VirtualFile& in_file);
        static std::optional<VAddr> LoadModule(Kernel::KProcess& process,
                                               Core::System& system,
                                               const FileSys::VfsFile& nso_file,
                                               VAddr load_base,
                                               bool should_pass_arguments,
                                               bool load_into_process,
                                               std::optional<FileSys::PatchManager> pm = std::nullopt);
  
        bool IsSegmentCompressed(size_t segment_num) const;
  
    private:
        Modules modules;
    };

}

#include "nso_loader.hpp"
  
#include <cstring>
#include <unordered_map>
  
#include "common/common_funcs.h"
#include "common/hex_util.h"
#include "common/logging/log.h"
#include "common/lz4_compression.h"
#include "common/settings.h"
#include "common/swap.h"
#include "core/core.h"
#include "core/file_sys/patch_manager.h"
#include "core/hle/kernel/code_set.h"
#include "core/hle/kernel/k_page_table.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/loader/nso.h"
  
namespace Loader {
  
    namespace {
        struct MODHeader {
            u32_le magic;
            u32_le dynamic_offset;
            u32_le bss_start_offset;
            u32_le bss_end_offset;
            u32_le eh_frame_hdr_start_offset;
            u32_le eh_frame_hdr_end_offset;
            u32_le module_offset; // Offset to runtime-generated module object. typically equal to .bss base
        };
        static constexpr std::size_t MOD_HEADER_SIZE = 0x1c;

        std::vector<u8> DecompressSegment(const std::vector<u8>& compressed_data,
                                           const NSOSegmentHeader& header) {
            std::vector<u8> uncompressed_data =
                Common::Compression::DecompressDataLZ4(compressed_data, header.size);
  
            ASSERT_MSG(uncompressed_data.size() == header.size,
                       "{} != {}", header.size, uncompressed_data.size());
  
            return uncompressed_data;
        }

        constexpr u32 PageAlignSize(u32 size) {
            return static_cast<u32>((size + Core::Memory::YUZU_PAGEMASK) & ~Core::Memory::YUZU_PAGEMASK);
        }

    } // Anonymous namespace

    AppLoader_NSO::AppLoader_NSO(FileSys::VirtualFile file_) : AppLoader(std::move(file_)) {}

    FileType::FileType AppLoader_NSO::IdentifyType(const FileSys::VirtualFile& in_file) {
        u32 magic = 0;
        if (in_file->ReadObject(&magic) != sizeof(magic)) {
            return FileType::Error;
        }
  
        if (NSO_HEADER_MAGIC != magic) {
            return FileType::Error;
        }
  
        return FileType::NSO;
    }

    std::optional<VAddr> AppLoader_NSO::LoadModule(Kernel::KProcess& process,
                                                   Core::System& system,
                                                   const FileSys::VfsFile& nso_file,
                                                   VAddr load_base,
                                                   bool should_pass_arguments,
                                                   bool load_into_process,
                                                   std::optional<FileSys::PatchManager> pm) {
        if (nso_file.GetSize() < NSO_HEADER_SIZE) {
            return std::nullopt;
        }
  
        NSOHeader nso_header{};
        if (NSO_HEADER_SIZE != nso_file.ReadObject(&nso_header)) {
            return std::nullopt;
        }
  
        if (nso_header.magic != NSO_HEADER_MAGIC) {
            return std::nullopt;
        }
  
        // Build program image
        Kernel::CodeSet codeset;
        Kernel::PhysicalMemory program_image;
        for (std::size_t i = 0; i < nso_header.segment_count; ++i) {
            std::vector<u8> data = nso_file.ReadBytes(nso_header.segments_compressed_size[i],
                                                      nso_header.segments[i].offset);
            if (IsSegmentCompressed(i)) {
                data = DecompressSegment(data, nso_header.segments[i]);
            }
            program_image.resize(nso_header.segments[i].location + static_cast<u32>(data.size()));
            std::memcpy(program_image.data() + nso_header.segments[i].location, data.data(),
                        data.size());
            codeset.segments[i].addr = nso_header.segments[i].location;
            codeset.segments[i].offset = nso_header.segments[i].location;
            codeset.segments[i].size = nso_header.segments[i].size;
        }
  
        if (should_pass_arguments && !Settings::values.program_args.GetValue().empty()) {
            const auto arg_data{Settings::values.program_args.GetValue()};
  
            codeset.DataSegment().size += NSO_ARGUMENT_DATA_ALLOCATION_SIZE;
            NSOArgumentHeader args_header{
                NSO_ARGUMENT_DATA_ALLOCATION_SIZE, static_cast<u32_le>(arg_data.size()), {}};
            const auto end_offset = program_image.size();
            program_image.resize(static_cast<u32>(program_image.size()) +
                                 NSO_ARGUMENT_DATA_ALLOCATION_SIZE);
            std::memcpy(program_image.data() + end_offset, &args_header, sizeof(NSOArgumentHeader));
            std::memcpy(program_image.data() + end_offset + sizeof(NSOArgumentHeader), arg_data.data(),
                        arg_data.size());
        }
  
        codeset.DataSegment().size += nso_header.segments[2].bss_size;
        const u32 image_size{
            PageAlignSize(static_cast<u32>(program_image.size()) + nso_header.segments[2].bss_size)};
        program_image.resize(image_size);
  
        for (std::size_t i = 0; i < nso_header.segment_count; ++i) {
            codeset.segments[i].size = PageAlignSize(codeset.segments[i].size);
        }
  
        // Apply patches if necessary
        if (pm && (pm->HasNSOPatch(nso_header.build_id) || Settings::values.dump_nso)) {
            std::vector<u8> pi_header(NSO_HEADER_SIZE + program_image.size());
            std::memcpy(pi_header.data(), &nso_header, NSO_HEADER_SIZE);
            std::memcpy(pi_header.data() + NSO_HEADER_SIZE, program_image.data(),
                        program_image.size());
  
            pi_header = pm->PatchNSO(pi_header, nso_file.GetName());
  
            std::copy(pi_header.begin() + NSO_HEADER_SIZE, pi_header.end(), program_image.data());
        }
  
        // If we aren't actually loading (i.e. just computing the process code layout), we are done
        if (!load_into_process) {
            return load_base + image_size;
        }
  
        // Apply cheats if they exist and the program has a valid title ID
        if (pm) {
            system.SetApplicationProcessBuildID(nso_header.build_id);
            const auto cheats = pm->CreateCheatList(nso_header.build_id);
            if (!cheats.empty()) {
                system.RegisterCheatList(cheats, nso_header.build_id, load_base, image_size);
            }
       

