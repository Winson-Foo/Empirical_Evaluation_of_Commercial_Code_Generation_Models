#include "nso_loader.h"
#include <cinttypes>
#include <cstring>
#include <vector>
#include "common.h"
#include "lz4_compression.h"
#include "settings.h"
#include "loader.h"
#include "module.h"
#include "module_header.h"
#include "memory.h"
#include "swap.h"
#include "common_brightness.h"
#include "patch_manager.h"
#include "log.h"
#include "k_process.h"
#include "k_thread.h"
#include "k_page_table.h"
#include "k_memory.h"
#include "code_set.h"

namespace Loader {

    namespace {

        // Struct to represent a MOD header
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

        // Helper function to decompress a segment
        std::vector<u8> DecompressSegment(const std::vector<u8>& compressed_data, const NSOSegmentHeader& header) {
            std::vector<u8> uncompressed_data = Common::Compression::DecompressDataLZ4(compressed_data, header.size);
            ASSERT_MSG(uncompressed_data.size() == header.size, "{} != {}", header.size, uncompressed_data.size());
            return uncompressed_data;
        }

        // Helper function to page-align a size
        constexpr u32 PageAlignSize(u32 size) {
            return static_cast<u32>((size + Core::Memory::YUZU_PAGEMASK) & ~Core::Memory::YUZU_PAGEMASK);
        }

    } // Anonymous namespace

    // Constructor for the NSO loader
    AppLoader_NSO::AppLoader_NSO(FileSys::VirtualFile file_) : AppLoader(std::move(file_)) {}

    // Function to identify the type of file
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

    // Function to load a module
    std::optional<VAddr> AppLoader_NSO::LoadModule(Kernel::KProcess& process, Core::System& system,
                                                   const FileSys::VfsFile& nso_file, VAddr load_base,
                                                   bool should_pass_arguments, bool load_into_process,
                                                   std::optional<FileSys::PatchManager> pm) {
        // Check if the file size is large enough to contain an NSO header
        if (nso_file.GetSize() < sizeof(NSOHeader)) {
            return std::nullopt;
        }

        // Read the NSO header
        NSOHeader nso_header{};
        if (sizeof(NSOHeader) != nso_file.ReadObject(&nso_header)) {
            return std::nullopt;
        }

        // Check that the NSO header has the correct magic value
        if (nso_header.magic != Common::MakeMagic('N', 'S', 'O', '0')) {
            return std::nullopt;
        }

        // Build program image
        Kernel::CodeSet codeset;
        Kernel::PhysicalMemory program_image;
        for (std::size_t i = 0; i < nso_header.segments.size(); ++i) {
            std::vector<u8> data = nso_file.ReadBytes(nso_header.segments_compressed_size[i], nso_header.segments[i].offset);
            if (nso_header.IsSegmentCompressed(i)) {
                data = DecompressSegment(data, nso_header.segments[i]);
            }
            program_image.resize(nso_header.segments[i].location + static_cast<u32>(data.size()));
            std::memcpy(program_image.data() + nso_header.segments[i].location, data.data(), data.size());
            codeset.segments[i].addr = nso_header.segments[i].location;
            codeset.segments[i].offset = nso_header.segments[i].location;
            codeset.segments[i].size = nso_header.segments[i].size;
        }

        // Add program arguments if applicable
        if (should_pass_arguments && !Settings::values.program_args.GetValue().empty()) {
            const auto arg_data{Settings::values.program_args.GetValue()};
            codeset.DataSegment().size += NSO_ARGUMENT_DATA_ALLOCATION_SIZE;
            NSOArgumentHeader args_header{
                NSO_ARGUMENT_DATA_ALLOCATION_SIZE, static_cast<u32_le>(arg_data.size()), {}};
            const auto end_offset = program_image.size();
            program_image.resize(static_cast<u32>(program_image.size()) +
                                 NSO_ARGUMENT_DATA_ALLOCATION_SIZE);
            std::memcpy(program_image.data() + end_offset, &args_header, sizeof(NSOArgumentHeader));
            std::memcpy(program_image.data() + end_offset + sizeof(NSOArgumentHeader), arg_data.data(), arg_data.size());
        }

        // Add BSS segment
        codeset.DataSegment().size += nso_header.segments[2].bss_size;
        const u32 image_size{
            PageAlignSize(static_cast<u32>(program_image.size()) + nso_header.segments[2].bss_size)};
        program_image.resize(image_size);

        for (std::size_t i = 0; i < nso_header.segments.size(); ++i) {
            codeset.segments[i].size = PageAlignSize(codeset.segments[i].size);
        }

        // Apply patches if necessary
        if (pm && (pm->HasNSOPatch(nso_header.build_id) || Settings::values.dump_nso)) {
            std::vector<u8> pi_header(sizeof(NSOHeader) + program_image.size());
            std::memcpy(pi_header.data(), &nso_header, sizeof(NSOHeader));
            std::memcpy(pi_header.data() + sizeof(NSOHeader), program_image.data(), program_image.size());
            pi_header = pm->PatchNSO(pi_header, nso_file.GetName());
            std::copy(pi_header.begin() + sizeof(NSOHeader), pi_header.end(), program_image.data());
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
        }

        // Load codeset for current process
        codeset.memory = std::move(program_image);
        process.LoadModule(std::move(codeset), load_base);

        return load_base + image_size;
    }

    // Function to load an NSO file
    AppLoader_NSO::LoadResult AppLoader_NSO::Load(Kernel::KProcess& process, Core::System& system) {
        // Check if the file is already loaded
        if (is_loaded) {
            return {ResultStatus::ErrorAlreadyLoaded, {}};
        }

        modules.clear();

        // Load module
        const VAddr base_address = GetInteger(process.PageTable().GetCodeRegionStart());
        if (!LoadModule(process, system, *file, base_address, true, true)) {
            return {ResultStatus::ErrorLoadingNSO, {}};
        }

        modules.insert_or_assign(base_address, file->GetName());
        LOG_DEBUG(Loader, "loaded module {} @ 0x{:X}", file->GetName(), base_address);

        is_loaded = true;
        return {ResultStatus::Success, LoadParameters{Kernel::KThread::DefaultThreadPriority, Core::Memory::DEFAULT_STACK_SIZE}};
    }

    // Function to read the modules in the loader
    ResultStatus AppLoader_NSO::ReadNSOModules(Modules& out_modules) {
        out_modules = this->modules;
        return ResultStatus::Success;
    }

} // namespace Loader