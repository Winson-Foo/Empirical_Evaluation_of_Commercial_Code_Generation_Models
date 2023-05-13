#include <vector>
#include <memory>
#include <tuple>
#include <algorithm>

#include "common/common_types.h"
#include "core/core.h"
#include "core/file_sys/card_image.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/submission_package.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/loader/nca.h"
#include "core/loader/xci.h"

namespace Loader {

    namespace {
        const auto CONTROL_NCA_CONTENT_TYPE = FileSys::NCAContentType::Control;
        const auto PACKED_UPDATE_CONTENT_TYPE = FileSys::ContentRecordType::Program;
        const auto HTML_DOC_CONTENT_TYPE = FileSys::ContentRecordType::HtmlDocument;
    }

    AppLoader_XCI::AppLoader_XCI(FileSys::VirtualFile file_,
                                 const Service::FileSystem::FileSystemController& fsc,
                                 const FileSys::ContentProvider& content_provider,
                                 u64 program_id,
                                 std::size_t program_index)
    : AppLoader(file_),
      xci_(std::make_unique<FileSys::XCI>(file_, program_id, program_index)),
      nca_loader_(std::make_unique<AppLoader_NCA>(xci_->GetProgramNCAFile()))
    {
        if (xci_->GetStatus() != ResultStatus::Success) {
            return;
        }

        const auto control_nca = xci_->GetNCAByType(CONTROL_NCA_CONTENT_TYPE);
        if (control_nca == nullptr || control_nca->GetStatus() != ResultStatus::Success) {
            return;
        }

        std::tie(nacp_file_, icon_file_) = [this, &content_provider, &control_nca, &fsc] {
            const FileSys::PatchManager pm{xci_->GetProgramTitleID(), fsc, content_provider};
            return pm.ParseControlNCA(*control_nca);
        }();
    }

    AppLoader_XCI::~AppLoader_XCI() = default;

    FileType AppLoader_XCI::IdentifyType(const FileSys::VirtualFile& xci_file) {
        const FileSys::XCI xci(xci_file);

        if (xci.GetStatus() == ResultStatus::Success &&
            xci.GetNCAByType(FileSys::NCAContentType::Program) != nullptr &&
            AppLoader_NCA::IdentifyType(xci.GetNCAFileByType(FileSys::NCAContentType::Program)) ==
                FileType::NCA) {
            return FileType::XCI;
        }

        return FileType::Error;
    }

    AppLoader_XCI::LoadResult AppLoader_XCI::Load(Kernel::KProcess& process, Core::System& system) {
        if (is_loaded_) {
            return {ResultStatus::ErrorAlreadyLoaded, {}};
        }

        if (xci_->GetStatus() != ResultStatus::Success) {
            return {xci_->GetStatus(), {}};
        }

        if (xci_->GetProgramNCAStatus() != ResultStatus::Success) {
            return {xci_->GetProgramNCAStatus(), {}};
        }

        if (!xci_->HasProgramNCA() && !Core::Crypto::KeyManager::KeyFileExists(false)) {
            return {ResultStatus::ErrorMissingProductionKeyFile, {}};
        }

        const auto result = nca_loader_->Load(process, system);
        if (result.first != ResultStatus::Success) {
            return result;
        }

        FileSys::VirtualFile update_raw;
        if (ReadUpdateRaw(update_raw) == ResultStatus::Success && update_raw != nullptr) {
            system.GetFileSystemController().SetPackedUpdate(std::move(update_raw));
        }

        is_loaded_ = true;
        return result;
    }

    ResultStatus AppLoader_XCI::ReadRomFS(FileSys::VirtualFile& out_file) {
        return nca_loader_->ReadRomFS(out_file);
    }

    u64 AppLoader_XCI::ReadRomFSIVFCOffset() const {
        return nca_loader_->ReadRomFSIVFCOffset();
    }

    ResultStatus AppLoader_XCI::ReadUpdateRaw(FileSys::VirtualFile& out_file) {
        u64 program_id{};
        nca_loader_->ReadProgramId(program_id);
        if (program_id == 0) {
            return ResultStatus::ErrorXCIMissingProgramNCA;
        }

        const auto read = xci_->GetSecurePartitionNSP()->GetNCAFile(
            FileSys::GetUpdateTitleID(program_id), PACKED_UPDATE_CONTENT_TYPE);
        if (read == nullptr) {
            return ResultStatus::ErrorNoPackedUpdate;
        }

        const auto nca_test = std::make_shared<FileSys::NCA>(read);
        if (nca_test->GetStatus() != ResultStatus::ErrorMissingBKTRBaseRomFS) {
            return nca_test->GetStatus();
        }

        out_file = read;
        return ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::ReadProgramId(u64& out_program_id) {
        return nca_loader_->ReadProgramId(out_program_id);
    }

    ResultStatus AppLoader_XCI::ReadProgramIds(std::vector<u64>& out_program_ids) {
        out_program_ids = xci_->GetProgramTitleIDs();
        return ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::ReadIcon(std::vector<u8>& buffer) {
        if (icon_file_ == nullptr) {
            return ResultStatus::ErrorNoControl;
        }

        buffer = icon_file_->ReadAllBytes();
        return ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::ReadTitle(std::string& title) {
        if (nacp_file_ == nullptr) {
            return ResultStatus::ErrorNoControl;
        }

        title = nacp_file_->GetApplicationName();
        return ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::ReadControlData(FileSys::NACP& control) {
        if (nacp_file_ == nullptr) {
            return ResultStatus::ErrorNoControl;
        }

        control = *nacp_file_;
        return ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::ReadManualRomFS(FileSys::VirtualFile& out_file) {
        const auto nca =
            xci_->GetSecurePartitionNSP()->GetNCA(xci_->GetSecurePartitionNSP()->GetProgramTitleID(),
                                                 HTML_DOC_CONTENT_TYPE);
        if (xci_->GetStatus() != ResultStatus::Success || nca == nullptr) {
            return ResultStatus::ErrorXCIMissingPartition;
        }

        out_file = nca->GetRomFS();
        return out_file == nullptr ? ResultStatus::ErrorNoRomFS : ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::ReadBanner(std::vector<u8>& buffer) {
        return nca_loader_->ReadBanner(buffer);
    }

    ResultStatus AppLoader_XCI::ReadLogo(std::vector<u8>& buffer) {
        return nca_loader_->ReadLogo(buffer);
    }

    ResultStatus AppLoader_XCI::ReadNSOModules(Modules& modules) {
        return nca_loader_->ReadNSOModules(modules);
    }

} // namespace Loader