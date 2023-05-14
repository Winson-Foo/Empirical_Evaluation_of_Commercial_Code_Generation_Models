// app_loader_xci.h

#pragma once

#include <vector>

#include "common/common_types.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/service/filesystem/filesystem.h"

namespace Loader {

    class AppLoader_XCI {

    public:
        explicit AppLoader_XCI(FileSys::VirtualFile file,
                               const Service::FileSystem::FileSystemController& fsc,
                               const FileSys::ContentProvider& contentProvider,
                               u64 programId, std::size_t programIndex);
        ~AppLoader_XCI();

        struct LoadResult {
            ResultStatus status;
            Kernel::KProcessHandle process;
        };

        FileType identifyType(const FileSys::VirtualFile& xciFile);
        LoadResult load(Kernel::KProcess& process, Core::System& system);
        ResultStatus readRomFS(FileSys::VirtualFile& outFile);
        u64 readRomFSIVFCOffset() const;
        ResultStatus readUpdateRaw(FileSys::VirtualFile& outFile);
        ResultStatus readProgramId(u64& outProgramId);
        ResultStatus readProgramIds(std::vector<u64>& outProgramIds);
        ResultStatus readIcon(std::vector<u8>& buffer);
        ResultStatus readTitle(std::string& title);
        ResultStatus readControlData(FileSys::NACP& control);
        ResultStatus readManualRomFS(FileSys::VirtualFile& outFile);
        ResultStatus readBanner(std::vector<u8>& buffer);
        ResultStatus readLogo(std::vector<u8>& buffer);
        ResultStatus readNSOModules(Modules& modules);

    private:
        AppLoader_XCI(const AppLoader_XCI&) = delete;
        AppLoader_XCI& operator=(const AppLoader_XCI&) = delete;

        ResultStatus setupControlNCA(const Service::FileSystem::FileSystemController& fsc,
                                     const FileSys::ContentProvider& contentProvider);
        ResultStatus setupNCA();
        ResultStatus setupPackedUpdate(Core::System& system);
        bool isLoaded{false};
        std::unique_ptr<FileSys::XCI> xci;
        std::unique_ptr<AppLoader_NCA> ncaLoader;
        const FileSys::NACP* nacpFile{nullptr};
        const FileSys::VirtualFile* iconFile{nullptr};
    };

} // namespace Loader

// app_loader_xci.cpp

#include "app_loader_xci.h"
#include "core/file_sys/card_image.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/submission_package.h"
#include "core/loader/nca.h"
#include "core/loader/xci.h"

namespace Loader {

    AppLoader_XCI::AppLoader_XCI(FileSys::VirtualFile file,
                                 const Service::FileSystem::FileSystemController& fsc,
                                 const FileSys::ContentProvider& contentProvider,
                                 u64 programId, std::size_t programIndex) :
            xci(std::make_unique<FileSys::XCI>(file, programId, programIndex)),
            ncaLoader(std::make_unique<AppLoader_NCA>(xci->GetProgramNCAFile())) {

        if (xci->GetStatus() != ResultStatus::Success) {
            return;
        }

        setupControlNCA(fsc, contentProvider);
    }

    AppLoader_XCI::~AppLoader_XCI() = default;

    FileType AppLoader_XCI::identifyType(const FileSys::VirtualFile& xciFile) {
        const FileSys::XCI xci(xciFile);

        if (xci.GetStatus() != ResultStatus::Success) {
            return FileType::Error;
        }

        if (xci.GetNCAByType(FileSys::NCAContentType::Program) != nullptr &&
            AppLoader_NCA::identifyType(xci.GetNCAFileByType(FileSys::NCAContentType::Program)) ==
            FileType::NCA) {
            return FileType::XCI;
        }

        return FileType::Error;
    }

    AppLoader_XCI::LoadResult AppLoader_XCI::load(Kernel::KProcess& process, Core::System& system) {
        if (isLoaded) {
            return {ResultStatus::ErrorAlreadyLoaded, {}};
        }

        if (xci->GetStatus() != ResultStatus::Success) {
            return {xci->GetStatus(), {}};
        }

        if (xci->GetProgramNCAStatus() != ResultStatus::Success) {
            return {xci->GetProgramNCAStatus(), {}};
        }

        if (!xci->HasProgramNCA() && !Core::Crypto::KeyManager::KeyFileExists(false)) {
            return {ResultStatus::ErrorMissingProductionKeyFile, {}};
        }

        const auto result = ncaLoader->Load(process, system);
        if (result.status != ResultStatus::Success) {
            return result;
        }

        setupPackedUpdate(system);

        isLoaded = true;
        return result;
    }

    ResultStatus AppLoader_XCI::readRomFS(FileSys::VirtualFile& outFile) {
        return ncaLoader->ReadRomFS(outFile);
    }

    u64 AppLoader_XCI::readRomFSIVFCOffset() const {
        return ncaLoader->ReadRomFSIVFCOffset();
    }

    ResultStatus AppLoader_XCI::readUpdateRaw(FileSys::VirtualFile& outFile) {
        u64 programId{};
        ncaLoader->ReadProgramId(programId);

        if (programId == 0) {
            return ResultStatus::ErrorXCIMissingProgramNCA;
        }

        const auto packedUpdate = xci->GetSecurePartitionNSP()->GetNCAFile(
                FileSys::GetUpdateTitleID(programId), FileSys::ContentRecordType::Program);

        if (packedUpdate == nullptr) {
            return ResultStatus::ErrorNoPackedUpdate;
        }

        const auto ncaTest = std::make_shared<FileSys::NCA>(packedUpdate);

        if (ncaTest->GetStatus() != ResultStatus::ErrorMissingBKTRBaseRomFS) {
            return ncaTest->GetStatus();
        }

        outFile = packedUpdate;
        return ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::readProgramId(u64& outProgramId) {
        return ncaLoader->ReadProgramId(outProgramId);
    }

    ResultStatus AppLoader_XCI::readProgramIds(std::vector<u64>& outProgramIds) {
        outProgramIds = xci->GetProgramTitleIDs();
        return ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::readIcon(std::vector<u8>& buffer) {
        if (iconFile == nullptr) {
            return ResultStatus::ErrorNoControl;
        }

        buffer = iconFile->ReadAllBytes();
        return ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::readTitle(std::string& title) {
        if (nacpFile == nullptr) {
            return ResultStatus::ErrorNoControl;
        }

        title = nacpFile->GetApplicationName();
        return ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::readControlData(FileSys::NACP& control) {
        if (nacpFile == nullptr) {
            return ResultStatus::ErrorNoControl;
        }

        control = *nacpFile;
        return ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::readManualRomFS(FileSys::VirtualFile& outFile) {
        const auto nca = xci->GetSecurePartitionNSP()->GetNCA(
                xci->GetSecurePartitionNSP()->GetProgramTitleID(),
                FileSys::ContentRecordType::HtmlDocument);

        if (xci->GetStatus() != ResultStatus::Success || nca == nullptr) {
            return ResultStatus::ErrorXCIMissingPartition;
        }

        outFile = nca->GetRomFS();
        return outFile == nullptr ? ResultStatus::ErrorNoRomFS : ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::readBanner(std::vector<u8>& buffer) {
        return ncaLoader->ReadBanner(buffer);
    }

    ResultStatus AppLoader_XCI::readLogo(std::vector<u8>& buffer) {
        return ncaLoader->ReadLogo(buffer);
    }

    ResultStatus AppLoader_XCI::readNSOModules(Modules& modules) {
        return ncaLoader->ReadNSOModules(modules);
    }

    ResultStatus AppLoader_XCI::setupControlNCA(const Service::FileSystem::FileSystemController& fsc,
                                                const FileSys::ContentProvider& contentProvider) {
        const auto controlNca = xci->GetNCAByType(FileSys::NCAContentType::Control);

        if (controlNca == nullptr || controlNca->GetStatus() != ResultStatus::Success) {
            return ResultStatus::Error;
        }

        const auto [foundNacp, foundIcon] = [&]() {
            const FileSys::PatchManager pm{xci->GetProgramTitleID(), fsc, contentProvider};
            return pm.ParseControlNCA(*controlNca);
        }();

        nacpFile = foundNacp;
        iconFile = foundIcon;

        return ResultStatus::Success;
    }

    ResultStatus AppLoader_XCI::setupNCA() {
        if (xci->GetStatus() != ResultStatus::Success) {
            return xci->GetStatus();
        }

        if (xci->GetProgramNCAStatus() != ResultStatus::Success) {
            return xci->GetProgramNCAStatus();
        }

        if (!xci->HasProgramNCA() && !Core::Crypto::KeyManager::KeyFileExists(false)) {
            return ResultStatus::ErrorMissingProductionKeyFile;
        }

        return ncaLoader->loadNCAFiles();
    }

    ResultStatus AppLoader_XCI::setupPackedUpdate(Core::System& system) {
        FileSys::VirtualFile packedUpdate;
        const auto readResult = readUpdateRaw(packedUpdate);

        if (readResult != ResultStatus::Success || packedUpdate == nullptr) {
            return readResult;
        }

        system.GetFileSystemController().SetPackedUpdate(std::move(packedUpdate));
        return ResultStatus::Success;
    }

} // namespace Loader

