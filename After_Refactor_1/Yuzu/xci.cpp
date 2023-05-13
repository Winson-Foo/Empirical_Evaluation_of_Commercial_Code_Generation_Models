namespace Loader {

  class AppLoader_XCI : public AppLoader {
  public:
    explicit AppLoader_XCI(FileSys::VirtualFile file,
                            const Service::FileSystem::FileSystemController& fs_controller,
                            const FileSys::ContentProvider& content_provider,
                            u64 program_id,
                            std::size_t program_index)
        : AppLoader(std::move(file)), xci_(std::make_shared<FileSys::XCI>(file, program_id, program_index)) {

      if (xci_->GetStatus() == ResultStatus::Success) {
        if (const auto control_nca = xci_->GetNCAByType(FileSys::NCAContentType::Control);
            control_nca != nullptr && control_nca->GetStatus() == ResultStatus::Success) {

          const auto [nacp_file, icon_file] = ParseControlNCA(*control_nca, fs_controller, content_provider);
          nacp_file_ = std::make_shared<FileSys::NACP>(*nacp_file);
          icon_file_ = std::move(icon_file);
        }
      }
    }

    ~AppLoader_XCI() override = default;

    FileType IdentifyType(const FileSys::VirtualFile& xci_file);

    LoadResult Load(Kernel::KProcess& process, Core::System& system) override;

    ResultStatus ReadRomFS(FileSys::VirtualFile& out_file);

    u64 ReadRomFSIVFCOffset() const;

    ResultStatus ReadUpdateRaw(FileSys::VirtualFile& out_file);

    ResultStatus ReadProgramId(u64& out_program_id);

    ResultStatus ReadProgramIds(std::vector<u64>& out_program_ids);

    ResultStatus ReadIcon(std::vector<u8>& buffer);

    ResultStatus ReadTitle(std::string& title);

    ResultStatus ReadControlData(FileSys::NACP& control);

    ResultStatus ReadManualRomFS(FileSys::VirtualFile& out_file);

    ResultStatus ReadBanner(std::vector<u8>& buffer);

    ResultStatus ReadLogo(std::vector<u8>& buffer);

    ResultStatus ReadNSOModules(Modules& modules);

  private:
    std::shared_ptr<FileSys::XCI> xci_;
    std::shared_ptr<FileSys::NACP> nacp_file_;
    std::unique_ptr<FileSys::VirtualFile> icon_file_;
    std::unique_ptr<AppLoader_NCA> nca_loader_;
    bool is_loaded_{};

    std::tuple<const FileSys::VirtualFile*, std::unique_ptr<FileSys::VirtualFile>>
    ParseControlNCA(const FileSys::VirtualFile& control_nca,
                    const Service::FileSystem::FileSystemController& fs_controller,
                    const FileSys::ContentProvider& content_provider) const;    
  };


  FileType AppLoader_XCI::IdentifyType(const FileSys::VirtualFile& xci_file) {
    const auto xci = std::make_shared<FileSys::XCI>(xci_file);

    if (xci->GetStatus() == ResultStatus::Success &&
        xci->GetNCAByType(FileSys::NCAContentType::Program) != nullptr &&
        AppLoader_NCA::IdentifyType(xci->GetNCAFileByType(FileSys::NCAContentType::Program)) == FileType::NCA) {
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

    const auto read = xci_->GetSecurePartitionNSP()->GetNCAFile(FileSys::GetUpdateTitleID(program_id),
                                                                   FileSys::ContentRecordType::Program);
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
                                                    FileSys::ContentRecordType::HtmlDocument);
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

  std::tuple<const FileSys::VirtualFile*, std::unique_ptr<FileSys::VirtualFile>>
  AppLoader_XCI::ParseControlNCA(const FileSys::VirtualFile& control_nca,
                                 const Service::FileSystem::FileSystemController& fs_controller,
                                 const FileSys::ContentProvider& content_provider) const {
    const FileSys::PatchManager pm{xci_->GetProgramTitleID(), fs_controller, content_provider};
    return pm.ParseControlNCA(control_nca);
  }

} // namespace Loader