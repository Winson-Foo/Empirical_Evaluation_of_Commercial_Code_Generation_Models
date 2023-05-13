#include <cstring>
#include <fmt/format.h>

#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "input_common/drivers/virtual_amiibo.h"

namespace InputCommon {

VirtualAmiibo::VirtualAmiibo(std::string input_engine_) : InputEngine(std::move(input_engine_)) {
    identifier.guid = Common::UUID{};
    identifier.port = 0;
    identifier.pad = 0;
    state = State::Initialized;
}

VirtualAmiibo::~VirtualAmiibo() = default;

Common::Input::DriverResult VirtualAmiibo::SetPollingMode(const PadIdentifier& identifier_, const Common::Input::PollingMode polling_mode_) {
    polling_mode = polling_mode_;

    switch (polling_mode) {
        case Common::Input::PollingMode::NFC:
            if (state == State::Initialized) {
                state = State::WaitingForAmiibo;
            }
            return Common::Input::DriverResult::Success;
        default:
            if (state == State::AmiiboIsOpen) {
                CloseAmiibo();
            }
            return Common::Input::DriverResult::NotSupported;
    }
}

Common::Input::NfcState VirtualAmiibo::SupportsNfc(const PadIdentifier& identifier_) const {
    return Common::Input::NfcState::Success;
}

Common::Input::NfcState VirtualAmiibo::WriteNfcData(const PadIdentifier& identifier_, const std::vector<u8>& data) {
    const Common::FS::IOFile nfc_file{file_path, Common::FS::FileAccessMode::ReadWrite, Common::FS::FileType::BinaryFile};

    if (!nfc_file.IsOpen()) {
        LOG_ERROR(Core, "Amiibo is already in use.");
        return Common::Input::NfcState::WriteFailed;
    }

    if (!nfc_file.Write(data)) {
        LOG_ERROR(Service_NFP, "Error writing to file.");
        return Common::Input::NfcState::WriteFailed;
    }

    nfc_data = data;

    return Common::Input::NfcState::Success;
}

VirtualAmiibo::State VirtualAmiibo::GetCurrentState() const {
    return state;
}

VirtualAmiibo::Info VirtualAmiibo::LoadAmiibo(const std::string& filename) {
    const Common::FS::IOFile nfc_file{filename, Common::FS::FileAccessMode::Read, Common::FS::FileType::BinaryFile};

    if (state != State::WaitingForAmiibo) {
        return Info::WrongDeviceState;
    }

    if (!nfc_file.IsOpen()) {
        return Info::UnableToLoad;
    }

    const int filesize = nfc_file.GetSize();
    if (filesize != AmiiboSize && filesize != AmiiboSizeWithoutPassword && filesize != MifareSize) {
        return Info::NotAnAmiibo;
    }

    nfc_data.resize(filesize);
    if (nfc_file.Read(nfc_data) < filesize) {
        return Info::NotAnAmiibo;
    }

    file_path = filename;
    state = State::AmiiboIsOpen;
    SetNfc(identifier, {Common::Input::NfcState::NewAmiibo, nfc_data});

    return Info::Success;
}

VirtualAmiibo::Info VirtualAmiibo::ReloadAmiibo() {
    if (state == State::AmiiboIsOpen) {
        SetNfc(identifier, {Common::Input::NfcState::NewAmiibo, nfc_data});

        return Info::Success;
    }

    return LoadAmiibo(file_path);
}

VirtualAmiibo::Info VirtualAmiibo::CloseAmiibo() {
    state = polling_mode == Common::Input::PollingMode::NFC ? State::WaitingForAmiibo : State::Initialized;
    SetNfc(identifier, {Common::Input::NfcState::AmiiboRemoved, {}});
    
    return Info::Success;
}

std::string VirtualAmiibo::GetLastFilePath() const {
    return file_path;
}

}  // namespace InputCommon