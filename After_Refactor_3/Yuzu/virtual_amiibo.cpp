// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <fmt/format.h>

#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "input_common/drivers/virtual_amiibo.h"

namespace InputCommon {

// Define commonly used constants
constexpr PadIdentifier identifier = {
    .guid = Common::UUID{},
    .port = 0,
    .pad = 0,
};

constexpr int AmiiboSize = 540;
constexpr int AmiiboSizeWithoutPassword = 532;
constexpr int MifareSize = 1024;

// Constructor takes an input engine string
VirtualAmiibo::VirtualAmiibo(std::string input_engine_)
    : InputEngine(std::move(input_engine_)), state(State::Initialized) {}

// Destructor
VirtualAmiibo::~VirtualAmiibo() = default;

// Set the polling mode
Common::Input::DriverResult VirtualAmiibo::SetPollingMode(const PadIdentifier& identifier_, 
    const Common::Input::PollingMode polling_mode_) {
    if (identifier_ != identifier) {
        return Common::Input::DriverResult::NotSupported;
    }

    polling_mode = polling_mode_;
    switch (polling_mode) {
        case Common::Input::PollingMode::NFC:
            if (state == State::Initialized) {
                state = State::WaitingForAmiibo;
            } else {
                return Common::Input::DriverResult::Failure;
            }
            break;
        default:
            if (state == State::AmiiboIsOpen) {
                CloseAmiibo();
            }
            break;
    }
    return Common::Input::DriverResult::Success;
}

// Check if NFC is supported
Common::Input::NfcState VirtualAmiibo::SupportsNfc(const PadIdentifier& identifier_) const {
    if (identifier_ != identifier) {
        return Common::Input::NfcState::NotSupported;
    } else {
        return Common::Input::NfcState::Success;
    }
}

// Write to the NFC data
Common::Input::NfcState VirtualAmiibo::WriteNfcData(const PadIdentifier& identifier_, const std::vector<u8>& data) {
    if (identifier_ != identifier) {
        return Common::Input::NfcState::WriteFailed;
    }

    const Common::FS::IOFile nfc_file{file_path, Common::FS::FileAccessMode::ReadWrite, Common::FS::FileType::BinaryFile};
    if (!nfc_file.IsOpen()) {
        return Common::Input::NfcState::WriteFailed;
    }

    if (!nfc_file.Write(data)) {
        LOG_ERROR(Service_NFP, "Error writing to file");
        return Common::Input::NfcState::WriteFailed;
    }

    nfc_data = data;
    return Common::Input::NfcState::Success;
}

// Get the current state
VirtualAmiibo::State VirtualAmiibo::GetCurrentState() const {
    return state;
}

// Load an Amiibo from file
VirtualAmiibo::Info VirtualAmiibo::LoadAmiibo(const std::string& filename) {
    if (state != State::WaitingForAmiibo) {
        return Info::WrongDeviceState;
    }

    const Common::FS::IOFile nfc_file{filename, Common::FS::FileAccessMode::Read, Common::FS::FileType::BinaryFile};
    if (!nfc_file.IsOpen()) {
        return Info::UnableToLoad;
    }

    switch (nfc_file.GetSize()) {
        case AmiiboSize:
        case AmiiboSizeWithoutPassword:
            nfc_data.resize(AmiiboSize);
            if (nfc_file.Read(nfc_data) < AmiiboSizeWithoutPassword) {
                return Info::NotAnAmiibo;
            }
            break;
        case MifareSize:
            nfc_data.resize(MifareSize);
            if (nfc_file.Read(nfc_data) < MifareSize) {
                return Info::NotAnAmiibo;
            }
            break;
        default:
            return Info::NotAnAmiibo;
    }

    file_path = filename;
    state = State::AmiiboIsOpen;
    SetNfc(identifier, {Common::Input::NfcState::NewAmiibo, nfc_data});
    return Info::Success;
}

// Reload an Amiibo from file
VirtualAmiibo::Info VirtualAmiibo::ReloadAmiibo() {
    if (state == State::AmiiboIsOpen) {
        SetNfc(identifier, {Common::Input::NfcState::NewAmiibo, nfc_data});
        return Info::Success;
    } else {
        return LoadAmiibo(file_path);
    }
}

// Close the current Amiibo
VirtualAmiibo::Info VirtualAmiibo::CloseAmiibo() {
    if (polling_mode == Common::Input::PollingMode::NFC) {
        state = State::WaitingForAmiibo;
    } else {
        state = State::Initialized;
    }
    SetNfc(identifier, {Common::Input::NfcState::AmiiboRemoved, {}});
    return Info::Success;
}

// Get the last loaded file path
std::string VirtualAmiibo::GetLastFilePath() const {
    return file_path;
}

} // namespace InputCommon

