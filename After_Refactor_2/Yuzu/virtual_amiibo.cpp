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

constexpr PadIdentifier kIdentifier = {
    .guid = Common::UUID{},
    .port = 0,
    .pad = 0,
};

VirtualAmiibo::VirtualAmiibo(std::string input_engine)
    : InputEngine(std::move(input_engine)) {}

VirtualAmiibo::~VirtualAmiibo() = default;

Common::Input::DriverResult VirtualAmiibo::SetPollingMode(
    [[maybe_unused]] const PadIdentifier& identifier,
    const Common::Input::PollingMode polling_mode) {
  polling_mode_ = polling_mode;

  if (polling_mode_ == Common::Input::PollingMode::NFC) {
    if (state_ == State::Initialized) {
      state_ = State::WaitingForAmiibo;
    }
    return Common::Input::DriverResult::Success;
  }

  if (state_ == State::AmiiboIsOpen) {
    CloseAmiibo();
  }
  return Common::Input::DriverResult::NotSupported;
}

Common::Input::NfcState VirtualAmiibo::SupportsNfc(
    [[maybe_unused]] const PadIdentifier& identifier) const {
  return Common::Input::NfcState::Success;
}

Common::Input::NfcState VirtualAmiibo::WriteNfcData(
    [[maybe_unused]] const PadIdentifier& identifier, const std::vector<u8>& data) {
  const Common::FS::IOFile nfc_file{file_path_, Common::FS::FileAccessMode::ReadWrite,
                                    Common::FS::FileType::BinaryFile};

  if (!nfc_file.IsOpen()) {
    LOG_ERROR(Core, "Amiibo is already in use");
    return Common::Input::NfcState::WriteFailed;
  }

  if (!nfc_file.Write(data)) {
    LOG_ERROR(Service_NFP, "Error writing to file");
    return Common::Input::NfcState::WriteFailed;
  }

  nfc_data_ = data;

  return Common::Input::NfcState::Success;
}

VirtualAmiibo::State VirtualAmiibo::GetCurrentState() const {
  return state_;
}

VirtualAmiibo::Info VirtualAmiibo::LoadAmiibo(const std::string& filename) {
  const Common::FS::IOFile nfc_file{filename, Common::FS::FileAccessMode::Read,
                                    Common::FS::FileType::BinaryFile};

  if (state_ != State::WaitingForAmiibo) {
    return Info::WrongDeviceState;
  }

  if (!nfc_file.IsOpen()) {
    return Info::UnableToLoad;
  }

  switch (nfc_file.GetSize()) {
  case AmiiboSize:
  case AmiiboSizeWithoutPassword:
    nfc_data_.resize(AmiiboSize);
    if (nfc_file.Read(nfc_data_) < AmiiboSizeWithoutPassword) {
      return Info::NotAnAmiibo;
    }
    break;
  case MifareSize:
    nfc_data_.resize(MifareSize);
    if (nfc_file.Read(nfc_data_) < MifareSize) {
      return Info::NotAnAmiibo;
    }
    break;
  default:
    return Info::NotAnAmiibo;
  }

  file_path_ = filename;
  state_ = State::AmiiboIsOpen;
  SetNfc(kIdentifier, {Common::Input::NfcState::NewAmiibo, nfc_data_});
  return Info::Success;
}

VirtualAmiibo::Info VirtualAmiibo::ReloadAmiibo() {
  if (state_ == State::AmiiboIsOpen) {
    SetNfc(kIdentifier, {Common::Input::NfcState::NewAmiibo, nfc_data_});
    return Info::Success;
  }

  return LoadAmiibo(file_path_);
}

VirtualAmiibo::Info VirtualAmiibo::CloseAmiibo() {
  state_ = polling_mode_ == Common::Input::PollingMode::NFC ? State::WaitingForAmiibo
                                                            : State::Initialized;
  SetNfc(kIdentifier, {Common::Input::NfcState::AmiiboRemoved, {}});
  return Info::Success;
}

std::string VirtualAmiibo::GetLastFilePath() const {
  return file_path_;
}

}  // namespace InputCommon