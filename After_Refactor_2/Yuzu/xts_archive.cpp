#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <vector>

#include "virtual_file.h"
#include "virtual_dir.h"
#include "loader/result_status.h"

namespace FileSys {

enum class NAXContentType {
    NCA,
    Unknown,
};

class NAX {
public:
    NAX(VirtualFile file);
    NAX(VirtualFile file, std::array<u8, 0x10> ncaId);

    Loader::ResultStatus GetStatus() const;
    VirtualFile GetDecrypted() const;
    std::unique_ptr<NCA> AsNCA() const;
    NAXContentType GetContentType() const;
    std::vector<VirtualFile> GetFiles() const;
    std::vector<VirtualDir> GetSubdirectories() const;
    std::string GetName() const;
    VirtualDir GetParentDirectory() const;

private:
    struct NAXHeader {
        u32 magic;
        u32 version;
        u8 reserved[0x50];
        std::array<u8, 0x10> key_area[2];
        Core::Crypto::SHA256Hash hmac;
        u64 file_size;
    };

    Loader::ResultStatus ParseHeader();
    Loader::ResultStatus DecryptFile(std::string_view path);
    Loader::ResultStatus ValidateHMAC(Core::Crypto::Key256 key);
    std::string GetNAXFilePath() const;

    std::unique_ptr<NAXHeader> header_;
    VirtualFile file_;
    VirtualFile decrypted_;
    Loader::ResultStatus status_;
    NAXContentType type_;
};

} // namespace FileSys


// nax.cpp

#include <algorithm>
#include <cstring>
#include <regex>

#include <mbedtls/md.h>
#include <mbedtls/sha256.h>

#include "nax.h"
#include "common/fs/path_util.h"
#include "common/hex_util.h"
#include "core/crypto/aes_util.h"
#include "core/crypto/key_manager.h"
#include "core/crypto/xts_encryption_layer.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/vfs_offset.h"
#include "core/file_sys/xts_archive.h"
#include "core/loader/loader.h"

namespace fs = std::filesystem;

namespace FileSys {

namespace {

    constexpr u64 NAX_HEADER_PADDING_SIZE = 0x4000;
    constexpr u32 NAX_MAGIC = Common::MakeMagic('N', 'A', 'X', '0');
    constexpr std::size_t NAX_KEY_AREA_SIZE = 0x20;
    constexpr std::size_t NAX_HMAC_SIZE = sizeof(Core::Crypto::SHA256Hash);

    template <typename SourceData, typename SourceKey, typename Destination>
    bool calculateHMAC(Destination* out, const SourceKey* key, std::size_t keyLength,
                       const SourceData* data, std::size_t dataLength) {
        mbedtls_md_context_t context;
        mbedtls_md_init(&context);

        if (mbedtls_md_setup(&context, mbedtls_md_info_from_type(MBEDTLS_MD_SHA256), 1) ||
            mbedtls_md_hmac_starts(&context, reinterpret_cast<const u8*>(key), keyLength) ||
            mbedtls_md_hmac_update(&context, reinterpret_cast<const u8*>(data), dataLength) ||
            mbedtls_md_hmac_finish(&context, reinterpret_cast<u8*>(out))) {
            mbedtls_md_free(&context);
            return false;
        }

        mbedtls_md_free(&context);
        return true;
    }

    Loader::ResultStatus deriveSDKeys(std::array<Core::Crypto::Key256, 2>& out,
                                      const Core::Crypto::KeyManager& keys) {
        keys.DeriveSDSeedLazy();
        return Core::Crypto::DeriveSDKeys(out, keys);
    }

    std::string sanitizeNAXFilePath(std::string_view path) {
        return Common::FS::SanitizePath(path);
    }

    Loader::ResultStatus verifyFileSize(VirtualFile file, u64 expectedFileSize) {
        if (file->GetSize() < NAX_HEADER_PADDING_SIZE + expectedFileSize) {
            return Loader::ResultStatus::ErrorIncorrectNAXFileSize;
        }
        return Loader::ResultStatus::Success;
    }

    Loader::ResultStatus verifyHeader(const NAX::NAXHeader& header) {
        if (header.magic != NAX_MAGIC) {
            return Loader::ResultStatus::ErrorBadNAXHeader;
        }
        return Loader::ResultStatus::Success;
    }

    Loader::ResultStatus decryptKeyArea(std::array<Core::Crypto::Key128, 2>& out,
                                         const Core::Crypto::Key256& naxKey,
                                         const std::array<Core::Crypto::Key128, 2>& encryptedKeyArea) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            Core::Crypto::AESCipher<Core::Crypto::Key128> cipher(naxKey.data(),
                    Core::Crypto::Mode::ECB);
            cipher.Transcode(out[i].data(), out[i].size(), encryptedKeyArea[i].data(),
                             Core::Crypto::Op::Decrypt);
        }
        return Loader::ResultStatus::Success;
    }

    Loader::ResultStatus validateHMAC(Core::Crypto::SHA256Hash& out,
                                      const NAX::NAXHeader& header, const Core::Crypto::Key256& key) {
        if (!calculateHMAC(out.data(), &header.magic, sizeof(NAX::NAXHeader) - NAX_HMAC_SIZE,
                           key.data() + NAX_KEY_AREA_SIZE, NAX_KEY_AREA_SIZE)) {
            return Loader::ResultStatus::ErrorNAXValidationHMACFailed;
        }
        if (header.hmac != out) {
            return Loader::ResultStatus::ErrorNAXValidationHMACFailed;
        }
        return Loader::ResultStatus::Success;
    }

} // anonymous namespace

NAX::NAX(VirtualFile file)
    : header_(std::make_unique<NAXHeader>()),
      file_(std::move(file)),
      status_{ParseHeader()},
      type_{NAXContentType::Unknown} {}

NAX::NAX(VirtualFile file, std::array<u8, 0x10> ncaId)
    : header_(std::make_unique<NAXHeader>()),
      file_(std::move(file)),
      status_{DecryptFile(GetNAXFilePath())},
      type_{NAXContentType::Unknown} {}

Loader::ResultStatus NAX::GetStatus() const {
    return status_;
}

VirtualFile NAX::GetDecrypted() const {
    return decrypted_;
}

std::unique_ptr<NCA> NAX::AsNCA() const {
    if (type_ == NAXContentType::NCA) {
        return std::make_unique<NCA>(GetDecrypted());
    }
    return nullptr;
}

NAXContentType NAX::GetContentType() const {
    return type_;
}

std::vector<VirtualFile> NAX::GetFiles() const {
    return {GetDecrypted()};
}

std::vector<VirtualDir> NAX::GetSubdirectories() const {
    return {};
}

std::string NAX::GetName() const {
    return file_->GetName();
}

VirtualDir NAX::GetParentDirectory() const {
    return file_->GetContainingDirectory();
}

Loader::ResultStatus NAX::ParseHeader() {
    if (file_ == nullptr) {
        return Loader::ResultStatus::ErrorNullFile;
    }
    if (file_->ReadObject(header_.get()) != sizeof(NAXHeader)) {
        return Loader::ResultStatus::ErrorBadNAXHeader;
    }
    return verifyHeader(*header_);
}

Loader::ResultStatus NAX::DecryptFile(std::string_view path) {
    const auto fileSize = header_->file_size;
    const auto expectedStatus = verifyFileSize(file_, fileSize);
    if (expectedStatus != Loader::ResultStatus::Success) {
        return expectedStatus;
    }

    std::array<Core::Crypto::Key256, 2> sdKeys{};
    const auto sdKeysRes = deriveSDKeys(sdKeys, Core::Crypto::KeyManager::Instance());
    if (sdKeysRes != Loader::ResultStatus::Success) {
        return sdKeysRes;
    }

    for (std::size_t i = 0; i < sdKeys.size(); ++i) {
        std::array<Core::Crypto::Key128, 2> naxKeys{};
        if (!calculateHMAC(naxKeys.data(), sdKeys[i].data(), sizeof(Core::Crypto::Key128),
                            path.data(), path.size())) {
            return Loader::ResultStatus::ErrorNAXKeyHMACFailed;
        }

        std::array<Core::Crypto::Key128, 2> encryptedKeyArea{};
        std::memcpy(&encryptedKeyArea, header_->key_area,
                    NAX_KEY_AREA_SIZE * encryptedKeyArea.size());
        const auto keyRes = decryptKeyArea(encryptedKeyArea, naxKeys, encryptedKeyArea);
        if (keyRes != Loader::ResultStatus::Success) {
            continue;
        }

        Core::Crypto::SHA256Hash validation{};
        const auto hmacRes = validateHMAC(validation, *header_, sdKeys[i]);
        if (hmacRes == Loader::ResultStatus::Success) {
            type_ = static_cast<NAXContentType>(i);
            Core::Crypto::Key256 finalKey{};
            std::memcpy(&finalKey, &encryptedKeyArea, sizeof(finalKey));
            const auto encFile =
                std::make_shared<VfsOffset>(file_, fileSize, NAX_HEADER_PADDING_SIZE);
            decrypted_ = std::make_shared<Core::Crypto::XTSEncryptionLayer>(encFile, finalKey);
            return Loader::ResultStatus::Success;
        }
    }

    return Loader::ResultStatus::ErrorNAXKeyDerivationFailed;
}

std::string NAX::GetNAXFilePath() const {
    std::string path = sanitizeNAXFilePath(file_->GetFullPath());
    static const std::regex naxPathRegex("/registered/(000000[0-9A-F]{2})/([0-9A-F]{32})\\.nca",
                                          std::regex_constants::ECMAScript |
                                              std::regex_constants::icase);
    std::smatch match;
    if (!std::regex_search(path, match, naxPathRegex)) {
        return std::string{};
    }

    const std::string twoDir = Common::ToUpper(match[1]);
    const std::string ncaId = Common::ToLower(match[2]);
    return fmt::format("/registered/{}/{}.nca", twoDir, ncaId);
}

} // namespace FileSys

