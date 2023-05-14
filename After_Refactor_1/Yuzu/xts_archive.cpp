#include <array>
#include <cstring>
#include <regex>
#include <string_view>
#include <variant>

#include <mbedtls/md.h>
#include <mbedtls/sha256.h>

#include "core/crypto/aes_util.h"
#include "core/crypto/key_manager.h"
#include "core/crypto/xts_encryption_layer.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/vfs_offset.h"
#include "core/file_sys/xts_archive.h"
#include "core/loader/loader.h"

namespace FileSys {

namespace fs = std::filesystem;

enum class NAXContentType {
    NCA,
    Other,
};

struct NAXHeader {
    u32 magic;
    u32 version;
    u8 reserved[0xC];
    u32 flags;
    u32 file_size;
    std::array<Core::Crypto::Key128, 4> key_area;
    Core::Crypto::SHA256Hash hmac;
};

constexpr u64 NAX_HEADER_PADDING_SIZE = 0x4000;

template <typename SourceData, typename SourceKey, typename Destination>
static bool CalculateHMAC256(Destination& out, const SourceKey& key, const SourceData& data) {
    mbedtls_md_context_t context;
    mbedtls_md_init(&context);

    if (mbedtls_md_setup(&context, mbedtls_md_info_from_type(MBEDTLS_MD_SHA256), 1) ||
        mbedtls_md_hmac_starts(&context, reinterpret_cast<const u8*>(&key), sizeof(key)) ||
        mbedtls_md_hmac_update(&context, reinterpret_cast<const u8*>(data.data()), data.size()) ||
        mbedtls_md_hmac_finish(&context, reinterpret_cast<u8*>(out.data()))) {
        mbedtls_md_free(&context);
        return false;
    }

    mbedtls_md_free(&context);
    return true;
}

class NAX {
public:
    explicit NAX(VirtualFile file)
        : file(std::move(file)), keys{Core::Crypto::KeyManager::Instance()} {
        const fs::path sanitized_path = Common::FS::SanitizePath(file->GetFullPath());
        static const std::regex nax_path_regex("/registered/(000000[0-9A-F]{2})/([0-9A-F]{32})\\.nca",
                                               std::regex_constants::ECMAScript |
                                                   std::regex_constants::icase);
        std::smatch match;
        if (!std::regex_search(sanitized_path.generic_string(), match, nax_path_regex)) {
            status = Loader::ResultStatus::ErrorBadNAXFilePath;
            return;
        }

        const std::string two_dir = Common::ToUpper(match[1]);
        const std::string nca_id = Common::ToLower(match[2]);
        status = Parse(fmt::format("/registered/{}/{}.nca", two_dir, nca_id));
    }

    explicit NAX(VirtualFile file, std::array<u8, 0x10> nca_id)
        : file(std::move(file)), keys{Core::Crypto::KeyManager::Instance()} {
        Core::Crypto::SHA256Hash hash{};
        mbedtls_sha256_ret(nca_id.data(), nca_id.size(), hash.data(), 0);
        status = Parse(fmt::format("/registered/000000{:02X}/{}.nca", hash[0],
                                   Common::HexToString(nca_id, false)));
    }

    ~NAX() = default;

    Loader::ResultStatus GetStatus() const {
        return status;
    }

    VirtualFile GetDecrypted() const {
        return dec_file.value();
    }

    std::unique_ptr<NCA> AsNCA() const {
        if (type == NAXContentType::NCA) {
            return std::make_unique<NCA>(GetDecrypted());
        }
        return nullptr;
    }

    NAXContentType GetContentType() const {
        return type;
    }

    std::vector<VirtualFile> GetFiles() const {
        return {GetDecrypted()};
    }

    std::vector<VirtualDir> GetSubdirectories() const {
        return {};
    }

    std::string GetName() const {
        return file->GetName();
    }

    VirtualDir GetParentDirectory() const {
        return file->GetContainingDirectory();
    }

private:
    Loader::ResultStatus Parse(const std::string_view path) {
        if (file == nullptr) {
            return Loader::ResultStatus::ErrorNullFile;
        }

        status = ReadHeader();
        if (status != Loader::ResultStatus::Success) {
            return status;
        }

        keys.DeriveSDSeedLazy();
        std::array<Core::Crypto::Key256, 2> sd_keys{};
        const Loader::ResultStatus sd_keys_res = Core::Crypto::DeriveSDKeys(sd_keys, keys);
        if (sd_keys_res != Loader::ResultStatus::Success) {
            return sd_keys_res;
        }

        const auto enc_keys = header.key_area;

        size_t i = 0;
        for (; i < sd_keys.size(); ++i) {
            std::array<Core::Crypto::Key128, 2> nax_keys{};
            if (!CalculateHMAC256(nax_keys, sd_keys[i], path)) {
                return Loader::ResultStatus::ErrorNAXKeyHMACFailed;
            }

            for (size_t j = 0; j < nax_keys.size(); ++j) {
                Core::Crypto::AESCipher<Core::Crypto::Key128> cipher(nax_keys[j],
                                                                     Core::Crypto::Mode::ECB);
                cipher.Transcode(enc_keys[j].data(), 0x10, header.key_area[j].data(),
                                 Core::Crypto::Op::Decrypt);
            }

            Core::Crypto::SHA256Hash validation{};
            if (!CalculateHMAC256(validation, header.magic, sd_keys[i].data() + 0x10, 0x10)) {
                return Loader::ResultStatus::ErrorNAXValidationHMACFailed;
            }
            if (header.hmac == validation) {
                break;
            }
        }

        if (i == 2) {
            return Loader::ResultStatus::ErrorNAXKeyDerivationFailed;
        }

        type = i == 0 ? NAXContentType::NCA : NAXContentType::Other;

        Core::Crypto::Key256 final_key{};
        std::memcpy(final_key.data(), &header.key_area, final_key.size());
        const auto enc_file =
            std::make_shared<OffsetVfsFile>(file, header.file_size, NAX_HEADER_PADDING_SIZE);
        dec_file = std::make_shared<Core::Crypto::XTSEncryptionLayer>(enc_file, final_key);

        return Loader::ResultStatus::Success;
    }

    Loader::ResultStatus ReadHeader() {
        header = std::make_unique<NAXHeader>();
        if (file->ReadObject(header.get()) != sizeof(NAXHeader)) {
            return Loader::ResultStatus::ErrorBadNAXHeader;
        }
        if (header->magic != Common::MakeMagic('N', 'A', 'X', '0')) {
            return Loader::ResultStatus::ErrorBadNAXHeader;
        }
        if (file->GetSize() < NAX_HEADER_PADDING_SIZE + header->file_size) {
            return Loader::ResultStatus::ErrorIncorrectNAXFileSize;
        }
        return Loader::ResultStatus::Success;
    }

    VirtualFile file;
    std::optional<NAXHeader> header;
    std::optional<VirtualFile> dec_file;
    NAXContentType type = NAXContentType::Other;
    Loader::ResultStatus status = Loader::ResultStatus::Success;
    const Core::Crypto::Keyset& keys;
};

} // namespace FileSys