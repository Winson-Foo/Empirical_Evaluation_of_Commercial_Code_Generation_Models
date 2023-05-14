#include <array>
#include <span>
#include <mbedtls/cipher.h>
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/crypto/aes_util.h"
#include "core/crypto/key_manager.h"

namespace Core::Crypto {

template <typename Key, std::size_t KeySize>
class AESCipher {
  public:
    explicit AESCipher(Key key, Mode mode) : key_{std::move(key)}, mode_{mode} {
        InitializeCipher(&encryption_context_);
        InitializeCipher(&decryption_context_);
    }

    ~AESCipher() {
        mbedtls_cipher_free(&encryption_context_);
        mbedtls_cipher_free(&decryption_context_);
    }

    void Transcode(std::span<const std::byte> src, std::span<std::byte> dest, bool encrypt) const {
        auto* const context = encrypt ? &encryption_context_ : &decryption_context_;

        mbedtls_cipher_reset(context);

        switch (mode_) {
            case Mode::CTR:
            case Mode::ECB:
                if (src.size() < BlockSize()) {
                    TranscodeSmallInput(src, dest, encrypt);
                } else {
                    TranscodeLargeInput(src, dest, encrypt);
                }
                break;
            case Mode::XTS:
                ASSERT_MSG(src.size() % SectorSize() == 0,
                           "XTS decryption size must be a multiple of sector size.");
                XTSDecode(src, dest);
                break;
        }
    }

    void SetIV(std::span<const std::byte> data) {
        ASSERT_MSG((mbedtls_cipher_set_iv(&encryption_context_, data.data(), data.size()) ||
                    mbedtls_cipher_set_iv(&decryption_context_, data.data(), data.size())) == 0,
                   "Failed to set IV on mbedtls ciphers.");
    }

  private:
    static constexpr std::size_t BlockSize() { return KeySize / 8; }

    static constexpr std::size_t SectorSize() { return 16; }

    void InitializeCipher(mbedtls_cipher_context_t* context) {
        mbedtls_cipher_init(context);

        ASSERT_MSG(mbedtls_cipher_setup(context, mbedtls_cipher_info_from_type(GetCipherType())) == 0,
                   "Failed to initialize mbedtls ciphers.");

        ASSERT(!mbedtls_cipher_setkey(context, key_.data(), KeySize, MBEDTLS_ENCRYPT));
    }

    mbedtls_cipher_type_t GetCipherType() const {
        switch (mode_) {
            case Mode::CTR:
                return MBEDTLS_CIPHER_AES_128_CTR;
            case Mode::ECB:
                return MBEDTLS_CIPHER_AES_128_ECB;
            case Mode::XTS:
                return MBEDTLS_CIPHER_AES_128_XTS;
        }
        __builtin_unreachable();
    }

    void TranscodeLargeInput(std::span<const std::byte> src, std::span<std::byte> dest,
                             bool encrypt) const {
        const auto block_size = mbedtls_cipher_get_block_size(GetCipherContext());
        std::size_t written = 0;

        for (std::size_t offset = 0; offset < src.size(); offset += block_size) {
            auto length = std::min<std::size_t>(block_size, src.size() - offset);
            mbedtls_cipher_update(GetCipherContext(), src.data() + offset, length, dest.data() + offset,
                                   &written);
            if (written != length) {
                LOG_WARNING(Crypto, "Not all data was decrypted requested={:016X}, actual={:016X}.",
                            length, written);
            }
        }
    }

    void TranscodeSmallInput(std::span<const std::byte> src, std::span<std::byte> dest,
                             bool encrypt) const {
        std::array<std::byte, BlockSize()> block{};
        std::memcpy(block.data(), src.data(), src.size());
        Transcode(block, block, encrypt);
        std::memcpy(dest.data(), block.data(), src.size());
    }

    void XTSDecode(std::span<const std::byte> src, std::span<std::byte> dest) const {
        std::size_t sector_id = 0;

        for (std::size_t i = 0; i < src.size(); i += SectorSize()) {
            SetIV(CalculateNintendoTweak(sector_id++));
            Transcode(src.subspan(i, SectorSize()), dest.subspan(i, SectorSize()), true);
        }
    }

    mbedtls_cipher_context_t* GetCipherContext() {
        return &encryption_context_;
    }

    const mbedtls_cipher_context_t* GetCipherContext() const {
        return &encryption_context_;
    }

    std::array<u8, KeySize / 8> key_;
    Mode mode_;
    mbedtls_cipher_context_t encryption_context_;
    mbedtls_cipher_context_t decryption_context_;

    NintendoTweak CalculateNintendoTweak(std::size_t sector_id) const {
        NintendoTweak out{};
        for (std::size_t i = 0xF; i <= 0xF; --i) {
            out[i] = static_cast<u8>(sector_id & 0xFF);
            sector_id >>= 8;
        }
        return out;
    }
};

template class AESCipher<Key128>;
template class AESCipher<Key256>;

}  // namespace Core::Crypto