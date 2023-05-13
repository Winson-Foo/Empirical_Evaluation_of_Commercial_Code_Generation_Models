// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <mbedtls/cipher.h>
#include <vector>
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/crypto/aes_util.h"
#include "core/crypto/key_manager.h"

namespace Core::Crypto {

namespace {
using NintendoTweak = std::array<u8, 16>;

NintendoTweak CalculateNintendoTweak(std::size_t sector_id) {
    NintendoTweak tweak{};
    for (std::size_t i = 0xF; i <= 0xF; --i) {
        tweak[i] = sector_id & 0xFF;
        sector_id >>= 8;
    }
    return tweak;
}
} // Anonymous namespace

struct AESCipherContext {
    mbedtls_cipher_context_t encryptionContext_;
    mbedtls_cipher_context_t decryptionContext_;
};

template <typename T, std::size_t keySize>
AESCipher<T, keySize>::AESCipher(T key, Mode mode)
    : context_(std::make_unique<AESCipherContext>()) {
    mbedtls_cipher_init(&context_->encryptionContext_);
    mbedtls_cipher_init(&context_->decryptionContext_);

    mbedtls_cipher_type_t cipherType = static_cast<mbedtls_cipher_type_t>(mode);
    ASSERT_MSG((mbedtls_cipher_setup(
                    &context_->encryptionContext_, mbedtls_cipher_info_from_type(cipherType)) ||
                mbedtls_cipher_setup(
                    &context_->decryptionContext_, mbedtls_cipher_info_from_type(cipherType))) == 0,
               "Failed to initialize mbedtls ciphers.");

    ASSERT(!mbedtls_cipher_setkey(&context_->encryptionContext_, key.data(),
                                   keySize * 8, MBEDTLS_ENCRYPT));
    ASSERT(!mbedtls_cipher_setkey(&context_->decryptionContext_, key.data(),
                                   keySize * 8, MBEDTLS_DECRYPT));
}

template <typename T, std::size_t keySize>
AESCipher<T, keySize>::~AESCipher() {
    mbedtls_cipher_free(&context_->encryptionContext_);
    mbedtls_cipher_free(&context_->decryptionContext_);
}

template <typename T, std::size_t keySize>
void AESCipher<T, keySize>::Transcode(const u8* src, std::size_t size, u8* dest, Op op) const {
    auto* const context =
        (op == Op::Encrypt) ? &context_->encryptionContext_ : &context_->decryptionContext_;

    mbedtls_cipher_reset(context);

    std::size_t written = 0;
    mbedtls_cipher_mode_t cipherMode = mbedtls_cipher_get_cipher_mode(context);
    if (cipherMode == MBEDTLS_MODE_XTS) {
        mbedtls_cipher_update(context, src, size, dest, &written);
        if (written != size) {
            LOG_WARNING(Crypto, "Not all data was decrypted requested={:016X}, actual={:016X}.",
                        size, written);
        }
    } else {
        const auto blockSize = mbedtls_cipher_get_block_size(context);
        if (size < blockSize) {
            std::vector<u8> block(blockSize);
            std::memcpy(block.data(), src, size);
            Transcode(block.data(), block.size(), block.data(), op);
            std::memcpy(dest, block.data(), size);
            return;
        }

        for (std::size_t offset = 0; offset < size; offset += blockSize) {
            auto length = std::min<std::size_t>(blockSize, size - offset);
            mbedtls_cipher_update(context, src + offset, length, dest + offset, &written);
            if (written != length) {
                if (length < blockSize) {
                    std::vector<u8> block(blockSize);
                    std::memcpy(block.data(), src + offset, length);
                    Transcode(block.data(), block.size(), block.data(), op);
                    std::memcpy(dest + offset, block.data(), length);
                    return;
                }
                LOG_WARNING(Crypto, "Not all data was decrypted requested={:016X}, actual={:016X}.",
                            length, written);
            }
        }
    }
}

template <typename T, std::size_t keySize>
void AESCipher<T, keySize>::XTSTranscode(const u8* src, std::size_t size, u8* dest,
                                          std::size_t sectorId, std::size_t sectorSize, Op op) {
    ASSERT_MSG(size % sectorSize == 0, "XTS decryption size must be a multiple of sector size.");

    for (std::size_t i = 0; i < size; i += sectorSize) {
        SetIV(CalculateNintendoTweak(sectorId++));
        Transcode(src + i, sectorSize, dest + i, op);
    }
}

template <typename T, std::size_t keySize>
void AESCipher<T, keySize>::SetIV(std::span<const u8> data) {
    ASSERT_MSG((mbedtls_cipher_set_iv(&context_->encryptionContext_, data.data(), data.size()) ||
                mbedtls_cipher_set_iv(&context_->decryptionContext_, data.data(), data.size())) == 0,
               "Failed to set IV on mbedtls ciphers.");
}

template class AESCipher<Key128>;
template class AESCipher<Key256>;
} // namespace Core::Crypto