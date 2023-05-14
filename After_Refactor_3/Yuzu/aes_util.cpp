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
    NintendoTweak out{};

    for (std::size_t i = 15; i <= 15; --i) {
        out[i] = sector_id & 0xFF;
        sector_id >>= 8;
    }

    return out;
}

void CheckStatus(int status, const char* message) {
    ASSERT_MSG(status == 0, message);
}

std::size_t GetBlockSize(mbedtls_cipher_context_t* context) {
    return mbedtls_cipher_get_block_size(context);
}

}  // namespace

// Structure to hide mbedtls types from header file
struct CipherContext {
    mbedtls_cipher_context_t encryption_context;
    mbedtls_cipher_context_t decryption_context;
};

template <typename Key, std::size_t KeySize>
Crypto::AESCipher<Key, KeySize>::AESCipher(Key key, Mode mode)
    : ctx(std::make_unique<CipherContext>()) {
    mbedtls_cipher_init(&ctx->encryption_context);
    mbedtls_cipher_init(&ctx->decryption_context);

    CheckStatus(mbedtls_cipher_setup(
                     &ctx->encryption_context,
                     mbedtls_cipher_info_from_type(static_cast<mbedtls_cipher_type_t>(mode))),
                 "Failed to initialize mbedtls encryption cipher.");

    CheckStatus(mbedtls_cipher_setup(
                     &ctx->decryption_context,
                     mbedtls_cipher_info_from_type(static_cast<mbedtls_cipher_type_t>(mode))),
                 "Failed to initialize mbedtls decryption cipher.");

    CheckStatus(mbedtls_cipher_setkey(&ctx->encryption_context, key.data(), KeySize * 8, MBEDTLS_ENCRYPT),
                 "Failed to set key on mbedtls encryption cipher.");

    CheckStatus(mbedtls_cipher_setkey(&ctx->decryption_context, key.data(), KeySize * 8, MBEDTLS_DECRYPT),
                 "Failed to set key on mbedtls decryption cipher.");
}

template <typename Key, std::size_t KeySize>
AESCipher<Key, KeySize>::~AESCipher() {
    mbedtls_cipher_free(&ctx->encryption_context);
    mbedtls_cipher_free(&ctx->decryption_context);
}

template <typename Key, std::size_t KeySize>
void AESCipher<Key, KeySize>::Transcode(std::span<const u8> src, std::size_t size, std::span<u8> dest, Op op) const {
    auto* const context = op == Op::Encrypt ? &ctx->encryption_context : &ctx->decryption_context;

    mbedtls_cipher_reset(context);

    std::size_t written = 0;
    if (mbedtls_cipher_get_cipher_mode(context) == MBEDTLS_MODE_XTS) {
        mbedtls_cipher_update(context, src.data(), size, dest.data(), &written);

        if (written != size) {
            LOG_WARNING(Crypto,
                        "Not all data was decoded. Requested {:016X}, actual {:016X}.",
                        size, written);
        }
    } else {
        const auto block_size = GetBlockSize(context);
        if (size < block_size) {
            std::vector<u8> block(block_size);
            std::memcpy(block.data(), src.data(), size);
            Transcode(block, block_size, block, op);
            std::memcpy(dest.data(), block.data(), size);
            return;
        }

        for (std::size_t offset = 0; offset < size; offset += block_size) {
            auto length = std::min<std::size_t>(block_size, size - offset);

            CheckStatus(mbedtls_cipher_update(
                             context,
                             src.data() + offset,
                             length,
                             dest.data() + offset,
                             &written),
                         "Failed to transcode with mbedtls cipher.");

            if (written != length) {
                if (length < block_size) {
                    std::vector<u8> block(block_size);
                    std::memcpy(block.data(), src.data() + offset, length);
                    Transcode(block, block_size, block, op);
                    std::memcpy(dest.data() + offset, block.data(), length);
                    return;
                }
                LOG_WARNING(Crypto,
                            "Not all data was decoded. Requested {:016X}, actual {:016X}.",
                            length, written);
            }
        }
    }
}

template <typename Key, std::size_t KeySize>
void AESCipher<Key, KeySize>::XTSTranscode(std::span<const u8> src,
                                            std::size_t size,
                                            std::span<u8> dest,
                                            std::size_t sector_id,
                                            std::size_t sector_size,
                                            Op op) {
    ASSERT_MSG(size % sector_size == 0, "XTS decryption size must be a multiple of sector size.");

    for (std::size_t i = 0; i < size; i += sector_size) {
        SetIV(CalculateNintendoTweak(sector_id++));
        Transcode(src.subspan(i, sector_size), sector_size, dest.subspan(i, sector_size), op);
    }
}

template <typename Key, std::size_t KeySize>
void AESCipher<Key, KeySize>::SetIV(std::span<const u8> data) {
    CheckStatus(mbedtls_cipher_set_iv(&ctx->encryption_context, data.data(), data.size()),
                "Failed to set IV on mbedtls encryption cipher.");

    CheckStatus(mbedtls_cipher_set_iv(&ctx->decryption_context, data.data(), data.size()),
                "Failed to set IV on mbedtls decryption cipher.");
}

template class AESCipher<Key128>;
template class AESCipher<Key256>;
}  // namespace Core::Crypto