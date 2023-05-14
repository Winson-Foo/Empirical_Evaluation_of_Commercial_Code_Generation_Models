#include <algorithm>
#include <cstring>
#include "core/crypto/xts_encryption_layer.h"

namespace Core::Crypto {

constexpr u64 XTS_SECTOR_SIZE = 0x4000;

// Constructor
XTSEncryptionLayer::XTSEncryptionLayer(FileSys::VirtualFile base, Key256 key)
    : EncryptionLayer(std::move(base)), cipher(key, Mode::XTS) {}

// Read a given number of bytes from the encrypted file
std::size_t XTSEncryptionLayer::Read(u8* data, std::size_t length, std::size_t offset) const {
    // If no bytes requested, return 0
    if (length == 0) {
        return 0;
    }

    // Read a single block
    if (offset % XTS_SECTOR_SIZE == 0 && length <= XTS_SECTOR_SIZE) {
        return readSingleBlock(data, length, offset);
    }

    // Read multiple blocks
    std::size_t bytesRead = 0;
    while (bytesRead < length) {
        std::size_t remaining = length - bytesRead;
        std::size_t blockOffset = offset + bytesRead;
        std::size_t bytesToRead = std::min(remaining, XTS_SECTOR_SIZE - (blockOffset % XTS_SECTOR_SIZE));
        bytesRead += readSingleBlock(data + bytesRead, bytesToRead, blockOffset);
    }
    return bytesRead;
}

// Read a single block of encrypted data
std::size_t XTSEncryptionLayer::readSingleBlock(u8* data, std::size_t length, std::size_t offset) const {
    // Read encrypted data from disk
    std::vector<u8> encrypted = base->ReadBytes(XTS_SECTOR_SIZE, offset);
    if (encrypted.size() < XTS_SECTOR_SIZE) {
        encrypted.resize(XTS_SECTOR_SIZE);
    }

    // Decrypt the data using XTS
    cipher.XTSTranscode(encrypted.data(), encrypted.size(), encrypted.data(),
                        offset / XTS_SECTOR_SIZE, XTS_SECTOR_SIZE, Op::Decrypt);

    // Copy the decrypted data to the output buffer
    std::memcpy(data, encrypted.data() + (offset % XTS_SECTOR_SIZE), length);
    return length;
}

} // namespace Core::Crypto