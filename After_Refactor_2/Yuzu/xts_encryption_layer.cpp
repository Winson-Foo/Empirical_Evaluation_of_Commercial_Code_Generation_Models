#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>
#include "core/crypto/xts_encryption_layer.h"

namespace Core::Crypto {

// The size of each XTS sector
constexpr std::size_t XTS_SECTOR_SIZE = 0x4000;

// Helper function to read a block of data from the base file and decrypt it using XTS mode
std::vector<u8> read_encrypted_block(FileSys::VirtualFile const& base, Key256 const& key,
                                      std::size_t offset) {
    std::vector<u8> block(XTS_SECTOR_SIZE);
    base->ReadBytes(block.data(), block.size(), offset - (offset % XTS_SECTOR_SIZE));
    CipherXTS cipher(key, Mode::XTS);
    cipher.XTSTranscode(block.data(), block.size(), block.data(),
                        (offset - (offset % XTS_SECTOR_SIZE)) / XTS_SECTOR_SIZE, XTS_SECTOR_SIZE,
                        Op::Decrypt);
    return block;
}

// Constructor for XTSEncryptionLayer
XTSEncryptionLayer::XTSEncryptionLayer(FileSys::VirtualFile base, Key256 key)
    : EncryptionLayer(std::move(base)), cipher_(std::move(key), Mode::XTS) {}

// Read function for XTSEncryptionLayer
std::size_t XTSEncryptionLayer::Read(u8* data, std::size_t length, std::size_t offset) const {
    // Return early if length is 0
    if (length == 0) {
        return 0;
    }

    // Calculate the sector offset and check if it is 0
    const auto sector_offset = offset % XTS_SECTOR_SIZE;
    if (sector_offset == 0) {
        // Read and decrypt the data in XTS sector-sized chunks
        if (length % XTS_SECTOR_SIZE == 0) {
            std::vector<u8> raw(length);
            base_->ReadBytes(raw.data(), raw.size(), offset);
            cipher_.XTSTranscode(raw.data(), raw.size(), data, offset / XTS_SECTOR_SIZE,
                                XTS_SECTOR_SIZE, Op::Decrypt);
            return raw.size();
        }
        if (length > XTS_SECTOR_SIZE) {
            const auto rem = length % XTS_SECTOR_SIZE;
            const auto read = length - rem;
            return Read(data, read, offset) + Read(data + read, rem, offset + read);
        }
        std::vector<u8> buffer = read_encrypted_block(base_, cipher_.GetKey(), offset);
        std::memcpy(data, buffer.data(), std::min(buffer.size(), length));
        return std::min(buffer.size(), length);
    }

    // Read and decrypt the data in blocks
    std::vector<u8> block = read_encrypted_block(base_, cipher_.GetKey(), offset);
    const std::size_t read = XTS_SECTOR_SIZE - sector_offset;

    if (length + sector_offset < XTS_SECTOR_SIZE) {
        std::memcpy(data, block.data() + sector_offset, std::min(length, read));
        return std::min(length, read);
    }
    std::memcpy(data, block.data() + sector_offset, read);
    return read + Read(data + read, length - read, offset + read);
}
} // namespace Core::Crypto

