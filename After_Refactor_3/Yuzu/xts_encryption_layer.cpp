namespace Core::Crypto {

// The size of each sector/block for XTS encryption
constexpr uint64_t kXtsSectorSize = 0x4000;

// Constructor
XTSEncryptionLayer::XTSEncryptionLayer(const FileSys::VirtualFile& base, const Key256& key)
    : EncryptionLayer(base), cipher_(key, Mode::XTS) {
}

// Reads data from the file and decrypts it using XTS mode
std::size_t XTSEncryptionLayer::Read(u8* data, std::size_t length, std::size_t offset) const {
    // If length is 0, return 0 since there is nothing to read
    if (length == 0) {
        return 0;
    }

    // Check if the offset is aligned with a sector boundary
    const uint64_t sector_offset = offset & (kXtsSectorSize - 1);
    if (sector_offset == 0) {
        // If the length is a multiple of the sector size, decrypt the entire buffer
        if (length % kXtsSectorSize == 0) {
            std::vector<u8> raw = base_->ReadBytes(length, offset);

            // Decrypt the data using XTS mode
            cipher_.XTSTranscode(raw.data(), raw.size(), data, offset / kXtsSectorSize,
                                kXtsSectorSize, Op::Decrypt);

            return raw.size();
        }

        // If the length is larger than the sector size, read in sector-size chunks and decrypt them recursively
        if (length > kXtsSectorSize) {
            const uint64_t rem = length % kXtsSectorSize;
            const uint64_t read = length - rem;
            return Read(data, read, offset) + Read(data + read, rem, offset + read);
        }

        // Otherwise, read in a single sector, decrypt it, and copy the decrypted data into the buffer
        std::vector<u8> buffer = base_->ReadBytes(kXtsSectorSize, offset);

        // Extend the buffer to the sector size if it is smaller
        if (buffer.size() < kXtsSectorSize) {
            buffer.resize(kXtsSectorSize);
        }

        // Decrypt the sector using XTS mode
        cipher_.XTSTranscode(buffer.data(), buffer.size(), buffer.data(), offset / kXtsSectorSize,
                            kXtsSectorSize, Op::Decrypt);

        // Copy the decrypted data into the buffer
        std::memcpy(data, buffer.data(), std::min(buffer.size(), length));
        return std::min(buffer.size(), length);
    }

    // If the offset is not aligned with a sector boundary, read in the entire sector
    std::vector<u8> block = base_->ReadBytes(kXtsSectorSize, offset - sector_offset);

    // Extend the block to the sector size if it is smaller
    if (block.size() < kXtsSectorSize) {
        block.resize(kXtsSectorSize);
    }

    // Decrypt the sector using XTS mode
    cipher_.XTSTranscode(block.data(), block.size(), block.data(),
                        (offset - sector_offset) / kXtsSectorSize, kXtsSectorSize, Op::Decrypt);

    // Copy the part of the sector that overlaps with the requested data into the buffer
    const std::size_t read = kXtsSectorSize - sector_offset;
    if (length + sector_offset < kXtsSectorSize) {
        std::memcpy(data, block.data() + sector_offset, std::min(length, read));
        return std::min(length, read);
    }

    // If we need to read more than one sector, copy the part of the sector that overlaps with the requested data into the buffer
    std::memcpy(data, block.data() + sector_offset, read);
    return read + Read(data + read, length - read, offset + read);
}

} // namespace Core::Crypto