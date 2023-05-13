namespace Common {
namespace {

constexpr size_t UUID_BYTE_SIZE = 16;
constexpr size_t UUID_RAW_STRING_SIZE = UUID_BYTE_SIZE * 2;
constexpr size_t UUID_FORMATTED_STRING_SIZE = UUID_RAW_STRING_SIZE + 4;

bool isHexChar(char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
}

u8 hexCharToByte(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return 0;
}

bool hasValidUUIDLength(size_t len) {
    return len == UUID_RAW_STRING_SIZE || len == UUID_FORMATTED_STRING_SIZE;
}

std::optional<std::array<u8, UUID_BYTE_SIZE>> constructFromRawString(std::string_view raw_string) {
    std::array<u8, UUID_BYTE_SIZE> uuid{};
    for (size_t i = 0, j = 0; i < UUID_RAW_STRING_SIZE; i += 2, ++j) {
        if (!isHexChar(raw_string[i]) || !isHexChar(raw_string[i + 1])) {
            return std::nullopt;
        }
        uuid[j] = (hexCharToByte(raw_string[i]) << 4) + hexCharToByte(raw_string[i + 1]);
    }
    return uuid;
}

std::optional<std::array<u8, UUID_BYTE_SIZE>> constructFromFormattedString(std::string_view formatted_string) {
    std::array<u8, UUID_BYTE_SIZE> uuid{};
    size_t i = 0;
    const auto* str = formatted_string.data();
    for (; i < 8; ++i) {
        if (*str != '-') {
            const u8 byte = (hexCharToByte(*str++) << 4) + hexCharToByte(*str++);
            uuid[i] = byte;
        } else {
            ++str;
        }
    }
    ++str;
    for (; i < 12; ++i) {
        if (*str != '-') {
            const u8 byte = (hexCharToByte(*str++) << 4) + hexCharToByte(*str++);
            uuid[i] = byte;
        } else {
            ++str;
        }
    }
    ++str;
    for (; i < 16; ++i) {
        if (*str != '-') {
            const u8 byte = (hexCharToByte(*str++) << 4) + hexCharToByte(*str++);
            uuid[i] = byte;
        } else {
            ++str;
        }
    }
    return uuid;
}

std::optional<std::array<u8, UUID_BYTE_SIZE>> constructUUID(std::string_view uuid_string) {
    const auto length = uuid_string.length();
    if (!hasValidUUIDLength(length)) {
        return std::nullopt;
    }
    if (length == UUID_RAW_STRING_SIZE) {
        return constructFromRawString(uuid_string);
    }
    if (length == UUID_FORMATTED_STRING_SIZE) {
        return constructFromFormattedString(uuid_string);
    }
    return std::nullopt;
}

} // Anonymous namespace

UUID::UUID(std::string_view uuid_string) 
    : uuid{constructUUID(uuid_string).value_or(std::array<u8, UUID_BYTE_SIZE>{})} {}

std::string UUID::rawString() const {
    std::string str;
    str.reserve(UUID_RAW_STRING_SIZE);
    for (const u8 byte : uuid) {
        str += fmt::format("{:02x}", byte);
    }
    return str;
}

std::string UUID::formattedString() const {
    return fmt::format("{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}", 
        uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7], 
        uuid[8], uuid[9], uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]);
}

size_t UUID::hash() const noexcept {
    u64 upper_hash;
    u64 lower_hash;
    std::memcpy(&upper_hash, uuid.data(), sizeof(u64));
    std::memcpy(&lower_hash, uuid.data() + sizeof(u64), sizeof(u64));
    return upper_hash ^ std::rotl(lower_hash, 1);
}

u128 UUID::asU128() const {
    u128 uuid_old;
    std::memcpy(&uuid_old, uuid.data(), sizeof(UUID));
    return uuid_old;
}

UUID UUID::makeRandom() {
    std::random_device device;
    return makeRandomWithSeed(device());
}

UUID UUID::makeRandomWithSeed(u32 seed) {
    TinyMT rng;
    rng.initialize(seed);
    UUID uuid;
    rng.generateRandomBytes(uuid.uuid.data(), sizeof(UUID));
    return uuid;
}

UUID UUID::makeRandomRFC4122V4() {
    auto uuid = makeRandom();
    // Set bits 6 and 7 of the clock-seq-hi-and-reserved byte to 0 and 1.
    uuid.uuid[8] = 0x80 | (uuid.uuid[8] & 0x3F);
    // Set bits 12 through 15 of the time_hi_and_version field to version 4.
    uuid.uuid[6] = 0x40 | (uuid.uuid[6] & 0xF);
    return uuid;
}

} // namespace Common