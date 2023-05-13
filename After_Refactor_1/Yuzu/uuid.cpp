namespace Common {
namespace {

constexpr size_t UUID_RAW_STRING_SIZE = sizeof(UUID) * 2;
constexpr size_t UUID_FORMATTED_STRING_SIZE = UUID_RAW_STRING_SIZE + 4;

template <typename T>
std::optional<u8> HexCharToByte(T c) {
    if (c >= '0' && c <= '9') {
        return static_cast<u8>(c - '0');
    }
    if (c >= 'a' && c <= 'f') {
        return static_cast<u8>(c - 'a' + 10);
    }
    if (c >= 'A' && c <= 'F') {
        return static_cast<u8>(c - 'A' + 10);
    }
    return std::nullopt;
}

std::array<u8, 0x10> ConstructFromRawString(std::string_view raw_string) {
    std::array<u8, 0x10> uuid;

    for (size_t i = 0; i < UUID_RAW_STRING_SIZE; i += 2) {
        const auto upper = HexCharToByte(raw_string[i]);
        const auto lower = HexCharToByte(raw_string[i + 1]);
        if (!upper || !lower) {
            return {};
        }
        uuid[i / 2] = static_cast<u8>((*upper << 4) | *lower);
    }

    return uuid;
}

std::array<u8, 0x10> ConstructFromFormattedString(std::string_view formatted_string) {
    std::array<u8, 0x10> uuid;
    size_t i = 0;

    const auto* string = formatted_string.data();

    // Process the first 8 characters.
    for (; i < 4; ++i) {
        const auto upper = HexCharToByte(*(string++));
        const auto lower = HexCharToByte(*(string++));
        if (!upper || !lower) {
            return {};
        }
        uuid[i] = static_cast<u8>((*upper << 4) | *lower);
    }

    // Process the next 4 characters.
    ++string;

    for (; i < 6; ++i) {
        const auto upper = HexCharToByte(*(string++));
        const auto lower = HexCharToByte(*(string++));
        if (!upper || !lower) {
            return {};
        }
        uuid[i] = static_cast<u8>((*upper << 4) | *lower);
    }

    // Process the next 4 characters.
    ++string;

    for (; i < 8; ++i) {
        const auto upper = HexCharToByte(*(string++));
        const auto lower = HexCharToByte(*(string++));
        if (!upper || !lower) {
            return {};
        }
        uuid[i] = static_cast<u8>((*upper << 4) | *lower);
    }

    // Process the next 4 characters.
    ++string;

    for (; i < 10; ++i) {
        const auto upper = HexCharToByte(*(string++));
        const auto lower = HexCharToByte(*(string++));
        if (!upper || !lower) {
            return {};
        }
        uuid[i] = static_cast<u8>((*upper << 4) | *lower);
    }

    // Process the last 12 characters.
    ++string;

    for (; i < 16; ++i) {
        const auto upper = HexCharToByte(*(string++));
        const auto lower = HexCharToByte(*(string++));
        if (!upper || !lower) {
            return {};
        }
        uuid[i] = static_cast<u8>((*upper << 4) | *lower);
    }

    return uuid;
}

std::array<u8, 0x10> ConstructUUID(std::string_view uuid_string) {
    const auto length = uuid_string.length();

    if (length == 0) {
        return {};
    }

    if (length == UUID_RAW_STRING_SIZE) {
        return ConstructFromRawString(uuid_string);
    }

    if (length == UUID_FORMATTED_STRING_SIZE) {
        return ConstructFromFormattedString(uuid_string);
    }

    throw std::invalid_argument(fmt::format("UUID string has an invalid length of {} characters!", length));
}

}  // Anonymous namespace

UUID::UUID(std::string_view uuid_string) : uuid{ConstructUUID(uuid_string)} {}

std::string UUID::RawString() const {
    return fmt::format("{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}"
                       "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                       uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5],
                       uuid[6], uuid[7], uuid[8], uuid[9], uuid[10], uuid[11],
                       uuid[12], uuid[13], uuid[14], uuid[15]);
}

std::string UUID::FormattedString() const {
    return fmt::format("{:02x}{:02x}{:02x}{:02x}"
                       "-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-"
                       "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                       uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5],
                       uuid[6], uuid[7], uuid[8], uuid[9], uuid[10], uuid[11],
                       uuid[12], uuid[13], uuid[14], uuid[15]);
}

size_t UUID::Hash() const noexcept {
    u64 upper_hash;
    u64 lower_hash;

    std::memcpy(&upper_hash, uuid.data(), sizeof(u64));
    std::memcpy(&lower_hash, uuid.data() + sizeof(u64), sizeof(u64));

    return upper_hash ^ std::rotl(lower_hash, 1);
}

u128 UUID::AsU128() const {
    u128 uuid_old;
    std::memcpy(&uuid_old, uuid.data(), sizeof(UUID));
    return uuid_old;
}

UUID UUID::MakeRandom() {
    std::random_device device;
    return MakeRandomWithSeed(device());
}

UUID UUID::MakeRandomWithSeed(u32 seed) {
    TinyMT rng;
    rng.Initialize(seed);

    UUID uuid;
    rng.GenerateRandomBytes(uuid.uuid.data(), sizeof(UUID));

    return uuid;
}

UUID UUID::MakeRandomRFC4122V4() {
    auto uuid = MakeRandom();

    uuid.uuid[8] = 0x80 | (uuid.uuid[8] & 0x3F);

    uuid.uuid[6] = 0x40 | (uuid.uuid[6] & 0xF);

    return uuid;
}

}  // namespace Common