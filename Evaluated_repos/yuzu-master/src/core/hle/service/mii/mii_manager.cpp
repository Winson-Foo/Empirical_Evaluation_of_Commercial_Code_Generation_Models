// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <random>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/string_util.h"

#include "core/hle/service/acc/profile_manager.h"
#include "core/hle/service/mii/mii_manager.h"
#include "core/hle/service/mii/raw_data.h"

namespace Service::Mii {

namespace {

constexpr Result ERROR_CANNOT_FIND_ENTRY{ErrorModule::Mii, 4};

constexpr std::size_t BaseMiiCount{2};
constexpr std::size_t DefaultMiiCount{RawData::DefaultMii.size()};

constexpr MiiStoreData::Name DefaultMiiName{u'y', u'u', u'z', u'u'};
constexpr std::array<u8, 8> HairColorLookup{8, 1, 2, 3, 4, 5, 6, 7};
constexpr std::array<u8, 6> EyeColorLookup{8, 9, 10, 11, 12, 13};
constexpr std::array<u8, 5> MouthColorLookup{19, 20, 21, 22, 23};
constexpr std::array<u8, 7> GlassesColorLookup{8, 14, 15, 16, 17, 18, 0};
constexpr std::array<u8, 62> EyeRotateLookup{
    {0x03, 0x04, 0x04, 0x04, 0x03, 0x04, 0x04, 0x04, 0x03, 0x04, 0x04, 0x04, 0x04, 0x03, 0x03, 0x04,
     0x04, 0x04, 0x03, 0x03, 0x04, 0x03, 0x04, 0x03, 0x03, 0x04, 0x03, 0x04, 0x04, 0x03, 0x04, 0x04,
     0x04, 0x03, 0x03, 0x03, 0x04, 0x04, 0x03, 0x03, 0x03, 0x04, 0x04, 0x03, 0x03, 0x03, 0x03, 0x03,
     0x03, 0x03, 0x03, 0x03, 0x04, 0x04, 0x04, 0x04, 0x03, 0x04, 0x04, 0x03, 0x04, 0x04}};
constexpr std::array<u8, 24> EyebrowRotateLookup{{0x06, 0x06, 0x05, 0x07, 0x06, 0x07, 0x06, 0x07,
                                                  0x04, 0x07, 0x06, 0x08, 0x05, 0x05, 0x06, 0x06,
                                                  0x07, 0x07, 0x06, 0x06, 0x05, 0x06, 0x07, 0x05}};

template <typename T, std::size_t SourceArraySize, std::size_t DestArraySize>
std::array<T, DestArraySize> ResizeArray(const std::array<T, SourceArraySize>& in) {
    std::array<T, DestArraySize> out{};
    std::memcpy(out.data(), in.data(), sizeof(T) * std::min(SourceArraySize, DestArraySize));
    return out;
}

CharInfo ConvertStoreDataToInfo(const MiiStoreData& data) {
    MiiStoreBitFields bf;
    std::memcpy(&bf, data.data.data.data(), sizeof(MiiStoreBitFields));

    return {
        .uuid = data.data.uuid,
        .name = ResizeArray<char16_t, 10, 11>(data.data.name),
        .font_region = static_cast<u8>(bf.font_region.Value()),
        .favorite_color = static_cast<u8>(bf.favorite_color.Value()),
        .gender = static_cast<u8>(bf.gender.Value()),
        .height = static_cast<u8>(bf.height.Value()),
        .build = static_cast<u8>(bf.build.Value()),
        .type = static_cast<u8>(bf.type.Value()),
        .region_move = static_cast<u8>(bf.region_move.Value()),
        .faceline_type = static_cast<u8>(bf.faceline_type.Value()),
        .faceline_color = static_cast<u8>(bf.faceline_color.Value()),
        .faceline_wrinkle = static_cast<u8>(bf.faceline_wrinkle.Value()),
        .faceline_make = static_cast<u8>(bf.faceline_makeup.Value()),
        .hair_type = static_cast<u8>(bf.hair_type.Value()),
        .hair_color = static_cast<u8>(bf.hair_color.Value()),
        .hair_flip = static_cast<u8>(bf.hair_flip.Value()),
        .eye_type = static_cast<u8>(bf.eye_type.Value()),
        .eye_color = static_cast<u8>(bf.eye_color.Value()),
        .eye_scale = static_cast<u8>(bf.eye_scale.Value()),
        .eye_aspect = static_cast<u8>(bf.eye_aspect.Value()),
        .eye_rotate = static_cast<u8>(bf.eye_rotate.Value()),
        .eye_x = static_cast<u8>(bf.eye_x.Value()),
        .eye_y = static_cast<u8>(bf.eye_y.Value()),
        .eyebrow_type = static_cast<u8>(bf.eyebrow_type.Value()),
        .eyebrow_color = static_cast<u8>(bf.eyebrow_color.Value()),
        .eyebrow_scale = static_cast<u8>(bf.eyebrow_scale.Value()),
        .eyebrow_aspect = static_cast<u8>(bf.eyebrow_aspect.Value()),
        .eyebrow_rotate = static_cast<u8>(bf.eyebrow_rotate.Value()),
        .eyebrow_x = static_cast<u8>(bf.eyebrow_x.Value()),
        .eyebrow_y = static_cast<u8>(bf.eyebrow_y.Value() + 3),
        .nose_type = static_cast<u8>(bf.nose_type.Value()),
        .nose_scale = static_cast<u8>(bf.nose_scale.Value()),
        .nose_y = static_cast<u8>(bf.nose_y.Value()),
        .mouth_type = static_cast<u8>(bf.mouth_type.Value()),
        .mouth_color = static_cast<u8>(bf.mouth_color.Value()),
        .mouth_scale = static_cast<u8>(bf.mouth_scale.Value()),
        .mouth_aspect = static_cast<u8>(bf.mouth_aspect.Value()),
        .mouth_y = static_cast<u8>(bf.mouth_y.Value()),
        .beard_color = static_cast<u8>(bf.beard_color.Value()),
        .beard_type = static_cast<u8>(bf.beard_type.Value()),
        .mustache_type = static_cast<u8>(bf.mustache_type.Value()),
        .mustache_scale = static_cast<u8>(bf.mustache_scale.Value()),
        .mustache_y = static_cast<u8>(bf.mustache_y.Value()),
        .glasses_type = static_cast<u8>(bf.glasses_type.Value()),
        .glasses_color = static_cast<u8>(bf.glasses_color.Value()),
        .glasses_scale = static_cast<u8>(bf.glasses_scale.Value()),
        .glasses_y = static_cast<u8>(bf.glasses_y.Value()),
        .mole_type = static_cast<u8>(bf.mole_type.Value()),
        .mole_scale = static_cast<u8>(bf.mole_scale.Value()),
        .mole_x = static_cast<u8>(bf.mole_x.Value()),
        .mole_y = static_cast<u8>(bf.mole_y.Value()),
        .padding = 0,
    };
}

u16 GenerateCrc16(const void* data, std::size_t size) {
    s32 crc{};
    for (std::size_t i = 0; i < size; i++) {
        crc ^= static_cast<const u8*>(data)[i] << 8;
        for (std::size_t j = 0; j < 8; j++) {
            crc <<= 1;
            if ((crc & 0x10000) != 0) {
                crc = (crc ^ 0x1021) & 0xFFFF;
            }
        }
    }
    return Common::swap16(static_cast<u16>(crc));
}

template <typename T>
T GetRandomValue(T min, T max) {
    std::random_device device;
    std::mt19937 gen(device());
    std::uniform_int_distribution<u64> distribution(static_cast<u64>(min), static_cast<u64>(max));
    return static_cast<T>(distribution(gen));
}

template <typename T>
T GetRandomValue(T max) {
    return GetRandomValue<T>({}, max);
}

MiiStoreData BuildRandomStoreData(Age age, Gender gender, Race race, const Common::UUID& user_id) {
    MiiStoreBitFields bf{};

    if (gender == Gender::All) {
        gender = GetRandomValue<Gender>(Gender::Maximum);
    }

    bf.gender.Assign(gender);
    bf.favorite_color.Assign(GetRandomValue<u8>(11));
    bf.region_move.Assign(0);
    bf.font_region.Assign(FontRegion::Standard);
    bf.type.Assign(0);
    bf.height.Assign(64);
    bf.build.Assign(64);

    if (age == Age::All) {
        const auto temp{GetRandomValue<int>(10)};
        if (temp >= 8) {
            age = Age::Old;
        } else if (temp >= 4) {
            age = Age::Normal;
        } else {
            age = Age::Young;
        }
    }

    if (race == Race::All) {
        const auto temp{GetRandomValue<int>(10)};
        if (temp >= 8) {
            race = Race::Black;
        } else if (temp >= 4) {
            race = Race::White;
        } else {
            race = Race::Asian;
        }
    }

    u32 axis_y{};
    if (gender == Gender::Female && age == Age::Young) {
        axis_y = GetRandomValue<u32>(3);
    }

    const std::size_t index{3 * static_cast<std::size_t>(age) +
                            9 * static_cast<std::size_t>(gender) + static_cast<std::size_t>(race)};

    const auto faceline_type_info{RawData::RandomMiiFaceline.at(index)};
    const auto faceline_color_info{RawData::RandomMiiFacelineColor.at(
        3 * static_cast<std::size_t>(gender) + static_cast<std::size_t>(race))};
    const auto faceline_wrinkle_info{RawData::RandomMiiFacelineWrinkle.at(index)};
    const auto faceline_makeup_info{RawData::RandomMiiFacelineMakeup.at(index)};
    const auto hair_type_info{RawData::RandomMiiHairType.at(index)};
    const auto hair_color_info{RawData::RandomMiiHairColor.at(3 * static_cast<std::size_t>(race) +
                                                              static_cast<std::size_t>(age))};
    const auto eye_type_info{RawData::RandomMiiEyeType.at(index)};
    const auto eye_color_info{RawData::RandomMiiEyeColor.at(static_cast<std::size_t>(race))};
    const auto eyebrow_type_info{RawData::RandomMiiEyebrowType.at(index)};
    const auto nose_type_info{RawData::RandomMiiNoseType.at(index)};
    const auto mouth_type_info{RawData::RandomMiiMouthType.at(index)};
    const auto glasses_type_info{RawData::RandomMiiGlassType.at(static_cast<std::size_t>(age))};

    bf.faceline_type.Assign(
        faceline_type_info.values[GetRandomValue<std::size_t>(faceline_type_info.values_count)]);
    bf.faceline_color.Assign(
        faceline_color_info.values[GetRandomValue<std::size_t>(faceline_color_info.values_count)]);
    bf.faceline_wrinkle.Assign(
        faceline_wrinkle_info
            .values[GetRandomValue<std::size_t>(faceline_wrinkle_info.values_count)]);
    bf.faceline_makeup.Assign(
        faceline_makeup_info
            .values[GetRandomValue<std::size_t>(faceline_makeup_info.values_count)]);

    bf.hair_type.Assign(
        hair_type_info.values[GetRandomValue<std::size_t>(hair_type_info.values_count)]);
    bf.hair_color.Assign(
        HairColorLookup[hair_color_info
                            .values[GetRandomValue<std::size_t>(hair_color_info.values_count)]]);
    bf.hair_flip.Assign(GetRandomValue<HairFlip>(HairFlip::Maximum));

    bf.eye_type.Assign(
        eye_type_info.values[GetRandomValue<std::size_t>(eye_type_info.values_count)]);

    const auto eye_rotate_1{gender != Gender::Male ? 4 : 2};
    const auto eye_rotate_2{gender != Gender::Male ? 3 : 4};
    const auto eye_rotate_offset{32 - EyeRotateLookup[eye_rotate_1] + eye_rotate_2};
    const auto eye_rotate{32 - EyeRotateLookup[bf.eye_type]};

    bf.eye_color.Assign(
        EyeColorLookup[eye_color_info
                           .values[GetRandomValue<std::size_t>(eye_color_info.values_count)]]);
    bf.eye_scale.Assign(4);
    bf.eye_aspect.Assign(3);
    bf.eye_rotate.Assign(eye_rotate_offset - eye_rotate);
    bf.eye_x.Assign(2);
    bf.eye_y.Assign(axis_y + 12);

    bf.eyebrow_type.Assign(
        eyebrow_type_info.values[GetRandomValue<std::size_t>(eyebrow_type_info.values_count)]);

    const auto eyebrow_rotate_1{race == Race::Asian ? 6 : 0};
    const auto eyebrow_y{race == Race::Asian ? 9 : 10};
    const auto eyebrow_rotate_offset{32 - EyebrowRotateLookup[eyebrow_rotate_1] + 6};
    const auto eyebrow_rotate{
        32 - EyebrowRotateLookup[static_cast<std::size_t>(bf.eyebrow_type.Value())]};

    bf.eyebrow_color.Assign(bf.hair_color);
    bf.eyebrow_scale.Assign(4);
    bf.eyebrow_aspect.Assign(3);
    bf.eyebrow_rotate.Assign(eyebrow_rotate_offset - eyebrow_rotate);
    bf.eyebrow_x.Assign(2);
    bf.eyebrow_y.Assign(axis_y + eyebrow_y);

    const auto nose_scale{gender == Gender::Female ? 3 : 4};

    bf.nose_type.Assign(
        nose_type_info.values[GetRandomValue<std::size_t>(nose_type_info.values_count)]);
    bf.nose_scale.Assign(nose_scale);
    bf.nose_y.Assign(axis_y + 9);

    const auto mouth_color{gender == Gender::Female ? GetRandomValue<int>(4) : 0};

    bf.mouth_type.Assign(
        mouth_type_info.values[GetRandomValue<std::size_t>(mouth_type_info.values_count)]);
    bf.mouth_color.Assign(MouthColorLookup[mouth_color]);
    bf.mouth_scale.Assign(4);
    bf.mouth_aspect.Assign(3);
    bf.mouth_y.Assign(axis_y + 13);

    bf.beard_color.Assign(bf.hair_color);
    bf.mustache_scale.Assign(4);

    if (gender == Gender::Male && age != Age::Young && GetRandomValue<int>(10) < 2) {
        const auto mustache_and_beard_flag{
            GetRandomValue<BeardAndMustacheFlag>(BeardAndMustacheFlag::All)};

        auto beard_type{BeardType::None};
        auto mustache_type{MustacheType::None};

        if ((mustache_and_beard_flag & BeardAndMustacheFlag::Beard) ==
            BeardAndMustacheFlag::Beard) {
            beard_type = GetRandomValue<BeardType>(BeardType::Beard1, BeardType::Beard5);
        }

        if ((mustache_and_beard_flag & BeardAndMustacheFlag::Mustache) ==
            BeardAndMustacheFlag::Mustache) {
            mustache_type =
                GetRandomValue<MustacheType>(MustacheType::Mustache1, MustacheType::Mustache5);
        }

        bf.mustache_type.Assign(mustache_type);
        bf.beard_type.Assign(beard_type);
        bf.mustache_y.Assign(10);
    } else {
        bf.mustache_type.Assign(MustacheType::None);
        bf.beard_type.Assign(BeardType::None);
        bf.mustache_y.Assign(axis_y + 10);
    }

    const auto glasses_type_start{GetRandomValue<std::size_t>(100)};
    u8 glasses_type{};
    while (glasses_type_start < glasses_type_info.values[glasses_type]) {
        if (++glasses_type >= glasses_type_info.values_count) {
            ASSERT(false);
            break;
        }
    }

    bf.glasses_type.Assign(glasses_type);
    bf.glasses_color.Assign(GlassesColorLookup[0]);
    bf.glasses_scale.Assign(4);
    bf.glasses_y.Assign(axis_y + 10);

    bf.mole_type.Assign(0);
    bf.mole_scale.Assign(4);
    bf.mole_x.Assign(2);
    bf.mole_y.Assign(20);

    return {DefaultMiiName, bf, user_id};
}

MiiStoreData BuildDefaultStoreData(const DefaultMii& info, const Common::UUID& user_id) {
    MiiStoreBitFields bf{};

    bf.font_region.Assign(info.font_region);
    bf.favorite_color.Assign(info.favorite_color);
    bf.gender.Assign(info.gender);
    bf.height.Assign(info.height);
    bf.build.Assign(info.weight);
    bf.type.Assign(info.type);
    bf.region_move.Assign(info.region);
    bf.faceline_type.Assign(info.face_type);
    bf.faceline_color.Assign(info.face_color);
    bf.faceline_wrinkle.Assign(info.face_wrinkle);
    bf.faceline_makeup.Assign(info.face_makeup);
    bf.hair_type.Assign(info.hair_type);
    bf.hair_color.Assign(HairColorLookup[info.hair_color]);
    bf.hair_flip.Assign(static_cast<HairFlip>(info.hair_flip));
    bf.eye_type.Assign(info.eye_type);
    bf.eye_color.Assign(EyeColorLookup[info.eye_color]);
    bf.eye_scale.Assign(info.eye_scale);
    bf.eye_aspect.Assign(info.eye_aspect);
    bf.eye_rotate.Assign(info.eye_rotate);
    bf.eye_x.Assign(info.eye_x);
    bf.eye_y.Assign(info.eye_y);
    bf.eyebrow_type.Assign(info.eyebrow_type);
    bf.eyebrow_color.Assign(HairColorLookup[info.eyebrow_color]);
    bf.eyebrow_scale.Assign(info.eyebrow_scale);
    bf.eyebrow_aspect.Assign(info.eyebrow_aspect);
    bf.eyebrow_rotate.Assign(info.eyebrow_rotate);
    bf.eyebrow_x.Assign(info.eyebrow_x);
    bf.eyebrow_y.Assign(info.eyebrow_y - 3);
    bf.nose_type.Assign(info.nose_type);
    bf.nose_scale.Assign(info.nose_scale);
    bf.nose_y.Assign(info.nose_y);
    bf.mouth_type.Assign(info.mouth_type);
    bf.mouth_color.Assign(MouthColorLookup[info.mouth_color]);
    bf.mouth_scale.Assign(info.mouth_scale);
    bf.mouth_aspect.Assign(info.mouth_aspect);
    bf.mouth_y.Assign(info.mouth_y);
    bf.beard_color.Assign(HairColorLookup[info.beard_color]);
    bf.beard_type.Assign(static_cast<BeardType>(info.beard_type));
    bf.mustache_type.Assign(static_cast<MustacheType>(info.mustache_type));
    bf.mustache_scale.Assign(info.mustache_scale);
    bf.mustache_y.Assign(info.mustache_y);
    bf.glasses_type.Assign(info.glasses_type);
    bf.glasses_color.Assign(GlassesColorLookup[info.glasses_color]);
    bf.glasses_scale.Assign(info.glasses_scale);
    bf.glasses_y.Assign(info.glasses_y);
    bf.mole_type.Assign(info.mole_type);
    bf.mole_scale.Assign(info.mole_scale);
    bf.mole_x.Assign(info.mole_x);
    bf.mole_y.Assign(info.mole_y);

    return {DefaultMiiName, bf, user_id};
}

} // namespace

MiiStoreData::MiiStoreData() = default;

MiiStoreData::MiiStoreData(const MiiStoreData::Name& name, const MiiStoreBitFields& bit_fields,
                           const Common::UUID& user_id) {
    data.name = name;
    data.uuid = Common::UUID::MakeRandomRFC4122V4();

    std::memcpy(data.data.data(), &bit_fields, sizeof(MiiStoreBitFields));
    data_crc = GenerateCrc16(data.data.data(), sizeof(data));
    device_crc = GenerateCrc16(&user_id, sizeof(Common::UUID));
}

MiiManager::MiiManager() : user_id{Service::Account::ProfileManager().GetLastOpenedUser()} {}

bool MiiManager::CheckAndResetUpdateCounter(SourceFlag source_flag, u64& current_update_counter) {
    if ((source_flag & SourceFlag::Database) == SourceFlag::None) {
        return false;
    }

    const bool result{current_update_counter != update_counter};

    current_update_counter = update_counter;

    return result;
}

bool MiiManager::IsFullDatabase() const {
    // TODO(bunnei): We don't implement the Mii database, so it cannot be full
    return false;
}

u32 MiiManager::GetCount(SourceFlag source_flag) const {
    std::size_t count{};
    if ((source_flag & SourceFlag::Database) != SourceFlag::None) {
        // TODO(bunnei): We don't implement the Mii database, but when we do, update this
        count += 0;
    }
    if ((source_flag & SourceFlag::Default) != SourceFlag::None) {
        count += (DefaultMiiCount - BaseMiiCount);
    }
    return static_cast<u32>(count);
}

ResultVal<CharInfo> MiiManager::UpdateLatest([[maybe_unused]] const CharInfo& info,
                                             SourceFlag source_flag) {
    if ((source_flag & SourceFlag::Database) == SourceFlag::None) {
        return ERROR_CANNOT_FIND_ENTRY;
    }

    // TODO(bunnei): We don't implement the Mii database, so we can't have an entry
    return ERROR_CANNOT_FIND_ENTRY;
}

CharInfo MiiManager::BuildRandom(Age age, Gender gender, Race race) {
    return ConvertStoreDataToInfo(BuildRandomStoreData(age, gender, race, user_id));
}

CharInfo MiiManager::BuildDefault(std::size_t index) {
    return ConvertStoreDataToInfo(BuildDefaultStoreData(RawData::DefaultMii.at(index), user_id));
}

CharInfo MiiManager::ConvertV3ToCharInfo(const Ver3StoreData& mii_v3) const {
    Service::Mii::MiiManager manager;
    auto mii = manager.BuildDefault(0);

    if (!ValidateV3Info(mii_v3)) {
        return mii;
    }

    // TODO: We are ignoring a bunch of data from the mii_v3

    mii.gender = static_cast<u8>(mii_v3.mii_information.gender);
    mii.favorite_color = static_cast<u8>(mii_v3.mii_information.favorite_color);
    mii.height = mii_v3.height;
    mii.build = mii_v3.build;

    // Copy name until string terminator
    mii.name = {};
    for (std::size_t index = 0; index < mii.name.size() - 1; index++) {
        mii.name[index] = mii_v3.mii_name[index];
        if (mii.name[index] == 0) {
            break;
        }
    }

    mii.font_region = mii_v3.region_information.character_set;

    mii.faceline_type = mii_v3.appearance_bits1.face_shape;
    mii.faceline_color = mii_v3.appearance_bits1.skin_color;
    mii.faceline_wrinkle = mii_v3.appearance_bits2.wrinkles;
    mii.faceline_make = mii_v3.appearance_bits2.makeup;

    mii.hair_type = mii_v3.hair_style;
    mii.hair_color = mii_v3.appearance_bits3.hair_color;
    mii.hair_flip = mii_v3.appearance_bits3.flip_hair;

    mii.eye_type = static_cast<u8>(mii_v3.appearance_bits4.eye_type);
    mii.eye_color = static_cast<u8>(mii_v3.appearance_bits4.eye_color);
    mii.eye_scale = static_cast<u8>(mii_v3.appearance_bits4.eye_scale);
    mii.eye_aspect = static_cast<u8>(mii_v3.appearance_bits4.eye_vertical_stretch);
    mii.eye_rotate = static_cast<u8>(mii_v3.appearance_bits4.eye_rotation);
    mii.eye_x = static_cast<u8>(mii_v3.appearance_bits4.eye_spacing);
    mii.eye_y = static_cast<u8>(mii_v3.appearance_bits4.eye_y_position);

    mii.eyebrow_type = static_cast<u8>(mii_v3.appearance_bits5.eyebrow_style);
    mii.eyebrow_color = static_cast<u8>(mii_v3.appearance_bits5.eyebrow_color);
    mii.eyebrow_scale = static_cast<u8>(mii_v3.appearance_bits5.eyebrow_scale);
    mii.eyebrow_aspect = static_cast<u8>(mii_v3.appearance_bits5.eyebrow_yscale);
    mii.eyebrow_rotate = static_cast<u8>(mii_v3.appearance_bits5.eyebrow_rotation);
    mii.eyebrow_x = static_cast<u8>(mii_v3.appearance_bits5.eyebrow_spacing);
    mii.eyebrow_y = static_cast<u8>(mii_v3.appearance_bits5.eyebrow_y_position);

    mii.nose_type = static_cast<u8>(mii_v3.appearance_bits6.nose_type);
    mii.nose_scale = static_cast<u8>(mii_v3.appearance_bits6.nose_scale);
    mii.nose_y = static_cast<u8>(mii_v3.appearance_bits6.nose_y_position);

    mii.mouth_type = static_cast<u8>(mii_v3.appearance_bits7.mouth_type);
    mii.mouth_color = static_cast<u8>(mii_v3.appearance_bits7.mouth_color);
    mii.mouth_scale = static_cast<u8>(mii_v3.appearance_bits7.mouth_scale);
    mii.mouth_aspect = static_cast<u8>(mii_v3.appearance_bits7.mouth_horizontal_stretch);
    mii.mouth_y = static_cast<u8>(mii_v3.appearance_bits8.mouth_y_position);

    mii.mustache_type = static_cast<u8>(mii_v3.appearance_bits8.mustache_type);
    mii.mustache_scale = static_cast<u8>(mii_v3.appearance_bits9.mustache_scale);
    mii.mustache_y = static_cast<u8>(mii_v3.appearance_bits9.mustache_y_position);

    mii.beard_type = static_cast<u8>(mii_v3.appearance_bits9.bear_type);
    mii.beard_color = static_cast<u8>(mii_v3.appearance_bits9.facial_hair_color);

    mii.glasses_type = static_cast<u8>(mii_v3.appearance_bits10.glasses_type);
    mii.glasses_color = static_cast<u8>(mii_v3.appearance_bits10.glasses_color);
    mii.glasses_scale = static_cast<u8>(mii_v3.appearance_bits10.glasses_scale);
    mii.glasses_y = static_cast<u8>(mii_v3.appearance_bits10.glasses_y_position);

    mii.mole_type = static_cast<u8>(mii_v3.appearance_bits11.mole_enabled);
    mii.mole_scale = static_cast<u8>(mii_v3.appearance_bits11.mole_scale);
    mii.mole_x = static_cast<u8>(mii_v3.appearance_bits11.mole_x_position);
    mii.mole_y = static_cast<u8>(mii_v3.appearance_bits11.mole_y_position);

    // TODO: Validate mii data

    return mii;
}

Ver3StoreData MiiManager::BuildFromStoreData(const CharInfo& mii) const {
    Service::Mii::MiiManager manager;
    Ver3StoreData mii_v3{};

    // TODO: We are ignoring a bunch of data from the mii_v3

    mii_v3.version = 1;
    mii_v3.mii_information.gender.Assign(mii.gender);
    mii_v3.mii_information.favorite_color.Assign(mii.favorite_color);
    mii_v3.height = mii.height;
    mii_v3.build = mii.build;

    // Copy name until string terminator
    mii_v3.mii_name = {};
    for (std::size_t index = 0; index < mii.name.size() - 1; index++) {
        mii_v3.mii_name[index] = mii.name[index];
        if (mii_v3.mii_name[index] == 0) {
            break;
        }
    }

    mii_v3.region_information.character_set.Assign(mii.font_region);

    mii_v3.appearance_bits1.face_shape.Assign(mii.faceline_type);
    mii_v3.appearance_bits2.wrinkles.Assign(mii.faceline_wrinkle);
    mii_v3.appearance_bits2.makeup.Assign(mii.faceline_make);

    mii_v3.hair_style = mii.hair_type;
    mii_v3.appearance_bits3.flip_hair.Assign(mii.hair_flip);

    mii_v3.appearance_bits4.eye_type.Assign(mii.eye_type);
    mii_v3.appearance_bits4.eye_scale.Assign(mii.eye_scale);
    mii_v3.appearance_bits4.eye_vertical_stretch.Assign(mii.eye_aspect);
    mii_v3.appearance_bits4.eye_rotation.Assign(mii.eye_rotate);
    mii_v3.appearance_bits4.eye_spacing.Assign(mii.eye_x);
    mii_v3.appearance_bits4.eye_y_position.Assign(mii.eye_y);

    mii_v3.appearance_bits5.eyebrow_style.Assign(mii.eyebrow_type);
    mii_v3.appearance_bits5.eyebrow_scale.Assign(mii.eyebrow_scale);
    mii_v3.appearance_bits5.eyebrow_yscale.Assign(mii.eyebrow_aspect);
    mii_v3.appearance_bits5.eyebrow_rotation.Assign(mii.eyebrow_rotate);
    mii_v3.appearance_bits5.eyebrow_spacing.Assign(mii.eyebrow_x);
    mii_v3.appearance_bits5.eyebrow_y_position.Assign(mii.eyebrow_y);

    mii_v3.appearance_bits6.nose_type.Assign(mii.nose_type);
    mii_v3.appearance_bits6.nose_scale.Assign(mii.nose_scale);
    mii_v3.appearance_bits6.nose_y_position.Assign(mii.nose_y);

    mii_v3.appearance_bits7.mouth_type.Assign(mii.mouth_type);
    mii_v3.appearance_bits7.mouth_scale.Assign(mii.mouth_scale);
    mii_v3.appearance_bits7.mouth_horizontal_stretch.Assign(mii.mouth_aspect);
    mii_v3.appearance_bits8.mouth_y_position.Assign(mii.mouth_y);

    mii_v3.appearance_bits8.mustache_type.Assign(mii.mustache_type);
    mii_v3.appearance_bits9.mustache_scale.Assign(mii.mustache_scale);
    mii_v3.appearance_bits9.mustache_y_position.Assign(mii.mustache_y);

    mii_v3.appearance_bits9.bear_type.Assign(mii.beard_type);

    mii_v3.appearance_bits10.glasses_scale.Assign(mii.glasses_scale);
    mii_v3.appearance_bits10.glasses_y_position.Assign(mii.glasses_y);

    mii_v3.appearance_bits11.mole_enabled.Assign(mii.mole_type);
    mii_v3.appearance_bits11.mole_scale.Assign(mii.mole_scale);
    mii_v3.appearance_bits11.mole_x_position.Assign(mii.mole_x);
    mii_v3.appearance_bits11.mole_y_position.Assign(mii.mole_y);

    // These types are converted to V3 from a table
    mii_v3.appearance_bits1.skin_color.Assign(Ver3FacelineColorTable[mii.faceline_color]);
    mii_v3.appearance_bits3.hair_color.Assign(Ver3HairColorTable[mii.hair_color]);
    mii_v3.appearance_bits4.eye_color.Assign(Ver3EyeColorTable[mii.eye_color]);
    mii_v3.appearance_bits5.eyebrow_color.Assign(Ver3HairColorTable[mii.eyebrow_color]);
    mii_v3.appearance_bits7.mouth_color.Assign(Ver3MouthlineColorTable[mii.mouth_color]);
    mii_v3.appearance_bits9.facial_hair_color.Assign(Ver3HairColorTable[mii.beard_color]);
    mii_v3.appearance_bits10.glasses_color.Assign(Ver3GlassColorTable[mii.glasses_color]);
    mii_v3.appearance_bits10.glasses_type.Assign(Ver3GlassTypeTable[mii.glasses_type]);

    mii_v3.crc = GenerateCrc16(&mii_v3, sizeof(Ver3StoreData) - sizeof(u16));

    // TODO: Validate mii_v3 data

    return mii_v3;
}

NfpStoreDataExtension MiiManager::SetFromStoreData(const CharInfo& mii) const {
    return {
        .faceline_color = static_cast<u8>(mii.faceline_color & 0xf),
        .hair_color = static_cast<u8>(mii.hair_color & 0x7f),
        .eye_color = static_cast<u8>(mii.eyebrow_color & 0x7f),
        .eyebrow_color = static_cast<u8>(mii.eyebrow_color & 0x7f),
        .mouth_color = static_cast<u8>(mii.mouth_color & 0x7f),
        .beard_color = static_cast<u8>(mii.beard_color & 0x7f),
        .glass_color = static_cast<u8>(mii.glasses_color & 0x7f),
        .glass_type = static_cast<u8>(mii.glasses_type & 0x1f),
    };
}

bool MiiManager::ValidateV3Info(const Ver3StoreData& mii_v3) const {
    bool is_valid = mii_v3.version == 0 || mii_v3.version == 3;

    is_valid = is_valid && (mii_v3.mii_name[0] != 0);

    is_valid = is_valid && (mii_v3.mii_information.birth_month < 13);
    is_valid = is_valid && (mii_v3.mii_information.birth_day < 32);
    is_valid = is_valid && (mii_v3.mii_information.favorite_color < 12);
    is_valid = is_valid && (mii_v3.height < 128);
    is_valid = is_valid && (mii_v3.build < 128);

    is_valid = is_valid && (mii_v3.appearance_bits1.face_shape < 12);
    is_valid = is_valid && (mii_v3.appearance_bits1.skin_color < 7);
    is_valid = is_valid && (mii_v3.appearance_bits2.wrinkles < 12);
    is_valid = is_valid && (mii_v3.appearance_bits2.makeup < 12);

    is_valid = is_valid && (mii_v3.hair_style < 132);
    is_valid = is_valid && (mii_v3.appearance_bits3.hair_color < 8);

    is_valid = is_valid && (mii_v3.appearance_bits4.eye_type < 60);
    is_valid = is_valid && (mii_v3.appearance_bits4.eye_color < 6);
    is_valid = is_valid && (mii_v3.appearance_bits4.eye_scale < 8);
    is_valid = is_valid && (mii_v3.appearance_bits4.eye_vertical_stretch < 7);
    is_valid = is_valid && (mii_v3.appearance_bits4.eye_rotation < 8);
    is_valid = is_valid && (mii_v3.appearance_bits4.eye_spacing < 13);
    is_valid = is_valid && (mii_v3.appearance_bits4.eye_y_position < 19);

    is_valid = is_valid && (mii_v3.appearance_bits5.eyebrow_style < 25);
    is_valid = is_valid && (mii_v3.appearance_bits5.eyebrow_color < 8);
    is_valid = is_valid && (mii_v3.appearance_bits5.eyebrow_scale < 9);
    is_valid = is_valid && (mii_v3.appearance_bits5.eyebrow_yscale < 7);
    is_valid = is_valid && (mii_v3.appearance_bits5.eyebrow_rotation < 12);
    is_valid = is_valid && (mii_v3.appearance_bits5.eyebrow_spacing < 12);
    is_valid = is_valid && (mii_v3.appearance_bits5.eyebrow_y_position < 19);

    is_valid = is_valid && (mii_v3.appearance_bits6.nose_type < 18);
    is_valid = is_valid && (mii_v3.appearance_bits6.nose_scale < 9);
    is_valid = is_valid && (mii_v3.appearance_bits6.nose_y_position < 19);

    is_valid = is_valid && (mii_v3.appearance_bits7.mouth_type < 36);
    is_valid = is_valid && (mii_v3.appearance_bits7.mouth_color < 5);
    is_valid = is_valid && (mii_v3.appearance_bits7.mouth_scale < 9);
    is_valid = is_valid && (mii_v3.appearance_bits7.mouth_horizontal_stretch < 7);
    is_valid = is_valid && (mii_v3.appearance_bits8.mouth_y_position < 19);

    is_valid = is_valid && (mii_v3.appearance_bits8.mustache_type < 6);
    is_valid = is_valid && (mii_v3.appearance_bits9.mustache_scale < 7);
    is_valid = is_valid && (mii_v3.appearance_bits9.mustache_y_position < 17);

    is_valid = is_valid && (mii_v3.appearance_bits9.bear_type < 6);
    is_valid = is_valid && (mii_v3.appearance_bits9.facial_hair_color < 8);

    is_valid = is_valid && (mii_v3.appearance_bits10.glasses_type < 9);
    is_valid = is_valid && (mii_v3.appearance_bits10.glasses_color < 6);
    is_valid = is_valid && (mii_v3.appearance_bits10.glasses_scale < 8);
    is_valid = is_valid && (mii_v3.appearance_bits10.glasses_y_position < 21);

    is_valid = is_valid && (mii_v3.appearance_bits11.mole_enabled < 2);
    is_valid = is_valid && (mii_v3.appearance_bits11.mole_scale < 9);
    is_valid = is_valid && (mii_v3.appearance_bits11.mole_x_position < 17);
    is_valid = is_valid && (mii_v3.appearance_bits11.mole_y_position < 31);

    return is_valid;
}

ResultVal<std::vector<MiiInfoElement>> MiiManager::GetDefault(SourceFlag source_flag) {
    std::vector<MiiInfoElement> result;

    if ((source_flag & SourceFlag::Default) == SourceFlag::None) {
        return result;
    }

    for (std::size_t index = BaseMiiCount; index < DefaultMiiCount; index++) {
        result.emplace_back(BuildDefault(index), Source::Default);
    }

    return result;
}

Result MiiManager::GetIndex([[maybe_unused]] const CharInfo& info, u32& index) {
    constexpr u32 INVALID_INDEX{0xFFFFFFFF};

    index = INVALID_INDEX;

    // TODO(bunnei): We don't implement the Mii database, so we can't have an index
    return ERROR_CANNOT_FIND_ENTRY;
}

} // namespace Service::Mii
