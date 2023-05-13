#pragma once

#include <vector>
#include "core/hle/service/nfc/common/device.h"
#include "core/hle/service/nfp.h"

namespace Service::AM::Applets {

struct CabinetInputData {
    Service::NFP::StartParamForAmiiboSettings amiibo_settings;
    std::vector<u8> stored_data;
};

} // namespace Service::AM::Applets

CabinetOutputData.h:

#pragma once

#include "core/hle/service/nfc/common/device.h"
#include "core/hle/service/nfp.h"

namespace Service::AM::Applets {

struct CabinetOutputData {
    Service::NFP::ReturnValueForAmiiboSettings amiibo_settings;
    bool cancelled = false;
};

} // namespace Service::AM::Applets

ICabinetExecuter.h:

#pragma once

#include <functional>
#include "CabinetInputData.h"
#include "CabinetOutputData.h"

namespace Service::AM::Applets {

class ICabinetExecuter {
public:
    virtual ~ICabinetExecuter() = default;

    virtual void execute(const CabinetInputData& input,
                         const std::function<void(CabinetOutputData)>& callback) = 0;
};

} // namespace Service::AM::Applets

CabinetExecuter.h:

#pragma once

#include "ICabinetExecuter.h"
#include "ICoreSystem.h"
#include "CabinetUI.h"
#include "CabinetNFC.h"
#include "CabinetMii.h"

namespace Service::AM::Applets {

class CabinetExecuter : public ICabinetExecuter {
public:
    CabinetExecuter(const ICoreSystem& core_system, const CabinetUI& ui, const CabinetNFC& nfc,
                    const CabinetMii& mii);

    void execute(const CabinetInputData& input,
                 const std::function<void(CabinetOutputData)>& callback) override;

private:
    const ICoreSystem& core_system_;
    const CabinetUI& ui_;
    const CabinetNFC& nfc_;
    const CabinetMii& mii_;
};

} // namespace Service::AM::Applets

CabinetExecuter.cpp:

#include "CabinetExecuter.h"

namespace Service::AM::Applets {

CabinetExecuter::CabinetExecuter(const ICoreSystem& core_system, const CabinetUI& ui,
                                 const CabinetNFC& nfc, const CabinetMii& mii)
    : core_system_{core_system}, ui_{ui}, nfc_{nfc}, mii_{mii} {}

void CabinetExecuter::execute(const CabinetInputData& input,
                              const std::function<void(CabinetOutputData)>& callback) {
    const auto callback_wrapper = [&](bool apply_changes, const std::string& amiibo_name) {
        CabinetOutputData output;
        if (apply_changes) {
            output.amiibo_settings.device_handle = input.amiibo_settings.device_handle;
            output.amiibo_settings.result = Service::NFP::CabinetResult::Cancel;

            switch (input.amiibo_settings.applet_mode) {
            case Service::NFP::CabinetMode::StartNicknameAndOwnerSettings: {
                Service::NFP::RegisterInfoPrivate register_info{};
                std::memcpy(register_info.amiibo_name.data(), amiibo_name.data(),
                            std::min(amiibo_name.size(), register_info.amiibo_name.size() - 1));
                nfc_.set_register_info_private(register_info);
                break;
            }
            case Service::NFP::CabinetMode::StartGameDataEraser:
                nfc_.delete_application_area();
                break;
            case Service::NFP::CabinetMode::StartRestorer:
                nfc_.restore_amiibo();
                break;
            case Service::NFP::CabinetMode::StartFormatter:
                nfc_.format();
                break;
            default:
                UNIMPLEMENTED_MSG("Unknown CabinetMode={}", input.amiibo_settings.applet_mode);
                break;
            }

            output.amiibo_settings.result |= Service::NFP::CabinetResult::Cancel;
            const auto reg_result = nfc_.get_register_info(output.amiibo_settings.register_info);
            const auto tag_result = nfc_.get_tag_info(output.amiibo_settings.tag_info, false);
            nfc_.finalize();

            if (reg_result.IsSuccess()) {
                output.amiibo_settings.result |= Service::NFP::CabinetResult::RegisterInfo;
            }

            if (tag_result.IsSuccess()) {
                output.amiibo_settings.result |= Service::NFP::CabinetResult::TagInfo;
            }
        } else {
            output.cancelled = true;
            nfc_.finalize();
        }

        callback(output);
    };

    if (nfc_.current_state() != Service::NFC::DeviceState::TagFound &&
        nfc_.current_state() != Service::NFC::DeviceState::TagMounted) {
        CabinetOutputData output;
        output.cancelled = true;
        callback(output);
        return;
    }

    if (nfc_.current_state() == Service::NFC::DeviceState::TagFound) {
        nfc_.mount(Service::NFP::ModelType::Amiibo, Service::NFP::MountTarget::All);
    }

    const auto parameters = Core::Frontend::CabinetParameters{
        .tag_info = input.amiibo_settings.tag_info,
        .register_info = input.amiibo_settings.register_info,
        .mode = input.amiibo_settings.applet_mode,
    };

    switch (input.amiibo_settings.applet_mode) {
    case Service::NFP::CabinetMode::StartNicknameAndOwnerSettings:
    case Service::NFP::CabinetMode::StartGameDataEraser:
    case Service::NFP::CabinetMode::StartRestorer:
    case Service::NFP::CabinetMode::StartFormatter:
        ui_.show_cabinet_applet(callback_wrapper, parameters, nfc_);
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown CabinetMode={}", input.amiibo_settings.applet_mode);
        CabinetOutputData output;
        output.cancelled = true;
        callback(output);
        break;
    }
}

} // namespace Service::AM::Applets

ICabinetUI.h:

#pragma once

#include <functional>
#include "core/hle/service/nfc/common/device.h"
#include "core/hle/service/nfp.h"

namespace Service::AM::Applets {

class ICabinetUI {
public:
    virtual ~ICabinetUI() = default;

    virtual void show_cabinet_applet(
        const std::function<void(bool, const std::string&)>& callback,
        const Core::Frontend::CabinetParameters& parameters, const CabinetNFC& nfc) = 0;
};

} // namespace Service::AM::Applets

CabinetUI.h:

#pragma once

#include "ICabinetUI.h"

namespace Service::AM::Applets {

class CabinetUI : public ICabinetUI {
public:
    void show_cabinet_applet(const std::function<void(bool, const std::string&)>& callback,
                             const Core::Frontend::CabinetParameters& parameters,
                             const CabinetNFC& nfc) override;
};

} // namespace Service::AM::Applets

CabinetUI.cpp:

#include "CabinetUI.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/frontend/applets/cabinet.h"

namespace Service::AM::Applets {

void CabinetUI::show_cabinet_applet(const std::function<void(bool, const std::string&)>& callback,
                                    const Core::Frontend::CabinetParameters& parameters,
                                    const CabinetNFC& nfc) {
    const auto npad_id = nfc.system().HIDCore().GetFirstNpadId();
    const Core::Frontend::CabinetApplet frontend{nfc.system().HIDCore(), npad_id};
    frontend.ShowCabinetApplet(
        callback, parameters,
        std::make_shared<Service::NFC::NfcDevice>(npad_id, nfc.system(), nfc.service_context(),
                                                  nfc.availability_change_event()));
}

} // namespace Service::AM::Applets

INFCDevice.h:

#pragma once

#include "core/hle/service/nfc/common/device.h"
#include "core/hle/service/nfp.h"

namespace Service::NFC {

class INFCDevice : public Service::NFC::Device {
public:
    INFCDevice(Core::Frontend::IController* controller, Core::System& system,
               Service::ServiceContext& service_context, Core::HLE::KReadableEvent* event);
};

} // namespace Service::NFC

NFCDeviceMock.h:

#pragma once

#include "INFCDevice.h"
#include "gmock/gmock.h"

namespace Service::NFC {

class NFCDeviceMock : public INFCDevice {
public:
    NFCDeviceMock(Core::Frontend::IController* controller, Core::System& system,
                  Service::ServiceContext& service_context, Core::HLE::KReadableEvent* event) :
        INFCDevice{controller, system, service_context, event} {}

    MOCK_METHOD(void, initialize, (), (override));
    MOCK_METHOD(void, start_detection, (Service::NFC::NfcProtocol), (override));
    MOCK_METHOD(void, mount, (Service::NFP::ModelType, Service::NFP::MountTarget), (override));
    MOCK_METHOD(void, set_register_info_private, (Service::NFP::RegisterInfoPrivate&), (override));
    MOCK_METHOD(void, delete_application_area, (), (override));
    MOCK_METHOD(void, restore_amiibo, (), (override));
    MOCK_METHOD(void, format, (), (override));
    MOCK_METHOD(void, get_register_info,
                (Service::NFP::ReturnValueForGetRegisterInfo&), (override));
    MOCK_METHOD(void, get_tag_info,
                (Service::NFP::ReturnValueForGetTagInfo&, bool), (override));
    MOCK_METHOD(void, finalize, (), (override));

    DeviceState current_state() const override { return DeviceState::TagMounted; }
};

} // namespace Service::NFC

CabinetNFC.h:

#pragma once

#include "INFCDevice.h"
#include "ServiceContext.h"
#include "ICoreSystem.h"

namespace Service::AM::Applets {

class CabinetNFC {
public:
    CabinetNFC(const ICoreSystem& core_system, ServiceContext& service_context);

    void set_register_info_private

