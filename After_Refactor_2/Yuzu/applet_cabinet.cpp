#include "common/assert.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "core/frontend/applets/cabinet.h"
#include "core/hid/hid_core.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/service/am/am.h"
#include "core/hle/service/am/applets/applet_cabinet.h"
#include "core/hle/service/mii/mii_manager.h"
#include "core/hle/service/nfc/common/device.h"

namespace Service::AM::Applets {

// Define constants for clarity
constexpr char kServiceName[] = "CabinetApplet";
constexpr char kAvailabilityChangeEventName[] = "CabinetApplet:AvailabilityChangeEvent";

class CabinetDevice {
public:
    CabinetDevice(Core::System& system) : nfp_device_{nullptr}, system_{system} {}

    void Initialize() {
        nfp_device_ = std::make_shared<Service::NFC::NfcDevice>(system_.HIDCore().GetFirstNpadId(),
                                                                system_, service_context_,
                                                                availability_change_event_);
        nfp_device_->Initialize();
        nfp_device_->StartDetection(Service::NFC::NfcProtocol::All);
    }

    void StartDetection() {
        nfp_device_->StartDetection(Service::NFC::NfcProtocol::All);
    }

    void StopDetection() {
        nfp_device_->StopDetection();
    }

    bool IsTagAvailable() const {
        return nfp_device_->GetCurrentState() == Service::NFC::DeviceState::TagFound ||
               nfp_device_->GetCurrentState() == Service::NFC::DeviceState::TagMounted;
    }

    void Mount(Service::NFP::ModelType model_type, Service::NFP::MountTarget mount_target) {
        nfp_device_->Mount(model_type, mount_target);
    }

    void SetRegisterInfoPrivate(const Service::NFP::RegisterInfoPrivate& register_info) {
        nfp_device_->SetRegisterInfoPrivate(register_info);
    }

    void DeleteApplicationArea() {
        nfp_device_->DeleteApplicationArea();
    }

    void RestoreAmiibo() {
        nfp_device_->RestoreAmiibo();
    }

    void Format() {
        nfp_device_->Format();
    }

    Service::Result GetRegisterInfo(Service::NFP::RegisterInfo& register_info) const {
        return nfp_device_->GetRegisterInfo(register_info);
    }

    Service::Result GetTagInfo(Service::NFP::TagInfo& tag_info, bool cache) const {
        return nfp_device_->GetTagInfo(tag_info, cache);
    }

    void Finalize() {
        nfp_device_->Finalize();
    }

private:
    std::shared_ptr<Service::NFC::NfcDevice> nfp_device_;
    Core::System& system_;
    Service::Context service_context_{system_, kServiceName};
    Service::KEvent availability_change_event_{system_, kAvailabilityChangeEventName};
};

class CabinetApplet {
public:
    CabinetApplet(Core::System& system, LibraryAppletMode applet_mode,
                  const Core::Frontend::CabinetApplet& frontend)
        : system_{system}, applet_mode_{applet_mode}, frontend_{frontend},
          cabinet_device_{system_} {}

    void Initialize() {
        LOG_INFO(Service_HID, "Initializing Cabinet Applet.");

        LOG_DEBUG(Service_HID,
                  "Initializing Applet with common_args: arg_version={}, lib_version={}, "
                  "play_startup_sound={}, size={}, system_tick={}, theme_color={}",
                  common_args_.arguments_version, common_args_.library_version,
                  common_args_.play_startup_sound, common_args_.size, common_args_.system_tick,
                  common_args_.theme_color);

        const auto storage = broker_.PopNormalDataToApplet();
        ASSERT(storage != nullptr);

        const auto applet_input_data = storage->GetData();
        ASSERT(applet_input_data.size() >= sizeof(StartParamForAmiiboSettings));

        std::memcpy(&applet_input_common_, applet_input_data.data(),
                    sizeof(StartParamForAmiiboSettings));
    }

    bool TransactionComplete() const {
        return is_complete_;
    }

    Result GetStatus() const {
        return ResultSuccess;
    }

    void ExecuteInteractive() {
        ASSERT_MSG(false, "Attempted to call interactive execution on non-interactive applet.");
    }

    void Execute() {
        if (is_complete_) {
            return;
        }

        const auto callback = [this](bool apply_changes, const std::string& amiibo_name) {
            DisplayCompleted(apply_changes, amiibo_name);
        };

        cabinet_device_.Initialize();

        const Core::Frontend::CabinetParameters parameters{
            .tag_info = applet_input_common_.tag_info,
            .register_info = applet_input_common_.register_info,
            .mode = applet_input_common_.applet_mode,
        };

        switch (applet_input_common_.applet_mode) {
        case Service::NFP::CabinetMode::StartNicknameAndOwnerSettings:
        case Service::NFP::CabinetMode::StartGameDataEraser:
        case Service::NFP::CabinetMode::StartRestorer:
        case Service::NFP::CabinetMode::StartFormatter:
            frontend_.ShowCabinetApplet(callback, parameters, cabinet_device_);
            break;
        default:
            UNIMPLEMENTED_MSG("Unknown CabinetMode={}", applet_input_common_.applet_mode);
            DisplayCompleted(false, {});
            break;
        }
    }

    void DisplayCompleted(bool apply_changes, const std::string& amiibo_name) {
        Service::Mii::MiiManager manager;
        ReturnValueForAmiiboSettings applet_output{};

        if (!apply_changes) {
            Cancel();
        }

        if (!cabinet_device_.IsTagAvailable()) {
            Cancel();
        }

        if (cabinet_device_.GetCurrentState() == Service::NFC::DeviceState::TagFound) {
            cabinet_device_.Mount(Service::NFP::ModelType::Amiibo, Service::NFP::MountTarget::All);
        }

        switch (applet_input_common_.applet_mode) {
        case Service::NFP::CabinetMode::StartNicknameAndOwnerSettings: {
            Service::NFP::RegisterInfoPrivate register_info{};
            std::memcpy(register_info.amiibo_name.data(), amiibo_name.data(),
                        std::min(amiibo_name.size(), register_info.amiibo_name.size() - 1));

            cabinet_device_.SetRegisterInfoPrivate(register_info);
            break;
        }
        case Service::NFP::CabinetMode::StartGameDataEraser:
            cabinet_device_.DeleteApplicationArea();
            break;
        case Service::NFP::CabinetMode::StartRestorer:
            cabinet_device_.RestoreAmiibo();
            break;
        case Service::NFP::CabinetMode::StartFormatter:
            cabinet_device_.Format();
            break;
        default:
            UNIMPLEMENTED_MSG("Unknown CabinetMode={}", applet_input_common_.applet_mode);
            break;
        }

        applet_output.device_handle = applet_input_common_.device_handle;
        applet_output.result = CabinetResult::Cancel;
        const auto reg_result = cabinet_device_.GetRegisterInfo(applet_output.register_info);
        const auto tag_result = cabinet_device_.GetTagInfo(applet_output.tag_info, false);
        cabinet_device_.Finalize();

        if (reg_result.IsSuccess()) {
            applet_output.result |= CabinetResult::RegisterInfo;
        }

        if (tag_result.IsSuccess()) {
            applet_output.result |= CabinetResult::TagInfo;
        }

        std::vector<u8> out_data(sizeof(ReturnValueForAmiiboSettings));
        std::memcpy(out_data.data(), &applet_output, sizeof(ReturnValueForAmiiboSettings));

        is_complete_ = true;

        broker_.PushNormalDataFromApplet(
            std::make_shared<IStorage>(system_, std::move(out_data)));
        broker_.SignalStateChanged();
    }

    void Cancel() {
        ReturnValueForAmiiboSettings applet_output{};
        applet_output.device_handle = applet_input_common_.device_handle;
        applet_output.result = CabinetResult::Cancel;
        cabinet_device_.Finalize();

        std::vector<u8> out_data(sizeof(ReturnValueForAmiiboSettings));
        std::memcpy(out_data.data(), &applet_output, sizeof(ReturnValueForAmiiboSettings));

        is_complete_ = true;

        broker_.PushNormalDataFromApplet(
            std::make_shared<IStorage>(system_, std::move(out_data)));
        broker_.SignalStateChanged();
    }

    Result RequestExit() {
        frontend_.Close();
        R_SUCCEED();
    }

private:
    Core::System& system_;
    LibraryAppletMode applet_mode_;
    const Core::Frontend::CabinetApplet& frontend_;
    const Service::Context service_context_{system_, kServiceName};
    Service::KEvent availability_change_event_{system_, kAvailabilityChangeEventName};
    CabinetDevice cabinet_device_;
    ReturnValueForAmiiboSettings applet_output_;
    StartParamForAmiiboSettings applet_input_common_;
    AppletCommonArguments common_args_;
    AppletHolder broker_;
    bool is_complete_{false};
};

Cabinet::Cabinet(Core::System& system, LibraryAppletMode applet_mode,
                 const Core::Frontend::CabinetApplet& frontend)
    : Applet{system, applet_mode}, cabinet_applet_{std::make_unique<CabinetApplet>(
                                       system, applet_mode, frontend)} {}

void Cabinet::Initialize() {
    cabinet_applet_->Initialize();
}

bool Cabinet::TransactionComplete() const {
    return cabinet_applet_->TransactionComplete();
}

Result Cabinet::GetStatus() const {
    return cabinet_applet_->GetStatus();
}

void Cabinet::ExecuteInteractive() {
    cabinet_applet_->ExecuteInteractive();
}

void Cabinet::Execute() {
    cabinet_applet_->Execute();
}

void Cabinet::DisplayCompleted(bool apply_changes, const std::string& amiibo_name) {
    cabinet_applet_->DisplayCompleted(apply_changes, amiibo_name);
}

void Cabinet::Cancel() {
    cabinet_applet_->Cancel();
}

Result Cabinet::RequestExit() {
    return cabinet_applet_->RequestExit();
}

} // namespace Service::AM::Applets

