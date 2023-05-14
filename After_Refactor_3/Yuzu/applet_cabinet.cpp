#include "core/frontend/applets/cabinet.h"
#include "core/hle/service/am/applets/applet_cabinet.h"
#include "core/hle/service/mii/mii_manager.h"
#include "core/hle/service/nfc/common/device.h"
#include "common/logging/log.h"

namespace Service::AM::Applets {

Cabinet::Cabinet(Core::System& system, LibraryAppletMode applet_mode, const Core::Frontend::CabinetApplet& frontend)
    : Applet{system, applet_mode}, m_frontend(frontend), m_system(system), m_service_context{system, "CabinetApplet"} {
    m_availability_change_event = CreateEvent("CabinetApplet:AvailabilityChangeEvent");
}

Cabinet::~Cabinet() = default;

std::shared_ptr<Service::Core::KReadableEvent> Cabinet::CreateEvent(const std::string& name) {
    return m_service_context.CreateEvent(name);
}

void Cabinet::Initialize() override {
    Applet::Initialize();

    LOG_INFO(Service_HID, "Initializing Cabinet Applet.");

    LOG_DEBUG(Service_HID,
              "Initializing Applet with common_args: arg_version={}, lib_version={}, "
              "play_startup_sound={}, size={}, system_tick={}, theme_color={}",
              m_common_args.arguments_version, m_common_args.library_version, m_common_args.play_startup_sound,
              m_common_args.size, m_common_args.system_tick, m_common_args.theme_color);

    auto storage = m_broker.PopNormalDataToApplet();
    ASSERT(storage != nullptr);

    std::copy(storage->GetData().data(), storage->GetData().data() + sizeof(StartParamForAmiiboSettings),
              &m_applet_input_common);
}

bool Cabinet::TransactionComplete() const override {
    return m_is_complete;
}

Result Cabinet::GetStatus() const override {
    return ResultSuccess;
}

void Cabinet::ExecuteInteractive() override {
    ASSERT_MSG(false, "Attempted to call interactive execution on non-interactive applet.");
}

void Cabinet::Execute() override {
    if (m_is_complete) {
        return;
    }

    const auto callback = [this](bool apply_changes, const std::string& amiibo_name) {
        DisplayCompleted(apply_changes, amiibo_name);
    };

    // TODO: listen on all controllers
    if (!m_nfp_device) {
        m_nfp_device = std::make_unique<Service::NFC::NfcDevice>(m_system.HIDCore().GetFirstNpadId(), m_system,
                                                                 m_service_context, m_availability_change_event);
        m_nfp_device->Initialize();
        m_nfp_device->StartDetection(Service::NFC::NfcProtocol::All);
    }

    const Core::Frontend::CabinetParameters parameters{
        .tag_info = m_applet_input_common.tag_info,
        .register_info = m_applet_input_common.register_info,
        .mode = m_applet_input_common.applet_mode,
    };

    switch (m_applet_input_common.applet_mode) {
    case Service::NFP::CabinetMode::StartNicknameAndOwnerSettings:
    case Service::NFP::CabinetMode::StartGameDataEraser:
    case Service::NFP::CabinetMode::StartRestorer:
    case Service::NFP::CabinetMode::StartFormatter:
        m_frontend.ShowCabinetApplet(callback, parameters, m_nfp_device.get());
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown CabinetMode={}", m_applet_input_common.applet_mode);
        DisplayCompleted(false, {});
        break;
    }
}

void Cabinet::DisplayCompleted(bool apply_changes, const std::string& amiibo_name) {
    ReturnValueForAmiiboSettings applet_output;

    if (!apply_changes || m_nfp_device->GetCurrentState() != Service::NFC::DeviceState::TagFound) {
        applet_output.result = CabinetResult::Cancel;
        nfp_device->Finalize();
    } else {
        switch (m_applet_input_common.applet_mode) {
        case Service::NFP::CabinetMode::StartNicknameAndOwnerSettings: {
            Service::NFP::RegisterInfoPrivate register_info{};
            std::copy(amiibo_name.begin(), amiibo_name.end(), register_info.amiibo_name.data());
            m_nfp_device->SetRegisterInfoPrivate(register_info);
            break;
        }
        case Service::NFP::CabinetMode::StartGameDataEraser:
            m_nfp_device->DeleteApplicationArea();
            break;
        case Service::NFP::CabinetMode::StartRestorer:
            m_nfp_device->RestoreAmiibo();
            break;
        case Service::NFP::CabinetMode::StartFormatter:
            m_nfp_device->Format();
            break;
        default:
            UNIMPLEMENTED_MSG("Unknown CabinetMode={}", m_applet_input_common.applet_mode);
            break;
        }

        applet_output.device_handle = m_applet_input_common.device_handle;
        applet_output.result |= CabinetResult::Cancel;

        const auto reg_result = m_nfp_device->GetRegisterInfo(applet_output.register_info);
        const auto tag_result = m_nfp_device->GetTagInfo(applet_output.tag_info, false);
        m_nfp_device->Finalize();

        if (reg_result.IsSuccess()) {
            applet_output.result |= CabinetResult::RegisterInfo;
        }

        if (tag_result.IsSuccess()) {
            applet_output.result |= CabinetResult::TagInfo;
        }
    }

    std::vector<u8> out_data(sizeof(ReturnValueForAmiiboSettings));
    std::copy(reinterpret_cast<const u8*>(&applet_output),
              reinterpret_cast<const u8*>(&applet_output) + sizeof(ReturnValueForAmiiboSettings), out_data.data());

    m_is_complete = true;
    m_broker.PushNormalDataFromApplet(std::make_shared<IStorage>(m_system, std::move(out_data)));
    m_broker.SignalStateChanged();
}

void Cabinet::Cancel() {
    DisplayCompleted(false, {});
}

Result Cabinet::RequestExit() {
    m_frontend.Close();
    R_SUCCEED();
}

} // namespace Service::AM::Applets