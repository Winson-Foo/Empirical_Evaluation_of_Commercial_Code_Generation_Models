#include "auth.h"
#include "photo_viewer.h"
#include "stub.h"

#include "common/assert.h"
#include "common/hex_util.h"
#include "common/logging/log.h"
#include "core/reporter.h"

namespace Service::AM::Applets {

// Constants and enums
constexpr Result kErrorInvalidPin{ErrorModule::PCTL, 221};
constexpr size_t kArgSize{0xC};
enum class AuthAppType : uint8_t {
    VerifyAuthenticatorForConfig = 1,
    VerifyParentalAuthentication,
    RegisterParentalPasscode,
    ChangeParentalPasscode,
};
enum class PhotoViewerAppMode : uint8_t {
    CurrentApp = 1,
    AllApps = 2,
};

// Helper functions and classes
static void LogCurrentStorage(AppletDataBroker& broker, std::string_view prefix) {
    auto log_storage = [&](std::shared_ptr<IStorage> storage, const std::string& type) {
        if (storage != nullptr) {
            const auto data = storage->GetData();
            LOG_INFO(Service_AM, "called (STUBBED), during {} received {} data with size={:08X}, data={}",
                     prefix, type, data.size(), Common::HexToString(data));
        }
    };
    log_storage(broker.PopNormalDataToApplet(), "normal");
    log_storage(broker.PopInteractiveDataToApplet(), "interactive");
}

// Auth applet implementation
Auth::Auth(Core::System& system, LibraryAppletMode applet_mode,
           Core::Frontend::ParentalControlsApplet& frontend)
    : Applet(system, applet_mode), frontend_(frontend), system_(system) {}

void Auth::Initialize() {
    Applet::Initialize();
    complete_ = false;

    const auto storage = broker_.PopNormalDataToApplet();
    ASSERT(storage != nullptr);
    const auto data = storage->GetData();
    ASSERT(data.size() >= kArgSize);

    struct Arg {
        INSERT_PADDING_BYTES(4);
        AuthAppType type;
        uint8_t arg0;
        uint8_t arg1;
        uint8_t arg2;
        INSERT_PADDING_BYTES(1);
    };
    static_assert(sizeof(Arg) == kArgSize, "Arg (AuthApplet) has incorrect size.");

    Arg arg{};
    std::memcpy(&arg, data.data(), sizeof(Arg));

    type_ = arg.type;
    arg0_ = arg.arg0;
    arg1_ = arg.arg1;
    arg2_ = arg.arg2;
}

void Auth::Execute() {
    if (complete_) {
        return;
    }

    auto unimplemented_log = [this] {
        UNIMPLEMENTED_MSG(
            "Unimplemented Auth applet type for type={:08X}, arg0={:02X}, arg1={:02X}, arg2={:02X}",
            static_cast<uint8_t>(type_), arg0_, arg1_, arg2_);
    };

    switch (type_) {
        case AuthAppType::VerifyAuthenticatorForConfig: {
            const auto callback = [this](bool is_successful) { AuthFinished(is_successful); };

            if (arg0_ == 1 && arg1_ == 0 && arg2_ == 1) {
                // ShowAuthenticatorForConfiguration
                frontend_.VerifyPINForSettings(callback);
            } else if (arg1_ == 0 && arg2_ == 0) {
                // ShowParentalAuthentication(bool)
                frontend_.VerifyPIN(callback, static_cast<bool>(arg0_));
            } else {
                unimplemented_log();
            }
            break;
        }
        case AuthAppType::RegisterParentalPasscode: {
            const auto callback = [this] { AuthFinished(true); };

            if (arg0_ == 0 && arg1_ == 0 && arg2_ == 0) {
                // RegisterParentalPasscode
                frontend_.RegisterPIN(callback);
            } else {
                unimplemented_log();
            }
            break;
        }
        case AuthAppType::ChangeParentalPasscode: {
            const auto callback = [this] { AuthFinished(true); };

            if (arg0_ == 0 && arg1_ == 0 && arg2_ == 0) {
                // ChangeParentalPasscode
                frontend_.ChangePIN(callback);
            } else {
                unimplemented_log();
            }
            break;
        }
        case AuthAppType::VerifyParentalAuthentication:
        default:
            unimplemented_log();
            break;
    }
}

void Auth::AuthFinished(bool is_successful) {
    successful_ = is_successful;
    complete_ = true;

    struct Return {
        Result result_code;
    };
    static_assert(sizeof(Return) == 0x4, "Return (AuthApplet) has incorrect size.");

    Return return_{GetStatus()};

    std::vector<uint8_t> out(sizeof(Return));
    std::memcpy(out.data(), &return_, sizeof(Return));

    broker_.PushNormalDataFromApplet(std::make_shared<IStorage>(system_, std::move(out)));
    broker_.SignalStateChanged();
}

Result Auth::RequestExit() {
    frontend_.Close();
    return ResultSuccess;
}

// PhotoViewer applet implementation
PhotoViewer::PhotoViewer(Core::System& system, LibraryAppletMode applet_mode,
                         const Core::Frontend::PhotoViewerApplet& frontend)
    : Applet(system, applet_mode), frontend_(frontend), system_(system) {}

void PhotoViewer::Initialize() {
    Applet::Initialize();
    complete_ = false;

    const auto storage = broker_.PopNormalDataToApplet();
    ASSERT(storage != nullptr);
    const auto data = storage->GetData();
    ASSERT(!data.empty());
    mode_ = static_cast<PhotoViewerAppMode>(data[0]);
}

void PhotoViewer::Execute() {
    if (complete_)
        return;

    const auto callback = [this] { ViewFinished(); };
    switch (mode_) {
        case PhotoViewerAppMode::CurrentApp:
            frontend_.ShowPhotosForApplication(system_.GetApplicationProcessProgramID(), callback);
            break;
        case PhotoViewerAppMode::AllApps:
            frontend_.ShowAllPhotos(callback);
            break;
        default:
            UNIMPLEMENTED_MSG("Unimplemented PhotoViewer applet mode={:02X}!", static_cast<uint8_t>(mode_));
            break;
    }
}

void PhotoViewer::ViewFinished() {
    broker_.PushNormalDataFromApplet(std::make_shared<IStorage>(system_, std::vector<uint8_t>()));
    broker_.SignalStateChanged();
    complete_ = true;
}

Result PhotoViewer::RequestExit() {
    frontend_.Close();
    return ResultSuccess;
}

// Stub applet implementation
StubApplet::StubApplet(Core::System& system, AppletId id, LibraryAppletMode applet_mode)
    : Applet(system, applet_mode), id_(id), system_(system) {}

void StubApplet::Initialize() {
    Applet::Initialize();
    LOG_WARNING(Service_AM, "called (STUBBED)");

    const auto data = broker_.PeekDataToAppletForDebug();
    system_.GetReporter().SaveUnimplementedAppletReport(
        static_cast<uint32_t>(id_), common_args.arguments_version, common_args.library_version,
        common_args.theme_color, common_args.play_startup_sound, common_args.system_tick,
        data.normal, data.interactive);

    LogCurrentStorage(broker_, "Initialize");
}

void StubApplet::ExecuteInteractive() {
    LogCurrentStorage(broker_, "ExecuteInteractive");

    auto storage = std::make_shared<IStorage>(system_, std::vector<uint8_t>(0x1000));
    broker_.PushNormalDataFromApplet(storage);

    broker_.PushInteractiveDataFromApplet(storage);
    broker_.SignalStateChanged();
}

void StubApplet::Execute() {
    LogCurrentStorage(broker_, "Execute");

    auto storage = std::make_shared<IStorage>(system_, std::vector<uint8_t>(0x1000));
    broker_.PushNormalDataFromApplet(storage);

    broker_.PushInteractiveDataFromApplet(storage);
    broker_.SignalStateChanged();
}

} // namespace Service::AM::Applets