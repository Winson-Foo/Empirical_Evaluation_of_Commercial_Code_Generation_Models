// File: auth_applet.hpp

#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "core/core.h"
#include "core/frontend/applets/general_frontend.h"
#include "core/hle/service/am/applets/applet.h"
#include "core/hle/service/am/applets/auth/auth_applet_types.h"
#include "core/hle/service/am/applets/auth/common.h"
#include "core/hle/result.h"

namespace Service::AM::Applets {

class Auth : public Applet {
public:
    Auth(Core::System& system, LibraryAppletMode applet_mode,
         const Core::Frontend::ParentalControlsApplet& frontend);
    ~Auth() override;

    void Initialize() override;
    bool TransactionComplete() const override;
    Result GetStatus() const override;
    void ExecuteInteractive() override;
    void Execute() override;
    Result RequestExit() override;

private:
    void AuthFinished(bool is_successful);

    const Core::Frontend::ParentalControlsApplet& frontend;
    Core::System& system;

    AuthAppletType type;
    u8 arg0;
    u8 arg1;
    u8 arg2;

    bool successful{false};
    bool complete{false};
};

} // namespace Service::AM::Applets


// File: auth_applet.cpp

#include "auth_applet.hpp"

#include <cstring>

#include "common/hex_util.h"
#include "common/logging/log.h"
#include "core/hle/service/am/applets/applet_general_backend.h"
#include "core/reporter.h"

namespace Service::AM::Applets {

constexpr Result kErrorInvalidPin{ErrorModule::PCTL, 221};

void LogCurrentStorage(AppletDataBroker& broker, std::string_view prefix) {
    auto storage = broker.PopNormalDataToApplet();
    while (storage != nullptr) {
        const auto data = storage->GetData();
        LOG_INFO(Service_AM,
                 "called (STUBBED), during {} received normal data with size={:08X}, data={}",
                 prefix, data.size(), Common::HexToString(data));
        storage = broker.PopNormalDataToApplet();
    }

    storage = broker.PopInteractiveDataToApplet();
    while (storage != nullptr) {
        const auto data = storage->GetData();
        LOG_INFO(Service_AM,
                 "called (STUBBED), during {} received interactive data with size={:08X}, data={}",
                 prefix, data.size(), Common::HexToString(data));
        storage = broker.PopInteractiveDataToApplet();
    }
}

Auth::Auth(Core::System& system_, LibraryAppletMode applet_mode_,
           const Core::Frontend::ParentalControlsApplet& frontend_)
    : Applet{system_, applet_mode_}, frontend{frontend_}, system{system_} {}

Auth::~Auth() = default;

void Auth::Initialize() {
    Applet::Initialize();
    complete = false;

    const auto storage = broker.PopNormalDataToApplet();
    if (storage == nullptr) {
        throw std::runtime_error("Failed to pop normal data in Auth::Initialize()");
    }

    const auto data = storage->GetData();
    if (data.size() < AuthApplet::kArgSize) {
        throw std::runtime_error("Invalid argument size in Auth::Initialize()");
    }

    struct Arg {
        INSERT_PADDING_BYTES(4);
        AuthAppletType type;
        u8 arg0;
        u8 arg1;
        u8 arg2;
        INSERT_PADDING_BYTES(1);
    };
    static_assert(sizeof(Arg) == AuthApplet::kArgSize, "AuthApplet::Arg has incorrect size.");

    const auto& arg = *reinterpret_cast<const Arg*>(data.data());

    type = arg.type;
    arg0 = arg.arg0;
    arg1 = arg.arg1;
    arg2 = arg.arg2;
}

bool Auth::TransactionComplete() const {
    return complete;
}

Result Auth::GetStatus() const {
    return successful ? ResultSuccess : kErrorInvalidPin;
}

void Auth::ExecuteInteractive() {
    throw std::runtime_error("Unexpected interactive applet data in Auth::ExecuteInteractive()");
}

void Auth::Execute() {
    if (complete) {
        return;
    }

    switch (type) {
    case AuthAppletType::ShowParentalAuthentication: {
        const auto callback = [this](bool is_successful) { AuthFinished(is_successful); };

        if (arg0 == 1 && arg1 == 0 && arg2 == 1) {
            // ShowAuthenticatorForConfiguration
            frontend.VerifyPINForSettings(callback);
        } else if (arg1 == 0 && arg2 == 0) {
            // ShowParentalAuthentication(bool)
            frontend.VerifyPIN(callback, static_cast<bool>(arg0));
        } else {
            throw std::runtime_error("Unimplemented Auth applet type in Auth::Execute()");
        }
        break;
    }
    case AuthAppletType::RegisterParentalPasscode: {
        const auto callback = [this] { AuthFinished(true); };

        if (arg0 == 0 && arg1 == 0 && arg2 == 0) {
            // RegisterParentalPasscode
            frontend.RegisterPIN(callback);
        } else {
            throw std::runtime_error("Unimplemented Auth applet type in Auth::Execute()");
        }
        break;
    }
    case AuthAppletType::ChangeParentalPasscode: {
        const auto callback = [this] { AuthFinished(true); };

        if (arg0 == 0 && arg1 == 0 && arg2 == 0) {
            // ChangeParentalPasscode
            frontend.ChangePIN(callback);
        } else {
            throw std::runtime_error("Unimplemented Auth applet type in Auth::Execute()");
        }
        break;
    }
    default:
        throw std::runtime_error("Unimplemented Auth applet type in Auth::Execute()");
    }
}

void Auth::AuthFinished(bool is_successful) {
    successful = is_successful;

    struct Return {
        Result result_code;
    };
    static_assert(sizeof(Return) == AuthApplet::kResultSize, "AuthApplet::Return has incorrect size.");

    Return return_{GetStatus()};

    std::vector<u8> out(sizeof(Return));
    std::memcpy(out.data(), &return_, sizeof(Return));

    broker.PushNormalDataFromApplet(std::make_shared<IStorage>(system, std::move(out)));
    broker.SignalStateChanged();
}

Result Auth::RequestExit() {
    frontend.Close();
    return ResultSuccess;
}


// File: photo_viewer_applet.hpp

#pragma once

#include "core/hle/service/am/applets/applet.h"
#include "core/hle/service/am/applets/photo_viewer/photo_viewer_applet_types.h"

namespace Core::Frontend {
class PhotoViewerApplet;
}

namespace Service::AM::Applets {

class PhotoViewer : public Applet {
public:
    PhotoViewer(Core::System& system, LibraryAppletMode applet_mode,
                const Core::Frontend::PhotoViewerApplet& frontend);
    ~PhotoViewer() override;

    void Initialize() override;
    bool TransactionComplete() const override;
    Result GetStatus() const override;
    void ExecuteInteractive() override;
    void Execute() override;
    Result RequestExit() override;

private:
    void ViewFinished();

    const Core::Frontend::PhotoViewerApplet& frontend;
    Core::System& system;

    PhotoViewerAppletMode mode;

    bool complete{false};
};

} // namespace Service::AM::Applets


// File: photo_viewer_applet.cpp

#include "photo_viewer_applet.hpp"

#include "core/frontend/photo_viewer_applet.h"

namespace Service::AM::Applets {

PhotoViewer::PhotoViewer(Core::System& system_, LibraryAppletMode applet_mode_,
                         const Core::Frontend::PhotoViewerApplet& frontend_)
    : Applet{system_, applet_mode_}, frontend{frontend_}, system{system_} {}

PhotoViewer::~PhotoViewer() = default;

void PhotoViewer::Initialize() {
    Applet::Initialize();
    complete = false;

    const auto storage = broker.PopNormalDataToApplet();
    if (storage == nullptr) {
        throw std::runtime_error("Failed to pop normal data in PhotoViewer::Initialize()");
    }

    const auto data = storage->GetData();
    if (data.empty()) {
        throw std::runtime_error("Invalid argument size in PhotoViewer::Initialize()");
    }

    mode = static_cast<PhotoViewerAppletMode>(data[0]);
}

bool PhotoViewer::TransactionComplete() const {
    return complete;
}

Result PhotoViewer::GetStatus() const {
    return ResultSuccess;
}

void PhotoViewer::ExecuteInteractive() {
    throw std::runtime_error("Unexpected interactive applet data in PhotoViewer::ExecuteInteractive()");
}

void PhotoViewer::Execute() {
    if (complete) {
        return;
    }

    const auto callback = [this] { ViewFinished(); };
    switch (mode) {
    case PhotoViewerAppletMode::CurrentApp:
        frontend.ShowPhotosForApplication(system.GetApplicationProcessProgramID(), callback);
        break;
    case PhotoViewerAppletMode::AllApps:
        frontend.ShowAllPhotos(callback);
        break;
    default:
        throw std::runtime_error("Unimplemented PhotoViewer applet mode in PhotoViewer::Execute()");
    }
}

void PhotoViewer::ViewFinished() {
    broker.PushNormalDataFromApplet(std::make_shared<IStorage>(system, std::vector<u8>{}));
    broker.SignalStateChanged();
}

Result PhotoViewer::RequestExit() {
    frontend.Close();
    return ResultSuccess;
}


// File: stub_applet.hpp

#pragma once

#include "core/hle/service/am/applets

