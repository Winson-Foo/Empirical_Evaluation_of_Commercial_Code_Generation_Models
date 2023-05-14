namespace Service::AM::Applets {

constexpr Result kErrorInvalidPin{ErrorModule::PCTL, 221};
constexpr int kAuthAppletArgsSize = 0xC;
constexpr int kMaxStorageDataSize = 0x1000;

static void LogCurrentStorage(AppletDataBroker& broker, std::string_view prefix) {
    std::shared_ptr<IStorage> storage = broker.PopNormalDataToApplet();
    for (; storage != nullptr; storage = broker.PopNormalDataToApplet()) {
        const auto data = storage->GetData();
        LOG_INFO(Service_AM,
                 "Received normal data during {} with size={:08X}, data={}",
                 prefix, data.size(), Common::HexToString(data));
    }

    storage = broker.PopInteractiveDataToApplet();
    for (; storage != nullptr; storage = broker.PopInteractiveDataToApplet()) {
        const auto data = storage->GetData();
        LOG_INFO(Service_AM,
                 "Received interactive data during {} with size={:08X}, data={}",
                 prefix, data.size(), Common::HexToString(data));
    }
}

Auth::Auth(Core::System& system, LibraryAppletMode applet_mode,
           Core::Frontend::ParentalControlsApplet& frontend)
    : Applet{system, applet_mode}, frontend{frontend}, system{system} {}

void Auth::Initialize() {
    Applet::Initialize();
    complete = false;

    const auto storage = broker.PopNormalDataToApplet();
    ASSERT(storage != nullptr);
    const auto data = storage->GetData();
    ASSERT(data.size() >= kAuthAppletArgsSize);

    struct AuthAppletArgs {
        INSERT_PADDING_BYTES(4);
        AuthAppletType type;
        u8 arg0;
        u8 arg1;
        u8 arg2;
        INSERT_PADDING_BYTES(1);
    };
    static_assert(sizeof(AuthAppletArgs) == kAuthAppletArgsSize, "AuthAppletArgs has incorrect size.");

    AuthAppletArgs args{};
    std::memcpy(&args, data.data(), sizeof(AuthAppletArgs));

    auth_applet_type = args.type;
    arg0 = args.arg0;
    arg1 = args.arg1;
    arg2 = args.arg2;
}

Result Auth::GetStatus() const {
    return is_successful ? ResultSuccess : kErrorInvalidPin;
}

void Auth::ExecuteInteractive() {
    ASSERT_MSG(false, "Unexpected interactive applet data.");
}

void Auth::Execute() {
    if (complete) {
        return;
    }

    const auto unimplemented_log = [this] {
        UNIMPLEMENTED_MSG(
            "Unimplemented Auth applet type for type={:08X}, arg0={:02X}, arg1={:02X}, arg2={:02X}",
            auth_applet_type, arg0, arg1, arg2);
    };

    switch (auth_applet_type) {
        // ...
    }
}

void Auth::AuthFinished(bool is_successful) {
    is_successful = is_successful;

    struct AuthAppletResult {
        Result result_code;
    };
    static_assert(sizeof(AuthAppletResult) == 0x4, "AuthAppletResult has incorrect size.");

    AuthAppletResult result{GetStatus()};

    std::vector<u8> out(sizeof(AuthAppletResult));
    std::memcpy(out.data(), &result, sizeof(AuthAppletResult));

    broker.PushNormalDataFromApplet(std::make_shared<IStorage>(
        system, std::move(out), kMaxStorageDataSize));
    broker.SignalStateChanged();
}

StubApplet::StubApplet(Core::System& system, AppletId id, LibraryAppletMode applet_mode)
    : Applet{system, applet_mode}, id{id}, system{system} {}

void StubApplet::Initialize() {
    Applet::Initialize();

    const auto data = broker.PeekDataToAppletForDebug();
    system.GetReporter().SaveUnimplementedAppletReport(
        static_cast<u32>(id), common_args.arguments_version, common_args.library_version,
        common_args.theme_color, common_args.play_startup_sound, common_args.system_tick,
        data.normal, data.interactive);

    LogCurrentStorage(broker, "Initialize");
}

Result StubApplet::GetStatus() const {
    return ResultSuccess;
}

void StubApplet::ExecuteInteractive() {
    LogCurrentStorage(broker, "ExecuteInteractive");

    broker.PushNormalDataFromApplet(std::make_shared<IStorage>(system, std::vector<u8>(kMaxStorageDataSize)));
    broker.PushInteractiveDataFromApplet(std::make_shared<IStorage>(system, std::vector<u8>(kMaxStorageDataSize)));
    broker.SignalStateChanged();
}

void StubApplet::Execute() {
    LogCurrentStorage(broker, "Execute");

    broker.PushNormalDataFromApplet(std::make_shared<IStorage>(system, std::vector<u8>(kMaxStorageDataSize)));
    broker.PushInteractiveDataFromApplet(std::make_shared<IStorage>(system, std::vector<u8>(kMaxStorageDataSize)));
    broker.SignalStateChanged();
}

} // namespace Service::AM::Applets```

