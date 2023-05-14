// Configuration class
class Config {
public:
    std::string username;
    std::string token;
    bool enableTelemetry;
    bool enableDiscordRpc;
};

// Token generator class
class TokenGenerator {
public:
    std::string GenerateDisplayToken(const std::string& username, const std::string& token) const {
        ...
    }
};

// Token extraction class
class TokenExtractor {
public:
    std::string UsernameFromDisplayToken(const std::string& display_token) const {
        ...
    }

    std::string TokenFromDisplayToken(const std::string& display_token) const {
        ...
    }
};

// Verification class
class Verification {
public:
    bool VerifyLogin(const std::string& username, const std::string& token) const {
        ...
    }
};

// Telemetry ID class
class TelemetryId {
public:
    void Regenerate() const {
        ...
    }

    u64 Get() const {
        ...
    }
};

// UI class
class Ui {
public:
    Ui(QWidget* parent) {
        ...
    }

    void SetConfiguration(const Config& config) {
        ...
    }

    Config GetConfiguration() const {
        ...
    }

    void ApplyConfiguration() {
        ...
    }

    void SetWebServiceConfigEnabled(bool enabled) {
        ...
    }

    void OnLoginChanged() {
        ...
    }

    void VerifyLogin() {
        ...
    }

    void OnLoginVerified() {
        ...
    }

    void RefreshTelemetryID() {
        ...
    }

    void RetranslateUI() {
        ...
    }

private:
    std::unique_ptr<Ui::ConfigureWeb> ui;
    bool userVerified;
    QFutureWatcher<bool> verifyWatcher;
};

// ConfigureWeb class
class ConfigureWeb : public QWidget {
public:
    ConfigureWeb(QWidget* parent) : QWidget(parent) {}

    void changeEvent(QEvent* event) {
        ...
    }

private:
    Config config_;
    Ui ui_;

    std::unique_ptr<TokenGenerator> tokenGenerator_;
    std::unique_ptr<TokenExtractor> tokenExtractor_;
    std::unique_ptr<Verification> verification_;
    std::unique_ptr<TelemetryId> telemetryId_;

    void SetConfiguration() {
        ...
    }

    void ApplyConfiguration() {
        ...
    }

    void SetWebServiceConfigEnabled(bool enabled) {
        ...
    }
};