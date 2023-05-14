#include <QIcon>
#include <QMessageBox>
#include <QtConcurrent/QtConcurrentRun>
#include "common/settings.h"
#include "core/telemetry_session.h"
#include "ui_configure_web.h"
#include "yuzu/configuration/configure_web.h"
#include "yuzu/uisettings.h"

constexpr char kTokenDelimiter{':'};
const QString kLearnMoreLink{"<a href='https://yuzu-emu.org/help/feature/telemetry/'><span style=\"text-decoration: underline; color:#039be5;\">Learn more</span></a>"};
const QString kSignupLink{"<a href='https://profile.yuzu-emu.org/'><span style=\"text-decoration: underline; color:#039be5;\">Sign up</span></a>"};
const QString kTokenInfoLink{"<a href='https://yuzu-emu.org/wiki/yuzu-web-service/'><span style=\"text-decoration: underline; color:#039be5;\">What is my token?</span></a>"};

std::string GenerateDisplayToken(const std::string& username, const std::string& token) {
    if (username.empty() || token.empty()) {
        return {};
    }

    const std::string unencoded_display_token{username + kTokenDelimiter + token};
    QByteArray b{unencoded_display_token.c_str()};
    QByteArray b64 = b.toBase64();
    return b64.toStdString();
}

std::string UsernameFromDisplayToken(const std::string& display_token) {
    const std::string unencoded_display_token{
        QByteArray::fromBase64(display_token.c_str()).toStdString()};
    return unencoded_display_token.substr(0, unencoded_display_token.find(kTokenDelimiter));
}

std::string TokenFromDisplayToken(const std::string& display_token) {
    const std::string unencoded_display_token{
        QByteArray::fromBase64(display_token.c_str()).toStdString()};
    return unencoded_display_token.substr(unencoded_display_token.find(kTokenDelimiter) + 1);
}

ConfigureWeb::ConfigureWeb(QWidget* parent) : 
    QWidget(parent), 
    ui(std::make_unique<Ui::ConfigureWeb>())
{
    ui->setupUi(this);
    ui->discord_group->setVisible(false);
    ui->web_credentials_disclaimer->setWordWrap(true);
    ui->telemetry_learn_more->setOpenExternalLinks(true);
    ui->web_signup_link->setOpenExternalLinks(true);
    ui->web_token_info_link->setOpenExternalLinks(true);
    ui->toggle_telemetry->setChecked(Settings::values.enable_telemetry.GetValue());
    ui->edit_token->setText(QString::fromStdString(GenerateDisplayToken(
        Settings::values.yuzu_username.GetValue(), Settings::values.yuzu_token.GetValue())));
    RetranslateUI();
    connect(ui->button_regenerate_telemetry_id, &QPushButton::clicked, this, &ConfigureWeb::RefreshTelemetryID);
    connect(ui->button_verify_login, &QPushButton::clicked, this, &ConfigureWeb::VerifyLogin);
    connect(ui->edit_token, &QLineEdit::textChanged, this, &ConfigureWeb::OnLoginChanged);
    connect(&verify_watcher, &QFutureWatcher<bool>::finished, this, &ConfigureWeb::OnLoginVerified);
}

ConfigureWeb::~ConfigureWeb() = default;

void ConfigureWeb::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QWidget::changeEvent(event);
}

void ConfigureWeb::RetranslateUI() {
    ui->retranslateUi(this);

    ui->telemetry_learn_more->setText(kLearnMoreLink);
    ui->web_signup_link->setText(kSignupLink);
    ui->web_token_info_link->setText(kTokenInfoLink);

    const QString telemetryIdStr = "Telemetry ID: 0x%1";
    const QString telemetryId = QString::number(Core::GetTelemetryId(), 16).toUpper();
    ui->label_telemetry_id->setText(tr(telemetryIdStr.toStdString().c_str(), telemetryId.toStdString().c_str()));
}

void ConfigureWeb::SetConfiguration() {
    if (Settings::values.yuzu_username.GetValue().empty()) {
        ui->username->setText(tr("Unspecified"));
    } else {
        ui->username->setText(QString::fromStdString(Settings::values.yuzu_username.GetValue()));
    }

    user_verified = true;

    ui->toggle_discordrpc->setChecked(UISettings::values.enable_discord_presence.GetValue());
}

void ConfigureWeb::ApplyConfiguration() {
    Settings::values.enable_telemetry = ui->toggle_telemetry->isChecked();
    UISettings::values.enable_discord_presence = ui->toggle_discordrpc->isChecked();
    if (user_verified) {
        Settings::values.yuzu_username = UsernameFromDisplayToken(ui->edit_token->text().toStdString());
        Settings::values.yuzu_token = TokenFromDisplayToken(ui->edit_token->text().toStdString());
    } else {
        QMessageBox::warning(
            this, tr("Token not verified"),
            tr("Token was not verified. The change to your token has not been saved."));
    }
}

void ConfigureWeb::RefreshTelemetryID() {
    const u64 new_telemetry_id{Core::RegenerateTelemetryId()};
    const QString telemetryIdStr = "Telemetry ID: 0x%1";
    const QString telemetryId = QString::number(new_telemetry_id, 16).toUpper();
    ui->label_telemetry_id->setText(tr(telemetryIdStr.toStdString().c_str(), telemetryId.toStdString().c_str()));
}

void ConfigureWeb::OnLoginChanged() {
    if (ui->edit_token->text().isEmpty()) {
        user_verified = true;
        ui->label_token_verified->setPixmap(QPixmap());
        ui->label_token_verified->setToolTip(QString());
    } else {
        user_verified = false;
        const QPixmap pixmap = QIcon::fromTheme(QStringLiteral("info")).pixmap(16);
        const QString tooltip = tr("Unverified, please click Verify before saving configuration", "Tooltip");
        ui->label_token_verified->setPixmap(pixmap);
        ui->label_token_verified->setToolTip(tooltip.toStdString().c_str());
    }
}

void ConfigureWeb::VerifyLogin() {
    const QString verifyingText = tr("Verifying...");
    ui->button_verify_login->setDisabled(true);
    ui->button_verify_login->setText(verifyingText);
    const QIcon syncingIcon = QIcon::fromTheme(QStringLiteral("sync"));
    ui->label_token_verified->setPixmap(syncingIcon.pixmap(16));
    ui->label_token_verified->setToolTip(verifyingText.toStdString().c_str());
    verify_watcher.setFuture(QtConcurrent::run(
        [username = std::cref(UsernameFromDisplayToken(ui->edit_token->text().toStdString())),
         token = std::cref(TokenFromDisplayToken(ui->edit_token->text().toStdString()))] {
            return Core::VerifyLogin(username.get(), token.get());
        }));
}

void ConfigureWeb::OnLoginVerified() {
    ui->button_verify_login->setEnabled(true);
    ui->button_verify_login->setText(tr("Verify"));
    const QIcon checkedIcon = QIcon::fromTheme(QStringLiteral("checked"));
    const QIcon failedIcon = QIcon::fromTheme(QStringLiteral("failed"));
    if (verify_watcher.result()) {
        user_verified = true;
        const QString username = QString::fromStdString(UsernameFromDisplayToken(ui->edit_token->text().toStdString()));
        const QString verifiedText = tr("Verified", "Tooltip");
        ui->label_token_verified->setPixmap(checkedIcon.pixmap(16));
        ui->label_token_verified->setToolTip(verifiedText);
        ui->username->setText(username);
    } else {
        const QString verificationFailedText = tr("Verification failed", "Tooltip");
        ui->label_token_verified->setPixmap(failedIcon.pixmap(16));
        ui->label_token_verified->setToolTip(verificationFailedText);
        ui->username->setText(tr("Unspecified"));
        QMessageBox::critical(this, tr("Verification failed"),
                              tr("Verification failed. Check that you have entered your token "
                                 "correctly, and that your internet connection is working."));
    }
}

void ConfigureWeb::SetWebServiceConfigEnabled(bool enabled) {
    ui->label_disable_info->setVisible(!enabled);
    ui->groupBoxWebConfig->setEnabled(enabled);
}

