#pragma once

#include <QWidget>
#include <QFutureWatcher>
#include <string>
#include <memory>

namespace Ui {
    class ConfigureWeb;
}

class ConfigureWeb : public QWidget {
    Q_OBJECT
public:
    explicit ConfigureWeb(QWidget* parent = nullptr);
    virtual ~ConfigureWeb();

    void SetWebServiceConfigEnabled(bool enabled);

signals:
    void OnConfigurationChanged();

private slots:
    void RefreshTelemetryID();
    void VerifyLogin();
    void OnLoginVerified();
    void OnLoginChanged();

protected:
    virtual void changeEvent(QEvent* event) override;

private:
    std::unique_ptr<Ui::ConfigureWeb> ui;
    QFutureWatcher<bool> verify_watcher;

    void RetranslateUI();
    void SetConfiguration();
    void ApplyConfiguration();
    bool IsLoginVerified() const;
    std::string GetUsername() const;
    std::string GetToken() const;
    void SetVerified(bool verified);
    void SetVerificationPending();
    static std::string GenerateDisplayToken(const std::string& username, const std::string& token);
    static std::string UsernameFromDisplayToken(const std::string& display_token);
    static std::string TokenFromDisplayToken(const std::string& display_token);
};

#include <QIcon>
#include <QMessageBox>
#include <QtConcurrent/QtConcurrentRun>
#include "common/settings.h"
#include "core/telemetry_session.h"
#include "ui_configure_web.h"
#include "yuzu/configuration/configure_web.h"
#include "yuzu/uisettings.h"

constexpr char kTokenDelimiter{':'};

std::string ConfigureWeb::GenerateDisplayToken(const std::string& username, const std::string& token) {
    if (username.empty() || token.empty()) {
        return {};
    }

    const std::string unencoded_display_token{username + kTokenDelimiter + token};
    QByteArray b{unencoded_display_token.c_str()};
    QByteArray b64 = b.toBase64();
    return b64.toStdString();
}

std::string ConfigureWeb::UsernameFromDisplayToken(const std::string& display_token) {
    const std::string unencoded_display_token{
        QByteArray::fromBase64(display_token.c_str()).toStdString()};
    return unencoded_display_token.substr(0, unencoded_display_token.find(kTokenDelimiter));
}

std::string ConfigureWeb::TokenFromDisplayToken(const std::string& display_token) {
    const std::string unencoded_display_token{
        QByteArray::fromBase64(display_token.c_str()).toStdString()};
    return unencoded_display_token.substr(unencoded_display_token.find(kTokenDelimiter) + 1);
}

ConfigureWeb::ConfigureWeb(QWidget* parent)
    : QWidget(parent), ui(std::make_unique<Ui::ConfigureWeb>()) {
    ui->setupUi(this);
    connect(ui->button_regenerate_telemetry_id, &QPushButton::clicked, this,
            &ConfigureWeb::RefreshTelemetryID);
    connect(ui->button_verify_login, &QPushButton::clicked, this, &ConfigureWeb::VerifyLogin);
    connect(&verify_watcher, &QFutureWatcher<bool>::finished, this, &ConfigureWeb::OnLoginVerified);

#ifndef USE_DISCORD_PRESENCE
    ui->discord_group->setVisible(false);
#endif

    SetConfiguration();
    RetranslateUI();
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

    ui->telemetry_learn_more->setText(
        tr("<a href='https://yuzu-emu.org/help/feature/telemetry/'><span style=\"text-decoration: "
           "underline; color:#039be5;\">Learn more</span></a>"));

    ui->web_signup_link->setText(
        tr("<a href='https://profile.yuzu-emu.org/'><span style=\"text-decoration: underline; "
           "color:#039be5;\">Sign up</span></a>"));

    ui->web_token_info_link->setText(
        tr("<a href='https://yuzu-emu.org/wiki/yuzu-web-service/'><span style=\"text-decoration: "
           "underline; color:#039be5;\">What is my token?</span></a>"));

    ui->label_telemetry_id->setText(
        tr("Telemetry ID: 0x%1").arg(QString::number(Core::GetTelemetryId(), 16).toUpper()));
}

void ConfigureWeb::SetConfiguration() {
    ui->web_credentials_disclaimer->setWordWrap(true);

    ui->telemetry_learn_more->setOpenExternalLinks(true);
    ui->web_signup_link->setOpenExternalLinks(true);
    ui->web_token_info_link->setOpenExternalLinks(true);

    if (Settings::values.yuzu_username.GetValue().empty()) {
        ui->username->setText(tr("Unspecified"));
    } else {
        ui->username->setText(QString::fromStdString(Settings::values.yuzu_username.GetValue()));
    }

    ui->toggle_telemetry->setChecked(Settings::values.enable_telemetry.GetValue());
    ui->edit_token->setText(QString::fromStdString(GenerateDisplayToken(GetUsername(), GetToken())));

    connect(ui->edit_token, &QLineEdit::textChanged, this, &ConfigureWeb::OnLoginChanged);

    SetVerified(IsLoginVerified());

    ui->toggle_discordrpc->setChecked(UISettings::values.enable_discord_presence.GetValue());
}

void ConfigureWeb::ApplyConfiguration() {
    Settings::values.enable_telemetry = ui->toggle_telemetry->isChecked();
    UISettings::values.enable_discord_presence = ui->toggle_discordrpc->isChecked();
    if (IsLoginVerified()) {
        Settings::values.yuzu_username = GetUsername();
        Settings::values.yuzu_token = GetToken();
    } else {
        QMessageBox::warning(
            this, tr("Token not verified"),
            tr("Token was not verified. The change to your token has not been saved."));
    }

    emit OnConfigurationChanged();
}

bool ConfigureWeb::IsLoginVerified() const {
    return !ui->edit_token->text().isEmpty() && ui->label_token_verified->toolTip() == tr("Verified", "Tooltip");
}

std::string ConfigureWeb::GetUsername() const {
    return UsernameFromDisplayToken(ui->edit_token->text().toStdString());
}

std::string ConfigureWeb::GetToken() const {
    return TokenFromDisplayToken(ui->edit_token->text().toStdString());
}

void ConfigureWeb::SetVerified(bool verified) {
    if (verified) {
        ui->label_token_verified->setPixmap(QIcon::fromTheme(QStringLiteral("checked")).pixmap(16));
        ui->label_token_verified->setToolTip(tr("Verified", "Tooltip"));
        ui->username->setText(QString::fromStdString(GetUsername()));
    } else {
        ui->label_token_verified->setPixmap(QPixmap());
        ui->label_token_verified->setToolTip(QString());
        ui->username->setText(tr("Unspecified"));
    }
}

void ConfigureWeb::SetVerificationPending() {
        ui->button_verify_login->setDisabled(true);
        ui->button_verify_login->setText(tr("Verifying..."));
        ui->label_token_verified->setPixmap(QIcon::fromTheme(QStringLiteral("sync")).pixmap(16));
        ui->label_token_verified->setToolTip(tr("Verifying..."));
}

void ConfigureWeb::RefreshTelemetryID() {
    const u64 new_telemetry_id{Core::RegenerateTelemetryId()};
    ui->label_telemetry_id->setText(
        tr("Telemetry ID: 0x%1").arg(QString::number(new_telemetry_id, 16).toUpper()));
}

void ConfigureWeb::VerifyLogin() {
    SetVerificationPending();
    verify_watcher.setFuture(QtConcurrent::run([this]() {
        return Core::VerifyLogin(GetUsername(), GetToken());
    }));
}

void ConfigureWeb::OnLoginVerified() {
    ui->button_verify_login->setEnabled(true);
    ui->button_verify_login->setText(tr("Verify"));
    SetVerified(verify_watcher.result());
    if (!IsLoginVerified()) {
        QMessageBox::critical(this, tr("Verification failed"),
                              tr("Verification failed. Check that you have entered your token "
                                 "correctly, and that your internet connection is working."));
        SetVerified(false);
    }

    emit OnConfigurationChanged();
}

void ConfigureWeb::OnLoginChanged() {
    SetVerified(false);
    if (!ui->edit_token->text().isEmpty()) {
        ui->label_token_verified->setPixmap(QIcon::fromTheme(QStringLiteral("info")).pixmap(16));
        ui->label_token_verified->setToolTip(
            tr("Unverified, please click Verify before saving configuration", "Tooltip"));
    } else {
        ui->label_token_verified->setPixmap(QPixmap());
        ui->label_token_verified->setToolTip(QString());
    }
}

void ConfigureWeb::SetWebServiceConfigEnabled(bool enabled) {
    ui->label_disable_info->setVisible(!enabled);
    ui->groupBoxWebConfig->setEnabled(enabled);
}