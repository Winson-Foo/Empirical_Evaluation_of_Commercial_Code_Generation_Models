// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/hle/service/service.h"

namespace Core {
class System;
}

namespace Service {

namespace FileSystem {
class FileSystemController;
} // namespace FileSystem

namespace NS {

class IAccountProxyInterface final : public ServiceFramework<IAccountProxyInterface> {
public:
    explicit IAccountProxyInterface(Core::System& system_);
    ~IAccountProxyInterface() override;
};

class IApplicationManagerInterface final : public ServiceFramework<IApplicationManagerInterface> {
public:
    explicit IApplicationManagerInterface(Core::System& system_);
    ~IApplicationManagerInterface() override;

    ResultVal<u8> GetApplicationDesiredLanguage(u32 supported_languages);
    ResultVal<u64> ConvertApplicationLanguageToLanguageCode(u8 application_language);

private:
    void GetApplicationControlData(HLERequestContext& ctx);
    void GetApplicationDesiredLanguage(HLERequestContext& ctx);
    void ConvertApplicationLanguageToLanguageCode(HLERequestContext& ctx);
};

class IApplicationVersionInterface final : public ServiceFramework<IApplicationVersionInterface> {
public:
    explicit IApplicationVersionInterface(Core::System& system_);
    ~IApplicationVersionInterface() override;
};

class IContentManagementInterface final : public ServiceFramework<IContentManagementInterface> {
public:
    explicit IContentManagementInterface(Core::System& system_);
    ~IContentManagementInterface() override;
};

class IDocumentInterface final : public ServiceFramework<IDocumentInterface> {
public:
    explicit IDocumentInterface(Core::System& system_);
    ~IDocumentInterface() override;
};

class IDownloadTaskInterface final : public ServiceFramework<IDownloadTaskInterface> {
public:
    explicit IDownloadTaskInterface(Core::System& system_);
    ~IDownloadTaskInterface() override;
};

class IECommerceInterface final : public ServiceFramework<IECommerceInterface> {
public:
    explicit IECommerceInterface(Core::System& system_);
    ~IECommerceInterface() override;
};

class IFactoryResetInterface final : public ServiceFramework<IFactoryResetInterface> {
public:
    explicit IFactoryResetInterface(Core::System& system_);
    ~IFactoryResetInterface() override;
};

class IReadOnlyApplicationControlDataInterface final
    : public ServiceFramework<IReadOnlyApplicationControlDataInterface> {
public:
    explicit IReadOnlyApplicationControlDataInterface(Core::System& system_);
    ~IReadOnlyApplicationControlDataInterface() override;

private:
    void GetApplicationControlData(HLERequestContext& ctx);
};

class NS final : public ServiceFramework<NS> {
public:
    explicit NS(const char* name, Core::System& system_);
    ~NS() override;

    std::shared_ptr<IApplicationManagerInterface> GetApplicationManagerInterface() const;

private:
    template <typename T, typename... Args>
    void PushInterface(HLERequestContext& ctx) {
        LOG_DEBUG(Service_NS, "called");

        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<T>(system);
    }

    void PushIApplicationManagerInterface(HLERequestContext& ctx) {
        LOG_DEBUG(Service_NS, "called");

        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<IApplicationManagerInterface>(system);
    }

    template <typename T, typename... Args>
    std::shared_ptr<T> GetInterface(Args&&... args) const {
        static_assert(std::is_base_of_v<SessionRequestHandler, T>,
                      "Not a base of ServiceFrameworkBase");

        return std::make_shared<T>(std::forward<Args>(args)...);
    }
};

void LoopProcess(Core::System& system);

} // namespace NS
} // namespace Service
