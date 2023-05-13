#include <array>
#include <cstring>
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/string_util.h"
#include "core/core.h"
#include "core/frontend/applets/error.h"
#include "core/hle/service/am/am.h"
#include "core/hle/service/am/applets/applet_error.h"
#include "core/reporter.h"

namespace Service::AM::Applets {

struct ErrorCode {
    u32 error_category{};
    u32 error_number{};

    static constexpr ErrorCode FromU64(u64 error_code) {
        return {
            .error_category{static_cast<u32>(error_code >> 32)},
            .error_number{static_cast<u32>(error_code & 0xFFFFFFFF)},
        };
    }

    static constexpr ErrorCode FromResult(Result result) {
        return {
            .error_category{2000 + static_cast<u32>(result.module.Value())},
            .error_number{result.description.Value()},
        };
    }

    constexpr Result ToResult() const {
        return Result{static_cast<ErrorModule>(error_category - 2000), error_number};
    }
};

struct ShowError {
    u8 mode;
    bool jump;
    bool use_64bit_error_code;
    std::byte padding[4];
    union {
        u64 error_code_64;
        u32 error_code_32;
    };
};

struct ShowErrorRecord {
    u8 mode;
    bool jump;
    std::byte padding[6];
    u64 error_code_64;
    u64 posix_time;
};

struct SystemErrorArg {
    u8 mode;
    bool jump;
    std::byte padding[6];
    u64 error_code_64;
    std::array<char, 8> language_code;
    std::array<char, 0x800> main_text;
    std::array<char, 0x800> detail_text;
};

struct ApplicationErrorArg {
    u8 mode;
    bool jump;
    std::byte padding[6];
    u32 error_code;
    std::array<char, 8> language_code;
    std::array<char, 0x800> main_text;
    std::array<char, 0x800> detail_text;
};

union Error::ErrorArguments {
    ShowError error;
    ShowErrorRecord error_record;
    SystemErrorArg system_error;
    ApplicationErrorArg application_error;
    std::array<std::byte, 0x1018> raw{};
};

namespace {
    void CopyArgumentData(const std::vector<u8>& data, ErrorArguments& variable) {
        ASSERT(data.size() >= sizeof(ErrorArguments));
        std::memcpy(&variable, data.data(), sizeof(ErrorArguments));
    }

    Result Decode64BitError(u64 error) {
        return ErrorCode::FromU64(error).ToResult();
    }
}

Error::Error(Core::System& system, LibraryAppletMode applet_mode, const Core::Frontend::ErrorApplet& frontend)
    : Applet{system, applet_mode}, m_frontend{frontend}, m_system{system} {}

Error::~Error() = default;

void Error::Initialize() {
    Applet::Initialize();
    m_error_arguments = std::make_unique<ErrorArguments>();
    m_complete = false;

    const auto storage = broker.PopNormalDataToApplet();
    ASSERT(storage != nullptr);
    const auto data = storage->GetData();

    ASSERT(!data.empty());
    std::memcpy(&m_applet_mode, data.data(), sizeof(ErrorAppletMode));

    switch (m_applet_mode) {
    case ErrorAppletMode::ShowError:
        CopyArgumentData(data, m_error_arguments->error);
        m_error_code = m_error_arguments->error.use_64bit_error_code ?
            Decode64BitError(m_error_arguments->error.error_code_64) :
            Result(m_error_arguments->error.error_code_32);
        break;
    case ErrorAppletMode::ShowSystemError:
        CopyArgumentData(data, m_error_arguments->system_error);
        m_error_code = Result(Decode64BitError(m_error_arguments->system_error.error_code_64));
        break;
    case ErrorAppletMode::ShowApplicationError:
        CopyArgumentData(data, m_error_arguments->application_error);
        m_error_code = Result(m_error_arguments->application_error.error_code);
        break;
    case ErrorAppletMode::ShowErrorRecord:
        CopyArgumentData(data, m_error_arguments->error_record);
        m_error_code = Decode64BitError(m_error_arguments->error_record.error_code_64);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented LibAppletError mode={:02X}!", m_applet_mode);
        break;
    }
}

bool Error::TransactionComplete() const {
    return m_complete;
}

Result Error::GetStatus() const {
    return ResultSuccess;
}

void Error::ExecuteInteractive() {
    ASSERT_MSG(false, "Unexpected interactive applet data!");
}

void Error::Execute() {
    if (m_complete) {
        return;
    }

    const auto callback = [this] { DisplayCompleted(); };
    const auto title_id = m_system.GetApplicationProcessProgramID();
    const auto& reporter {m_system.GetReporter()};

    switch (m_applet_mode) {
    case ErrorAppletMode::ShowError:
        reporter.SaveErrorReport(title_id, m_error_code);
        m_frontend.ShowError(m_error_code, callback);
        break;
    case ErrorAppletMode::ShowSystemError:
    case ErrorAppletMode::ShowApplicationError: {
        const auto is_system = m_applet_mode == ErrorAppletMode::ShowSystemError;
        const auto& main_text =
            is_system ? m_error_arguments->system_error.main_text : m_error_arguments->application_error.main_text;
        const auto& detail_text =
            is_system ? m_error_arguments->system_error.detail_text : m_error_arguments->application_error.detail_text;

        const auto main_text_string =
            Common::StringFromFixedZeroTerminatedBuffer(main_text.data(), main_text.size());
        const auto detail_text_string =
            Common::StringFromFixedZeroTerminatedBuffer(detail_text.data(), detail_text.size());

        reporter.SaveErrorReport(title_id, m_error_code, main_text_string, detail_text_string);
        m_frontend.ShowCustomErrorText(m_error_code, main_text_string, detail_text_string, callback);
        break;
    }
    case ErrorAppletMode::ShowErrorRecord:
        reporter.SaveErrorReport(title_id, m_error_code,
                                 fmt::format("{:016X}", m_error_arguments->error_record.posix_time));
        m_frontend.ShowErrorWithTimestamp(
            m_error_code, std::chrono::seconds{m_error_arguments->error_record.posix_time}, callback);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented LibAppletError mode={:02X}!", m_applet_mode);
        DisplayCompleted();
    }
}

void Error::DisplayCompleted() {
    m_complete = true;
    broker.PushNormalDataFromApplet(std::make_shared<IStorage>(m_system, std::vector<u8>{}));
    broker.SignalStateChanged();
}

Result Error::RequestExit() {
    m_frontend.Close();
    R_SUCCEED();
}

} // namespace Service::AM::Applets