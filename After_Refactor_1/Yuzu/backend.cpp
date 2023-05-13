#include <atomic>
#include <chrono>
#include <climits>
#include <memory>
#include <thread>

#include <fmt/format.h>

#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/literals.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"

#include "common/logging/backend.h"
#include "common/logging/log.h"
#include "common/logging/log_entry.h"
#include "common/logging/text_formatter.h"
#include "common/settings.h"
#include "common/bounded_threadsafe_queue.h"

namespace {

constexpr auto MAX_LOG_SIZE = Settings::values.extended_logging ? 1_GiB : 100_MiB;
constexpr auto MAX_LOGS_TO_WRITE = Settings::values.log_filter.IsDebug() ? INT_MAX : 100;

}

namespace Common::Log {

namespace Constants {

const auto LOG_FILE = "yuzu_log.txt";

}

class Backend {
public:
    virtual ~Backend() = default;

    virtual void Write(const Entry& entry) = 0;
    virtual void EnableForStacktrace() = 0;
    virtual void Flush() = 0;
};

class ColorConsoleBackend final : public Backend {
public:
    explicit ColorConsoleBackend() = default;

    ~ColorConsoleBackend() override = default;

    void Write(const Entry& entry) override {
        if (enabled.load(std::memory_order_relaxed)) {
            PrintColoredMessage(entry);
        }
    }

    void Flush() override {
        // stderr shouldn't be buffered
    }

    void EnableForStacktrace() override {
        enabled = true;
    }

    void SetEnabled(bool enabled_) {
        enabled = enabled_;
    }

private:
    std::atomic_bool enabled{false};
};

class FileBackend final : public Backend {
public:
    explicit FileBackend(const std::filesystem::path& filename) {
        auto old_filename = filename;
        old_filename += ".old.txt";

        // Existence checks are done within the functions themselves.
        // We don't particularly care if these succeed or not.
        static_cast<void>(FS::RemoveFile(old_filename));
        static_cast<void>(FS::RenameFile(filename, old_filename));

        file = std::make_unique<FS::IOFile>(filename, FS::FileAccessMode::Write,
                                            FS::FileType::TextFile);
    }

    ~FileBackend() override = default;

    void Write(const Entry& entry) override {
        if (!enabled) {
            return;
        }

        bytes_written += file->WriteString(FormatLogMessage(entry).append(1, '\n'));

        const bool write_limit_exceeded = bytes_written > MAX_LOG_SIZE;
        if (entry.log_level >= Level::Error || write_limit_exceeded) {
            if (write_limit_exceeded) {
                enabled = false;
            }
            file->Flush();
        }
    }

    void Flush() override {
        file->Flush();
    }

    void EnableForStacktrace() override {
        enabled = true;
        bytes_written = 0;
    }

private:
    std::unique_ptr<FS::IOFile> file;
    bool enabled = true;
    std::size_t bytes_written = 0;
};

class DebuggerBackend final : public Backend {
public:
    explicit DebuggerBackend() = default;

    ~DebuggerBackend() override = default;

    void Write(const Entry& entry) override {
#ifdef _WIN32
        ::OutputDebugStringW(UTF8ToUTF16W(FormatLogMessage(entry).append(1, '\n')).c_str());
#endif
    }

    void Flush() override {}

    void EnableForStacktrace() override {}
};

class LoggerImpl {
public:
    explicit LoggerImpl(const std::filesystem::path& file_backend_filename, const Filter& filter_)
        : filter{filter_}, file_backend{file_backend_filename} {}

    void SetGlobalFilter(const Filter& f) {
        filter = f;
    }

    void SetColorConsoleBackendEnabled(bool enabled) {
        color_console_backend.SetEnabled(enabled);
    }

    void PushEntry(Class log_class, Level log_level, const char* filename, unsigned int line_num,
                   const char* function, std::string&& message) {
        if (!filter.CheckMessage(log_class, log_level)) {
            return;
        }
        message_queue.EmplaceWait(
            CreateEntry(log_class, log_level, filename, line_num, function, std::move(message)));
    }

    void ForEachBackend(auto lambda) {
        lambda(static_cast<Backend&>(debugger_backend));
        lambda(static_cast<Backend&>(color_console_backend));
        lambda(static_cast<Backend&>(file_backend));
    }

    void StartBackendThread() {
        backend_thread = std::jthread([this](std::stop_token stop_token) {
            Common::SetCurrentThreadName("Logger");
            Entry entry;
            const auto write_logs = [this, &entry]() {
                ForEachBackend([&entry](Backend& backend) { backend.Write(entry); });
            };
            while (!stop_token.stop_requested()) {
                message_queue.PopWait(entry, stop_token);
                if (entry.filename != nullptr) {
                    write_logs();
                }
            }
            int max_logs_to_write = MAX_LOGS_TO_WRITE;
            while (max_logs_to_write-- && message_queue.TryPop(entry)) {
                write_logs();
            }
        });
    }

private:
    Entry CreateEntry(Class log_class, Level log_level, const char* filename, unsigned int line_nr,
                      const char* function, std::string&& message) const {
        using std::chrono::duration_cast;
        using std::chrono::microseconds;
        using std::chrono::steady_clock;

        return {
            .timestamp = duration_cast<microseconds>(steady_clock::now() - time_origin),
            .log_class = log_class,
            .log_level = log_level,
            .filename = filename,
            .line_num = line_nr,
            .function = function,
            .message = std::move(message),
        };
    }

private:
    std::chrono::steady_clock::time_point time_origin{std::chrono::steady_clock::now()};
    Filter filter;
    DebuggerBackend debugger_backend{};
    ColorConsoleBackend color_console_backend{};
    FileBackend file_backend;

    MPSCQueue<Entry> message_queue{};
    std::jthread backend_thread;
};

class Logger {
public:
    static Logger& Instance() {
        static Logger instance;
        return instance;
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    LoggerImpl& GetLoggerImpl() {
        return *logger_impl;
    }

    void Initialize() {
        if (!logger_impl) {
            auto const& log_dir = FS::GetYuzuPath(FS::YuzuPath::LogDir);

            void(FS::CreateDir(log_dir));

            auto filter = Filter{};
            filter.ParseFilterString(Settings::values.log_filter.GetValue());

            logger_impl = std::make_unique<LoggerImpl>(log_dir / Constants::LOG_FILE, filter);
        }
    }

    void Start() {
        logger_impl->StartBackendThread();
    }

    void SetGlobalFilter(const Filter& filter) {
        logger_impl->SetGlobalFilter(filter);
    }

    void SetColorConsoleBackendEnabled(bool enabled) {
        logger_impl->SetColorConsoleBackendEnabled(enabled);
    }

    void PushEntry(Class log_class, Level log_level, const char* filename, unsigned int line_num,
                   const char* function, std::string&& message) {
        logger_impl->PushEntry(log_class, log_level, filename, line_num, function, std::move(message));
    }

private:
    Logger() = default;

    std::unique_ptr<LoggerImpl> logger_impl;
};

void Initialize() {
    Logger::Instance().Initialize();
}

void Start() {
    Logger::Instance().Start();
}

void DisableLoggingInTests() {
    Logger::Instance().GetLoggerImpl().ForEachBackend([](Backend& backend) {
        backend.EnableForStacktrace();
    });
}

void SetGlobalFilter(const Filter& filter) {
    Logger::Instance().SetGlobalFilter(filter);
}

void SetColorConsoleBackendEnabled(bool enabled) {
    Logger::Instance().SetColorConsoleBackendEnabled(enabled);
}

void FmtLogMessageImpl(Class log_class, Level log_level, const char* filename,
                       unsigned int line_num, const char* function, const char* format,
                       const fmt::format_args& args) {
    Logger::Instance().PushEntry(log_class, log_level, filename, line_num, function,
                                 fmt::vformat(format, args));
}
} // namespace Common::Log

