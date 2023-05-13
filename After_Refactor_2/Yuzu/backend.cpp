#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <thread>

#include <fmt/format.h>

#ifdef _WIN32
#include <windows.h> // For OutputDebugStringW
#endif

#include "common/fs/fs.h"
#include "common/fs/fs_paths.h"
#include "common/literals.h"
#include "common/settings.h"
#include "common/thread.h"
#include "common/logging/backend.h"
#include "common/logging/log.h"
#include "common/logging/log_entry.h"
#include "common/logging/text_formatter.h"
#include "common/bounded_threadsafe_queue.h"

namespace fs = std::filesystem;

namespace Common::Log {

namespace {

/**
 * Interface for logging backends.
 */
class Backend {
public:
    virtual ~Backend() = default;

    virtual void Write(const Entry& entry) = 0;

    virtual void EnableForStacktrace() = 0;

    virtual void Flush() = 0;
};

/**
 * Backend that writes to stderr and with color
 */
class ColorConsoleBackend final : public Backend {
public:
    explicit ColorConsoleBackend() = default;

    ~ColorConsoleBackend() override = default;

    void Write(const Entry& entry) override {
        if (enabled_.load(std::memory_order_relaxed)) {
            PrintColoredMessage(entry);
        }
    }

    void Flush() override {
        // stderr shouldn't be buffered
    }

    void EnableForStacktrace() override {
        enabled_ = true;
    }

    void SetEnabled(bool enabled) {
        enabled_ = enabled;
    }

private:
    std::atomic_bool enabled_{ false };
};

/**
 * Backend that writes to a file passed into the constructor
 */
class FileBackend final : public Backend {
public:
    explicit FileBackend(const fs::path& filename) {
        auto old_filename = filename;
        old_filename += ".old.txt";

        std::error_code ec;
        fs::remove(old_filename, ec);
        fs::rename(filename, old_filename, ec);

        file_ = std::make_unique<FS::IOFile>(filename, FS::FileAccessMode::Write, FS::FileType::TextFile);
    }

    ~FileBackend() override = default;

    void Write(const Entry& entry) override {
        if (!enabled_) {
            return;
        }

        bytes_written_ += file_->WriteString(FormatLogMessage(entry).append(1, '\n'));

        using namespace Common::Literals;
        const auto write_limit = Settings::values.extended_logging ? 1_GiB : 100_MiB;
        const bool write_limit_exceeded = bytes_written_ > write_limit;
        if (entry.log_level >= Level::Error || write_limit_exceeded) {
            if (write_limit_exceeded) {
                enabled_ = false;
            }
            file_->Flush();
        }
    }

    void Flush() override {
        file_->Flush();
    }

    void EnableForStacktrace() override {
        enabled_ = true;
        bytes_written_ = 0;
    }

private:
    std::unique_ptr<FS::IOFile> file_;
    bool enabled_ = true;
    std::size_t bytes_written_ = 0;
};

/**
 * Backend that writes to Visual Studio's output window
 */
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

class Impl {
public:
    static Impl& Instance() {
        if (!instance_) {
            throw std::runtime_error("Using Logging instance before its initialization");
        }
        return *instance_;
    }

    static void Initialize() {
        if (instance_) {
            LOG_WARNING(Log, "Reinitializing logging backend");
            return;
        }

        fs::create_directories(fs::path(GetYuzuPath(YuzuPath::LogDir)));
        instance_ = std::unique_ptr<Impl>(new Impl(fs::path(GetYuzuPath(YuzuPath::LogDir)) / LOG_FILE));
        initialization_in_progress_suppress_logging_ = false;
    }

    static void Start() {
        instance_->StartBackendThread();
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    void SetGlobalFilter(const Filter& filter) {
        filter_ = filter;
    }

    void SetColorConsoleBackendEnabled(bool enabled) {
        color_console_backend_.SetEnabled(enabled);
    }

    void PushEntry(Class log_class, Level log_level, const char* filename, unsigned int line_num,
                   const char* function, std::string&& message) {
        if (!filter_.CheckMessage(log_class, log_level)) {
            return;
        }
        message_queue_.EmplaceWait(
            CreateEntry(log_class, log_level, filename, line_num, function, std::move(message)));
    }

private:
    Impl(const fs::path& file_backend_filename)
        : file_backend_(file_backend_filename) {}

    ~Impl() = default;

    Entry CreateEntry(Class log_class, Level log_level, const char* filename, unsigned int line_nr,
                      const char* function, std::string&& message) const {
        using std::chrono::duration_cast;
        using std::chrono::microseconds;
        using std::chrono::steady_clock;

        return {
            .timestamp = duration_cast<microseconds>(steady_clock::now() - time_origin_),
            .log_class = log_class,
            .log_level = log_level,
            .filename = filename,
            .line_num = line_nr,
            .function = function,
            .message = std::move(message),
        };
    }

    void WriteLogs(const Entry& entry) {
        ForEachBackend([&entry](Backend& backend) { backend.Write(entry); });
    }

    void ForEachBackend(auto lambda) {
        lambda(static_cast<Backend&>(debugger_backend_));
        lambda(static_cast<Backend&>(color_console_backend_));
        lambda(static_cast<Backend&>(file_backend_));
    }

    void StartBackendThread() {
        backend_thread_ = std::jthread([this](std::stop_token stop_token) {
            Common::SetCurrentThreadName("Logger");
            Entry entry;
            while (!stop_token.stop_requested()) {
                message_queue_.PopWait(entry, stop_token);
                if (entry.filename != nullptr) {
                    WriteLogs(entry);
                }
            }
            int max_logs_to_write = filter_.IsDebug() ? INT_MAX : 100;
            while (max_logs_to_write-- && message_queue_.TryPop(entry)) {
                WriteLogs(entry);
            }
        });
    }

    static inline std::unique_ptr<Impl> instance_;
    static inline bool initialization_in_progress_suppress_logging_ = true;

    Filter filter_;
    DebuggerBackend debugger_backend_{};
    ColorConsoleBackend color_console_backend_{};
    FileBackend file_backend_;

    std::chrono::steady_clock::time_point time_origin_{ std::chrono::steady_clock::now() };
    MPSCQueue<Entry> message_queue_{};
    std::jthread backend_thread_;
};

} // namespace

void Initialize() {
    Impl::Initialize();
}

void Start() {
    Impl::Start();
}

void DisableLoggingInTests() {
    initialization_in_progress_suppress_logging_ = true;
}

void SetGlobalFilter(const Filter& filter) {
    Impl::Instance().SetGlobalFilter(filter);
}

void SetColorConsoleBackendEnabled(bool enabled) {
    Impl::Instance().SetColorConsoleBackendEnabled(enabled);
}

void FmtLogMessageImpl(Class log_class, Level log_level, const char* filename,
                       unsigned int line_num, const char* function, const char* format,
                       const fmt::format_args& args) {
    if (!initialization_in_progress_suppress_logging_) {
        Impl::Instance().PushEntry(log_class, log_level, filename, line_num, function,
                                   fmt::vformat(format, args));
    }
}

} // namespace Common::Log