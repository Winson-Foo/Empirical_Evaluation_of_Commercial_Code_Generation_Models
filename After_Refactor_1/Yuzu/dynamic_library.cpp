namespace Common {

using Handle = void*;

std::string GetUnprefixedFilename(const std::string& filename) {
#if defined(_WIN32)
    return filename + ".dll";
#elif defined(__APPLE__)
    return filename + ".dylib";
#else
    return filename + ".so";
#endif
}

std::string GetVersionedFilename(const std::string& libname, int major, int minor) {
#if defined(_WIN32)
    if (major >= 0 && minor >= 0)
        return fmt::format("{}-{}-{}.dll", libname, major, minor);
    else if (major >= 0)
        return fmt::format("{}-{}.dll", libname, major);
    else
        return fmt::format("{}.dll", libname);
#elif defined(__APPLE__)
    const auto prefix = std::strncmp(libname.c_str(), "lib", 3) ? "lib" : "";
    if (major >= 0 && minor >= 0)
        return fmt::format("{}{}.{}.{}.dylib", prefix, libname, major, minor);
    else if (major >= 0)
        return fmt::format("{}{}.{}.dylib", prefix, libname, major);
    else
        return fmt::format("{}{}.dylib", prefix, libname);
#else
    const auto prefix = std::strncmp(libname.c_str(), "lib", 3) ? "lib" : "";
    if (major >= 0 && minor >= 0)
        return fmt::format("{}{}.so.{}.{}", prefix, libname, major, minor);
    else if (major >= 0)
        return fmt::format("{}{}.so.{}", prefix, libname, major);
    else
        return fmt::format("{}{}.so", prefix, libname);
#endif
}

class DynamicLibrary {
public:
    DynamicLibrary() = default;

    explicit DynamicLibrary(const std::string& filename) {
        Open(filename);
    }

    DynamicLibrary(const DynamicLibrary& other) = delete;
    DynamicLibrary& operator=(const DynamicLibrary& other) = delete;

    DynamicLibrary(DynamicLibrary&& other) noexcept : handle_{std::exchange(other.handle_, nullptr)} {}
    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept {
        Close();
        handle_ = std::exchange(other.handle_, nullptr);
        return *this;
    }

    ~DynamicLibrary() noexcept {
        Close();
    }

    bool IsOpen() const noexcept {
        return handle_ != nullptr;
    }

    bool Open(const std::string& filename) noexcept {
        const auto unprefixed = GetUnprefixedFilename(filename);
#ifdef _WIN32
        handle_ = reinterpret_cast<Handle>(LoadLibraryA(unprefixed.c_str()));
#else
        handle_ = dlopen(unprefixed.c_str(), RTLD_NOW);
#endif
        return IsOpen();
    }

    void Close() noexcept {
        if (IsOpen()) {
#ifdef _WIN32
            FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
            dlclose(handle_);
#endif
            handle_ = nullptr;
        }
    }

    void* GetSymbolAddress(const std::string& name) const noexcept {
#ifdef _WIN32
        return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle_), name.c_str()));
#else
        return dlsym(handle_, name.c_str());
#endif
    }

private:
    Handle handle_{nullptr};
};

} // namespace Common