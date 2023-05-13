// dynamic_library.h

#ifndef DYNAMIC_LIBRARY_H
#define DYNAMIC_LIBRARY_H

#include <string>

namespace Common {

class DynamicLibrary {
public:
    DynamicLibrary();
    explicit DynamicLibrary(const char* filename);
    DynamicLibrary(DynamicLibrary&& rhs) noexcept;
    DynamicLibrary& operator=(DynamicLibrary&& rhs) noexcept;
    ~DynamicLibrary();

    bool Open(const char* filename);
    void Close();
    bool IsOpen() const { return handle != nullptr; }
    void* GetSymbolAddress(const char* name) const;

    static std::string GetUnprefixedFilename(const char* filename);
    static std::string GetVersionedFilename(const char* libname, int major = -1, int minor = -1);

private:
    void* handle = nullptr;
};

} // namespace Common

#endif // DYNAMIC_LIBRARY_H


// dynamic_library.cpp

#include "dynamic_library.h"

#include <utility>

#ifdef _WIN32
#include <windows.h>
#include "windows_dynamic_library.h"
#else
#include <dlfcn.h>
#include "unix_dynamic_library.h"
#endif

namespace Common {

DynamicLibrary::DynamicLibrary() = default;

DynamicLibrary::DynamicLibrary(const char* filename) {
    Open(filename);
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& rhs) noexcept
    : handle(std::exchange(rhs.handle, nullptr)) {}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& rhs) noexcept {
    Close();
    handle = std::exchange(rhs.handle, nullptr);
    return *this;
}

DynamicLibrary::~DynamicLibrary() {
    Close();
}

std::string DynamicLibrary::GetUnprefixedFilename(const char* filename) {
    return std::string(filename) + UNPREFIXED_EXTENSION;
}

std::string DynamicLibrary::GetVersionedFilename(const char* libname, int major, int minor) {
    std::string prefix = "";
    if (std::strncmp(libname, LIB_PREFIX, LIB_PREFIX_LENGTH) != 0) {
        prefix = LIB_PREFIX;
    }

    std::string version = "";
    if (major >= 0 && minor >= 0) {
        version = fmt::format(".{}.{}", major, minor);
    } else if (major >= 0) {
        version = fmt::format(".{}", major);
    }

    return prefix + std::string(libname) + version + VERSIONED_EXTENSION;
}

#ifdef _WIN32
#include "windows_dynamic_library.cpp"
#else
#include "unix_dynamic_library.cpp"
#endif

} // namespace Common


// windows_dynamic_library.h

#ifndef WINDOWS_DYNAMIC_LIBRARY_H
#define WINDOWS_DYNAMIC_LIBRARY_H

#define UNPREFIXED_EXTENSION ".dll"
#define LIB_PREFIX ""
#define LIB_PREFIX_LENGTH 0
#define VERSIONED_EXTENSION ""

#endif // WINDOWS_DYNAMIC_LIBRARY_H


// unix_dynamic_library.h

#ifndef UNIX_DYNAMIC_LIBRARY_H
#define UNIX_DYNAMIC_LIBRARY_H

#define UNPREFIXED_EXTENSION ".so"
#define LIB_PREFIX "lib"
#define LIB_PREFIX_LENGTH 3
#define VERSIONED_EXTENSION ".so"

#endif // UNIX_DYNAMIC_LIBRARY_H


// windows_dynamic_library.cpp

#include "windows_dynamic_library.h"

namespace Common {

bool DynamicLibrary::Open(const char* filename) {
    handle = reinterpret_cast<void*>(LoadLibraryA(filename));
    return handle != nullptr;
}

void DynamicLibrary::Close() {
    if (!IsOpen())
        return;

    FreeLibrary(reinterpret_cast<HMODULE>(handle));
    handle = nullptr;
}

void* DynamicLibrary::GetSymbolAddress(const char* name) const {
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
}

} // namespace Common


// unix_dynamic_library.cpp

#include "unix_dynamic_library.h"

namespace Common {

bool DynamicLibrary::Open(const char* filename) {
    handle = dlopen(filename, RTLD_NOW);
    return handle != nullptr;
}

void DynamicLibrary::Close() {
    if (!IsOpen())
        return;

    dlclose(handle);
    handle = nullptr;
}

void* DynamicLibrary::GetSymbolAddress(const char* name) const {
    return dlsym(handle, name);
}

} // namespace Common

