#include <string>
#include <utility>

#include <fmt/format.h>

#include "common/dynamic_library.h"

#ifdef _WIN32

#include <windows.h>

namespace Common {
namespace {

void* LoadLibrary(const char* filename) { return reinterpret_cast<void*>(::LoadLibraryA(filename)); }

void FreeLibrary(void* handle) { ::FreeLibrary(reinterpret_cast<HMODULE>(handle)); }

void* GetProcAddress(void* handle, const char* name) {
  return reinterpret_cast<void*>(::GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
}

}  // namespace

DynamicLibrary::DynamicLibrary() = default;

DynamicLibrary::DynamicLibrary(const char* filename) { Open(filename); }

DynamicLibrary::DynamicLibrary(DynamicLibrary&& rhs) noexcept : handle_(std::move(rhs.handle_)) {}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& rhs) noexcept {
  Close();
  handle_ = std::move(rhs.handle_);
  return *this;
}

DynamicLibrary::~DynamicLibrary() { Close(); }

std::string DynamicLibrary::GetUnprefixedFilename(const char* filename) {
  return std::string(filename) + ".dll";
}

std::string DynamicLibrary::GetVersionedFilename(const char* libname, int major, int minor) {
  if (major >= 0 && minor >= 0)
    return fmt::format("{}-{}-{}.dll", libname, major, minor);
  else if (major >= 0)
    return fmt::format("{}-{}.dll", libname, major);
  else
    return fmt::format("{}.dll", libname);
}

bool DynamicLibrary::Open(const char* filename) {
  handle_.reset(LoadLibrary(filename));
  return IsOpen();
}

void DynamicLibrary::Close() {
  if (!IsOpen()) return;
  FreeLibrary(handle_.release());
}

void* DynamicLibrary::GetSymbolAddress(const char* name) const {
  return IsOpen() ? GetProcAddress(handle_.get(), name) : nullptr;
}

}  // namespace Common

#else

#include <dlfcn.h>

namespace Common {
namespace {

void* dlopen(const char* filename, int flag) { return ::dlopen(filename, flag); }

void dlclose(void* handle) { ::dlclose(handle); }

void* dlsym(void* handle, const char* name) { return ::dlsym(handle, name); }

}  // namespace

DynamicLibrary::DynamicLibrary() = default;

DynamicLibrary::DynamicLibrary(const char* filename) { Open(filename); }

DynamicLibrary::DynamicLibrary(DynamicLibrary&& rhs) noexcept : handle_(std::move(rhs.handle_)) {}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& rhs) noexcept {
  Close();
  handle_ = std::move(rhs.handle_);
  return *this;
}

DynamicLibrary::~DynamicLibrary() { Close(); }

std::string DynamicLibrary::GetUnprefixedFilename(const char* filename) {
#if defined(__APPLE__)
  return std::string(filename) + ".dylib";
#else
  return std::string(filename) + ".so";
#endif
}

std::string DynamicLibrary::GetVersionedFilename(const char* libname, int major, int minor) {
#if defined(__APPLE__)
  const char* prefix = std::strncmp(libname, "lib", 3) ? "lib" : "";
  if (major >= 0 && minor >= 0)
    return fmt::format("{}{}.{}.{}.dylib", prefix, libname, major, minor);
  else if (major >= 0)
    return fmt::format("{}{}.{}.dylib", prefix, libname, major);
  else
    return fmt::format("{}{}.dylib", prefix, libname);
#else
  const char* prefix = std::strncmp(libname, "lib", 3) ? "lib" : "";
  if (major >= 0 && minor >= 0)
    return fmt::format("{}{}.so.{}.{}", prefix, libname, major, minor);
  else if (major >= 0)
    return fmt::format("{}{}.so.{}", prefix, libname, major);
  else
    return fmt::format("{}{}.so", prefix, libname);
#endif
}

bool DynamicLibrary::Open(const char* filename) {
  handle_.reset(dlopen(filename, RTLD_NOW));
  return IsOpen();
}

void DynamicLibrary::Close() {
  if (!IsOpen()) return;
  dlclose(handle_.release());
}

void* DynamicLibrary::GetSymbolAddress(const char* name) const {
  return IsOpen() ? dlsym(handle_.get(), name) : nullptr;
}

}  // namespace Common

#endif

