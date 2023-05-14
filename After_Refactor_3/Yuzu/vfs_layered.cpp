// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright 2018 yuzu Emulator Project

#include <algorithm>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace FileSys {

// Definition of a layered virtual file system directory
class LayeredVfsDirectory final : public VirtualDir {
public:
    // Constructor
    explicit LayeredVfsDirectory(std::vector<VirtualDir> dirs, std::string name = "");

    // Destructor
    ~LayeredVfsDirectory() override = default;

    // Returns a layered virtual directory from the given list of directories
    static VirtualDir MakeLayeredDirectory(std::vector<VirtualDir> dirs, std::string name = "");

    // Returns the virtual file with the given relative path
    VirtualFile GetFileRelative(std::string_view path) const override;

    // Returns the virtual directory with the given relative path
    VirtualDir GetDirectoryRelative(std::string_view path) const override;

    // Returns the virtual file with the given name
    VirtualFile GetFile(std::string_view name) const override;

    // Returns the virtual subdirectory with the given name
    VirtualDir GetSubdirectory(std::string_view name) const override;

    // Returns the full path of this directory
    std::string GetFullPath() const override;

    // Returns all files in this directory and its subdirectories
    std::vector<VirtualFile> GetFiles() const override;

    // Returns all subdirectories in this directory
    std::vector<VirtualDir> GetSubdirectories() const override;

    // Returns true if this directory is writable, false otherwise
    bool IsWritable() const override;

    // Returns true if this directory is readable, false otherwise
    bool IsReadable() const override;

    // Returns the name of this directory
    std::string GetName() const override;

    // Returns the parent directory of this directory
    VirtualDir GetParentDirectory() const override;

    // Creates a subdirectory with the given name
    VirtualDir CreateSubdirectory(std::string_view name) override;

    // Creates a file with the given name
    VirtualFile CreateFile(std::string_view name) override;

    // Deletes the subdirectory with the given name
    bool DeleteSubdirectory(std::string_view name) override;

    // Deletes the file with the given name
    bool DeleteFile(std::string_view name) override;

    // Renames this directory to the given name
    bool Rename(std::string_view name) override;

private:
    std::vector<VirtualDir> dirs;     // List of directories in this layered directory
    std::string name;                 // Name of this directory
};

LayeredVfsDirectory::LayeredVfsDirectory(std::vector<VirtualDir> dirs_, std::string name_)
    : dirs(std::move(dirs_)), name(std::move(name_)) {}

VirtualDir LayeredVfsDirectory::MakeLayeredDirectory(std::vector<VirtualDir> dirs, std::string name) {
    if (dirs.empty())
        return nullptr;
    if (dirs.size() == 1)
        return dirs[0];
    return VirtualDir(std::make_shared<LayeredVfsDirectory>(std::move(dirs), std::move(name)));
}

VirtualFile LayeredVfsDirectory::GetFileRelative(std::string_view path) const {
    for (const auto& layer : dirs) {
        if (auto file = layer->GetFileRelative(path))
            return file;
    }
    return nullptr;
}

VirtualDir LayeredVfsDirectory::GetDirectoryRelative(std::string_view path) const {
    std::vector<VirtualDir> out;
    for (const auto& layer : dirs) {
        if (auto dir = layer->GetDirectoryRelative(path))
            out.push_back(std::move(dir));
    }
    return MakeLayeredDirectory(std::move(out));
}

VirtualFile LayeredVfsDirectory::GetFile(std::string_view name) const {
    return GetFileRelative(name);
}

VirtualDir LayeredVfsDirectory::GetSubdirectory(std::string_view name) const {
    return GetDirectoryRelative(name);
}

std::string LayeredVfsDirectory::GetFullPath() const {
    return dirs[0]->GetFullPath();
}

std::vector<VirtualFile> LayeredVfsDirectory::GetFiles() const {
    std::vector<VirtualFile> out;
    std::set<std::string> out_names;
    for (const auto& layer : dirs) {
        for (const auto& file : layer->GetFiles()) {
            if (out_names.emplace(file->GetName()).second)
                out.push_back(file);
        }
    }
    return out;
}

std::vector<VirtualDir> LayeredVfsDirectory::GetSubdirectories() const {
    std::vector<std::string> names;
    for (const auto& layer : dirs) {
        for (const auto& sd : layer->GetSubdirectories()) {
            const auto name = sd->GetName();
            if (std::find(names.cbegin(), names.cend(), name) == names.cend())
                names.push_back(name);
        }
    }
    std::vector<VirtualDir> out;
    out.reserve(names.size());
    for (const auto& name : names)
        out.emplace_back(GetSubdirectory(name));
    return out;
}

bool LayeredVfsDirectory::IsWritable() const { return false; }

bool LayeredVfsDirectory::IsReadable() const { return true; }

std::string LayeredVfsDirectory::GetName() const { return !name.empty() ? name : dirs[0]->GetName(); }

VirtualDir LayeredVfsDirectory::GetParentDirectory() const { return dirs[0]->GetParentDirectory(); }

VirtualDir LayeredVfsDirectory::CreateSubdirectory(std::string_view name) { return nullptr; }

VirtualFile LayeredVfsDirectory::CreateFile(std::string_view name) { return nullptr; }

bool LayeredVfsDirectory::DeleteSubdirectory(std::string_view name) { return false; }

bool LayeredVfsDirectory::DeleteFile(std::string_view name) { return false; }

bool LayeredVfsDirectory::Rename(std::string_view name) {
    this->name = std::string(name);
    return true;
}

}  // namespace FileSys

