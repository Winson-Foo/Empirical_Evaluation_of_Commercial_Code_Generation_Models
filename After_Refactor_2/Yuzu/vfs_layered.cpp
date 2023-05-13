#include <algorithm>
#include <set>
#include <utility>
#include <vector>
#include "core/file_sys/vfs_layered.h"

namespace FileSys {

LayeredVfsDirectory::LayeredVfsDirectory(const std::vector<VirtualDir>& dirs, const std::string& name)
    : dirs(dirs), name(name) {}

VirtualDir LayeredVfsDirectory::MakeLayeredDirectory(const std::vector<VirtualDir>& dirs, const std::string& name) {
    if (dirs.empty())
        return nullptr;
    return dirs.size() == 1 ? dirs[0] : VirtualDir(new LayeredVfsDirectory(dirs, name));
}

VirtualFile LayeredVfsDirectory::GetFileRelative(const std::string_view& path) const {
    for (const auto& layer : dirs) {
        const auto file = layer->GetFileRelative(path);
        if (file != nullptr)
            return file;
    }
    return nullptr;
}

VirtualDir LayeredVfsDirectory::GetDirectoryRelative(const std::string_view& path) const {
    std::vector<VirtualDir> dirs_out;
    for (const auto& layer : dirs) {
        const auto sub_dir = layer->GetDirectoryRelative(path);
        if (sub_dir != nullptr) {
            dirs_out.emplace_back(sub_dir);
        }
    }
    return MakeLayeredDirectory(dirs_out);
}

VirtualFile LayeredVfsDirectory::GetFile(const std::string_view& file_name) const {
    return GetFileRelative(file_name);
}

VirtualDir LayeredVfsDirectory::GetSubdirectory(const std::string_view& subdir_name) const {
    return GetDirectoryRelative(subdir_name);
}

std::string LayeredVfsDirectory::GetFullPath() const {
    return dirs[0]->GetFullPath();
}

std::vector<VirtualFile> LayeredVfsDirectory::GetFiles() const {
    std::vector<VirtualFile> files_out;
    std::set<std::string> files_out_names;
    for (const auto& layer : dirs) {
        const auto files = layer->GetFiles();
        for (const auto& file : files) {
            const auto file_name = file->GetName();
            if (!files_out_names.contains(file_name)) {
                files_out_names.emplace(file_name);
                files_out.emplace_back(file);
            }
        }
    }
    return files_out;
}

std::vector<VirtualDir> LayeredVfsDirectory::GetSubdirectories() const {
    std::vector<std::string> subdirs_names;
    for (const auto& layer : dirs) {
        const auto subdirs = layer->GetSubdirectories();
        for (const auto& subdir : subdirs) {
            if (std::find(subdirs_names.begin(), subdirs_names.end(), subdir->GetName()) == subdirs_names.end()) {
                subdirs_names.emplace_back(subdir->GetName());
            }
        }
    }
    std::vector<VirtualDir> subdirs_out;
    subdirs_out.reserve(subdirs_names.size());
    for (const auto& subdir : subdirs_names) {
        subdirs_out.emplace_back(GetSubdirectory(subdir));
    }
    return subdirs_out;
}

bool LayeredVfsDirectory::IsWritable() const {
    return false;
}

bool LayeredVfsDirectory::IsReadable() const {
    return true;
}

std::string LayeredVfsDirectory::GetName() const {
    return name.empty() ? dirs[0]->GetName() : name;
}

VirtualDir LayeredVfsDirectory::GetParentDirectory() const {
    return dirs[0]->GetParentDirectory();
}

VirtualDir LayeredVfsDirectory::CreateSubdirectory(const std::string_view& subdir_name) {
    return nullptr;
}

VirtualFile LayeredVfsDirectory::CreateFile(const std::string_view& file_name) {
    return nullptr;
}

bool LayeredVfsDirectory::DeleteSubdirectory(const std::string_view& subdir_name) {
    return false;
}

bool LayeredVfsDirectory::DeleteFile(const std::string_view& file_name) {
    return false;
}

bool LayeredVfsDirectory::Rename(const std::string_view& new_name) {
    name = new_name;
    return true;
}

}  // namespace FileSys