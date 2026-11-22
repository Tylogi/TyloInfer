#pragma once
#include <filesystem>

namespace mfq {
// Native file operations must support the full path after cache hash suffixes
// are appended, including applications without a Windows long-path manifest.
inline std::filesystem::path native_file_path(const std::filesystem::path& path) {
#ifdef _WIN32
    auto absolute=std::filesystem::absolute(path).lexically_normal();
    const auto& text=absolute.native();
    if(text.rfind(LR"(\\?\)",0)==0 || text.rfind(LR"(\\.\)",0)==0)return absolute;
    if(text.rfind(LR"(\\)",0)==0)return std::filesystem::path(LR"(\\?\UNC\)"+text.substr(2));
    return std::filesystem::path(LR"(\\?\)"+text);
#else
    return path;
#endif
}
} // namespace mfq
