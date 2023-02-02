#pragma once
#include <string>
#include <vector>

namespace FileSystemUtils
{
    void DeleteDirectoryRecursively(const std::string& in_path);
    void DeleteFile(const std::string& in_path);
    //This function is named 'Create_Directory' and not 'CreateDirectory' due to a weird issue resulting from add Eigen tensor include
    void Create_Directory(const std::string& in_path);
    bool DoesFileExist(const std::string& in_path);
    std::vector<std::string> GetAllFilesInDirectory(const std::string& in_path);
    void WriteToFile(const std::string& in_filePath, const std::stringstream& in_strStream);
    void AppendToFile(const std::string& in_filePath, const std::stringstream& in_strStream);
    void SetCurrentPath(const std::string& in_path);
    std::string GetFileNameFromPath(const std::string& in_path);
    std::string GetPathInFolder(const std::string& in_folder, const std::string& in_path);

    std::string GetRunFolderName(const int32_t in_runNumber);
    std::string GetPathInRunFolder(const int32_t in_runNumber, const std::string& in_path);
    
    std::string GetRunOutputFolderPath(const int32_t in_runNumber);
}