#include "FileSystemUtils.h"
#include <filesystem>
#include "RelaxationLabelingSolverBaseConstants.h"
#include "Utilities.h"

/************************************************************************/
void FileSystemUtils::DeleteDirectoryRecursively(const std::string& in_path)
/************************************************************************/
{
    std::filesystem::remove_all(in_path);
}

/************************************************************************/
void FileSystemUtils::DeleteFile(const std::string& in_path)
/************************************************************************/
{
    std::filesystem::remove(in_path);
}

/************************************************************************/
void FileSystemUtils::Create_Directory(const std::string& in_path)
/************************************************************************/
{
    const bool bWasDirCreated = std::filesystem::create_directory(in_path);
    Utilities::LogAndAbortIf(bWasDirCreated == false, "FileSystemUtils::Create_Directory() failed");
}

/************************************************************************/
bool FileSystemUtils::DoesFileExist(const std::string& in_path)
/************************************************************************/
{
    return std::filesystem::exists(in_path);
}

/************************************************************************/
std::vector<std::string> FileSystemUtils::GetAllFilesInDirectory(const std::string& in_path)
/************************************************************************/
{
    //1. Get items as paths vector
    std::vector<std::filesystem::path> pathsVec;
    for (const std::filesystem::directory_entry& p : std::filesystem::directory_iterator(in_path))
        pathsVec.push_back(p.path());

    //2. Sort items by numbers or by strings
    std::sort(pathsVec.begin(), pathsVec.end(), [](const std::filesystem::path& in_rhs, const std::filesystem::path& in_lhs) -> bool
    {
        const std::string rhsFileNameString = in_rhs.filename().string();
        const std::string lhsFileNameString = in_lhs.filename().string();
        
        if (std::isdigit(rhsFileNameString[0]) && std::isdigit(lhsFileNameString[0]))
            return std::stoi(rhsFileNameString) < std::stoi(lhsFileNameString);
        else
            return rhsFileNameString < lhsFileNameString;
    });

    //3. Convert to string vector
    std::vector<std::string> retVal;
    std::transform(pathsVec.begin(), pathsVec.end(), std::back_inserter(retVal),
        [](const std::filesystem::path& in_path) -> std::string {return in_path.string();});

    return retVal;
}

/************************************************************************/
void FileSystemUtils::WriteToFile(const std::string& in_filePath, const std::stringstream& in_strStream)
/************************************************************************/
{
    std::ofstream file(in_filePath);
    file << in_strStream.str() << std::endl;
    file.flush();
    file.close();
}

/************************************************************************/
void FileSystemUtils::AppendToFile(const std::string& in_filePath, const std::stringstream& in_strStream)
/************************************************************************/
{
    std::ofstream file;
    file.open(in_filePath, std::ios_base::app);
    file << in_strStream.str() << std::endl;
    file.flush();
    file.close();
}

/************************************************************************/
void FileSystemUtils::SetCurrentPath(const std::string& in_path)
/************************************************************************/
{
    std::filesystem::current_path(in_path);
}

/************************************************************************/
std::string FileSystemUtils::GetFileNameFromPath(const std::string& in_path)
/************************************************************************/
{
    return std::filesystem::path(in_path).filename().string();
}

/************************************************************************/
std::string FileSystemUtils::GetPathInFolder(const std::string& in_folder, const std::string& in_path)
/************************************************************************/
{
    return in_folder + in_path;
}

/************************************************************************/
std::string FileSystemUtils::GetRunFolderName(const int32_t in_runNumber)
/************************************************************************/
{
    std::stringstream name;

    if (-1 != in_runNumber)
        name << "run " << in_runNumber << "\\";

    return name.str();
}

/************************************************************************/
std::string FileSystemUtils::GetPathInRunFolder(const int32_t in_runNumber, const std::string& in_path)
/************************************************************************/
{
    std::stringstream fileName;
    fileName << GetRunOutputFolderPath(in_runNumber) << in_path;
    return fileName.str();
}

/************************************************************************/
std::string FileSystemUtils::GetRunOutputFolderPath(const int32_t in_runNumber)
/************************************************************************/
{
    std::stringstream fileName;
    fileName << OUTPUT_DIRECTORY_PATH.string() << GetRunFolderName(in_runNumber);
    return fileName.str();
}