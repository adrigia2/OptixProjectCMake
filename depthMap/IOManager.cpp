// ======================================================================== //
// IOManager - Utility class for binary file I/O operations               //
// ======================================================================== //

#include "IOManager.h"
#include "LogManager.h"
#include <fstream>
#include <sys/stat.h>

#ifdef _WIN32
    #include <direct.h>
    #define MKDIR(dir) _mkdir(dir)
#else
    #include <sys/types.h>
    #define MKDIR(dir) mkdir(dir, 0755)
#endif

namespace osc {

    IOManager::IOResult IOManager::saveBinaryFile(const std::string& filepath, 
                                                   const void* data, 
                                                   size_t size)
    {
        if (data == nullptr) {
            LogManager::LogError("IOManager: Cannot save null data pointer");
            return IOResult(false, "Data pointer is null");
        }

        if (size == 0) {
            LogManager::LogWarning("IOManager: Attempting to save 0 bytes to %s", filepath.c_str());
            return IOResult(false, "Data size is zero");
        }

        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            LogManager::LogError("IOManager: Failed to open file for writing: %s", filepath.c_str());
            return IOResult(false, "Failed to open file: " + filepath);
        }

        file.write(reinterpret_cast<const char*>(data), size);
        
        if (!file.good()) {
            LogManager::LogError("IOManager: Error writing to file: %s", filepath.c_str());
            file.close();
            return IOResult(false, "Error writing to file");
        }

        file.close();

        LogManager::LogDebug("IOManager: Successfully saved %zu bytes to %s", size, filepath.c_str());
        return IOResult(true, "", size);
    }

    bool IOManager::fileExists(const std::string& filepath)
    {
        std::ifstream file(filepath);
        return file.good();
    }

    long long IOManager::getFileSize(const std::string& filepath)
    {
        struct stat stat_buf;
        int rc = stat(filepath.c_str(), &stat_buf);
        return rc == 0 ? stat_buf.st_size : -1;
    }

    bool IOManager::createDirectory(const std::string& dirpath)
    {
        struct stat info;

        if (stat(dirpath.c_str(), &info) == 0) {
            if (info.st_mode & S_IFDIR) {
                return true;  // Directory already exists
            }
            LogManager::LogError("IOManager: Path exists but is not a directory: %s", dirpath.c_str());
            return false;
        }

        // Try to create the directory
        if (MKDIR(dirpath.c_str()) == 0) {
            LogManager::LogInfo("IOManager: Created directory: %s", dirpath.c_str());
            return true;
        }

        LogManager::LogError("IOManager: Failed to create directory: %s", dirpath.c_str());
        return false;
    }

} // namespace osc
