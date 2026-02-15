// ======================================================================== //
// IOManager - Utility class for binary file I/O operations               //
// ======================================================================== //

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

namespace osc {

    /*! Utility class for handling binary file input/output operations */
    class IOManager
    {
    public:
        /*! Result structure for I/O operations */
        struct IOResult {
            bool success;
            std::string errorMessage;
            size_t bytesWritten;

            IOResult() : success(false), bytesWritten(0) {}
            IOResult(bool s, const std::string& msg = "", size_t bytes = 0)
                : success(s), errorMessage(msg), bytesWritten(bytes) {}
        };

        /*! Save binary data to file
         *  @param filepath Path where the file should be saved
         *  @param data Pointer to the data buffer
         *  @param size Size of the data in bytes
         *  @return IOResult with success status and error message if any
         */
        static IOResult saveBinaryFile(const std::string& filepath, 
                                       const void* data, 
                                       size_t size);

        /*! Save binary data from a vector to file
         *  @param filepath Path where the file should be saved
         *  @param data Vector containing the binary data
         *  @return IOResult with success status and error message if any
         */
        template<typename T>
        static IOResult saveBinaryFile(const std::string& filepath, 
                                       const std::vector<T>& data)
        {
            return saveBinaryFile(filepath, data.data(), data.size() * sizeof(T));
        }

        /*! Load binary data from file
         *  @param filepath Path to the file to load
         *  @param data Vector to store the loaded data
         *  @return IOResult with success status and error message if any
         */
        template<typename T>
        static IOResult loadBinaryFile(const std::string& filepath, 
                                       std::vector<T>& data)
        {
            std::ifstream file(filepath, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                return IOResult(false, "Failed to open file: " + filepath);
            }

            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            if (size % sizeof(T) != 0) {
                return IOResult(false, "File size is not a multiple of element size");
            }

            size_t numElements = size / sizeof(T);
            data.resize(numElements);

            if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
                return IOResult(false, "Failed to read file data");
            }

            file.close();
            return IOResult(true, "", static_cast<size_t>(size));
        }

        /*! Check if a file exists
         *  @param filepath Path to the file to check
         *  @return true if the file exists, false otherwise
         */
        static bool fileExists(const std::string& filepath);

        /*! Get the size of a file
         *  @param filepath Path to the file
         *  @return Size of the file in bytes, or -1 if file doesn't exist
         */
        static long long getFileSize(const std::string& filepath);

        /*! Create directory if it doesn't exist
         *  @param dirpath Path to the directory
         *  @return true if directory exists or was created successfully
         */
        static bool createDirectory(const std::string& dirpath);

    private:
        IOManager() = delete;  // Static class, no instantiation
    };

} // namespace osc
