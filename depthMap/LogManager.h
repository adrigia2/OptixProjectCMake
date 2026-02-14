#pragma once
#include <cstdarg>
#include <string>

class LogManager
{

public:

    enum class Level { Debug = 0, Info = 1, Warning = 2, Error = 3, Default = 4 };


    static void Log(const char* format, ...);
    static void LogError(const char* format, ...);
    static void LogWarning(const char* format, ...);
    static void LogInfo(const char* format, ...);
    static void LogDebug(const char* format, ...);

    // Extra utili (opzionali ma consigliati)
    static void SetLogFile(const std::string& path, bool append = true);
    static void EnableConsole(bool enabled);
    static void EnableFile(bool enabled);
    static void SetMinLevel(int level); // 0=Debug,1=Info,2=Warning,3=Error

private:

    static void LogImpl(Level lvl, const char* format, va_list args);
};
