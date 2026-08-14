#pragma once
#include <cstdarg>
#include <string>

class LogManager
{

public:

    enum class Level { Disabled = 0, Fatal = 1, Error = 2, Warning = 3, Verbose = 4 };


    static void Log(const char* format, ...);
    static void LogError(const char* format, ...);
    static void LogWarning(const char* format, ...);
    static void LogFatal(const char* format, ...);

    // Useful extras (optional, but recommended)
    static void SetLogFile(const std::string& path, bool append = true);
    static void EnableConsole(bool enabled);
    static void EnableFile(bool enabled);
	static void SetMinLevel(Level level); 

private:

    static void LogImpl(Level lvl, const char* format, va_list args);
};
