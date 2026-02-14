#include "LogManager.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace
{
    std::mutex g_logMutex;

    bool g_consoleEnabled = true;
    bool g_fileEnabled = false;
    int  g_minLevel = 0; // Debug

    std::ofstream g_file;

    const char* LevelToString(LogManager::Level lvl)
    {
        switch (lvl)
        {
        case LogManager::Level::Debug:   return "DEBUG";
        case LogManager::Level::Info:    return "INFO";
        case LogManager::Level::Warning: return "WARN";
        case LogManager::Level::Error:   return "ERROR";
        case LogManager::Level::Default: return "LOG";
        }
        return "LOG";
    }

    std::string NowTimestamp()
    {
        using namespace std::chrono;
        const auto now = system_clock::now();
        const auto t = system_clock::to_time_t(now);

        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif

        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S")
            << '.' << std::setw(3) << std::setfill('0') << ms.count();
        return oss.str();
    }

    // printf-style formatting in std::string
    std::string VFormat(const char* format, va_list args)
    {
        va_list argsCopy;
        va_copy(argsCopy, args);

        int needed = std::vsnprintf(nullptr, 0, format, argsCopy);
        va_end(argsCopy);

        if (needed <= 0) return {};

        std::string result;
        result.resize(static_cast<size_t>(needed));

        std::vsnprintf(&result[0], static_cast<size_t>(needed) + 1, format, args);
        return result;
    }

#if defined(_WIN32)
    WORD WinColorForLevel(LogManager::Level lvl)
    {
        switch (lvl)
        {
        case LogManager::Level::Error:   return FOREGROUND_RED | FOREGROUND_INTENSITY;
        case LogManager::Level::Warning: return FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
        case LogManager::Level::Info:    return FOREGROUND_GREEN | FOREGROUND_INTENSITY;
        case LogManager::Level::Debug:   return FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED; // default gray
        default:                         return FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED;
        }
    }
#else
    const char* AnsiColorForLevel(LogManager::Level lvl)
    {
        switch (lvl)
        {
        case LogManager::Level::Error:   return "\x1b[31;1m"; // bright red
        case LogManager::Level::Warning: return "\x1b[33;1m"; // bright yellow
        case LogManager::Level::Info:    return "\x1b[32;1m"; // bright green
        case LogManager::Level::Debug:   return "\x1b[0m";    // reset
        default:                         return "\x1b[0m";
        }
    }
#endif

    bool PassesFilter(LogManager::Level lvl)
    {
        // Default e Log() li facciamo passare sempre, oppure li mappiamo a Info.
        int levelInt = 1;
        switch (lvl)
        {
        case LogManager::Level::Debug:   levelInt = 0; break;
        case LogManager::Level::Info:    levelInt = 1; break;
        case LogManager::Level::Warning: levelInt = 2; break;
        case LogManager::Level::Error:   levelInt = 3; break;
        case LogManager::Level::Default: levelInt = 1; break;
        }
        return levelInt >= g_minLevel;
    }
} // namespace

// ---- API pubblica extra ----

void LogManager::SetLogFile(const std::string& path, bool append)
{
    std::lock_guard<std::mutex> lock(g_logMutex);

    if (g_file.is_open())
        g_file.close();

    std::ios::openmode mode = std::ios::out;
    mode |= append ? std::ios::app : std::ios::trunc;

    g_file.open(path, mode);
    g_fileEnabled = g_file.good();
}

void LogManager::EnableConsole(bool enabled)
{
    std::lock_guard<std::mutex> lock(g_logMutex);
    g_consoleEnabled = enabled;
}

void LogManager::EnableFile(bool enabled)
{
    std::lock_guard<std::mutex> lock(g_logMutex);
    g_fileEnabled = enabled;
}

void LogManager::SetMinLevel(int level)
{
    std::lock_guard<std::mutex> lock(g_logMutex);
    g_minLevel = level;
}

// ---- Funzioni come nel tuo header ----

void LogManager::Log(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    LogImpl(Level::Default, format, args);
    va_end(args);
}

void LogManager::LogError(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    LogImpl(Level::Error, format, args);
    va_end(args);
}

void LogManager::LogWarning(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    LogImpl(Level::Warning, format, args);
    va_end(args);
}

void LogManager::LogInfo(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    LogImpl(Level::Info, format, args);
    va_end(args);
}

void LogManager::LogDebug(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    LogImpl(Level::Debug, format, args);
    va_end(args);
}

// ---- Core ----

void LogManager::LogImpl(Level lvl, const char* format, va_list args)
{
    std::lock_guard<std::mutex> lock(g_logMutex);

    if (!PassesFilter(lvl))
        return;

    const std::string message = VFormat(format, args);
    const std::string ts = NowTimestamp();

    std::ostringstream line;
    line << "[" << ts << "]"
        << "[" << LevelToString(lvl) << "] "
        << message;

    const std::string out = line.str();

    // Console
    if (g_consoleEnabled)
    {
#if defined(_WIN32)
        HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO info{};
        GetConsoleScreenBufferInfo(h, &info);
        WORD oldAttr = info.wAttributes;

        SetConsoleTextAttribute(h, WinColorForLevel(lvl));
        std::cout << out << "\n";
        SetConsoleTextAttribute(h, oldAttr);
#else
        std::cout << AnsiColorForLevel(lvl) << out << "\x1b[0m" << "\n";
#endif
        std::cout.flush();
    }

    // File
    if (g_fileEnabled && g_file.is_open())
    {
        g_file << out << "\n";
        g_file.flush();
    }
}
