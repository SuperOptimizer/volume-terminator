#pragma once

#include <spdlog/spdlog.h>

#include <filesystem>

void AddLogFile(const std::filesystem::path& path);
void SetLogLevel(const std::string& s);
std::shared_ptr<spdlog::logger> Logger();


