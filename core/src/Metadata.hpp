#pragma once

/** @file */

#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include <filesystem>


class Metadata
{

public:
    Metadata() = default;
    explicit Metadata(std::filesystem::path fileLocation);
    std::filesystem::path path() const { return _path; }
void setPath(const std::filesystem::path& path) { _path = path; }
    void save() { save(_path); }
    void save(const std::filesystem::path& path);
    bool hasKey(const std::string& key) const { return _json.count(key) > 0; }
    template <typename T>
    T get(const std::string& key) const
    {
        if (_json.find(key) == _json.end()) {
            auto msg = "could not find key '" + key + "' in metadata";
            throw std::runtime_error(msg);
        }
        return _json[key].get<T>();
    }

    template <typename T>
    void set(const std::string& key, T value)
    {
        _json[key] = value;
    }

    void printString() const { std::cout << _json << std::endl; }
    void printObject() const { std::cout << _json.dump(4) << std::endl; }
protected:
    nlohmann::json _json;
    std::filesystem::path _path;
};

