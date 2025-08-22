#pragma once

#include <filesystem>

class Segmentation
{
public:
    std::string _name;
    std::string _uuid;
    std::string _format;

    Segmentation(const std::filesystem::path &path);

    Segmentation(const std::filesystem::path &path, const std::string &uuid, const std::string &name);

    static std::shared_ptr<Segmentation> New(const std::filesystem::path &path);

    static std::shared_ptr<Segmentation> New(
        const std::filesystem::path &path, const std::string &uuid, const std::string &name);


    std::string id() {return _uuid;}
    std::string name() {return _name;}
    std::filesystem::path path() {return _path;}

    std::filesystem::path _path;
};
