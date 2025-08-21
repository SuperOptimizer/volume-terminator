#pragma once

#include <filesystem>
#include <Metadata.hpp>

class Segmentation
{
public:
    std::string _name;
    std::string _uuid;

    explicit Segmentation(std::filesystem::path path);
    Segmentation(std::filesystem::path path, std::string uuid, std::string name);

    static std::shared_ptr<Segmentation> New(std::filesystem::path path);

    static std::shared_ptr<Segmentation> New(
        std::filesystem::path path, std::string uuid, std::string name);

    Metadata _metadata;

    Metadata metadata() {return _metadata;}

    std::string id() {return _uuid;}
    std::string name() {return _name;}
    std::filesystem::path path() {return _path;}

    std::filesystem::path _path;
};
