#pragma once

#include <filesystem>

class Segmentation
{
public:
    std::string _name;
    std::string _uuid;
    std::string _format;

    explicit Segmentation(std::filesystem::path path);
    Segmentation(std::filesystem::path path, std::string uuid, std::string name);

    static std::shared_ptr<Segmentation> New(std::filesystem::path path);

    static std::shared_ptr<Segmentation> New(
        std::filesystem::path path, std::string uuid, std::string name);


    std::string id() {return _uuid;}
    std::string name() {return _name;}
    std::filesystem::path path() {return _path;}

    std::filesystem::path _path;
};
