#pragma once

/** @file */

#include <cstddef>
#include <cstdint>
#include <mutex>

#include <filesystem>
#include <opencv2/core/matx.hpp>
#include <z5/dataset.hxx>
#include <z5/filesystem/handle.hxx>

#include "Metadata.hpp"

#include "z5/types/types.hxx"


class Volume
{
public:

    Volume() = delete;

    explicit Volume(std::filesystem::path path);
    Volume(std::filesystem::path path, std::string uuid, std::string name);
    static  std::shared_ptr<Volume> New(std::filesystem::path path);
    static  std::shared_ptr<Volume> New(
        std::filesystem::path path, std::string uuid, std::string name);

    bool isZarr{false};

    int sliceWidth() const;
    int sliceHeight() const;
    int numSlices() const;
    double voxelSize() const;
    double min() const;
    double max() const;

    bool isInBounds(double x, double y, double z) const;
    bool isInBounds(const cv::Vec3d& v) const;

    z5::Dataset *zarrDataset(int level = 0);
    size_t numScales();

    std::string id() {return _uuid;}
    std::string name() {return _name;}
    std::filesystem::path path() {return _path;}
    Metadata metadata() {return _metadata;}

protected:
    std::filesystem::path _path;
    std::string _name;
    std::string _uuid;
    int _width{0};
    int _height{0};
    int _slices{0};
    int _numSliceCharacters{0};
    Metadata _metadata;

    z5::filesystem::handle::File *_zarrFile;
    std::vector<std::unique_ptr<z5::Dataset>> _zarrDs;
    nlohmann::json _zarrGroup;

    void zarrOpen();
};

