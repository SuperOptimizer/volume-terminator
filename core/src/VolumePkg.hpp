#pragma once

/** @file */

#include <cstddef>
#include <iostream>
#include <map>

#include <filesystem>
#include "Segmentation.hpp"
#include "Volume.hpp"

class VolumePkg
{
public:

    VolumePkg(std::filesystem::path fileLocation, int version);
    explicit VolumePkg(const std::filesystem::path& fileLocation);
    static std::shared_ptr<VolumePkg> New(std::filesystem::path fileLocation, int version);
    static std::shared_ptr<VolumePkg> New(std::filesystem::path fileLocation);
    [[nodiscard]] std::string name() const;
    [[nodiscard]] int version() const;
    [[nodiscard]] double materialThickness() const;
    void saveMetadata();
    void saveMetadata(const std::filesystem::path& filePath);
    bool hasVolumes() const;
    [[nodiscard]] bool hasVolume(const std::string& id) const;
    std::size_t numberOfVolumes() const;
    [[nodiscard]]  std::vector<std::string> volumeIDs() const;
    [[nodiscard]] std::vector<std::string> volumeNames() const;
    std::shared_ptr<Volume> newVolume(std::string name = "");
    [[nodiscard]] const std::shared_ptr<Volume> volume() const;
    std::shared_ptr<Volume> volume();
    [[nodiscard]] const std::shared_ptr<Volume> volume(const std::string& id) const;
    std::shared_ptr<Volume> volume(const std::string& id);
    bool hasSegmentations() const;
    std::size_t numberOfSegmentations() const;
    [[nodiscard]] std::vector<std::string> segmentationIDs() const;
    [[nodiscard]] std::vector<std::string> segmentationNames() const;
    [[nodiscard]] const std::shared_ptr<Segmentation> segmentation(const std::string& id) const;
    std::vector<std::filesystem::path> segmentationFiles();
    std::shared_ptr<Segmentation> segmentation(const std::string& id);
    void removeSegmentation(const std::string& id);
    void setSegmentationDirectory(const std::string& dirName);
    [[nodiscard]] std::string getSegmentationDirectory() const;
    [[nodiscard]] std::vector<std::string> getAvailableSegmentationDirectories() const;

    void refreshSegmentations();


private:
    std::filesystem::path _rootdir;
    std::map<std::string, std::shared_ptr<Volume>> _volumes;
    std::map<std::string, std::shared_ptr<Segmentation>> _segmentations;
    std::vector<std::filesystem::path> _segmentation_files;
    std::string _currentSegmentationDir = "paths";
    std::map<std::string, std::string> _segmentationDirectories;

    void loadSegmentationsFromDirectory(const std::string& dirName);
};

