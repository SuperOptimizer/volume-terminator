#include "VolumePkg.hpp"

#include <set>
#include <utility>

#include "DateTime.hpp"
#include "Logging.hpp"



////// Convenience vars and fns for accessing VolumePkg sub-paths //////
constexpr auto CONFIG = "config.json";

inline auto VolsDir(const std::filesystem::path& baseDir) -> std::filesystem::path
{
    return baseDir / "volumes";
}

inline auto SegsDir(const std::filesystem::path& baseDir, const std::string& dirName = "paths") -> std::filesystem::path
{
    return baseDir / dirName;
}

inline auto RendDir(const std::filesystem::path& baseDir) -> std::filesystem::path
{
    return baseDir / "renders";
}

inline auto TfmDir(const std::filesystem::path& baseDir) -> std::filesystem::path
{
    return baseDir / "transforms";
}

inline auto PreviewDirs(const std::filesystem::path& baseDir) -> std::vector<std::filesystem::path>
{
    return { baseDir / "volumes_preview_half", baseDir / "volumes_masked", baseDir / "volumes_previews"};
}

inline auto ReqDirs(const std::filesystem::path& baseDir) -> std::vector<std::filesystem::path>
{
    return {
        baseDir, ::VolsDir(baseDir), ::SegsDir(baseDir), ::RendDir(baseDir),
        ::TfmDir(baseDir)};
}

inline void keep(const std::filesystem::path& dir)
{
    if (not std::filesystem::exists(dir / ".vckeep")) {
        std::ofstream(dir / ".vckeep", std::ostream::ate);
    }
}


VolumePkg::VolumePkg(std::filesystem::path fileLocation, int version)
    : _rootdir{std::move(fileLocation)}
{


    // Make directories
    for (const auto& d : ::ReqDirs(_rootdir)) {
        if (not std::filesystem::exists(d)) {
            std::filesystem::create_directory(d);
        }
        if (d != _rootdir) {
            ::keep(d);
        }
    }

}

// Use this when reading a volpkg from a file
VolumePkg::VolumePkg(const std::filesystem::path& fileLocation) : _rootdir{fileLocation}
{
    for (const auto& d : ::ReqDirs(_rootdir)) {
        if (not std::filesystem::exists(d)) {
            Logger()->warn(
                "Creating missing VolumePkg directory: {}",
                d.filename().string());
            std::filesystem::create_directory(d);
        }
        if (d != _rootdir) {
            ::keep(d);
        }
    }

    // Load volumes into volumes_
    for (const auto& entry : std::filesystem::directory_iterator(::VolsDir(_rootdir))) {
        std::filesystem::path dirpath = std::filesystem::canonical(entry);
        if (std::filesystem::is_directory(dirpath)) {
            auto v = Volume::New(dirpath);
            _volumes.emplace(v->id(), v);
        }
    }

    // Load segmentations from ALL available directories at startup
    auto availableDirs = getAvailableSegmentationDirectories();
    for (const auto& dirName : availableDirs) {
        loadSegmentationsFromDirectory(dirName);
    }
}

std::shared_ptr<VolumePkg> VolumePkg::New(const std::filesystem::path& fileLocation, int version)
{
    return std::make_shared<VolumePkg>(fileLocation, version);
}

// Shared pointer volumepkg construction
std::shared_ptr<VolumePkg> VolumePkg::New(const std::filesystem::path& fileLocation)
{
    return std::make_shared<VolumePkg>(fileLocation);
}


std::string VolumePkg::name() const
{
    return "asdfasdfasdf";
}

auto VolumePkg::version() const -> int { return 1; }

void VolumePkg::saveMetadata() { }

void VolumePkg::saveMetadata(const std::filesystem::path& filePath)
{
}

// VOLUME FUNCTIONS //
auto VolumePkg::hasVolumes() const -> bool { return !_volumes.empty(); }

auto VolumePkg::hasVolume(const std::string& id) const -> bool
{
    return _volumes.contains(id);
}

auto VolumePkg::numberOfVolumes() const -> std::size_t
{
    return _volumes.size();
}

auto VolumePkg::volumeIDs() const -> std::vector<std::string>
{
    std::vector<std::string> ids;
    for (const auto &key: _volumes | std::views::keys) {
        ids.emplace_back(key);
    }
    return ids;
}

auto VolumePkg::volumeNames() const -> std::vector<std::string>
{
    std::vector<std::string> names;
    for (const auto &val: _volumes | std::views::values) {
        names.emplace_back(val->name());
    }
    return names;
}

auto VolumePkg::newVolume(std::string name) -> std::shared_ptr<Volume>
{
    // Generate a uuid
    auto uuid = DateTime();

    // Get dir name if not specified
    if (name.empty()) {
        name = uuid;
    }

    // Make the volume directory
    auto volDir = ::VolsDir(_rootdir) / uuid;
    if (!std::filesystem::exists(volDir)) {
        std::filesystem::create_directory(volDir);
    } else {
        throw std::runtime_error("Volume directory already exists");
    }

    // Make the volume
    auto r = _volumes.emplace(uuid, Volume::New(volDir, uuid, name));
    if (!r.second) {
        auto msg = "Volume already exists with ID " + uuid;
        throw std::runtime_error(msg);
    }

    // Return the Volume Pointer
    return r.first->second;
}

auto VolumePkg::volume() const -> std::shared_ptr<Volume> {
    if (_volumes.empty()) {
        throw std::out_of_range("No volumes in VolPkg");
    }
    return _volumes.begin()->second;
}

auto VolumePkg::volume() -> std::shared_ptr<Volume>
{
    if (_volumes.empty()) {
        throw std::out_of_range("No volumes in VolPkg");
    }
    return _volumes.begin()->second;
}

auto VolumePkg::volume(const std::string &id) const
    -> std::shared_ptr<Volume> {
    return _volumes.at(id);
}

auto VolumePkg::volume(const std::string& id) -> std::shared_ptr<Volume>
{
    return _volumes.at(id);
}

// SEGMENTATION FUNCTIONS //
auto VolumePkg::hasSegmentations() const -> bool
{
    return !_segmentations.empty();
}

auto VolumePkg::numberOfSegmentations() const -> std::size_t
{
    return _segmentations.size();
}

auto VolumePkg::segmentation(const std::string &id)
const -> std::shared_ptr<Segmentation> {
    return _segmentations.at(id);
}

std::vector<std::filesystem::path> VolumePkg::segmentationFiles()
{
    return _segmentation_files;
}

auto VolumePkg::segmentation(const std::string& id)
    -> std::shared_ptr<Segmentation>
{
    return _segmentations.at(id);
}

auto VolumePkg::segmentationIDs() const -> std::vector<std::string>
{
    std::vector<std::string> ids;
    // Only return IDs from the current directory
    for (const auto &key: _segmentations | std::views::keys) {
        if (auto it = _segmentationDirectories.find(key); it != _segmentationDirectories.end() && it->second == _currentSegmentationDir) {
            ids.emplace_back(key);
        }
    }
    return ids;
}

auto VolumePkg::segmentationNames() const -> std::vector<std::string>
{
    std::vector<std::string> names;
    for (const auto &val: _segmentations | std::views::values) {
        names.emplace_back(val->name());
    }
    return names;
}



// SEGMENTATION DIRECTORY METHODS //
void VolumePkg::loadSegmentationsFromDirectory(const std::string& dirName)
{
    // DO NOT clear existing segmentations - we keep all directories in memory
    // Only remove segmentations from this specific directory
    std::vector<std::string> toRemove;
    for (const auto& pair : _segmentationDirectories) {
        if (pair.second == dirName) {
            toRemove.push_back(pair.first);
        }
    }
    
    // Remove old segmentations from this directory
    for (const auto& id : toRemove) {
        _segmentations.erase(id);
        _segmentationDirectories.erase(id);
        
        // Remove from files vector
        auto it = std::ranges::remove_if(_segmentation_files,
                                         [&id, this](const std::filesystem::path& path) {
                                             auto segIt = _segmentations.find(id);
                                             return segIt != _segmentations.end() && segIt->second->path() == path;
                                         }).begin();
        _segmentation_files.erase(it, _segmentation_files.end());
    }
    
    // Check if directory exists
    const auto segDir = ::SegsDir(_rootdir, dirName);
    if (!std::filesystem::exists(segDir)) {
        Logger()->warn("Segmentation directory '{}' does not exist", dirName);
        return;
    }
    
    // Load segmentations from the specified directory
    for (const auto& entry : std::filesystem::directory_iterator(segDir)) {
        std::filesystem::path dirpath = std::filesystem::canonical(entry);
        if (std::filesystem::is_directory(dirpath)) {
            try {
                auto s = Segmentation::New(dirpath);
                _segmentations.emplace(s->id(), s);
                _segmentation_files.push_back(dirpath);
                // Track which directory this segmentation came from
                _segmentationDirectories[s->id()] = dirName;
            }
            catch (const std::exception &exc) {
                std::cout << "WARNING: some exception occured, skipping segment dir: " << dirpath << std::endl;
                std::cerr << exc.what();
            }
        }
    }
}

void VolumePkg::setSegmentationDirectory(const std::string& dirName)
{
    // Just change the current directory - all segmentations are already loaded
    _currentSegmentationDir = dirName;
}

auto VolumePkg::getSegmentationDirectory() const -> std::string
{
    return _currentSegmentationDir;
}

auto VolumePkg::getAvailableSegmentationDirectories() const -> std::vector<std::string>
{
    std::vector<std::string> dirs;
    
    // Check for common segmentation directories
    const std::vector<std::string> commonDirs = {"paths", "traces"};
    for (const auto& dir : commonDirs) {
        if (std::filesystem::exists(_rootdir / dir) && std::filesystem::is_directory(_rootdir / dir)) {
            dirs.push_back(dir);
        }
    }
    
    return dirs;
}

void VolumePkg::removeSegmentation(const std::string& id)
{
    // Check if segmentation exists
    auto it = _segmentations.find(id);
    if (it == _segmentations.end()) {
        throw std::runtime_error("Segmentation not found: " + id);
    }
    
    // Get the path before removing
    std::filesystem::path segPath = it->second->path();
    
    // Remove from internal map
    _segmentations.erase(it);
    
    // Remove from files vector
    auto fileIt = std::ranges::find(_segmentation_files, segPath);
    if (fileIt != _segmentation_files.end()) {
        _segmentation_files.erase(fileIt);
    }
    
    // Delete the physical folder
    if (std::filesystem::exists(segPath)) {
        std::filesystem::remove_all(segPath);
    }
}

void VolumePkg::refreshSegmentations()
{
    const auto segDir = ::SegsDir(_rootdir, _currentSegmentationDir);
    if (!std::filesystem::exists(segDir)) {
        Logger()->warn("Segmentation directory '{}' does not exist", _currentSegmentationDir);
        return;
    }
    
    // Build a set of current segmentation paths on disk for the current directory
    std::set<std::filesystem::path> diskPaths;
    for (const auto& entry : std::filesystem::directory_iterator(segDir)) {
        std::filesystem::path dirpath = std::filesystem::canonical(entry);
        if (std::filesystem::is_directory(dirpath)) {
            diskPaths.insert(dirpath);
        }
    }
    
    // Find segmentations to remove (loaded from current directory but not on disk anymore)
    std::vector<std::string> toRemove;
    for (const auto& seg : _segmentations) {
        auto dirIt = _segmentationDirectories.find(seg.first);
        if (dirIt != _segmentationDirectories.end() && dirIt->second == _currentSegmentationDir) {
            // This segmentation belongs to the current directory
            // Check if it still exists on disk
            if (!diskPaths.contains(seg.second->path())) {
                // Not on disk anymore - mark for removal
                toRemove.push_back(seg.first);
            }
        }
    }
    
    // Remove segmentations that no longer exist
    for (const auto& id : toRemove) {
        Logger()->info("Removing segmentation '{}' - no longer exists on disk", id);
        
        // Get the path before removing the segmentation
        std::filesystem::path segPath;
        auto segIt = _segmentations.find(id);
        if (segIt != _segmentations.end()) {
            segPath = segIt->second->path();
        }
        
        // Remove from segmentations map
        _segmentations.erase(id);
        
        // Remove from directories map
        _segmentationDirectories.erase(id);
        
        // Remove from files vector if we have a path
        if (!segPath.empty()) {
            auto fileIt = std::ranges::find(_segmentation_files,
                                            segPath);
            if (fileIt != _segmentation_files.end()) {
                _segmentation_files.erase(fileIt);
            }
        }
    }
    
    // Find and add new segmentations (on disk but not in memory)
    for (const auto& diskPath : diskPaths) {
        bool found = false;
        for (const auto &val: _segmentations | std::views::values) {
            if (val->path() == diskPath) {
                found = true;
                break;
            }
        }
        
        if (!found) {
            try {
                auto s = Segmentation::New(diskPath);
                _segmentations.emplace(s->id(), s);
                _segmentation_files.push_back(diskPath);
                _segmentationDirectories[s->id()] = _currentSegmentationDir;
                Logger()->info("Added new segmentation '{}'", s->id());
            }
            catch (const std::exception &exc) {
                Logger()->warn("Failed to load segment dir: {} - {}", diskPath.string(), exc.what());
            }
        }
    }
}
