#include "Volume.hpp"

#include <opencv2/imgproc.hpp>

#include "z5/attributes.hxx"
#include "z5/dataset.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/handle.hxx"
#include "z5/types/types.hxx"
#include "z5/factory.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include "xtensor/containers/xarray.hpp"

static const std::filesystem::path CONFIG_FILE = "meta.json";


// Load a Volume from disk
Volume::Volume(std::filesystem::path path)
{
    _path = path;
    _uuid = path.stem();
    _name = path.stem();

    std::ifstream config_file(path / CONFIG_FILE);

    if (!config_file.is_open()) {
        throw std::runtime_error("Cannot open metadata file: " + (path / CONFIG_FILE).string());
    }
    nlohmann::json metadata;
    config_file >> metadata;  // Parse JSON directly
    config_file.close();

    _width = metadata["width"].get<int>();
    _height = metadata["height"].get<int>();
    _slices = metadata["slices"].get<int>();

    zarrOpen();
}

// Setup a Volume from a folder of slices
Volume::Volume(std::filesystem::path path, std::string uuid, std::string name)
{
    _path = path;
    _uuid = uuid;
    _name = name;
    zarrOpen();
}

void Volume::zarrOpen()
{

    _zarrFile = new z5::filesystem::handle::File(_path);
    z5::filesystem::handle::Group group(_path, z5::FileMode::FileMode::r);
    z5::readAttributes(group, _zarrGroup);
    
    std::vector<std::string> groups;
    _zarrFile->keys(groups);
    std::sort(groups.begin(), groups.end());
    
    //FIXME hardcoded assumption that groups correspond to power-2 scaledowns ...
    for(auto name : groups) {
        z5::filesystem::handle::Dataset ds_handle(group, name, nlohmann::json::parse(std::ifstream(_path/name/".zarray")).value<std::string>("dimension_separator","."));

        _zarrDs.push_back(z5::filesystem::openDataset(ds_handle));
        if (_zarrDs.back()->getDtype() != z5::types::Datatype::uint8 && _zarrDs.back()->getDtype() != z5::types::Datatype::uint16)
            throw std::runtime_error("only uint8 & uint16 is currently supported for zarr datasets incompatible type found in "+_path.string()+" / " +name);
    }
}

// Load a Volume from disk, return a pointer
auto Volume::New(std::filesystem::path path) -> std::shared_ptr<Volume>
{
    return std::make_shared<Volume>(path);
}

// Set a Volume from a folder of slices, return a pointer
auto Volume::New(std::filesystem::path path, std::string uuid, std::string name)
    -> std::shared_ptr<Volume>
{
    return std::make_shared<Volume>(path, uuid, name);
}

auto Volume::sliceWidth() const -> int { return _width; }
auto Volume::sliceHeight() const -> int { return _height; }
auto Volume::numSlices() const -> int { return _slices; }
auto Volume::voxelSize() const -> double { return 1.0; }
auto Volume::min() const -> double { return 0.0; }
auto Volume::max() const -> double { return 65536.0; }




auto Volume::isInBounds(double x, double y, double z) const -> bool
{
    return x >= 0 && x < _width && y >= 0 && y < _height && z >= 0 &&
           z < _slices;
}

auto Volume::isInBounds(const cv::Vec3d& v) const -> bool
{
    return isInBounds(v(0), v(1), v(2));
}

void throw_run_path(const std::filesystem::path &path, const std::string msg)
{
    throw std::runtime_error(msg + " for " + path.string());
}

std::ostream& operator<< (std::ostream& out, const xt::xarray<uint8_t>::shape_type &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b\b]"; // use two ANSI backspace characters '\b' to overwrite final ", "
    }
    return out;
}

z5::Dataset *Volume::zarrDataset(int level)
{
    if (level >= _zarrDs.size())
        return nullptr;

    return _zarrDs[level].get();
}