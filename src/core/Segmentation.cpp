#include "Segmentation.hpp"



static const std::filesystem::path METADATA_FILE = "meta.json";


Segmentation::Segmentation(const std::filesystem::path& path)
{
_path = path;
    _uuid = path.stem();
    _name = path.stem();
    _format = "tifxyz"; //FIXME: pull this from metadata
    //_metadata = Metadata(path / METADATA_FILE);
}

Segmentation::Segmentation(const std::filesystem::path &path, const std::string &uuid, const std::string &name)
{
_path = path;
    _uuid = uuid;
    _name = name;
    _format = "tifxyz"; //FIXME: pull this from metadata
}

auto Segmentation::New(const std::filesystem::path& path) -> std::shared_ptr<Segmentation>
{
    return std::make_shared<Segmentation>(path);
}

auto Segmentation::New(const std::filesystem::path& path, const std::string& uuid, const std::string& name)
    -> std::shared_ptr<Segmentation>
{
    return std::make_shared<Segmentation>(path, uuid, name);
}

