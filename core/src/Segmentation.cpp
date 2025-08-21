#include "Segmentation.hpp"



static const std::filesystem::path METADATA_FILE = "meta.json";


Segmentation::Segmentation(std::filesystem::path path)
{
_path = path;
    _uuid = path.stem();
    _name = path.stem();
    _metadata = Metadata(path / METADATA_FILE);
}

Segmentation::Segmentation(std::filesystem::path path, std::string uuid, std::string name)
{
_path = path;
    _uuid = uuid;
    _name = name;
}

auto Segmentation::New(std::filesystem::path path) -> std::shared_ptr<Segmentation>
{
    return std::make_shared<Segmentation>(path);
}

auto Segmentation::New(std::filesystem::path path, std::string uuid, std::string name)
    -> std::shared_ptr<Segmentation>
{
    return std::make_shared<Segmentation>(path, uuid, name);
}




template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
