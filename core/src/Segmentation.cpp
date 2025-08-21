#include "Segmentation.hpp"




// Load a Segmentation directory from disk
// Reads and verifies metadata
Segmentation::Segmentation(std::filesystem::path path)
    : DiskBasedObjectBaseClass(std::move(path))
{
    if (metadata_.get<std::string>("type") != "seg") {
        throw std::runtime_error("File not of type: seg");
    }
}

// Make a new Segmentation file on disk
Segmentation::Segmentation(std::filesystem::path path, Identifier uuid, std::string name)
    : DiskBasedObjectBaseClass(
          std::move(path), std::move(uuid), std::move(name))
{
    metadata_.set("type", "seg");
    metadata_.set("vcps", std::string{});
    metadata_.set("vcano", std::string{});
    metadata_.set("volume", Volume::Identifier{});
    metadata_.save();
}

// Load a Segmentation from disk, return a pointer
auto Segmentation::New(std::filesystem::path path) -> std::shared_ptr<Segmentation>
{
    return std::make_shared<Segmentation>(path);
}

// Make a new segmentation on disk, return a pointer
auto Segmentation::New(std::filesystem::path path, std::string uuid, std::string name)
    -> std::shared_ptr<Segmentation>
{
    return std::make_shared<Segmentation>(path, uuid, name);
}




template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
