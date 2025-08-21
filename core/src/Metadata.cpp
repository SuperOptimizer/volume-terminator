#include "Metadata.hpp"




Metadata::Metadata(std::filesystem::path fileLocation) : _path{fileLocation}
{
    // open the file
    if (!std::filesystem::exists(fileLocation)) {
        auto msg = "could not find json file '" + fileLocation.string() + "'";
        throw std::runtime_error(msg);
    }
    std::ifstream jsonFile(fileLocation.string());
    if (!jsonFile) {
        auto msg = "could not open json file '" + fileLocation.string() + "'";
        throw std::runtime_error(msg);
    }

    jsonFile >> _json;
    if (jsonFile.bad()) {
        auto msg = "could not read json file '" + fileLocation.string() + "'";
        throw std::runtime_error(msg);
    }
}

void Metadata::save(const std::filesystem::path& path)
{
     std::ofstream jsonFile(path.string(), std::ofstream::out);
    jsonFile << _json << '\n';
    if (jsonFile.fail()) {
        auto msg = "could not write json file '" + path.string() + "'";
        throw std::runtime_error(msg);
    }
}
