#include "Metadata.hpp"




// Read a json config from disk
Metadata::Metadata(std::filesystem::path fileLocation) : path_{fileLocation}
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

    jsonFile >> json_;
    if (jsonFile.bad()) {
        auto msg = "could not read json file '" + fileLocation.string() + "'";
        throw std::runtime_error(msg);
    }
}

// save the JSON file to disk
void Metadata::save(const std::filesystem::path& path)
{
    // open the file
    std::ofstream jsonFile(path.string(), std::ofstream::out);

    // try to push into the json file
    jsonFile << json_ << '\n';
    if (jsonFile.fail()) {
        auto msg = "could not write json file '" + path.string() + "'";
        throw std::runtime_error(msg);
    }
}
