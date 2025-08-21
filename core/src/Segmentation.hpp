#pragma once

/** @file */

#include <filesystem>
#include "DiskBasedObjectBaseClass.hpp"
#include "Volume.hpp"

#include <variant>

/**
 * @class Segmentation
 * @author Seth Parker
 *
 * @brief Segmentation data
 *
 * Provides access to Segmentation information stored on disk, usually inside of
 * a VolumePkg.
 *
 * A Segmentation is generated within the coordinate frame of a Volume. Use the
 * `[has\|get\|set]VolumeID()` methods to retrieve the ID of the Volume with
 * which the Segmentation is associated.
 *
 * @ingroup Types
 */
class Segmentation : public DiskBasedObjectBaseClass
{
public:



    /** @brief Load a Segmentation from file */
    explicit Segmentation(std::filesystem::path path);

    /** @brief Make a new Segmentation in a directory */
    Segmentation(std::filesystem::path path, Identifier uuid, std::string name);

    /** @copydoc Segmentation(std::filesystem::path path) */
    static std::shared_ptr<Segmentation> New(std::filesystem::path path);

    /** @copydoc Segmentation(std::filesystem::path path, Identifier uuid,
     * std::string name) */
    static std::shared_ptr<Segmentation> New(
        std::filesystem::path path, Identifier uuid, std::string name);


};
