#pragma once

#include "../ui/VCCollection.hpp"
#include "../core/Surface.hpp"

#include <nlohmann/json.hpp>

nlohmann::json calc_point_metrics(const VCCollection& collection, QuadSurface* surface, const cv::Mat_<float>& winding);
