#pragma once

#include "VCCollection.hpp"
#include "Surface.hpp"

#include <nlohmann/json.hpp>

namespace apps
{

#include <opencv2/core.hpp>

nlohmann::json calc_point_metrics(const VCCollection& collection, QuadSurface* surface, const cv::Mat_<float>& winding);

} // namespace apps