#pragma once

#include <opencv2/core.hpp>

// External declarations for configurable parameters
extern float local_cost_inl_th;
extern float same_surface_th;
extern float straight_weight;
extern float straight_min_count;
extern int inlier_base_threshold;

cv::Mat_<cv::Vec3f> upsample_with_grounding(cv::Mat_<cv::Vec3f> &small, cv::Mat_<cv::Vec2f> &locs, const cv::Size &tgt_size, const cv::Mat_<cv::Vec3f> &points, double sx, double sy);
void refine_normal(const std::vector<std::pair<cv::Vec2i,cv::Vec3f>> &refs, cv::Vec3f &point, cv::Vec3f &normal, cv::Vec3f &vx, cv::Vec3f &vy, const std::vector<float> &ws);
std::string get_surface_time_str();

#pragma once

#include "CostFunctions.hpp"

#define OPTIMIZE_ALL 1
#define SURF_LOSS 2
#define SPACE_LOSS 2 //SURF and SPACE are never used together
#define LOSS_3D_INDIRECT 4
#define LOSS_ZLOC 8
#define FLAG_GEN0 16
#define LOSS_ON_SURF 32
#define LOSS_ON_NORMALS 64

#define STATE_UNUSED 0
#define STATE_LOC_VALID 1
#define STATE_PROCESSING 2
#define STATE_COORD_VALID 4
#define STATE_FAIL 8
#define STATE_PHYS_ONLY 16
