#include <omp.h>

#include "SurfaceHelpers.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "Slicing.hpp"
#include "Surface.hpp"
#include "ChunkedTensor.hpp"


#include "xtensor/views/xview.hpp"
#include <fstream>
#include <iostream>


static int gen_straight_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, bool optimize_all, float w = 0.5);
static int gen_dist_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, float unit, bool optimize_all, ceres::ResidualBlockId *res, float w = 1.0);

// Forward declarations of global variables used across functions
extern float straight_weight;
extern float straight_weight_3D;
extern float sliding_w_scale;
extern float z_loc_loss_w;
extern float dist_loss_2d_w;
extern float dist_loss_3d_w;


class ALifeTime
{
public:
    ALifeTime(const std::string &msg = "")
    {
        if (msg.size())
            std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    double unit = 0;
    std::string del_msg;
    std::string unit_string;
    ~ALifeTime()
    {
        auto end = std::chrono::high_resolution_clock::now();
        if (del_msg.size())
            std::cout << del_msg << std::chrono::duration<double>(end-start).count() << " s";
        else
            std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s";

        if (unit)
            std::cout << " " << unit/std::chrono::duration<double>(end-start).count() << unit_string << "/s" << "\n";
        else
            std::cout << "\n";

    }
    double seconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end-start).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

static cv::Vec3f at_int_inv(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f p)
{
    int x = p[1];
    int y = p[0];
    float fx = p[1]-x;
    float fy = p[0]-y;

    const cv::Vec3f& p00 = points(y,x);
    const cv::Vec3f& p01 = points(y,x+1);
    const cv::Vec3f& p10 = points(y+1,x);
    const cv::Vec3f& p11 = points(y+1,x+1);

    cv::Vec3f p0 = (1-fx)*p00 + fx*p01;
    cv::Vec3f p1 = (1-fx)*p10 + fx*p11;

    return (1-fy)*p0 + fy*p1;
}

static bool loc_valid(int state)
{
    return state & STATE_LOC_VALID;
}

static bool coord_valid(int state)
{
    return (state & STATE_COORD_VALID) || (state & STATE_LOC_VALID);
}

//gen straigt loss given point and 3 offsets
static int gen_straight_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2,
    const cv::Vec2i &o3, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, bool optimize_all, float w)
{
    if (!coord_valid(state(p+o1)))
        return 0;
    if (!coord_valid(state(p+o2)))
        return 0;
    if (!coord_valid(state(p+o3)))
        return 0;

    problem.AddResidualBlock(StraightLoss::Create(w), nullptr, &dpoints(p+o1)[0], &dpoints(p+o2)[0], &dpoints(p+o3)[0]);

    if (!optimize_all) {
        if (o1 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&dpoints(p+o1)[0]);
        if (o2 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&dpoints(p+o2)[0]);
        if (o3 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&dpoints(p+o3)[0]);
    }

    return 1;
}

static int gen_dist_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints,
    float unit, bool optimize_all, ceres::ResidualBlockId *res, float w)
{
    // Add a loss saying that dpoints(p) and dpoints(p+off) should themselves be distance |off| apart
    // Here dpoints is a 2D grid mapping surface-space points to 3D volume space
    // So this says that distances should be preserved from volume to surface

    if (!coord_valid(state(p)))
        return 0;
    if (!coord_valid(state(p+off)))
        return 0;

    ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss::Create(unit*cv::norm(off),w), nullptr, &dpoints(p)[0], &dpoints(p+off)[0]);

    if (res)
        *res = tmp;

    if (!optimize_all)
        problem.SetParameterBlockConstant(&dpoints(p+off)[0]);

    return 1;
}

static cv::Vec2i lower_p(const cv::Vec2i &point, const cv::Vec2i &offset)
{
    if (offset[0] == 0) {
        if (offset[1] < 0)
            return point+offset;
        else
            return point;
    }
    if (offset[0] < 0)
        return point+offset;
    else
        return point;
}

static bool loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status)
{
    return loss_status(lower_p(p, off)) & (1 << bit);
}

static int set_loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status, int set)
{
    if (set)
        loss_status(lower_p(p, off)) |= (1 << bit);
    return set;
}

static int conditional_dist_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, float unit, bool optimize_all, float w = 1.0)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_dist_loss(problem, p, off, state, out, unit, optimize_all, nullptr, w));
    return set;
};

static int conditional_straight_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3,
    cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, bool optimize_all)
{
    int set = 0;
    if (!loss_mask(bit, p, o2, loss_status))
        set += set_loss_mask(bit, p, o2, loss_status, gen_straight_loss(problem, p, o1, o2, o3, state, out, optimize_all));
    return set;
};

struct vec2i_hash {
    size_t operator()(cv::Vec2i p) const
    {
        size_t hash1 = std::hash<float>{}(p[0]);
        size_t hash2 = std::hash<float>{}(p[1]);

        //magic numbers from boost. should be good enough
        return hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
    }
};

static void freeze_inner_params(ceres::Problem &problem, int edge_dist, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out,
    cv::Mat_<cv::Vec2d> &loc, cv::Mat_<uint16_t> &loss_status, int inner_flags)
{
    cv::Mat_<float> dist(state.size());

    edge_dist = std::min(edge_dist,254);


    cv::Mat_<uint8_t> masked;
    bitwise_and(state, cv::Scalar(inner_flags), masked);


    cv::distanceTransform(masked, dist, cv::DIST_L1, cv::DIST_MASK_3);

    for(int j=0;j<dist.rows;j++)
        for(int i=0;i<dist.cols;i++) {
            if (dist(j,i) >= edge_dist && !loss_mask(7, {j,i}, {0,0}, loss_status)) {
                if (problem.HasParameterBlock(&out(j,i)[0]))
                    problem.SetParameterBlockConstant(&out(j,i)[0]);
                if (!loc.empty() && problem.HasParameterBlock(&loc(j,i)[0]))
                    problem.SetParameterBlockConstant(&loc(j,i)[0]);
                set_loss_mask(7, {j,i}, {0,0}, loss_status, 1);
            }
            if (dist(j,i) >= edge_dist+1 && !loss_mask(8, {j,i}, {0,0}, loss_status)) {
                if (problem.HasParameterBlock(&out(j,i)[0]))
                    problem.RemoveParameterBlock(&out(j,i)[0]);
                if (!loc.empty() && problem.HasParameterBlock(&loc(j,i)[0]))
                    problem.RemoveParameterBlock(&loc(j,i)[0]);
                set_loss_mask(8, {j,i}, {0,0}, loss_status, 1);
            }
        }
}


struct DSReader
{
    z5::Dataset *ds;
    float scale;
    ChunkCache *cache;
};


template <typename T, typename C>
static int gen_space_loss(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, float w = 0.1)
{
    // Add a loss saying that value of 3D volume tensor t at location loc(p) should be near-zero

    if (!loc_valid(state(p)))
        return 0;

    problem.AddResidualBlock(SpaceLossAcc<T,C>::Create(t, w), nullptr, &loc(p)[0]);

    return 1;
}

template <typename T, typename C>
static int gen_space_line_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, int steps, float w = 0.1, float dist_th = 2)
{
    // Add a loss saying that value of 3D volume tensor t should be near-zero for all locations along
    // the line from loc(p) to loc(p + off)

    if (!loc_valid(state(p)))
        return 0;
    if (!loc_valid(state(p+off)))
        return 0;

    problem.AddResidualBlock(SpaceLineLossAcc<T,C>::Create(t, steps, w), nullptr, &loc(p)[0], &loc(p+off)[0]);

    return 1;
}

static float space_trace_dist_w = 1.0;

//create all valid losses for this point
template <typename I, typename T, typename C>
static int emptytrace_create_centered_losses(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &loc, const I &interp, Chunked3d<T,C> &t, float unit, int flags = 0)
{
    //generate losses for point p
    int count = 0;

    //horizontal
    count += gen_straight_loss(problem, p, {0,-2},{0,-1},{0,0}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,-1},{0,0},{0,1}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,0},{0,1},{0,2}, state, loc, flags & OPTIMIZE_ALL);

    //vertical
    count += gen_straight_loss(problem, p, {-2,0},{-1,0},{0,0}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {-1,0},{0,0},{1,0}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,0},{1,0},{2,0}, state, loc, flags & OPTIMIZE_ALL);

    //direct neighboars
    count += gen_dist_loss(problem, p, {0,-1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {0,1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {-1,0}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {1,0}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);

    //diagonal neighbors
    count += gen_dist_loss(problem, p, {1,-1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {-1,1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {1,1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {-1,-1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);

    if (flags & SPACE_LOSS) {
        count += gen_space_loss(problem, p, state, loc, t);

        count += gen_space_line_loss(problem, p, {1,0}, state, loc, t, unit);
        count += gen_space_line_loss(problem, p, {-1,0}, state, loc, t, unit);
        count += gen_space_line_loss(problem, p, {0,1}, state, loc, t, unit);
        count += gen_space_line_loss(problem, p, {0,-1}, state, loc, t, unit);
    }

    return count;
}

template <typename T, typename C>
static int conditional_spaceline_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, int steps)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_space_line_loss(problem, p, off, state, loc, t, steps));
    return set;
};

//create only missing losses so we can optimize the whole problem
template <typename I, typename T, typename C>
static int emptytrace_create_missing_centered_losses(ceres::Problem &problem, cv::Mat_<uint16_t> &loss_status, const cv::Vec2i &p,
    cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, const I &interp, Chunked3d<T,C> &t, float unit,
    int flags = SPACE_LOSS | OPTIMIZE_ALL)
{
    //generate losses for point p
    int count = 0;

    //horizontal
    // if (flags & SPACE_LOSS) {
        count += conditional_straight_loss(0, p, {0,-2},{0,-1},{0,0}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(0, p, {0,-1},{0,0},{0,1}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(0, p, {0,0},{0,1},{0,2}, loss_status, problem, state, loc, flags);

        //vertical
        count += conditional_straight_loss(1, p, {-2,0},{-1,0},{0,0}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(1, p, {-1,0},{0,0},{1,0}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(1, p, {0,0},{1,0},{2,0}, loss_status, problem, state, loc, flags);
    // }

    //direct neighboars h
    count += conditional_dist_loss(2, p, {0,-1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(2, p, {0,1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    //direct neighbors v
    count += conditional_dist_loss(3, p, {-1,0}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(3, p, {1,0}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    //diagonal neighbors
    count += conditional_dist_loss(4, p, {1,-1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(4, p, {-1,1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    count += conditional_dist_loss(5, p, {1,1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(5, p, {-1,-1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    if (flags & SPACE_LOSS) {
        if (!loss_mask(6, p, {0,0}, loss_status))
            count += set_loss_mask(6, p, {0,0}, loss_status, gen_space_loss(problem, p, state, loc, t));

        count += conditional_spaceline_loss(7, p, {1,0}, loss_status, problem, state, loc, t, unit);
        count += conditional_spaceline_loss(7, p, {-1,0}, loss_status, problem, state, loc, t, unit);

        count += conditional_spaceline_loss(8, p, {0,1}, loss_status, problem, state, loc, t, unit);
        count += conditional_spaceline_loss(8, p, {0,-1}, loss_status, problem, state, loc, t, unit);
    }

    return count;
}

//optimize within a radius, setting edge points to constant
template <typename I, typename T, typename C>
static float local_optimization(int radius, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &locs,
    const I &interp, Chunked3d<T,C> &t, float unit, bool quiet = false)
{
    ceres::Problem problem;
    cv::Mat_<uint16_t> loss_status(state.size());

    int r_outer = radius+3;

    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,locs.cols-1);ox++)
            loss_status(oy,ox) = 0;

    for(int oy=std::max(p[0]-radius,0);oy<=std::min(p[0]+radius,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-radius,0);ox<=std::min(p[1]+radius,locs.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) <= radius)
                emptytrace_create_missing_centered_losses(problem, loss_status, op, state, locs, interp, t, unit);
        }
    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,locs.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) > radius && problem.HasParameterBlock(&locs(op)[0]))
                problem.SetParameterBlockConstant(&locs(op)[0]);
        }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;
    options.function_tolerance = 1e-4;
    options.use_nonmonotonic_steps = true;

//    if (problem.NumParameterBlocks() > 1) {
//        options.use_inner_iterations = true;
//    }
#ifdef VC_USE_CUDA_SPARSE
    // Check if Ceres was actually built with CUDA sparse support
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;

        // Enable mixed precision for SPARSE_SCHUR
        if (options.linear_solver_type == ceres::SPARSE_SCHUR) {
            options.use_mixed_precision_solves = true;
        }
    } else {
        std::cerr << "Warning: CUDA_SPARSE requested but Ceres was not built with CUDA sparse support. Falling back to default solver." << "\n";
    }
#endif
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!quiet)
        std::cout << "local solve radius " << radius << " " << summary.BriefReport() << "\n";

    return sqrt(summary.final_cost/summary.num_residual_blocks);
}

static float min_dist(const cv::Vec2i &p, const std::vector<cv::Vec2i> &list)
{
    double dist = 10000000000;
    for(auto &o : list) {
        if (o[0] == -1 || o == p)
            continue;
        dist = std::min(cv::norm(o-p), dist);
    }

    return dist;
}

static cv::Point2i extract_point_min_dist(std::vector<cv::Vec2i> &cands, std::vector<cv::Vec2i> &blocked, int &idx, float dist)
{
    for(int i=0;i<cands.size();i++) {
        cv::Vec2i p = cands[(i + idx) % cands.size()];

        if (p[0] == -1)
            continue;

        if (min_dist(p, blocked) >= dist) {
            cands[(i + idx) % cands.size()] = {-1,-1};
            idx = (i + idx + 1) % cands.size();

            return p;
        }
    }

    return {-1,-1};
}

//collection of points which can be retrieved with minimum distance requirement
class OmpThreadPointCol
{
public:
    OmpThreadPointCol(float dist, const std::vector<cv::Vec2i> &src) :
        _thread_count(omp_get_max_threads()),
        _dist(dist),
        _points(src),
        _thread_points(_thread_count,{-1,-1}),
        _thread_idx(_thread_count, -1) {};

    template <typename T>
    OmpThreadPointCol(float dist, T src) :
        _thread_count(omp_get_max_threads()),
        _dist(dist),
        _points(src.begin(), src.end()),
        _thread_points(_thread_count,{-1,-1}),
        _thread_idx(_thread_count, -1) {};

    cv::Point2i next()
    {
        int t_id = omp_get_thread_num();
        if (_thread_idx[t_id] == -1)
            _thread_idx[t_id] = rand() % _thread_count;
        _thread_points[t_id] = {-1,-1};
#pragma omp critical
        _thread_points[t_id] = extract_point_min_dist(_points, _thread_points, _thread_idx[t_id], _dist);
        return _thread_points[t_id];
    }

protected:
    int _thread_count;
    float _dist;
    std::vector<cv::Vec2i> _points;
    std::vector<cv::Vec2i> _thread_points;
    std::vector<int> _thread_idx;
};

template <typename E>
static E _max_d_ign(const E &a, const E &b)
{
    if (a == E(-1))
        return b;
    if (b == E(-1))
        return a;
    return std::max(a,b);
}

template <typename T, typename E>
static void _dist_iteration(T &from, T &to, int s)
{
    E magic = -1;
#pragma omp parallel for
    for(int k=0;k<s;k++)
        for(int j=0;j<s;j++)
            for(int i=0;i<s;i++) {
                E dist = from(k,j,i);
                if (dist == magic) {
                    if (k) dist = _max_d_ign(dist, from(k-1,j,i));
                    if (k < s-1) dist = _max_d_ign(dist, from(k+1,j,i));
                    if (j) dist = _max_d_ign(dist, from(k,j-1,i));
                    if (j < s-1) dist = _max_d_ign(dist, from(k,j+1,i));
                    if (i) dist = _max_d_ign(dist, from(k,j,i-1));
                    if (i < s-1) dist = _max_d_ign(dist, from(k,j,i+1));
                    if (dist != magic)
                        to(k,j,i) = dist+1;
                    else
                        to(k,j,i) = dist;
                }
                else
                    to(k,j,i) = dist;

            }
}

template <typename T, typename E>
static T distance_transform(const T &chunk, int steps, int size)
{
    T c1 = xt::empty<E>(chunk.shape());
    T c2 = xt::empty<E>(chunk.shape());

    c1 = chunk;

    E magic = -1;

    for(int n=0;n<steps/2;n++) {
        _dist_iteration<T,E>(c1,c2,size);
        _dist_iteration<T,E>(c2,c1,size);
    }

#pragma omp parallel for
    for(int z=0;z<size;z++)
        for(int y=0;y<size;y++)
            for(int x=0;x<size;x++)
                if (c1(z,y,x) == magic)
                    c1(z,y,x) = steps;

    return c1;
}

struct thresholdedDistance
{
    enum {BORDER = 16};
    enum {CHUNK_SIZE = 64};
    enum {FILL_V = 0};
    enum {TH = 170};
    const std::string UNIQUE_ID_STRING = "dqk247q6vz_"+std::to_string(BORDER)+"_"+std::to_string(CHUNK_SIZE)+"_"+std::to_string(FILL_V)+"_"+std::to_string(TH);
    template <typename T, typename E> void compute(const T &large, T &small, const cv::Vec3i &offset_large)
    {
        T outer = xt::empty<E>(large.shape());

        int s = CHUNK_SIZE+2*BORDER;
        E magic = -1;

        int good_count = 0;

#pragma omp parallel for
        for(int z=0;z<s;z++)
            for(int y=0;y<s;y++)
                for(int x=0;x<s;x++)
                    if (large(z,y,x) < TH)
                        outer(z,y,x) = magic;
        else {
            good_count++;
            outer(z,y,x) = 0;
        }

        outer = distance_transform<T,E>(outer, 15, s);

        int low = int(BORDER);
        int high = int(BORDER)+int(CHUNK_SIZE);

        auto crop_outer = view(outer, xt::range(low,high),xt::range(low,high),xt::range(low,high));

        small = crop_outer;
    }

};

float dist_th = 1.5;

QuadSurface *space_tracing_quad_phys(z5::Dataset *ds, float scale, ChunkCache *cache, cv::Vec3f origin, int stop_gen, float step, const std::string &cache_root, float voxelsize)
{
    ALifeTime f_timer("empty space tracing\n");
    DSReader reader = {ds,scale,cache};

    // Calculate the maximum possible size the patch might grow to
    //FIXME show and handle area edge!
    int w = 2*stop_gen+50;
    int h = w;
    cv::Size size = {w,h};
    cv::Rect bounds(0,0,w,h);

    int x0 = w/2;
    int y0 = h/2;

    // Together these represent the cached distance-transform of the thresholded surface volume
    thresholdedDistance compute;
    Chunked3d<uint8_t,thresholdedDistance> proc_tensor(compute, ds, cache, cache_root);

    // Debug: test the chunk cache by reading one voxel
    passTroughComputor pass;
    Chunked3d<uint8_t,passTroughComputor> dbg_tensor(pass, ds, cache);
    std::cout << "seed val " << origin << " " <<
    (int)dbg_tensor(origin[2],origin[1],origin[0]) << "\n";

    ALifeTime *timer = new ALifeTime("search & optimization ...");

    // This provides a cached interpolated version of the original surface volume
    CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp_global(proc_tensor);

    // fringe contains all 2D points around the edge of the patch where we might expand it
    // cands will contain new points adjacent to the fringe that are candidates to expand into
    std::vector<cv::Vec2i> fringe;
    std::vector<cv::Vec2i> cands;

    float T = step;
    float Ts = step*reader.scale;

    int r = 1;
    int r2 = 2;

    // The following track the state of the patch; they are each as big as the largest possible patch but initially empty
    // - locs defines the patch! It says for each 2D position, which 3D position it corresponds to
    // - state tracks whether each 2D position is part of the patch yet, and whether its 3D position has been found
    cv::Mat_<cv::Vec3d> locs(size,cv::Vec3f(-1,-1,-1));
    cv::Mat_<uint8_t> state(size,0);
    cv::Mat_<uint8_t> phys_fail(size,0);
    cv::Mat_<float> init_dist(size,0);
    cv::Mat_<uint16_t> loss_status(cv::Size(w,h),0);

    cv::Vec3f vx = {1,0,0};
    cv::Vec3f vy = {0,1,0};

    // Initialise the trace at the center of the available area, as a tiny single-quad patch at the seed point
    cv::Rect used_area(x0,y0,2,2);
    //these are locations in the local volume!
    locs(y0,x0) = origin;
    locs(y0,x0+1) = origin+vx*0.1;
    locs(y0+1,x0) = origin+vy*0.1;
    locs(y0+1,x0+1) = origin+vx*0.1 + vy*0.1;

    state(y0,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
    state(y0+1,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
    state(y0,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;
    state(y0+1,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;

    // This Ceres problem is parameterised by locs; residuals are progressively added as the patch grows enforcing that
    // all points in the patch are correct distance in 2D vs 3D space, not too high curvature, near surface prediction, etc.
    ceres::Problem big_problem;

    // Add losses for every 'active' surface point (just the four currently) that doesn't yet have them
    int loss_count = 0;
    loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0,x0}, state, locs, interp_global, proc_tensor, Ts);
    loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0+1,x0}, state, locs,  interp_global, proc_tensor, Ts);
    loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0,x0+1}, state, locs,  interp_global, proc_tensor, Ts);
    loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0+1,x0+1}, state, locs,  interp_global, proc_tensor, Ts);

    std::cout << "init loss count " << loss_count << "\n";

    ceres::Solver::Options options_big;
    options_big.linear_solver_type = ceres::SPARSE_SCHUR;
    options_big.use_nonmonotonic_steps = true;
#ifdef VC_USE_CUDA_SPARSE
    // Check if Ceres was actually built with CUDA sparse support
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
        options_big.linear_solver_type = ceres::SPARSE_SCHUR;
        options_big.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;

        // Enable mixed precision for SPARSE_SCHUR
        if (options_big.linear_solver_type == ceres::SPARSE_SCHUR) {
            options_big.use_mixed_precision_solves = true;
        }
    } else {
        std::cerr << "Warning: CUDA_SPARSE requested but Ceres was not built with CUDA sparse support. Falling back to default solver." << "\n";
    }
#endif
    options_big.minimizer_progress_to_stdout = false;
    options_big.max_num_iterations = 10000;

    // Solve the initial optimisation problem, just placing the first four vertices around the seed
    ceres::Solver::Summary big_summary;
    ceres::Solve(options_big, &big_problem, &big_summary);
    std::cout << big_summary.BriefReport() << "\n";

    // Prepare a new set of Ceres options used later during local solves
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 200;
    options.function_tolerance = 1e-3;

    // Record the four vertices in the (previously empty) fringe, i.e. the current edge of the patch
    fringe.push_back({y0,x0});
    fringe.push_back({y0+1,x0});
    fringe.push_back({y0,x0+1});
    fringe.push_back({y0+1,x0+1});

    std::vector<cv::Vec2i> neighs = {{1,0},{0,1},{-1,0},{0,-1}};

    int succ = 0;  // number of quads successfully added to the patch (each of size approx. step**2)

    int generation = 0;  // each generation considers expanding by one step at every fringe point
    int phys_fail_count = 0;
    double phys_fail_th = 0.1;

    int max_local_opt_r = 4;

    std::vector<float> gen_max_cost;
    std::vector<float> gen_avg_cost;
    
    int ref_max = 6;
    int curr_ref_min = ref_max;

    while (fringe.size()) {
        bool global_opt = generation <= 20;

        ALifeTime timer_gen;
        timer_gen.del_msg = "time per generation ";

        int phys_fail_count_gen = 0;
        generation++;
        if (stop_gen && generation >= stop_gen)
            break;

        std::vector<cv::Vec2i> rest_ps;  // contains candidates we didn't fully consider for some reason (write-only!)

        // For every point in the fringe (where we might expand the patch outwards), add to cands all
        // new 2D points we might add to the patch (and later find the corresponding 3D point for)
        for(const auto& p : fringe)
        {
            if ((state(p) & STATE_LOC_VALID) == 0) {
                if (state(p) & STATE_COORD_VALID)
                    for(const auto& n : neighs)
                        if (bounds.contains(cv::Point(p+n))
                            && (state(p+n) & (STATE_PROCESSING | STATE_LOC_VALID | STATE_COORD_VALID)) == 0) {
                            rest_ps.push_back(p+n);
                            }
                continue;
            }

            for(const auto& n : neighs)
                if (bounds.contains(cv::Point(p+n))
                    && (state(p+n) & STATE_PROCESSING) == 0
                    && (state(p+n) & STATE_LOC_VALID) == 0) {
                    state(p+n) |= STATE_PROCESSING;
                    cands.push_back(p+n);
                }
        }
        std::cout << "gen " << generation << " processing " << cands.size() << " fringe cands (total done " << succ << " fringe: " << fringe.size() << ")" << "\n";
        fringe.resize(0);

        std::cout << "cands " << cands.size() << "\n";
        
        if (generation % 10 == 0)
            curr_ref_min = std::min(curr_ref_min, 5);

        int succ_gen = 0;
        std::vector<cv::Vec2i> succ_gen_ps;

        // Build a structure that allows parallel iteration over cands, while avoiding any two threads simultaneously
        // considering two points that are too close to each other...
        OmpThreadPointCol cands_threadcol(max_local_opt_r*2+1, cands);

        // ...then start iterating over candidates in parallel using the above to yield points
#pragma omp parallel
        {
            CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor);
//             int idx = rand() % cands.size();
            while (true) {
                cv::Vec2i p = cands_threadcol.next();
                if (p[0] == -1)
                    break;

                if (state(p) & (STATE_LOC_VALID | STATE_COORD_VALID))
                    continue;

                // p is now a 2D point we consider adding to the patch; find the best 3D point to map it to

                // Iterate all adjacent points that are in the patch, and find their 3D locations
                int ref_count = 0;
                cv::Vec3d avg = {0,0,0};
                std::vector<cv::Vec2i> srcs;
                for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,locs.rows-1);oy++)
                    for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,locs.cols-1);ox++)
                        if (state(oy,ox) & STATE_LOC_VALID) {
                            ref_count++;
                            avg += locs(oy,ox);
                            srcs.push_back({oy,ox});
                        }
                        
                // Of those adjacent points, find the one that itself has most adjacent in-patch points
                cv::Vec2i best_l = srcs[0];
                int best_ref_l = -1;
                int rec_ref_sum = 0;
                for(cv::Vec2i l : srcs) {
                    int ref_l = 0;
                    for(int oy=std::max(l[0]-r,0);oy<=std::min(l[0]+r,locs.rows-1);oy++)
                        for(int ox=std::max(l[1]-r,0);ox<=std::min(l[1]+r,locs.cols-1);ox++)
                            if (state(oy,ox) & STATE_LOC_VALID)
                                ref_l++;
                    
                    rec_ref_sum += ref_l;
                    
                    if (ref_l > best_ref_l) {
                        best_l = l;
                        best_ref_l = ref_l;
                    }
                }

                // Unused
                int ref_count2 = 0;
                for(int oy=std::max(p[0]-r2,0);oy<=std::min(p[0]+r2,locs.rows-1);oy++)
                    for(int ox=std::max(p[1]-r2,0);ox<=std::min(p[1]+r2,locs.cols-1);ox++)
                        // if (state(oy,ox) & (STATE_LOC_VALID | STATE_COORD_VALID)) {
                        if (state(oy,ox) & STATE_LOC_VALID) {
                            ref_count2++;
                        }

                // If the candidate 2D point is too 'loosely' connected to the patch, skip it; thus we prefer to keep the patch
                // compact rather than growing tendrils
                if (ref_count < 2 || ref_count+0.35*rec_ref_sum < curr_ref_min /*|| (generation > 3 && ref_count2 < 14)*/) {
                    state(p) &= ~STATE_PROCESSING;
#pragma omp critical
                    rest_ps.push_back(p);
                    continue;
                }

                // Initial guess for the corresponding 3D location is a perturbation of the position of the best-connected neighbor
                avg /= ref_count;
                cv::Vec3d init = locs(best_l)+cv::Vec3d((rand()%1000)/10000.0-0.05,(rand()%1000)/10000.0-0.05,(rand()%1000)/10000.0-0.05);
                locs(p) = init;

                // Set up a new local optimzation problem for the candidate point and its neighbors (initially just distance
                // and curvature losses, not nearness-to-surface)
                ceres::Problem problem;

                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                //int local_loss_count = emptytrace_create_centered_losses(problem, p, state, locs, interp, proc_tensor, Ts);

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                double loss1 = summary.final_cost;

                // If the solve couldn't find a good 3D position for the new point, try again with a different random initialisation (this time
                // around the average neighbor location instead of the best one)
                if (loss1 > phys_fail_th) {
                    cv::Vec3d best_loc = locs(p);
                    double best_loss = loss1;
                    for (int n=0;n<100;n++) {
                        int range = step*10;
                        locs(p) = avg + cv::Vec3d((rand()%(range*2))-range,(rand()%(range*2))-range,(rand()%(range*2))-range);
                        ceres::Solve(options, &problem, &summary);
                        loss1 = summary.final_cost;
                        if (loss1 < best_loss) {
                            best_loss = loss1;
                            best_loc = locs(p);
                        }
                        if (loss1 < phys_fail_th)
                            break;
                    }
                    loss1 = best_loss;
                    locs(p) = best_loc;
                }

                cv::Vec3d phys_only_loc = locs(p);
                // locs(p) = init;

                // Add to the local problem losses saying the new point should fall near the surface predictions
                gen_space_loss(problem, p, state, locs, proc_tensor);

                gen_space_line_loss(problem, p, {1,0}, state, locs, proc_tensor, T, 0.1, 100);
                gen_space_line_loss(problem, p, {-1,0}, state, locs, proc_tensor, T, 0.1, 100);
                gen_space_line_loss(problem, p, {0,1}, state, locs, proc_tensor, T, 0.1, 100);
                gen_space_line_loss(problem, p, {0,-1}, state, locs, proc_tensor, T, 0.1, 100);

                // Re-solve the updated local problem
                ceres::Solve(options, &problem, &summary);

                // Measure the worst-case distance from the surface predictions, of edges between the new point and its neighbors
                double dist;
                interp.Evaluate(locs(p)[2],locs(p)[1],locs(p)[0], &dist);
                int count = 0;
                for (auto &off : neighs) {
                    if (state(p+off) & STATE_LOC_VALID) {
                        for(int i=1;i<T;i++) {
                            float f1 = float(i)/T;
                            float f2 = 1-f1;
                            cv::Vec3d l = locs(p)*f1 + locs(p+off)*f2;
                            double d2;
                            interp.Evaluate(l[2],l[1],l[0], &d2);
                            dist = std::max(dist, d2);
                            count++;
                        }
                    }
                }

                init_dist(p) = dist;

                if (dist >= dist_th || summary.final_cost >= 0.1) {
                    // The solution to the local problem is bad -- large loss, or too far from the surface; still add to the global
                    // problem (as below) but don't mark the point location as valid
                    locs(p) = phys_only_loc;
                    state(p) = STATE_COORD_VALID;
                    if (global_opt) {
#pragma omp critical
                        loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, p, state, locs,
                                                                                interp_global, proc_tensor, Ts, OPTIMIZE_ALL);
                    }
                    if (loss1 > phys_fail_th) {
                        // If even the first local solve (geometry only) was bad, try a less-local re-solve, that also adjusts the
                        // neighbors at progressively increasing radii as needed
                        phys_fail(p) = 1;

                        float err = 0;
                        for(int range = 1; range<=max_local_opt_r;range++) {
                            err = local_optimization(range, p, state, locs, interp, proc_tensor, Ts);
                            if (err <= phys_fail_th)
                                break;
                        }
                        if (err > phys_fail_th) {
                            std::cout << "local phys fail! " << err << "\n";
#pragma omp atomic
                            phys_fail_count++;
#pragma omp atomic
                            phys_fail_count_gen++;
                        }
                    }
                }
                else {
                    // We found a good solution to the local problem; add losses for the new point to the global problem, add the
                    // new point to the fringe, and record as successful
                    if (global_opt) {
#pragma omp critical
                        loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, p, state, locs,
                                                                                interp_global, proc_tensor, Ts);
                    }
#pragma omp atomic
                    succ++;
#pragma omp atomic
                    succ_gen++;
#pragma omp critical
                    {
                        if (!used_area.contains(cv::Point(p[1],p[0]))) {
                            used_area = used_area | cv::Rect(p[1],p[0],1,1);
                        }
                    }
                    
#pragma omp critical
                    {
                        fringe.push_back(p);
                        succ_gen_ps.push_back(p);
                    }
                }
            }  // end parallel iteration over cands
        }

        // If there are now no fringe points, reduce the required compactness when considering cands for the next generation
        if (!fringe.size() && curr_ref_min > 2) {
            curr_ref_min--;
            std::cout << used_area << "\n";
            for(int j=used_area.y;j<used_area.br().y;j++)
                for(int i=used_area.x;i<used_area.br().x;i++) {
                    if (state(j, i) & STATE_LOC_VALID)
                        fringe.push_back({j,i});
                }
            std::cout << "new limit " << curr_ref_min << " " << fringe.size() << "\n";
        }
        else if (fringe.size())
            curr_ref_min = ref_max;

        for(const auto& p: fringe)
            if (locs(p)[0] == -1)
                std::cout << "impossible! " << p << " " << cv::Vec2i(y0,x0) << "\n";

        if (generation >= 3) {
            options_big.max_num_iterations = 10;
        }

        //this actually did work (but was slow ...)
        if (phys_fail_count_gen) {
            options_big.minimizer_progress_to_stdout = true;
            options_big.max_num_iterations = 100;
        }
        else
            options_big.minimizer_progress_to_stdout = false;

        if (!global_opt) {
            // For late generations, instead of re-solving the global problem, solve many local-ish problems, around each
            // of the newly added points
            std::vector<cv::Vec2i> opt_local;
            for(auto p : succ_gen_ps)
                if (p[0] % 4 == 0 && p[1] % 4 == 0)
                    opt_local.push_back(p);

            if (opt_local.size()) {
                OmpThreadPointCol opt_local_threadcol(17, opt_local);

#pragma omp parallel
                while (true)
                {
                    CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor);
                    cv::Vec2i p = opt_local_threadcol.next();
                    if (p[0] == -1)
                        break;

                    local_optimization(8, p, state, locs, interp, proc_tensor, Ts, true);
                }
            }
        }
        else {
            // For early generations, re-solve the big problem, jointly optimising the locations of all points in the patch
            std::cout << "running big solve" << "\n";
            ceres::Solve(options_big, &big_problem, &big_summary);
            std::cout << big_summary.BriefReport() << "\n";
            std::cout << "avg err: " << sqrt(big_summary.final_cost/big_summary.num_residual_blocks) << "\n";
        }

        if (generation > 10 && global_opt) {
            // Beyond 10 generations but while still trying global re-solves, simplify the big problem by fixing locations
            // of points that are already 'certain', in the sense they are not near any other points that don't yet have valid
            // locations
            cv::Mat_<cv::Vec2d> _empty;
            freeze_inner_params(big_problem, 10, state, locs, _empty, loss_status, STATE_LOC_VALID | STATE_COORD_VALID);
        }

        cands.resize(0);

        // Record the cost of the current patch, by re-evaluating all losses within the patch bbox region
        cv::Mat_<cv::Vec3d> locs_crop = locs(used_area);
        cv::Mat_<uint8_t> state_crop = state(used_area);
        double max_cost = 0;
        double avg_cost = 0;
        int cost_count = 0;
        for(int j=0;j<locs_crop.rows;j++)
            for(int i=0;i<locs_crop.cols;i++) {
                ceres::Problem problem;
                emptytrace_create_centered_losses(problem, {j,i}, state_crop, locs_crop, interp_global, proc_tensor, Ts);
                double cost = 0.0;
                problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
                max_cost = std::max(max_cost, cost);
                avg_cost += cost;
                cost_count++;
            }
        gen_avg_cost.push_back(avg_cost/cost_count);
        gen_max_cost.push_back(max_cost);

        float const current_area_vx2 = double(succ)*step*step;
        float const current_area_cm2 = current_area_vx2 * voxelsize * voxelsize / 1e8;
        printf("-> total done %d/ fringe: %ld surf: %fG vx^2 (%f cm^2)\n", succ, (long)fringe.size(), current_area_vx2/1e9, current_area_cm2);

        timer_gen.unit = succ_gen*step*step;
        timer_gen.unit_string = "vx^2";
        print_accessor_stats();

    }  // end while fringe is non-empty
    delete timer;

    locs = locs(used_area);
    state = state(used_area);

    double max_cost = 0;
    double avg_cost = 0;
    int count = 0;
    for(int j=0;j<locs.rows;j++)
        for(int i=0;i<locs.cols;i++) {
            ceres::Problem problem;
            emptytrace_create_centered_losses(problem, {j,i}, state, locs, interp_global, proc_tensor, Ts);
            double cost = 0.0;
            problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
            max_cost = std::max(max_cost, cost);
            avg_cost += cost;
            count++;
        }
    avg_cost /= count;

    float const area_est_vx2 = succ*step*step;
    float const area_est_cm2 = area_est_vx2 * voxelsize * voxelsize / 1e8;
    printf("generated approximate surface %f vx^2 (%f cm^2)\n", area_est_vx2, area_est_cm2);

    QuadSurface *surf = new QuadSurface(locs, {1/T, 1/T});

    surf->meta = new nlohmann::json;
    (*surf->meta)["area_vx2"] = area_est_vx2;
    (*surf->meta)["area_cm2"] = area_est_cm2;
    (*surf->meta)["max_cost"] = max_cost;
    (*surf->meta)["avg_cost"] = avg_cost;
    (*surf->meta)["max_gen"] = generation;
    (*surf->meta)["gen_avg_cost"] = gen_avg_cost;
    (*surf->meta)["gen_max_cost"] = gen_max_cost;
    (*surf->meta)["seed"] = {origin[0],origin[1],origin[2]};
    (*surf->meta)["elapsed_time_s"] = f_timer.seconds();

    return surf;
}

using SurfPoint = std::pair<QuadSurface*,cv::Vec2i>;

class resId_t
{
public:
    resId_t() {};
    resId_t(int type, QuadSurface* qs, const cv::Vec2i& p) : _type(type), _sm(qs), _p(p) {};
    resId_t(int type, QuadSurface* qs, const cv::Vec2i &a, const cv::Vec2i &b) : _type(type), _sm(qs)
    {
        if (a[0] == b[0]) {
            if (a[1] <= b[1])
                _p = a;
            else
                _p = b;
        }
        else if (a[0] < b[0])
            _p = a;
        else
            _p = b;

    }
    bool operator==(const resId_t &o) const
    {
        if (_type != o._type)
            return false;
        if (_sm != o._sm)
            return false;
        if (_p != o._p)
            return false;
        return true;
    }

    int _type;
    QuadSurface* _sm;
    cv::Vec2i _p;
};

struct resId_hash {
    size_t operator()(resId_t id) const
    {
        size_t hash1 = std::hash<int>{}(id._type);
        size_t hash2 = std::hash<void*>{}(id._sm);
        size_t hash3 = std::hash<int>{}(id._p[0]);
        size_t hash4 = std::hash<int>{}(id._p[1]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash =  hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
        hash =  hash  ^ (hash4 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};


struct SurfPoint_hash {
    size_t operator()(SurfPoint p) const
    {
        size_t hash1 = std::hash<void*>{}(p.first);
        size_t hash2 = std::hash<int>{}(p.second[0]);
        size_t hash3 = std::hash<int>{}(p.second[1]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash =  hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};

//Surface tracking data for loss functions
class SurfTrackerData
{

public:
    cv::Vec2d &loc(QuadSurface *qs, const cv::Vec2i &loc)
    {
        return _data[{qs,loc}];
    }
    ceres::ResidualBlockId &resId(const resId_t &id)
    {
        return _res_blocks[id];
    }
    bool hasResId(const resId_t &id)
    {
        // std::cout << "check hasResId " << id._sm << " " << id._type << " " << id._p << "\n";
        return _res_blocks.count(id);
    }
    bool has(QuadSurface *qs, const cv::Vec2i &loc)
    {
        return _data.count({qs,loc});
    }
    void erase(QuadSurface *qs, const cv::Vec2i &loc)
    {
        _data.erase({qs,loc});
    }
    void eraseSurf(QuadSurface *qs, const cv::Vec2i &loc)
    {
        _surfs[loc].erase(qs);
    }
    std::set<QuadSurface*> &surfs(const cv::Vec2i &loc)
    {
        return _surfs[loc];
    }
    const std::set<QuadSurface*> &surfsC(const cv::Vec2i &loc) const
    {
        if (!_surfs.count(loc))
            return _emptysurfs;
        else
            return _surfs.find(loc)->second;
    }
    cv::Vec3d lookup_int(QuadSurface *qs, const cv::Vec2i &p)
    {
        auto id = std::make_pair(qs,p);
        if (!_data.count(id))
            throw std::runtime_error("error, lookup failed!");
        cv::Vec2d l = loc(qs, p);
        if (l[0] == -1)
            return {-1,-1,-1};
        else {
            cv::Rect bounds = {0, 0, qs->rawPoints().rows-2,qs->rawPoints().cols-2};
            cv::Vec2i li = {floor(l[0]),floor(l[1])};
            if (bounds.contains(cv::Point(li)))
                return at_int_inv(qs->rawPoints(), l);
            else
                return {-1,-1,-1};
        }
    }
    bool valid_int(QuadSurface *qs, const cv::Vec2i &p)
    {
        auto id = std::make_pair(qs,p);
        if (!_data.count(id))
            return false;
        cv::Vec2d l = loc(qs, p);
        if (l[0] == -1)
            return false;
        else {
            cv::Rect bounds = {0, 0, qs->rawPoints().rows-2,qs->rawPoints().cols-2};
            cv::Vec2i li = {floor(l[0]),floor(l[1])};
            if (bounds.contains(cv::Point(li)))
            {
                if (qs->rawPoints()(li[0],li[1])[0] == -1)
                    return false;
                if (qs->rawPoints()(li[0]+1,li[1])[0] == -1)
                    return false;
                if (qs->rawPoints()(li[0],li[1]+1)[0] == -1)
                    return false;
                if (qs->rawPoints()(li[0]+1,li[1]+1)[0] == -1)
                    return false;
                return true;
            }
            else
                return false;
        }
    }
    cv::Vec3d lookup_int_loc(QuadSurface *qs, const cv::Vec2f &l)
    {
        if (l[0] == -1)
            return {-1,-1,-1};
        else {
            cv::Rect bounds = {0, 0, qs->rawPoints().rows-2,qs->rawPoints().cols-2};
            if (bounds.contains(cv::Point(l)))
                return at_int_inv(qs->rawPoints(), l);
            else
                return {-1,-1,-1};
        }
    }
    void flip_x(int x0)
    {
        std::cout << " src sizes " << _data.size() << " " << _surfs.size() << "\n";
        SurfTrackerData old = *this;
        _data.clear();
        _res_blocks.clear();
        _surfs.clear();
        
        for(auto &it : old._data)
            _data[{it.first.first,{it.first.second[0],x0+x0-it.first.second[1]}}] = it.second;
        
        for(auto &it : old._surfs)
            _surfs[{it.first[0],x0+x0-it.first[1]}] = it.second;
        
        std::cout << " flipped sizes " << _data.size() << " " << _surfs.size() << "\n";
    }
// protected:
    std::unordered_map<SurfPoint,cv::Vec2d,SurfPoint_hash> _data;
    std::unordered_map<resId_t,ceres::ResidualBlockId,resId_hash> _res_blocks;
    std::unordered_map<cv::Vec2i,std::set<QuadSurface*>,vec2i_hash> _surfs;
    std::set<QuadSurface*> _emptysurfs;
    cv::Vec3d seed_coord;
    cv::Vec2i seed_loc;
};

static void copy(const SurfTrackerData &src, SurfTrackerData &tgt, const cv::Rect &roi_)
{
    cv::Rect roi(roi_.y,roi_.x,roi_.height,roi_.width);
    
    {
        auto it = tgt._data.begin();
        while (it != tgt._data.end()) {
            if (roi.contains(cv::Point(it->first.second)))
                it = tgt._data.erase(it);
            else
                it++;
        }
    }
    
    {
        auto it = tgt._surfs.begin();
        while (it != tgt._surfs.end()) {
            if (roi.contains(cv::Point(it->first)))
                it = tgt._surfs.erase(it);
            else
                it++;
        }
    }
    
    for(auto &it : src._data)
        if (roi.contains(cv::Point(it.first.second)))
            tgt._data[it.first] = it.second;
    for(auto &it : src._surfs)
        if (roi.contains(cv::Point(it.first)))
            tgt._surfs[it.first] = it.second;
    
    // tgt.seed_loc = src.seed_loc;
    // tgt.seed_coord = src.seed_coord;
}

static int add_surftrack_distloss(QuadSurface *qs, const cv::Vec2i &p, const cv::Vec2i &off, SurfTrackerData &data,
    ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0, ceres::ResidualBlockId *res = nullptr, float w = 1.0)
{
    if ((state(p) & STATE_LOC_VALID) == 0 || !data.has(qs, p))
        return 0;
    if ((state(p+off) & STATE_LOC_VALID) == 0 || !data.has(qs, p+off))
        return 0;

    // Use the global parameter if w is default value (1.0), otherwise use the provided value
    float weight = (w == 1.0f) ? dist_loss_2d_w : w;
    ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss2D::Create(unit*cv::norm(off), weight), nullptr, &data.loc(qs, p)[0], &data.loc(qs, p+off)[0]);
    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0)
        problem.SetParameterBlockConstant(&data.loc(qs, p+off)[0]);

    return 1;
}


static int add_surftrack_distloss_3D(cv::Mat_<cv::Vec3d> &points, const cv::Vec2i &p, const cv::Vec2i &off, ceres::Problem &problem,
    const cv::Mat_<uint8_t> &state, float unit, int flags = 0, ceres::ResidualBlockId *res = nullptr, float w = 2.0)
{
    if ((state(p) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;
    if ((state(p+off) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;

    // Use the global parameter if w is default value (2.0), otherwise use the provided value
    float weight = (w == 2.0f) ? dist_loss_3d_w : w;
    ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss::Create(unit*cv::norm(off), weight), nullptr, &points(p)[0], &points(p+off)[0]);

    // std::cout << cv::norm(points(p)-points(p+off)) << " tgt " << unit << points(p) << points(p+off) << "\n";
    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0)
        problem.SetParameterBlockConstant(&points(p+off)[0]);

    return 1;
}

static int cond_surftrack_distloss_3D(int type, QuadSurface *qs, cv::Mat_<cv::Vec3d> &points, const cv::Vec2i &p, const cv::Vec2i &off,
    SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0)
{
    resId_t id(type, qs, p, p+off);
    if (data.hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = add_surftrack_distloss_3D(points, p, off, problem, state, unit, flags, &res);

    data.resId(id) = res;

    return count;
}

static int cond_surftrack_distloss(int type, QuadSurface *qs, const cv::Vec2i &p, const cv::Vec2i &off, SurfTrackerData &data,
    ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0)
{
    resId_t id(type, qs, p, p+off);
    if (data.hasResId(id))
        return 0;

    add_surftrack_distloss(qs, p, off, data, problem, state, unit, flags, &data.resId(id));

    return 1;
}

static int add_surftrack_straightloss(QuadSurface *qs, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3,
    SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, int flags = 0, float w = 0.7f)
{
    if ((state(p+o1) & STATE_LOC_VALID) == 0 || !data.has(qs, p+o1))
        return 0;
    if ((state(p+o2) & STATE_LOC_VALID) == 0 || !data.has(qs, p+o2))
        return 0;
    if ((state(p+o3) & STATE_LOC_VALID) == 0 || !data.has(qs, p+o3))
        return 0;

    // Always use the global straight_weight for 2D
    w = straight_weight;

    // std::cout << "add straight " << qs << p << o1 << o2 << o3 << "\n";
    problem.AddResidualBlock(StraightLoss2D::Create(w), nullptr, &data.loc(qs, p+o1)[0], &data.loc(qs, p+o2)[0], &data.loc(qs, p+o3)[0]);

    if ((flags & OPTIMIZE_ALL) == 0) {
        if (o1 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&data.loc(qs, p+o1)[0]);
        if (o2 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&data.loc(qs, p+o2)[0]);
        if (o3 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&data.loc(qs, p+o3)[0]);
    }

    return 1;
}

static int add_surftrack_straightloss_3D(const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3, cv::Mat_<cv::Vec3d> &points,
    ceres::Problem &problem, const cv::Mat_<uint8_t> &state, int flags = 0, ceres::ResidualBlockId *res = nullptr, float w = 4.0f)
{
    if ((state(p+o1) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    if ((state(p+o2) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    if ((state(p+o3) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;

    // Always use the global straight_weight_3D for 3D
    w = straight_weight_3D;

    // std::cout << "add straight " << qs << p << o1 << o2 << o3 << "\n";
    ceres::ResidualBlockId tmp =
    problem.AddResidualBlock(StraightLoss::Create(w), nullptr, &points(p+o1)[0], &points(p+o2)[0], &points(p+o3)[0]);
    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0) {
        if (o1 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&points(p+o1)[0]);
        if (o2 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&points(p+o2)[0]);
        if (o3 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&points(p+o3)[0]);
    }

    return 1;
}

static int cond_surftrack_straightloss_3D(int type, QuadSurface *qs, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3,
    cv::Mat_<cv::Vec3d> &points, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, int flags = 0)
{
    resId_t id(type, qs, p);
    if (data.hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = add_surftrack_straightloss_3D(p, o1, o2, o3, points ,problem, state, flags, &res);

    if (count)
        data.resId(id) = res;

    return count;
}

static int add_surftrack_surfloss(QuadSurface *qs, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &points, float step, ceres::ResidualBlockId *res = nullptr, float w = 0.1)
{
    if ((state(p) & STATE_LOC_VALID) == 0 || !data.valid_int(qs, p))
        return 0;

    ceres::ResidualBlockId tmp;
    tmp = problem.AddResidualBlock(SurfaceLossD::Create(qs->rawPoints(), w), nullptr, &points(p)[0], &data.loc(qs, p)[0]);
    
    if (res)
        *res = tmp;

    return 1;
}

//gen straigt loss given point and 3 offsets
static int cond_surftrack_surfloss(int type, QuadSurface *qs, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem,
    const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, float step)
{
    resId_t id(type, qs, p);
    if (data.hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = add_surftrack_surfloss(qs, p, data, problem, state, points, step, &res);

    if (count)
        data.resId(id) = res;

    return count;
}

//will optimize only the center point
static int surftrack_add_local(QuadSurface *qs, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem,
    const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, float step, float src_step, int flags = 0, int *straigh_count_ptr = nullptr)
{
    int count = 0;
    int count_straight = 0;
    //direct
    if (flags & LOSS_3D_INDIRECT) {
        // h
        count += add_surftrack_distloss_3D(points, p, {0,1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {1,0}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {0,-1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {-1,0}, problem, state, step*src_step, flags);

        //v
        count += add_surftrack_distloss_3D(points, p, {1,1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {1,-1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {-1,1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {-1,-1}, problem, state, step*src_step, flags);

        //horizontal
        count_straight += add_surftrack_straightloss_3D(p, {0,-2},{0,-1},{0,0}, points, problem, state);
        count_straight += add_surftrack_straightloss_3D(p, {0,-1},{0,0},{0,1}, points, problem, state);
        count_straight += add_surftrack_straightloss_3D(p, {0,0},{0,1},{0,2}, points, problem, state);

        //vertical
        count_straight += add_surftrack_straightloss_3D(p, {-2,0},{-1,0},{0,0}, points, problem, state);
        count_straight += add_surftrack_straightloss_3D(p, {-1,0},{0,0},{1,0}, points, problem, state);
        count_straight += add_surftrack_straightloss_3D(p, {0,0},{1,0},{2,0}, points, problem, state);
    }
    else {
        count += add_surftrack_distloss(qs, p, {0,1}, data, problem, state, step);
        count += add_surftrack_distloss(qs, p, {1,0}, data, problem, state, step);
        count += add_surftrack_distloss(qs, p, {0,-1}, data, problem, state, step);
        count += add_surftrack_distloss(qs, p, {-1,0}, data, problem, state, step);

        //diagonal
        count += add_surftrack_distloss(qs, p, {1,1}, data, problem, state, step);
        count += add_surftrack_distloss(qs, p, {1,-1}, data, problem, state, step);
        count += add_surftrack_distloss(qs, p, {-1,1}, data, problem, state, step);
        count += add_surftrack_distloss(qs, p, {-1,-1}, data, problem, state, step);

        //horizontal
        count_straight += add_surftrack_straightloss(qs, p, {0,-2},{0,-1},{0,0}, data, problem, state);
        count_straight += add_surftrack_straightloss(qs, p, {0,-1},{0,0},{0,1}, data, problem, state);
        count_straight += add_surftrack_straightloss(qs, p, {0,0},{0,1},{0,2}, data, problem, state);

        //vertical
        count_straight += add_surftrack_straightloss(qs, p, {-2,0},{-1,0},{0,0}, data, problem, state);
        count_straight += add_surftrack_straightloss(qs, p, {-1,0},{0,0},{1,0}, data, problem, state);
        count_straight += add_surftrack_straightloss(qs, p, {0,0},{1,0},{2,0}, data, problem, state);
    }

    if (flags & LOSS_ZLOC)
        problem.AddResidualBlock(ZLocationLoss<cv::Vec3f>::Create(
            qs->rawPoints(),
            data.seed_coord[2] - (p[0]-data.seed_loc[0])*step*src_step, z_loc_loss_w), 
            new ceres::HuberLoss(1.0), &data.loc(qs, p)[0]);

    if (flags & SURF_LOSS) {
        count += add_surftrack_surfloss(qs, p, data, problem, state, points, step);
    }

    if (straigh_count_ptr)
        *straigh_count_ptr += count_straight;

    return count + count_straight;
}

//will optimize only the center point
static int surftrack_add_global(QuadSurface *qs, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &points, float step, int flags = 0, float step_onsurf = 0)
{
    if ((state(p) & (STATE_LOC_VALID | STATE_COORD_VALID)) == 0)
        return 0;

    int count = 0;
    //losses are defind in 3D
    if (flags & LOSS_3D_INDIRECT) {
        // h
        count += cond_surftrack_distloss_3D(0, qs, points, p, {0,1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(0, qs, points, p, {1,0}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(1, qs, points, p, {0,-1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(1, qs, points, p, {-1,0}, data, problem, state, step, flags);

        //v
        count += cond_surftrack_distloss_3D(2, qs, points, p, {1,1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(2, qs, points, p, {1,-1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(3, qs, points, p, {-1,1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(3, qs, points, p, {-1,-1}, data, problem, state, step, flags);

        //horizontal
        count += cond_surftrack_straightloss_3D(4, qs, p, {0,-2},{0,-1},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(4, qs, p, {0,-1},{0,0},{0,1}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(4, qs, p, {0,0},{0,1},{0,2}, points, data, problem, state, flags);

        //vertical
        count += cond_surftrack_straightloss_3D(5, qs, p, {-2,0},{-1,0},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(5, qs, p, {-1,0},{0,0},{1,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(5, qs, p, {0,0},{1,0},{2,0}, points, data, problem, state, flags);
        
        //dia1
        count += cond_surftrack_straightloss_3D(6, qs, p, {-2,-2},{-1,-1},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(6, qs, p, {-1,-1},{0,0},{1,1}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(6, qs, p, {0,0},{1,1},{2,2}, points, data, problem, state, flags);
        
        //dia1
        count += cond_surftrack_straightloss_3D(7, qs, p, {-2,2},{-1,1},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(7, qs, p, {-1,1},{0,0},{1,-1}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(7, qs, p, {0,0},{1,-1},{2,-2}, points, data, problem, state, flags);
    }
    
    //losses on surface
    if (flags & LOSS_ON_SURF)
    {
        if (step_onsurf == 0)
            throw std::runtime_error("oops step_onsurf == 0");
        
        //direct
        count += cond_surftrack_distloss(8, qs, p, {0,1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(8, qs, p, {1,0}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(9, qs, p, {0,-1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(9, qs, p, {-1,0}, data, problem, state, step_onsurf);
        
        //diagonal
        count += cond_surftrack_distloss(10, qs, p, {1,1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(10, qs, p, {1,-1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(11, qs, p, {-1,1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(11, qs, p, {-1,-1}, data, problem, state, step_onsurf);
    }

    if (flags & SURF_LOSS && state(p) & STATE_LOC_VALID)
        count += cond_surftrack_surfloss(14, qs, p, data, problem, state, points, step);

    return count;
}

static double local_cost_destructive(QuadSurface *qs, const cv::Vec2i& p, SurfTrackerData &data, cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &points, float step, float src_step, cv::Vec3f loc, int *ref_count = nullptr, int *straight_count_ptr = nullptr)
{
    uint8_t state_old = state(p);
    state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
    int count;
    int straigh_count;
    if (!straight_count_ptr)
        straight_count_ptr = &straigh_count;
    
        double test_loss = 0.0;
    {
        ceres::Problem problem_test;

        data.loc(qs, p) = {loc[1], loc[0]};

        count = surftrack_add_local(qs, p, data, problem_test, state, points, step, src_step, 0, straight_count_ptr);
        if (ref_count)
            *ref_count = count;

        problem_test.Evaluate(ceres::Problem::EvaluateOptions(), &test_loss, nullptr, nullptr, nullptr);
    } //destroy problme before data
    data.erase(qs, p);
    state(p) = state_old;

    if (!count)
        return 0;
    else
        return sqrt(test_loss/count);
}


static double local_cost(QuadSurface *qs, const cv::Vec2i& p, SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points,
    float step, float src_step, int *ref_count = nullptr, int *straight_count_ptr = nullptr)
{
    int count;
    int straigh_count;
    if (!straight_count_ptr)
        straight_count_ptr = &straigh_count;
    
        double test_loss = 0.0;
    ceres::Problem problem_test;

    count = surftrack_add_local(qs, p, data, problem_test, state, points, step, src_step, 0, straight_count_ptr);
    if (ref_count)
        *ref_count = count;

    problem_test.Evaluate(ceres::Problem::EvaluateOptions(), &test_loss, nullptr, nullptr, nullptr);

    if (!count)
        return 0;
    else
        return sqrt(test_loss/count);
}

static double local_solve(QuadSurface *qs, const cv::Vec2i& p, SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points,
    float step, float src_step, int flags)
{
    ceres::Problem problem;

    surftrack_add_local(qs, p, data, problem, state, points, step, src_step, flags);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (summary.num_residual_blocks < 3)
        return 10000;

    return summary.final_cost/summary.num_residual_blocks;
}


static cv::Mat_<cv::Vec3d> surftrack_genpoints_hr(SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, cv::Rect &used_area,
    float step, float step_src, bool inpaint = false)
{
    cv::Mat_<cv::Vec3f> points_hr(state.rows*step, state.cols*step, {0,0,0});
    cv::Mat_<int> counts_hr(state.rows*step, state.cols*step, 0);
#pragma omp parallel for //FIXME data access is just not threading friendly ...
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++) {
            if (state(j,i) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j,i+1) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j+1,i) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j+1,i+1) & (STATE_LOC_VALID|STATE_COORD_VALID))
            {
            for(auto &qs : data.surfsC({j,i})) {
                if (data.valid_int(qs,{j,i})
                    && data.valid_int(qs,{j,i+1})
                    && data.valid_int(qs,{j+1,i})
                    && data.valid_int(qs,{j+1,i+1}))
                {
                    cv::Vec2f l00 = data.loc(qs,{j,i});
                    cv::Vec2f l01 = data.loc(qs,{j,i+1});
                    cv::Vec2f l10 = data.loc(qs,{j+1,i});
                    cv::Vec2f l11 = data.loc(qs,{j+1,i+1});

                    for(int sy=0;sy<=step;sy++)
                        for(int sx=0;sx<=step;sx++) {
                            float fx = sx/step;
                            float fy = sy/step;
                            cv::Vec2f l0 = (1-fx)*l00 + fx*l01;
                            cv::Vec2f l1 = (1-fx)*l10 + fx*l11;
                            cv::Vec2f l = (1-fy)*l0 + fy*l1;
                            if (loc_valid(qs->rawPoints(), l)) {
                                points_hr(j*step+sy,i*step+sx) += data.lookup_int_loc(qs,l);
                                counts_hr(j*step+sy,i*step+sx) += 1;
                            }
                        }
                }
            }
            if (!counts_hr(j*step+1,i*step+1) && inpaint) {
                const cv::Vec3d& c00 = points(j,i);
                const cv::Vec3d& c01 = points(j,i+1);
                const cv::Vec3d& c10 = points(j+1,i);
                const cv::Vec3d& c11 = points(j+1,i+1);
            
                for(int sy=0;sy<=step;sy++)
                    for(int sx=0;sx<=step;sx++) {
                        if (!counts_hr(j*step+sy,i*step+sx)) {
                            float fx = sx/step;
                            float fy = sy/step;
                            cv::Vec3d c0 = (1-fx)*c00 + fx*c01;
                            cv::Vec3d c1 = (1-fx)*c10 + fx*c11;
                            cv::Vec3d c = (1-fy)*c0 + fy*c1;
                            points_hr(j*step+sy,i*step+sx) = c;
                            counts_hr(j*step+sy,i*step+sx) = 1;
                        }
                    }
            }
        }
    }
#pragma omp parallel for
    for(int j=0;j<points_hr.rows;j++)
        for(int i=0;i<points_hr.cols;i++)
            if (counts_hr(j,i))
                points_hr(j,i) /= counts_hr(j,i);
            else
                points_hr(j,i) = {-1,-1,-1};

    return points_hr;
}


int static dbg_counter = 0;
// Default values for thresholds Will be configurable through JSON
float local_cost_inl_th = 0.2;
float same_surface_th = 2.0;
float straight_weight = 0.7f;       // Weight for 2D straight line constraints
float straight_weight_3D = 4.0f;    // Weight for 3D straight line constraints
float sliding_w_scale = 1.0f;       // Scale factor for sliding window
float z_loc_loss_w = 0.1f;          // Weight for Z location loss constraints
float dist_loss_2d_w = 1.0f;        // Weight for 2D distance constraints
float dist_loss_3d_w = 2.0f;        // Weight for 3D distance constraints
float straight_min_count = 1.0f;    // Minimum number of straight constraints
int inlier_base_threshold = 20;     // Starting threshold for inliers

//try flattening the current surface mapping assuming direct 3d distances
//this is basically just a reparametrization
static void optimize_surface_mapping(SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, cv::Rect used_area,
    cv::Rect static_bounds, float step, float src_step, const cv::Vec2i &seed, int closing_r, bool keep_inpainted = false, 
    const std::filesystem::path& tgt_dir = std::filesystem::path())
{
    std::cout << "optimizer: optimizing surface " << state.size() << " " << used_area <<  " " << static_bounds << "\n";

    cv::Mat_<cv::Vec3d> points_new = points.clone();
    QuadSurface* qs = new QuadSurface(points, {1,1});
    
    std::shared_mutex mutex;
    
    SurfTrackerData data_new;
    data_new._data = data._data;
    
    used_area = cv::Rect(used_area.x-2,used_area.y-2,used_area.size().width+4,used_area.size().height+4);
    cv::Rect used_area_hr = {used_area.x*step, used_area.y*step, used_area.width*step, used_area.height*step};
    
    ceres::Problem problem_inpaint;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
#ifdef VC_USE_CUDA_SPARSE
    // Check if Ceres was actually built with CUDA sparse support
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;

        // Enable mixed precision for SPARSE_SCHUR
        if (options.linear_solver_type == ceres::SPARSE_SCHUR) {
            options.use_mixed_precision_solves = true;
        }
    } else {
        std::cerr << "Warning: CUDA_SPARSE requested but Ceres was not built with CUDA sparse support. Falling back to default solver." << "\n";
    }
#endif
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.num_threads = omp_get_max_threads();
    options.use_nonmonotonic_steps = true;

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (state(j,i) & STATE_LOC_VALID) {
                data_new.surfs({j,i}).insert(qs);
                data_new.loc(qs, {j,i}) = {j,i};
            }
            
    cv::Mat_<uint8_t> new_state = state.clone();
        
    //generate closed version of state
    cv::Mat m = cv::getStructuringElement(cv::MORPH_RECT, {3,3});
    
    uint8_t STATE_VALID = STATE_LOC_VALID | STATE_COORD_VALID;
    
    int res_count = 0;
    //slowly inpaint physics only points
    for(int r=0;r<closing_r+2;r++) {
        cv::Mat_<uint8_t> masked;
        bitwise_and(state, STATE_VALID, masked);
        cv::dilate(masked, masked, m, {-1,-1}, r);
        cv::erode(masked, masked, m, {-1,-1}, std::min(r,closing_r));
        // cv::imwrite("masked.tif", masked);
        
        for(int j=used_area.y;j<used_area.br().y;j++)
            for(int i=used_area.x;i<used_area.br().x;i++)
                if ((masked(j,i) & STATE_VALID) && (~new_state(j,i) & STATE_VALID)) {
                    new_state(j, i) = STATE_COORD_VALID;
                    points_new(j,i) = {-3,-2,-4};
                    //TODO add local area solve
                    //double err = local_solve(qs, {j,i}, data_new, new_state, points_new, step, src_step, LOSS_3D_INDIRECT | SURF_LOSS);
                    if (points_new(j,i)[0] == -3) {
                        //FIXME actually check for solver failure?
                        new_state(j, i) = 0;
                        points_new(j,i) = {-1,-1,-1};
                    }
                    else
                        res_count += surftrack_add_global(qs, {j,i}, data_new, problem_inpaint, new_state, points_new, step*src_step, LOSS_3D_INDIRECT | OPTIMIZE_ALL);
                }
    }
    
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (state(j,i) & STATE_LOC_VALID)
                if (problem_inpaint.HasParameterBlock(&points_new(j,i)[0]))
                    problem_inpaint.SetParameterBlockConstant(&points_new(j,i)[0]);
    
    ceres::Solve(options, &problem_inpaint, &summary);
    std::cout << summary.BriefReport() << "\n";
    
    cv::Mat_<cv::Vec3d> points_inpainted = points_new.clone();
    
    //TODO we could directly use higher res here?
    QuadSurface* sm_inp = new QuadSurface(points_inpainted, {1,1});
    
    SurfTrackerData data_inp;
    data_inp._data = data_new._data;
    
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (new_state(j,i) & STATE_LOC_VALID) {
                data_inp.surfs({j,i}).insert(sm_inp);
                data_inp.loc(sm_inp, {j,i}) = {j,i};
            }
            
    ceres::Problem problem;
        
    std::cout << "optimizer: using " << used_area.tl() << used_area.br() << "\n";
        
    int fix_points = 0;
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++) {
            res_count += surftrack_add_global(sm_inp, {j,i}, data_inp, problem, new_state, points_new, step*src_step, LOSS_3D_INDIRECT | SURF_LOSS | OPTIMIZE_ALL);
            fix_points++;
            if (problem.HasParameterBlock(&data_inp.loc(sm_inp, {j,i})[0]))
                problem.AddResidualBlock(LinChkDistLoss::Create(data_inp.loc(sm_inp, {j,i}), 1.0), nullptr, &data_inp.loc(sm_inp, {j,i})[0]);
        }
        
    std::cout << "optimizer: num fix points " << fix_points << "\n";
            
    data_inp.seed_loc = seed;
    data_inp.seed_coord = points_new(seed);

    int fix_points_z = 0;
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++) {
            fix_points_z++;
            if (problem.HasParameterBlock(&data_inp.loc(sm_inp, {j,i})[0]))
                problem.AddResidualBlock(ZLocationLoss<cv::Vec3d>::Create(points_new, data_inp.seed_coord[2] - (j-data.seed_loc[0])*step*src_step, z_loc_loss_w), new ceres::HuberLoss(1.0), &data_inp.loc(sm_inp, {j,i})[0]);
        }
        
    std::cout << "optimizer: optimizing " << res_count << " residuals, seed " << seed << "\n";
            
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (static_bounds.contains(cv::Point(i,j))) {
                if (problem.HasParameterBlock(&data_inp.loc(sm_inp, {j,i})[0]))
                    problem.SetParameterBlockConstant(&data_inp.loc(sm_inp, {j,i})[0]);
                if (problem.HasParameterBlock(&points_new(j, i)[0]))
                    problem.SetParameterBlockConstant(&points_new(j, i)[0]);
            }
    
    options.max_num_iterations = 1000;
    options.use_nonmonotonic_steps = true;
    options.use_inner_iterations = true;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    std::cout << "optimizer: rms " << sqrt(summary.final_cost/summary.num_residual_blocks) << " count " << summary.num_residual_blocks << "\n";
    
    {
        cv::Mat_<cv::Vec3d> points_hr_inp = surftrack_genpoints_hr(data, new_state, points_inpainted, used_area, step, src_step, true);
        try {
            QuadSurface *dbg_surf = new QuadSurface(points_hr_inp(used_area_hr), {1/src_step,1/src_step});
            std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str()+"_inp_hr";
            dbg_surf->save(tgt_dir / uuid, uuid);
            delete dbg_surf;
        } catch (cv::Exception) {
            // We did not find a valid region of interest to expand to
            std::cout << "optimizer: no valid region of interest found" << "\n";
        }        
    }
            
    cv::Mat_<cv::Vec3d> points_hr = surftrack_genpoints_hr(data, new_state, points_inpainted, used_area, step, src_step);
    SurfTrackerData data_out;
    cv::Mat_<cv::Vec3d> points_out(points.size(), {-1,-1,-1});
    cv::Mat_<uint8_t> state_out(state.size(), 0);
    cv::Mat_<uint8_t> support_count(state.size(), 0);
#pragma omp parallel for
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (static_bounds.contains(cv::Point(i,j))) {
                points_out(j, i) = points(j, i);
                state_out(j, i) = state(j, i);
                //FIXME copy surfs and locs
                mutex.lock();
                data_out.surfs({j,i}) = data.surfsC({j,i});
                for(auto &s : data_out.surfs({j,i}))
                    data_out.loc(s, {j,i}) = data.loc(s, {j,i});
                mutex.unlock();
            }
            else if (new_state(j,i) & STATE_VALID) {
                cv::Vec2d l = data_inp.loc(sm_inp ,{j,i});
                int y = l[0];
                int x = l[1];
                l *= step;
                if (loc_valid(points_hr, l)) {
                    // mutex.unlock();
                    int src_loc_valid_count = 0;
                    if (state(y,x) & STATE_LOC_VALID)
                        src_loc_valid_count++;
                    if (state(y,x+1) & STATE_LOC_VALID)
                        src_loc_valid_count++;
                    if (state(y+1,x) & STATE_LOC_VALID)
                        src_loc_valid_count++;
                    if (state(y+1,x+1) & STATE_LOC_VALID)
                        src_loc_valid_count++;
                    
                    support_count(j,i) = src_loc_valid_count;
    
                    points_out(j, i) = interp_lin_2d(points_hr, l);
                    state_out(j, i) = STATE_LOC_VALID | STATE_COORD_VALID;
                    
                    std::set<QuadSurface*> surfs;
                    surfs.insert(data.surfsC({y,x}).begin(), data.surfsC({y,x}).end());
                    surfs.insert(data.surfsC({y,x+1}).begin(), data.surfsC({y,x+1}).end());
                    surfs.insert(data.surfsC({y+1,x}).begin(), data.surfsC({y+1,x}).end());
                    surfs.insert(data.surfsC({y+1,x+1}).begin(), data.surfsC({y+1,x+1}).end());
                    
                    for(auto &s : surfs) {
                        auto ptr = s->pointer();
                        float res = s->pointTo(ptr, points_out(j, i), same_surface_th, 10);
                        if (res <= same_surface_th) {
                            mutex.lock();
                            data_out.surfs({j,i}).insert(s);
                            cv::Vec3f loc = s->loc_raw(ptr);
                            data_out.loc(s, {j,i}) = {loc[1], loc[0]};
                            mutex.unlock();
                        }
                    }
                }
            }
                    
    //now filter by consistency
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++)
            if (!static_bounds.contains(cv::Point(i,j)) && state_out(j,i) & STATE_VALID) {
                std::set<QuadSurface*> surf_src = data_out.surfs({j,i});
                for (auto s : surf_src) {
                    int count;
                    float cost = local_cost(s, {j,i}, data_out, state_out, points_out, step, src_step, &count);
                    if (cost >= local_cost_inl_th /*|| count < 1*/) {
                        data_out.erase(s, {j,i});
                        data_out.eraseSurf(s, {j,i});
                    }
                }
            }

    cv::Mat_<uint8_t> fringe(state.size());
    cv::Mat_<uint8_t> fringe_next(state.size(), 1);
    int added = 1;
    for(int r=0;r<30 && added;r++) {
        ALifeTime timer("optimizer: add iteration\n");
        
        fringe_next.copyTo(fringe);
        fringe_next.setTo(0);
        
        added = 0;
#pragma omp parallel for collapse(2) schedule(dynamic)
        for(int j=used_area.y;j<used_area.br().y-1;j++)
            for(int i=used_area.x;i<used_area.br().x-1;i++)
                if (!static_bounds.contains(cv::Point(i,j)) && state_out(j,i) & STATE_LOC_VALID && (fringe(j, i) || fringe_next(j, i))) {
                    mutex.lock_shared();
                    std::set<QuadSurface*> surf_cands = data_out.surfs({j,i});
                    for(auto s : data_out.surfs({j,i}))
                        surf_cands.insert(s->overlapping.begin(), s->overlapping.end());
                        mutex.unlock();
                        
                    for(auto test_surf : surf_cands) {
                        mutex.lock_shared();
                        if (data_out.has(test_surf, {j,i})) {
                            mutex.unlock();
                            continue;
                        }
                        mutex.unlock();
                        
                        auto ptr = test_surf->pointer();
                        if (test_surf->pointTo(ptr, points_out(j, i), same_surface_th, 10) > same_surface_th)
                            continue;
                        
                        int count = 0;
                        cv::Vec3f loc_3d = test_surf->loc_raw(ptr);
                        int straight_count = 0;
                        float cost;
                        mutex.lock();
                        cost = local_cost_destructive(test_surf, {j,i}, data_out, state_out, points_out, step, src_step, loc_3d, &count, &straight_count);
                        mutex.unlock();
                        
                        if (cost > local_cost_inl_th)
                            continue;
                        
                        mutex.lock();
#pragma omp atomic
                        added++;
                        data_out.surfs({j,i}).insert(test_surf);
                        data_out.loc(test_surf, {j,i}) = {loc_3d[1], loc_3d[0]};
                        mutex.unlock();
                        
                        for(int y=j-2;y<=j+2;y++)
                            for(int x=i-2;x<=i+2;x++)
                                fringe_next(y,x) = 1;
                    }
                }
        std::cout << "optimizer: added " << added << "\n";
    }
                     
    //reset unsupported points
#pragma omp parallel for
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++)
            if (!static_bounds.contains(cv::Point(i,j))) {
                if (state_out(j,i) & STATE_LOC_VALID) {
                    if (data_out.surfs({j,i}).size() < 1) {
                        state_out(j,i) = 0;
                        points_out(j, i) = {-1,-1,-1};
                    }
                }
                else {
                    state_out(j,i) = 0;
                    points_out(j, i) = {-1,-1,-1};
                }
            }
            
    points = points_out;
    state = state_out;
    data = data_out;
    data.seed_loc = seed;
    data.seed_coord = points(seed);
            
    {
        cv::Mat_<cv::Vec3d> points_hr_inp = surftrack_genpoints_hr(data, state, points, used_area, step, src_step, true);
        try {
            QuadSurface *dbg_surf = new QuadSurface(points_hr_inp(used_area_hr), {1/src_step,1/src_step});
            std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str()+"_opt_inp_hr";
            dbg_surf->save(tgt_dir / uuid, uuid);
            delete dbg_surf;
        } catch (cv::Exception) {
            // We did not find a valid region of interest to expand to
            std::cout << "optimizer: no valid region of interest found" << "\n";
        }        
    }
            
    dbg_counter++;
}

QuadSurface *grow_surf_from_surfs(QuadSurface *seed, const std::vector<QuadSurface*> &surfs_v, const nlohmann::json &params, float voxelsize)
{
    bool flip_x = params.value("flip_x", 0);
    int global_steps_per_window = params.value("global_steps_per_window", 0);


    std::cout << "global_steps_per_window: " << global_steps_per_window << "\n";
    std::cout << "flip_x: " << flip_x << "\n";
    std::filesystem::path tgt_dir = params["tgt_dir"];
    
    std::unordered_map<std::string,QuadSurface*> surfs;
    float src_step = params.value("src_step", 20);
    float step = params.value("step", 10);
    int max_width = params.value("max_width", 80000);
    
    local_cost_inl_th = params.value("local_cost_inl_th", 0.2f);
    same_surface_th = params.value("same_surface_th", 2.0f);
    straight_weight = params.value("straight_weight", 0.7f);            // Weight for 2D straight line constraints
    straight_weight_3D = params.value("straight_weight_3D", 4.0f);      // Weight for 3D straight line constraints
    sliding_w_scale = params.value("sliding_w_scale", 1.0f);            // Scale factor for sliding window
    z_loc_loss_w = params.value("z_loc_loss_w", 0.1f);                  // Weight for Z location loss constraints
    dist_loss_2d_w = params.value("dist_loss_2d_w", 1.0f);              // Weight for 2D distance constraints
    dist_loss_3d_w = params.value("dist_loss_3d_w", 2.0f);              // Weight for 3D distance constraints
    straight_min_count = params.value("straight_min_count", 1.0f);      // Minimum number of straight constraints
    inlier_base_threshold = params.value("inlier_base_threshold", 20);  // Starting threshold for inliers

    std::cout << "  local_cost_inl_th: " << local_cost_inl_th << "\n";
    std::cout << "  same_surface_th: " << same_surface_th << "\n";
    std::cout << "  straight_weight: " << straight_weight << "\n";
    std::cout << "  straight_weight_3D: " << straight_weight_3D << "\n";
    std::cout << "  straight_min_count: " << straight_min_count << "\n";
    std::cout << "  inlier_base_threshold: " << inlier_base_threshold << "\n";
    std::cout << "  sliding_w_scale: " << sliding_w_scale << "\n";
    std::cout << "  z_loc_loss_w: " << z_loc_loss_w << "\n";
    std::cout << "  dist_loss_2d_w: " << dist_loss_2d_w << "\n";
    std::cout << "  dist_loss_3d_w: " << dist_loss_3d_w << "\n";

    std::cout << "total surface count: " << surfs_v.size() << "\n";

    std::set<QuadSurface*> approved_sm;

    std::set<std::string> used_approved_names;
    std::string log_filename = "/tmp/vc_grow_seg_from_segments_" + get_surface_time_str() + "_used_approved_segments.txt";
    std::ofstream approved_log(log_filename);
    
    for(auto &qs : surfs_v) {
        if (qs->meta->contains("tags") && qs->meta->at("tags").contains("approved"))
            approved_sm.insert(qs);
        if (!qs->meta->contains("tags") || !qs->meta->at("tags").contains("defective")) {
            surfs[qs->name()] = qs;
        }
    }
    
    for(auto qs : approved_sm)
        std::cout << "approved: " << qs->name() << "\n";

    for(auto &qs : surfs_v)
        for(const auto& name : qs->overlapping_str)
            if (surfs.count(name))
                qs->overlapping.insert(surfs[name]);

    std::cout << "total surface count (after defective filter): " << surfs.size() << "\n";
    std::cout << "seed " << seed << " name " << seed->name() << " seed overlapping: " 
              << seed->overlapping.size() << "/" << seed->overlapping_str.size() << "\n";

    cv::Mat_<cv::Vec3f> seed_points = seed->rawPoints();

    int stop_gen = 100000;
    int closing_r = 20; //FIXME dont forget to reset!

    // Get sliding window scale from params (set earlier from JSON)

    //1k ~ 1cm, scaled by sliding_w_scale parameter
    int sliding_w = static_cast<int>(1000/src_step/step*2 * sliding_w_scale);
    int w = 2000/src_step/step*2+10+2*closing_r;
    int h = 15000/src_step/step*2+10+2*closing_r;
    cv::Size size = {w,h};
    cv::Rect bounds(0,0,w-1,h-1);
    cv::Rect save_bounds_inv(closing_r+5,closing_r+5,h-closing_r-10,w-closing_r-10);
    cv::Rect active_bounds(closing_r+5,closing_r+5,w-closing_r-10,h-closing_r-10);
    cv::Rect static_bounds(0,0,0,h);

    int x0 = w/2;
    int y0 = h/2;
    int r = 1;
    
    std::cout << "starting with size " << size << " seed " << cv::Vec2i(y0,x0) << "\n";

    std::vector<cv::Vec2i> neighs = {{1,0},{0,1},{-1,0},{0,-1}};

    std::unordered_set<cv::Vec2i,vec2i_hash> fringe;

    cv::Mat_<uint8_t> state(size,0);
    cv::Mat_<uint16_t> inliers_sum_dbg(size,0);
    cv::Mat_<cv::Vec3d> points(size,{-1,-1,-1});

    cv::Rect used_area(x0,y0,2,2);
    cv::Rect used_area_hr = {used_area.x*step, used_area.y*step, used_area.width*step, used_area.height*step};

    SurfTrackerData data;
    
    cv::Vec2i seed_loc = {seed_points.rows/2, seed_points.cols/2};
    
    while (seed_points(seed_loc)[0] == -1) {
        seed_loc = {rand() % seed_points.rows, rand() % seed_points.cols };
        std::cout << "try loc " << seed_loc << "\n";
    }

    data.loc(seed,{y0,x0}) = {seed_loc[0], seed_loc[1]};
    data.surfs({y0,x0}).insert(seed);
    points(y0,x0) = data.lookup_int(seed,{y0,x0});

    
    data.seed_coord = points(y0,x0);
    data.seed_loc = cv::Point2i(y0,x0);
    
    std::cout << "seed coord " << data.seed_coord << " at " << data.seed_loc << "\n";

    state(y0,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
    fringe.insert(cv::Vec2i(y0,x0));

    //insert initial surfs per location
    for(const auto& p : fringe) {
        data.surfs(p).insert(seed);
        cv::Vec3f coord = points(p);
        std::cout << "testing " << p << " from cands: " << seed->overlapping.size() << coord << "\n";
        for(auto s : seed->overlapping) {
            auto ptr = s->pointer();
            if (s->pointTo(ptr, coord, same_surface_th) <= same_surface_th) {
                cv::Vec3f loc = s->loc_raw(ptr);
                data.surfs(p).insert(s);
                data.loc(s, p) = {loc[1], loc[0]};
            }
        }
        std::cout << "fringe point " << p << " surfcount " << data.surfs(p).size() << " init " << data.loc(seed, p) << data.lookup_int(seed, p) << "\n";
    }

    std::cout << "starting from " << x0 << " " << y0 << "\n";

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 200;
    
    int final_opts = global_steps_per_window;
    
    int loc_valid_count = 0;
    int succ = 0;
    int curr_best_inl_th = inlier_base_threshold;
    int last_succ_parametrization = 0;
    
    std::vector<SurfTrackerData> data_ths(omp_get_max_threads());
    std::vector<std::vector<cv::Vec2i>> added_points_threads(omp_get_max_threads());
    for(int i=0;i<omp_get_max_threads();i++)
        data_ths[i] = data;
    
    bool at_right_border = false;
    for(int generation=0;generation<stop_gen;generation++) {
        std::unordered_set<cv::Vec2i,vec2i_hash> cands;
        if (generation == 0) {
            cands.insert(cv::Vec2i(y0-1,x0));
        }
        else
            for(const auto& p : fringe)
            {
                if ((state(p) & STATE_LOC_VALID) == 0)
                    continue;

                for(const auto& n : neighs) {
                    cv::Vec2i pn = p+n;
                    if (save_bounds_inv.contains(cv::Point(pn))
                        && (state(pn) & STATE_PROCESSING) == 0
                        && (state(pn) & STATE_LOC_VALID) == 0)
                    {
                        state(pn) |= STATE_PROCESSING;
                        cands.insert(pn);
                    }
                    else if (!save_bounds_inv.contains(cv::Point(pn)) && save_bounds_inv.br().y <= pn[1]) {
                        at_right_border = true;
                    }
                }
            }
            fringe.clear();

            std::cout << "go with cands " << cands.size() << " inl_th " << curr_best_inl_th << "\n";

            OmpThreadPointCol threadcol(3, cands);

            std::shared_mutex mutex;
            int best_inliers_gen = 0;
#pragma omp parallel
        while (true)
        {
            cv::Vec2i p = threadcol.next();
            
            if (p[0] == -1)
                break;

            if (state(p) & STATE_LOC_VALID)
                continue;
            
            if (points(p)[0] != -1)
                throw std::runtime_error("oops points(p)[0]");

            std::set<QuadSurface*> local_surfs = {seed};
            
            mutex.lock_shared();
            SurfTrackerData &data_th = data_ths[omp_get_thread_num()];
            for(const auto& added : added_points_threads[omp_get_thread_num()]) {
                data_th.surfs(added) = data.surfs(added);
                for (auto &s : data.surfsC(added)) {
                    if (!data.has(s, added))
                        std::cout << "where the heck is our data?" << "\n";
                    else
                        data_th.loc(s, added) = data.loc(s, added);
                }
            }
            mutex.unlock();
            mutex.lock();
            added_points_threads[omp_get_thread_num()].resize(0);
            mutex.unlock();
            
            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                    if (state(oy,ox) & STATE_LOC_VALID) {
                        auto p_surfs = data_th.surfsC({oy,ox});
                        local_surfs.insert(p_surfs.begin(), p_surfs.end());
                    }

            cv::Vec3d best_coord = {-1,-1,-1};
            int best_inliers = -1;
            QuadSurface *best_surf = nullptr;
            cv::Vec2d best_loc = {-1,-1};
            bool best_ref_seed = false;
            bool best_approved = false;

            for(auto ref_surf : local_surfs) {
                int ref_count = 0;
                cv::Vec2d avg = {0,0};
                bool ref_seed = false;
                for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                    for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                        if ((state(oy,ox) & STATE_LOC_VALID) && data_th.valid_int(ref_surf,{oy,ox})) {
                            ref_count++;
                            avg += data_th.loc(ref_surf,{oy,ox});
                            if (data_th.seed_loc == cv::Vec2i(oy,ox))
                                ref_seed = true;
                        }

                if (ref_count < 2 && !ref_seed)
                    continue;

                avg /= ref_count;
                
                data_th.loc(ref_surf,p) = avg + cv::Vec2d((rand() % 1000)/500.0-1, (rand() % 1000)/500.0-1);

                ceres::Problem problem;

                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;

                int straight_count_init = 0;
                int count_init = surftrack_add_local(ref_surf, p, data_th, problem, state, points, step, src_step, LOSS_ZLOC, &straight_count_init);
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                bool fail = false;
                cv::Vec2d ref_loc = data_th.loc(ref_surf,p);

                if (!data_th.valid_int(ref_surf,p))
                    fail = true;
                
                cv::Vec3d coord;

                if (!fail) {
                    coord = data_th.lookup_int(ref_surf,p);
                    if (coord[0] == -1)
                        fail = true;
                }

                if (fail) {
                    data_th.erase(ref_surf, p);
                    continue;
                }

                state(p) = 0;
                
                int inliers_sum = 0;
                int inliers_count = 0;

                //TODO could also have priorities!
                if (approved_sm.count(ref_surf) && straight_count_init >= 2 && count_init >= 4) {
                    std::cout << "found approved qs " << ref_surf->name() << "\n";

                    // Log approved surface if not already logged
                    if (used_approved_names.insert(ref_surf->name()).second) {
                        mutex.lock();
                        approved_log << ref_surf->name() << "\n";
                        approved_log.flush();
                        mutex.unlock();
                    }

                    best_inliers = 1000;
                    best_coord = coord;
                    best_surf = ref_surf;
                    best_loc = ref_loc;
                    best_ref_seed = ref_seed;
                    data_th.erase(ref_surf, p);
                    best_approved = true;
                    break;
                }

                for(auto test_surf : local_surfs) {
                    auto ptr = test_surf->pointer();
                    //FIXME this does not check geometry, only if its also on the surfaces (which might be good enough...)
                    if (test_surf->pointTo(ptr, coord, same_surface_th, 10) <= same_surface_th) {
                        int count = 0;
                        int straight_count = 0;
                        state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                        cv::Vec3f loc = test_surf->loc_raw(ptr);
                        data_th.loc(test_surf, p) = {loc[1], loc[0]};
                        float cost = local_cost(test_surf, p, data_th, state, points, step, src_step, &count, &straight_count);
                        state(p) = 0;
                        data_th.erase(test_surf, p);
                        if (cost < local_cost_inl_th && (ref_seed || (count >= 2 && straight_count >= straight_min_count))) {
                            inliers_sum += count;
                            inliers_count++;
                        }
                    }
                }
                if ((inliers_count >= 2 || ref_seed) && inliers_sum > best_inliers) {
                    best_inliers = inliers_sum;
                    best_coord = coord;
                    best_surf = ref_surf;
                    best_loc = ref_loc;
                    best_ref_seed = ref_seed;
                }
                data_th.erase(ref_surf, p);
            }

            if (points(p)[0] != -1)
                throw std::runtime_error("oops points(p)[0]");
            
            if (!best_approved && (best_inliers >= curr_best_inl_th || best_ref_seed))
            {
                cv::Vec2f tmp_loc_;
                cv::Rect used_th = used_area;
                float dist = pointTo(tmp_loc_, points(used_th), best_coord, same_surface_th, 1000, 1.0/(step*src_step));
                tmp_loc_ += cv::Vec2f(used_th.x,used_th.y);
                if (dist <= same_surface_th) {
                    int state_sum = state(tmp_loc_[1],tmp_loc_[0]) + state(tmp_loc_[1]+1,tmp_loc_[0]) + state(tmp_loc_[1],tmp_loc_[0]+1) + state(tmp_loc_[1]+1,tmp_loc_[0]+1);
                    best_inliers = -1;
                    best_ref_seed = false;
                    if (!state_sum)
                        throw std::runtime_error("this should not have any location?!");
                }
            }
            
            if (best_inliers >= curr_best_inl_th || best_ref_seed) {
                if (best_coord[0] == -1)
                    throw std::runtime_error("oops best_cord[0]");
                
                data_th.surfs(p).insert(best_surf);
                data_th.loc(best_surf, p) = best_loc;
                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                points(p) = best_coord;
                inliers_sum_dbg(p) = best_inliers;
                
                ceres::Problem problem;
                surftrack_add_local(best_surf, p, data_th, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC);
                
                std::set<QuadSurface*> more_local_surfs;
                
                for(auto test_surf : local_surfs) {
                    for(auto s : test_surf->overlapping)
                        if (!local_surfs.count(s) && s != best_surf)
                            more_local_surfs.insert(s);
                    
                    if (test_surf == best_surf)
                        continue;
                    
                    auto ptr = test_surf->pointer();
                    if (test_surf->pointTo(ptr, best_coord, same_surface_th, 10) <= same_surface_th) {
                        cv::Vec3f loc = test_surf->loc_raw(ptr);
                        data_th.loc(test_surf, p) = {loc[1], loc[0]};
                        int count = 0;
                        float cost = local_cost(test_surf, p, data_th, state, points, step, src_step, &count);
                        //FIXME opt then check all in extra again!
                        if (cost < local_cost_inl_th) {
                            data_th.surfs(p).insert(test_surf);
                            surftrack_add_local(test_surf, p, data_th, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC);
                        }
                        else
                            data_th.erase(test_surf, p);
                    }
                }
                
                ceres::Solver::Summary summary;
                
                ceres::Solve(options, &problem, &summary);
                
                //TODO only add/test if we have 2 neighs which both find locations
                for(auto test_surf : more_local_surfs) {
                    auto ptr = test_surf->pointer();
                    float res = test_surf->pointTo(ptr, best_coord, same_surface_th, 10);
                    if (res <= same_surface_th) {
                        cv::Vec3f loc = test_surf->loc_raw(ptr);
                        cv::Vec3f coord = data_th.lookup_int_loc(test_surf, {loc[1], loc[0]});
                        if (coord[0] == -1) {
                            continue;
                        }
                        int count = 0;
                        float cost = local_cost_destructive(test_surf, p, data_th, state, points, step, src_step, loc, &count);
                        if (cost < local_cost_inl_th) {
                            data_th.loc(test_surf, p) = {loc[1], loc[0]};
                            data_th.surfs(p).insert(test_surf);
                        };
                    }
                }
                
                mutex.lock();
                succ++;
                
                data.surfs(p) = data_th.surfs(p);
                for(auto &s : data.surfs(p))
                    if (data_th.has(s, p))
                        data.loc(s, p) = data_th.loc(s, p);
                
                for(int t=0;t<omp_get_max_threads();t++)
                    added_points_threads[t].push_back(p);
                
                if (!used_area.contains(cv::Point(p[1],p[0]))) {
                    used_area = used_area | cv::Rect(p[1],p[0],1,1);
                    used_area_hr = {used_area.x*step, used_area.y*step, used_area.width*step, used_area.height*step};
                }
                fringe.insert(p);
                mutex.unlock();
            }
            else if (best_inliers == -1) {
                //just try again some other time
                state(p) = 0;
                points(p) = {-1,-1,-1};
            }
            else {
                state(p) = 0;
                points(p) = {-1,-1,-1};
#pragma omp critical
                best_inliers_gen = std::max(best_inliers_gen, best_inliers);
            }
        }
        
        if (generation == 1 && flip_x) {
            data.flip_x(x0);
            
            for(int i=0;i<omp_get_max_threads();i++) {
                data_ths[i] = data;
                added_points_threads[i].clear();
            }

            cv::Mat_<uint8_t> state_orig = state.clone();
            cv::Mat_<cv::Vec3d> points_orig = points.clone();
            state.setTo(0);
            points.setTo(cv::Vec3d(-1,-1,-1));
            cv::Rect new_used_area = used_area;
            for(int j=used_area.y;j<=used_area.br().y+1;j++)
                for(int i=used_area.x;i<=used_area.br().x+1;i++)
                    if (state_orig(j, i)) {
                        int nx = x0+x0-i;
                        int ny = j;
                        state(ny, nx) = state_orig(j, i);
                        points(ny, nx) = points_orig(j, i);
                        new_used_area = new_used_area | cv::Rect(nx,ny,1,1);
                    }
                    
            used_area = new_used_area;
            used_area_hr = {used_area.x*step, used_area.y*step, used_area.width*step, used_area.height*step};
            
            fringe.clear();
            for(int j=used_area.y-2;j<=used_area.br().y+2;j++)
                for(int i=used_area.x-2;i<=used_area.br().x+2;i++)
                    if (state(j,i) & STATE_LOC_VALID)
                        fringe.insert(cv::Vec2i(j,i));
        }
        
        int inl_lower_bound_reg = params.value("consensus_default_th", 10);
        int inl_lower_bound_b = params.value("consensus_limit_th", 2);
        int inl_lower_bound = inl_lower_bound_reg;
        
        if (!at_right_border && curr_best_inl_th <= inl_lower_bound)
            inl_lower_bound = inl_lower_bound_b;
        
        if (!fringe.size() && curr_best_inl_th > inl_lower_bound) {
            curr_best_inl_th -= (1+curr_best_inl_th-inl_lower_bound)/2;
            curr_best_inl_th = std::min(curr_best_inl_th, std::max(best_inliers_gen,inl_lower_bound));
            if (curr_best_inl_th >= inl_lower_bound) {
                cv::Rect active = active_bounds & used_area;
                for(int j=active.y-2;j<=active.br().y+2;j++)
                    for(int i=active.x-2;i<=active.br().x+2;i++)
                        if (state(j,i) & STATE_LOC_VALID)
                                fringe.insert(cv::Vec2i(j,i));
            }
        }
        else
            curr_best_inl_th = inlier_base_threshold;
   
        loc_valid_count = 0;
        for(int j=used_area.y;j<used_area.br().y-1;j++)
            for(int i=used_area.x;i<used_area.br().x-1;i++)
                if (state(j,i) & STATE_LOC_VALID)
                    loc_valid_count++;
        
        bool update_mapping = (succ >= 1000 && (loc_valid_count-last_succ_parametrization) >= std::max(100.0, 0.3*last_succ_parametrization));
        if (!fringe.size() && final_opts) {
            final_opts--;
            update_mapping = true;
        }
        
        if (!global_steps_per_window)
            update_mapping = false;

        if (generation % 50 == 0 || update_mapping /*|| generation < 10*/) {
            {
                cv::Mat_<cv::Vec3d> points_hr = surftrack_genpoints_hr(data, state, points, used_area, step, src_step);
                QuadSurface *dbg_surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});
                dbg_surf->meta = new nlohmann::json;
                (*dbg_surf->meta)["vc_grow_seg_from_segments_params"] = params;

                float const area_est_vx2 = loc_valid_count*src_step*src_step*step*step;
                float const area_est_cm2 = area_est_vx2 * voxelsize * voxelsize / 1e8;
                (*dbg_surf->meta)["area_vx2"] = area_est_vx2;
                (*dbg_surf->meta)["area_cm2"] = area_est_cm2;
                (*dbg_surf->meta)["used_approved_segments"] = std::vector<std::string>(used_approved_names.begin(), used_approved_names.end());
                std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str();
                dbg_surf->save(tgt_dir / uuid, uuid);
                delete dbg_surf;
            }
        }

        //lets just see what happens
        if (update_mapping) {
            dbg_counter = generation;
            SurfTrackerData opt_data = data;
            cv::Mat_<uint8_t> opt_state = state.clone();
            cv::Mat_<cv::Vec3d> opt_points = points.clone(); 
            
            cv::Rect active = active_bounds & used_area;
            optimize_surface_mapping(opt_data, opt_state, opt_points, active, static_bounds, step, src_step, {y0,x0}, closing_r, true, tgt_dir);
            if (active.area() > 0) {
                copy(opt_data, data, active);
                opt_points(active).copyTo(points(active));
                opt_state(active).copyTo(state(active));

                for(int i=0;i<omp_get_max_threads();i++) {
                    data_ths[i] = data;
                    added_points_threads[i].resize(0);
                }
            }
            
            last_succ_parametrization = loc_valid_count;
            //recalc fringe after surface optimization (which often shrinks the surf)
            fringe.clear();
            curr_best_inl_th = inlier_base_threshold;
            for(int j=active.y-2;j<=active.br().y+2;j++)
                for(int i=active.x-2;i<=active.br().x+2;i++)
                    if (state(j,i) & STATE_LOC_VALID)
                        fringe.insert(cv::Vec2i(j,i));

            {
                cv::Mat_<cv::Vec3d> points_hr = surftrack_genpoints_hr(data, state, points, used_area, step, src_step);
                QuadSurface *dbg_surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});
                dbg_surf->meta = new nlohmann::json;
                (*dbg_surf->meta)["vc_grow_seg_from_segments_params"] = params;

                std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str()+"_opt";
                float const area_est_vx2 = loc_valid_count*src_step*src_step*step*step;
                float const area_est_cm2 = area_est_vx2 * voxelsize * voxelsize / 1e8;
                (*dbg_surf->meta)["area_vx2"] = area_est_vx2;
                (*dbg_surf->meta)["area_cm2"] = area_est_cm2;
                (*dbg_surf->meta)["used_approved_segments"] = std::vector<std::string>(used_approved_names.begin(), used_approved_names.end());
                dbg_surf->save(tgt_dir / uuid, uuid);
                delete dbg_surf;
            }
        }

        float const current_area_vx2 = loc_valid_count*src_step*src_step*step*step;
        float const current_area_cm2 = current_area_vx2 * voxelsize * voxelsize / 1e8;
        printf("gen %d processing %lu fringe cands (total done %d fringe: %lu) area %.0f vx^2 (%f cm^2) best th: %d\n", 
               generation, static_cast<unsigned long>(cands.size()), succ, static_cast<unsigned long>(fringe.size()), 
               current_area_vx2, current_area_cm2, best_inliers_gen);
        
        //continue expansion
        if (!fringe.size() && w < max_width/step)
        {
            at_right_border = false;
            std::cout << "expanding by " << sliding_w << "\n";
            
            std::cout << size << bounds << save_bounds_inv << used_area << active_bounds << (used_area & active_bounds) << static_bounds << "\n";
            final_opts = global_steps_per_window;
            w += sliding_w;
            size = {w,h};
            bounds = {0,0,w-1,h-1};
            save_bounds_inv = {closing_r+5,closing_r+5,h-closing_r-10,w-closing_r-10};

            cv::Mat_<cv::Vec3d> old_points = points;
            points = cv::Mat_<cv::Vec3d>(size, {-1,-1,-1});
            old_points.copyTo(points(cv::Rect(0,0,old_points.cols,h)));
            
            cv::Mat_<uint8_t> old_state = state;
            state = cv::Mat_<uint8_t>(size, 0);
            old_state.copyTo(state(cv::Rect(0,0,old_state.cols,h)));

            cv::Mat_<uint16_t> old_inliers_sum_dbg = inliers_sum_dbg;
            inliers_sum_dbg = cv::Mat_<uint16_t>(size, 0);
            old_inliers_sum_dbg.copyTo(inliers_sum_dbg(cv::Rect(0,0,old_inliers_sum_dbg.cols,h)));

            int overlap = 5;
            active_bounds = {w-sliding_w-2*closing_r-10-overlap,closing_r+5,sliding_w+2*closing_r+10+overlap,h-closing_r-10};
            static_bounds = {0,0,w-sliding_w-2*closing_r-10,h};

            cv::Rect active = active_bounds & used_area;

            std::cout << size << bounds << save_bounds_inv << used_area << active_bounds << (used_area & active_bounds) << static_bounds << "\n";
            fringe.clear();
            curr_best_inl_th = inlier_base_threshold;
            for(int j=active.y-2;j<=active.br().y+2;j++)
                for(int i=active.x-2;i<=active.br().x+2;i++)
                    //FIXME why isn't this working?!'
                    if (state(j,i) & STATE_LOC_VALID)
                        fringe.insert(cv::Vec2i(j,i));
        }
        
        cv::imwrite(tgt_dir / "inliers_sum.tif", inliers_sum_dbg(used_area));
        
        if (!fringe.size())
            break;
    }

    approved_log.close();
    
    float const area_est_vx2 = loc_valid_count*src_step*src_step*step*step;
    float const area_est_cm2 = area_est_vx2 * voxelsize * voxelsize / 1e8;
    std::cout << "area est: " << area_est_vx2 << " vx^2 (" << area_est_cm2 << " cm^2)" << "\n";

    cv::Mat_<cv::Vec3d> points_hr = surftrack_genpoints_hr(data, state, points, used_area, step, src_step);

    QuadSurface *surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});

    surf->meta = new nlohmann::json;
    (*surf->meta)["area_vx2"] = area_est_vx2;
    (*surf->meta)["area_cm2"] = area_est_cm2;
    (*surf->meta)["used_approved_segments"] = std::vector<std::string>(used_approved_names.begin(), used_approved_names.end());

    return surf;
}

std::string get_surface_time_str()
{
    using namespace std::chrono;

    // get current time
    auto now = system_clock::now();

    // get number of milliseconds for the current second
    // (remainder after division into seconds)
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream oss;

    oss << std::put_time(&bt, "%Y%m%d%H%M%S"); // HH:MM:SS
    oss << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
}
