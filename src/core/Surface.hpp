#pragma once
#include <filesystem>
#include <set>

#include <opencv2/core.hpp> 
#include <nlohmann/json_fwd.hpp>
#include <z5/dataset.hxx>

#include "Slicing.hpp"

#define Z_DBG_GEN_PREFIX "auto_grown_"

struct Rect3D {
    cv::Vec3f low = {0,0,0};
    cv::Vec3f high = {0,0,0};
};

bool intersect(const Rect3D &a, const Rect3D &b);
Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p);


// Base surface class with lazy loading support
class Surface
{
public:
    virtual ~Surface();

    // Lazy loading interface
    virtual bool isLoaded() const { return true; }
    virtual void load() { }
    virtual void unload() { }

    // Core surface API
    virtual cv::Vec3f pointer() = 0;
    virtual void move(cv::Vec3f &ptr, const cv::Vec3f &offset) = 0;
    virtual bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    virtual cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    virtual cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    virtual cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    virtual float pointTo(cv::Vec3f &ptr, const cv::Vec3f &coord, float th, int max_iters = 1000) = 0;
    virtual void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size,
                     const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) = 0;

    // Metadata and bounding box
    virtual Rect3D bbox() { return _bbox; }
    virtual std::string name() const;



    // Public data members
    nlohmann::json *meta = nullptr;
    std::filesystem::path path;
    std::string id;

    Rect3D _bbox = {{-1,-1,-1},{-1,-1,-1}};
    mutable bool _loading = false;  // Prevent recursive loads
};

class PlaneSurface : public Surface
{
public:
    PlaneSurface() {};
    PlaneSurface(const cv::Vec3f& origin_, const cv::Vec3f &normal_);

    // Surface API
    cv::Vec3f pointer() override;
    void move(cv::Vec3f &ptr, const cv::Vec3f &offset) override;
    bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override { return true; };
    cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    float pointTo(cv::Vec3f &ptr, const cv::Vec3f &coord, float th, int max_iters = 1000) override { abort(); };
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size,
             const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override;

    // Plane-specific methods
    float pointDist(const cv::Vec3f& wp) const;
    cv::Vec3f project(const cv::Vec3f& wp, float render_scale = 1.0, float coord_scale = 1.0) const;
    void setNormal(const cv::Vec3f& normal);
    void setOrigin(const cv::Vec3f& origin);
    cv::Vec3f origin();
    float scalarp(const cv::Vec3f& point) const;

protected:
    void update();
    cv::Vec3f _normal = {0,0,1};
    cv::Vec3f _origin = {0,0,0};
    cv::Matx33d _M;
    cv::Vec3d _T;
};

// Quad surface with lazy loading support
class QuadSurface : public Surface
{
public:
    QuadSurface() {};
    // Direct construction with points
    QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale);
    QuadSurface(cv::Mat_<cv::Vec3f> *points, const cv::Vec2f &scale);
    // Lazy construction from file
    QuadSurface(const std::filesystem::path &path);
    ~QuadSurface() override;

    // Lazy loading implementation
    bool isLoaded() const override { return _points != nullptr; }
    void load() override;
    void unload() override;

    // Surface API
    cv::Vec3f pointer() override;
    void move(cv::Vec3f &ptr, const cv::Vec3f &offset) override;
    bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc_raw(const cv::Vec3f &ptr) const;
    cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size,
             const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override;
    float pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters = 1000) override;

    // QuadSurface specific
    cv::Size size();
    cv::Vec2f scale() const;
    Rect3D bbox() override;

    // Saving methods
    void save(const std::string &path, const std::string &uuid);
    void save(const std::filesystem::path &path);
    void save_meta() const;

    // Direct access (triggers load if needed)
    cv::Mat_<cv::Vec3f> rawPoints() const;
    cv::Mat_<cv::Vec3f>* rawPointsPtr() const;

    friend QuadSurface *regularized_local_quad(QuadSurface *src, const cv::Vec3f &ptr, int w, int h, int step_search, int step_out);
    friend QuadSurface *smooth_vc_segmentation(QuadSurface *src);
    friend class ControlPointSurface;

    cv::Vec2f _scale;

    std::set<QuadSurface*> overlapping;  // Populated externally
    std::set<std::string> overlapping_str;
    // Overlapping surface management
    void readOverlapping();
    const std::set<std::string>& overlappingNames() const { return overlapping_str; }



    void ensureLoaded() const;
    void initFromPoints();  // Common initialization after points are set
protected:

    cv::Mat_<cv::Vec3f>* _points = nullptr;
    cv::Rect _bounds;
    cv::Vec3f _center;
};

// Delta surface base class - operates on another surface
class DeltaSurface : public Surface
{
public:
    DeltaSurface(Surface *base);
    virtual void setBase(Surface *base);

    // Most operations forward to base
    cv::Vec3f pointer() override;
    void move(cv::Vec3f &ptr, const cv::Vec3f &offset) override;
    bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    float pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters = 1000) override;

    // Derived classes must implement gen()
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size,
             const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override = 0;

    // Forward lazy loading to base
    bool isLoaded() const override { return _base ? _base->isLoaded() : true; }
    void load() override { if (_base) _base->load(); }
    void unload() override { if (_base) _base->unload(); }
    Rect3D bbox() override { return _base ? _base->bbox() : Rect3D(); }

protected:
    Surface *_base = nullptr;
};

// Control point surface
class SurfaceControlPoint
{
public:
    SurfaceControlPoint(Surface *base, const cv::Vec3f &ptr_, const cv::Vec3f &control);
    cv::Vec3f ptr;
    cv::Vec3f orig_wp;
    cv::Vec3f normal;
    cv::Vec3f control_point;
};

class ControlPointSurface : public DeltaSurface
{
public:
    ControlPointSurface(Surface *base) : DeltaSurface(base) {};
    void addControlPoint(const cv::Vec3f &base_ptr, const cv::Vec3f& control_point);
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size,
             const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override;
    void setBase(Surface *base) override;

protected:
    std::vector<SurfaceControlPoint> _controls;
};

// Refinement surface
class RefineCompSurface : public DeltaSurface
{
public:
    RefineCompSurface(z5::Dataset *ds, ChunkCache *cache, QuadSurface *base = nullptr);
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size,
             const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override;

    float start = 0;
    float stop = -100;
    float step = 2.0;
    float low = 0.1;
    float high = 1.0;

protected:
    z5::Dataset *_ds;
    ChunkCache *_cache;
};

// Utility functions
bool overlap(Surface &a, Surface &b, int max_iters = 1000);
bool contains(Surface &a, const cv::Vec3f &loc, int max_iters = 1000);
bool contains(Surface &a, const std::vector<cv::Vec3f> &locs);
bool contains_any(Surface &a, const std::vector<cv::Vec3f> &locs);

void find_intersect_segments(std::vector<std::vector<cv::Vec3f>> &seg_vol,
                             std::vector<std::vector<cv::Vec2f>> &seg_grid,
                             const cv::Mat_<cv::Vec3f> &points, PlaneSurface *plane,
                             const cv::Rect &plane_roi, float step, int min_tries = 10);

float min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out,
              const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds,
              const PlaneSurface *plane, float init_step = 16.0, float min_step = 0.125);


float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt,
              float th, int max_iters, float scale);
float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt,
              float th, int max_iters, float scale);

void write_overlapping_json(const std::filesystem::path& seg_path, const std::set<std::string>& overlapping_names);
std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path);

QuadSurface *load_quad_from_obj(const std::string &path);
QuadSurface *load_quad_from_tifxyz(const std::string &path);
QuadSurface *space_tracing_quad_phys(z5::Dataset *ds, float scale, ChunkCache *cache,
                                     cv::Vec3f origin, int generations = 100, float step = 10,
                                     const std::string &cache_root = "", float voxelsize = 1.0);
QuadSurface *regularized_local_quad(QuadSurface *src, const cv::Vec3f &ptr, int w, int h,
                                    int step_search = 100, int step_out = 5);
QuadSurface *smooth_vc_segmentation(QuadSurface *src);