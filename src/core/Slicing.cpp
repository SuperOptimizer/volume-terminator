#include "Slicing.hpp"

#include <nlohmann/json.hpp>

#include "xtensor/containers/xarray.hpp"
#include "xtensor/io/xio.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor/views/xview.hpp"
#include "z5/common.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <shared_mutex>

#include <algorithm>
#include <unordered_set>


int ChunkCache::groupIdx(const std::string& name)
{
    if (!_group_store.contains(name))
        _group_store[name] = _group_store.size()+1;
    
     return _group_store[name];
}
    
void ChunkCache::put(const cv::Vec4i& idx, xt::xarray<uint8_t> *ar)
{
    if (_stored >= _size) {
        using KP = std::pair<cv::Vec4i, uint64_t>;
        std::vector<KP> gen_list(_gen_store.begin(), _gen_store.end());
        std::sort(gen_list.begin(), gen_list.end(), [](const KP &a, const KP &b){ return a.second < b.second; });
        for(const auto &key: gen_list | std::views::keys) {
            std::shared_ptr<xt::xarray<uint8_t>> ar = _store[key];
            //TODO we could remove this with lower probability so we dont store infiniteyl empty blocks but also keep more of them as they are cheap
            if (ar.get()) {
                size_t size = ar.get()->storage().size();
                ar.reset();
                _stored -= size;
            
                _store.erase(key);
                _gen_store.erase(key);
            }

            //we delete 10% of cache content to amortize sorting costs
            if (_stored < 0.9*_size) {
                break;
            }
        }
    }

    if (ar) {
        if (_store.contains(idx)) {
            assert(_store[idx].get());
            _stored -= ar->size();
        }
        _stored += ar->size();
    }
    _store[idx].reset(ar);
    _generation++;
    _gen_store[idx] = _generation;
}

// Add these methods to ChunkCache class
void ChunkCache::putST(const cv::Vec4i& idx, xt::xarray<uint8_t> *ar) {
    // No mutex needed for single-threaded version
    if (_stored >= _size) {
        // Eviction logic
        using KP = std::pair<cv::Vec4i, uint64_t>;
        std::vector<KP> gen_list(_gen_store.begin(), _gen_store.end());
        std::sort(gen_list.begin(), gen_list.end(),
                  [](const KP &a, const KP &b){ return a.second < b.second; });

        for(const auto &key: gen_list | std::views::keys) {
            std::shared_ptr<xt::xarray<uint8_t>> ar = _store[key];
            if (ar.get()) {
                size_t size = ar.get()->storage().size();
                ar.reset();
                _stored -= size;
                _store.erase(key);
                _gen_store.erase(key);
            }

            if (_stored < 0.9*_size) {
                break;
            }
        }
    }

    if (ar) {
        if (_store.contains(idx) && _store[idx].get()) {
            _stored -= _store[idx]->size();
        }
        _stored += ar->size();
    }
    _store[idx].reset(ar);
    _generation++;
    _gen_store[idx] = _generation;
}

std::shared_ptr<xt::xarray<uint8_t>> ChunkCache::getST(const cv::Vec4i& idx) {
    // No mutex needed
    auto res = _store.find(idx);
    if (res == _store.end())
        return nullptr;

    _generation++;
    _gen_store[idx] = _generation;
    return res->second;
}

bool ChunkCache::hasST(const cv::Vec4i& idx) {
    // No mutex needed
    return _store.contains(idx);
}



ChunkCache::~ChunkCache()
{
    for(auto &it : _store)
        it.second.reset();
}

void ChunkCache::reset()
{
    _gen_store.clear();
    _group_store.clear();
    _store.clear();

    _generation = 0;
    _stored = 0;
}

std::shared_ptr<xt::xarray<uint8_t>> ChunkCache::get(const cv::Vec4i& idx)
{
    auto res = _store.find(idx);
    if (res == _store.end())
        return nullptr;

    _generation++;
    _gen_store[idx] = _generation;
    
    return res->second;
}

bool ChunkCache::has(const cv::Vec4i& idx)
{
    return _store.contains(idx);
}


void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds,
                               const cv::Mat_<cv::Vec3f> &coords, ChunkCache *cache) {
    out = cv::Mat_<uint8_t>(coords.size(), 0);

    if (!cache) {
        std::cout << "ERROR should use a shared chunk cache!" << std::endl;
        abort();
    }

    int group_idx = cache->groupIdx(ds->path());

    auto cw = ds->chunking().blockShape()[0];
    auto ch = ds->chunking().blockShape()[1];
    auto cd = ds->chunking().blockShape()[2];

    int w = coords.cols;
    int h = coords.rows;

    std::unordered_map<cv::Vec4i,std::shared_ptr<xt::xarray<uint8_t>>,vec4i_hash> chunks;

    // Lambda for retrieving single values (unchanged)
    auto retrieve_single_value_cached = [&cw,&ch,&cd,&group_idx,&chunks](
        int ox, int oy, int oz) -> uint8_t {

            int ix = int(ox)/cw;
            int iy = int(oy)/ch;
            int iz = int(oz)/cd;

            cv::Vec4i idx = {group_idx,ix,iy,iz};

            xt::xarray<uint8_t> *chunk  = chunks[idx].get();

            if (!chunk)
                return 0;

            int lx = ox-ix*cw;
            int ly = oy-iy*ch;
            int lz = oz-iz*cd;

            return chunk->operator()(lx,ly,lz);
        };

        // size_t done = 0;

        #pragma omp parallel
        {
            cv::Vec4i last_idx = {-1,-1,-1,-1};
            std::shared_ptr<xt::xarray<uint8_t>> chunk_ref;
            xt::xarray<uint8_t> *chunk = nullptr;
            std::unordered_map<cv::Vec4i,std::shared_ptr<xt::xarray<uint8_t>>,vec4i_hash> chunks_local;

            #pragma omp for collapse(2)
            for(size_t y = 0;y<h;y++) {
                for(size_t x = 0;x<w;x++) {
                    float ox = coords(y,x)[2];
                    float oy = coords(y,x)[1];
                    float oz = coords(y,x)[0];

                    if (ox < 0 || oy < 0 || oz < 0)
                        continue;

                    int ix = int(ox)/cw;
                    int iy = int(oy)/ch;
                    int iz = int(oz)/cd;

                    cv::Vec4i idx = {group_idx,ix,iy,iz};

                    if (idx != last_idx) {
                        last_idx = idx;
                        chunks_local[idx] = nullptr;
                    }

                    int lx = ox-ix*cw;
                    int ly = oy-iy*ch;
                    int lz = oz-iz*cd;

                    if (lx+1 >= cw || ly+1 >= ch || lz+1 >= cd) {
                        if (lx+1>=cw) {
                            cv::Vec4i idx2 = idx;
                            idx2[1]++;
                            chunks_local[idx2] = nullptr;
                        }
                        if (ly+1>=ch) {
                            cv::Vec4i idx2 = idx;
                            idx2[2]++;
                            chunks_local[idx2] = nullptr;
                        }

                        if (lz+1>=cd) {
                            cv::Vec4i idx2 = idx;
                            idx2[3]++;
                            chunks_local[idx2] = nullptr;
                        }
                    }
                }
            }

#pragma omp barrier
#pragma omp critical
            chunks.merge(chunks_local);

        }

#pragma omp paralle for schedule(dynamic)
    for(auto &it : chunks) {
        xt::xarray<uint8_t> *chunk = nullptr;
        std::shared_ptr<xt::xarray<uint8_t>> chunk_ref;

        cv::Vec4i idx = it.first;

        cache->mutex.lock();
        if (!cache->has(idx)) {
            cache->mutex.unlock();
            chunk = readChunk<uint8_t>(*ds, {size_t(idx[1]),size_t(idx[2]),size_t(idx[3])});
            cache->mutex.lock();
            cache->put(idx, chunk);
            chunk_ref = cache->get(idx);
        } else {
            chunk_ref = cache->get(idx);
            // chunk = chunk_ref.get();
        }
        chunks[idx] = chunk_ref;
        cache->mutex.unlock();
    }

    #pragma omp parallel
    {
        cv::Vec4i last_idx = {-1,-1,-1,-1};
        std::shared_ptr<xt::xarray<uint8_t>> chunk_ref;
        xt::xarray<uint8_t> *chunk = nullptr;

        #pragma omp for collapse(2)
        for(size_t y = 0;y<h;y++) {
            for(size_t x = 0;x<w;x++) {
                float ox = coords(y,x)[2];
                float oy = coords(y,x)[1];
                float oz = coords(y,x)[0];

                if (ox < 0 || oy < 0 || oz < 0)
                    continue;

                int ix = int(ox)/cw;
                int iy = int(oy)/ch;
                int iz = int(oz)/cd;

                cv::Vec4i idx = {group_idx,ix,iy,iz};

                if (idx != last_idx) {
                    last_idx = idx;
                    chunk = chunks[idx].get();
                }

                int lx = ox-ix*cw;
                int ly = oy-iy*ch;
                int lz = oz-iz*cd;

                //valid - means zero!
                if (!chunk)
                    continue;

                float c000 = chunk->operator()(lx,ly,lz);
                float c100, c010, c110, c001, c101, c011, c111;

                // Handle edge cases for interpolation
                if (lx+1 >= cw || ly+1 >= ch || lz+1 >= cd) {
                    if (lx+1>=cw)
                        c100 = retrieve_single_value_cached(ox+1,oy,oz);
                    else
                        c100 = chunk->operator()(lx+1,ly,lz);

                    if (ly+1 >= ch)
                        c010 = retrieve_single_value_cached(ox,oy+1,oz);
                    else
                        c010 = chunk->operator()(lx,ly+1,lz);
                    if (lz+1 >= cd)
                        c001 = retrieve_single_value_cached(ox,oy,oz+1);
                    else
                        c001 = chunk->operator()(lx,ly,lz+1);

                    c110 = retrieve_single_value_cached(ox+1,oy+1,oz);
                    c101 = retrieve_single_value_cached(ox+1,oy,oz+1);
                    c011 = retrieve_single_value_cached(ox,oy+1,oz+1);
                    c111 = retrieve_single_value_cached(ox+1,oy+1,oz+1);
                } else {
                    c100 = chunk->operator()(lx+1,ly,lz);
                    c010 = chunk->operator()(lx,ly+1,lz);
                    c110 = chunk->operator()(lx+1,ly+1,lz);
                    c001 = chunk->operator()(lx,ly,lz+1);
                    c101 = chunk->operator()(lx+1,ly,lz+1);
                    c011 = chunk->operator()(lx,ly+1,lz+1);
                    c111 = chunk->operator()(lx+1,ly+1,lz+1);
                }

                // Trilinear interpolation
                float fx = ox-int(ox);
                float fy = oy-int(oy);
                float fz = oz-int(oz);

                float c00 = (1-fz)*c000 + fz*c001;
                float c01 = (1-fz)*c010 + fz*c011;
                float c10 = (1-fz)*c100 + fz*c101;
                float c11 = (1-fz)*c110 + fz*c111;

                float c0 = (1-fy)*c00 + fy*c01;
                float c1 = (1-fy)*c10 + fy*c11;

                float c = (1-fx)*c0 + fx*c1;

                out(y,x) = c;
            }
        }
    }
}

struct vec3i_hash {
    size_t operator()(cv::Vec3i p) const {
        size_t const hash1 = std::hash<int>{}(p[0]);
        size_t const hash2 = std::hash<int>{}(p[1]);
        size_t const hash3 = std::hash<int>{}(p[2]);

        size_t const hash = hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        return hash ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
    }
};

void readInterpolated2D(cv::Mat_<uint8_t> &out, z5::Dataset *ds,
                                  const cv::Mat_<cv::Vec3f> &coords, ChunkCache *cache) {
    out = cv::Mat_<uint8_t>(coords.size(), 0);

    int group_idx = cache->groupIdx(ds->path());

    // Bit operations setup
    const int chunk_size = ds->chunking().blockShape()[0];
    const int chunk_shift = __builtin_ctz(chunk_size);
    const int chunk_mask = chunk_size - 1;

    int w = coords.cols;
    int h = coords.rows;

    cv::Vec4i last_idx = {-1,-1,-1,-1};
    xt::xarray<uint8_t> *chunk = nullptr;
    std::shared_ptr<xt::xarray<uint8_t>> chunk_ref;

    // Process in tiles for better cache locality
    const int TILE_SIZE = 32;

    for(size_t tile_y = 0; tile_y < h; tile_y += TILE_SIZE) {
        size_t y_end = std::min(tile_y + TILE_SIZE, (size_t)h);

        for(size_t tile_x = 0; tile_x < w; tile_x += TILE_SIZE) {
            size_t x_end = std::min(tile_x + TILE_SIZE, (size_t)w);

            for(size_t y = tile_y; y < y_end; y++) {
                // Prefetch next row
                if (y + 1 < y_end) {
                    __builtin_prefetch(&coords(y+1, tile_x), 0, 1);
                }

                for(size_t x = tile_x; x < x_end; x++) {
                    // Use fast rounding (could use SIMD here)
                    int ox = int(coords(y,x)[2] + 0.5f);
                    int oy = int(coords(y,x)[1] + 0.5f);
                    int oz = int(coords(y,x)[0] + 0.5f);

                    // Branchless bounds check
                    if ((ox | oy | oz) < 0)
                        continue;

                    // Bit operations for chunk index
                    int ix = ox >> chunk_shift;
                    int iy = oy >> chunk_shift;
                    int iz = oz >> chunk_shift;

                    cv::Vec4i idx = {group_idx, ix, iy, iz};

                    // Only reload chunk if needed
                    if (idx != last_idx) {
                        last_idx = idx;

                        if (!cache->hasST(idx)) {
                            chunk = readChunk<uint8_t>(*ds, {size_t(ix), size_t(iy), size_t(iz)});
                            cache->putST(idx, chunk);
                            chunk_ref = cache->getST(idx);
                        } else {
                            chunk_ref = cache->getST(idx);
                        }
                        chunk = chunk_ref.get();
                    }

                    if (!chunk)
                        continue;

                    // Bit mask for local coordinates
                    int lx = ox & chunk_mask;
                    int ly = oy & chunk_mask;
                    int lz = oz & chunk_mask;

                    out(y,x) = chunk->operator()(lx, ly, lz);
                }
            }
        }
    }
}

//somehow opencvs functions are pretty slow 
static cv::Vec3f normed(const cv::Vec3f& v)
{
    return v/sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
}

static cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f p)
{
    int x = p[0];
    int y = p[1];
    float fx = p[0]-x;
    float fy = p[1]-y;
    
    const cv::Vec3f& p00 = points(y,x);
    const cv::Vec3f& p01 = points(y,x+1);
    const cv::Vec3f& p10 = points(y+1,x);
    const cv::Vec3f& p11 = points(y+1,x+1);
    
    cv::Vec3f p0 = (1-fx)*p00 + fx*p01;
    cv::Vec3f p1 = (1-fx)*p10 + fx*p11;
    
    return (1-fy)*p0 + fy*p1;
}

static cv::Vec2f vmin(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return {std::min(a[0],b[0]),std::min(a[1],b[1])};
}

static cv::Vec2f vmax(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return {std::max(a[0],b[0]),std::max(a[1],b[1])};
}

cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &loc)
{
    cv::Vec2f inb_loc = {loc[0], loc[1]};
    //move inside from the grid border so w can access required locations
    inb_loc = vmax(inb_loc, {1,1});
    inb_loc = vmin(inb_loc, {points.cols-3,points.rows-3});
    
    if (!loc_valid_xy(points, inb_loc))
        return {NAN,NAN,NAN};
    
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(1,0)))
        return {NAN,NAN,NAN};
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(-1,0)))
        return {NAN,NAN,NAN};
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(0,1)))
        return {NAN,NAN,NAN};
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(0,-1)))
        return {NAN,NAN,NAN};
    
    cv::Vec3f xv = normed(at_int(points,inb_loc+cv::Vec2f(1,0))-at_int(points,inb_loc-cv::Vec2f(1,0)));
    cv::Vec3f yv = normed(at_int(points,inb_loc+cv::Vec2f(0,1))-at_int(points,inb_loc-cv::Vec2f(0,1)));
    
    cv::Vec3f n = yv.cross(xv);

    if (std::isnan(n[0]))
        return {NAN,NAN,NAN};
    
    return normed(n);
}

static float sdist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a-b;
    return d.dot(d);
}

static void min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const cv::Vec3f& tgt)
{
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains(cv::Point(loc))) {
        out = {-1,-1,-1};
        loc = {-1,-1};
        return;
    }
    
    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = sdist(val, tgt);
    float res;
    
    std::vector<cv::Vec2f> search;
    search = {{1,0},{-1,0}};
    
    float step = 1.0;
    
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;
            
            if (!boundary.contains(cv::Point(cand))) {
                out = {-1,-1,-1};
                loc = {-1,-1};
                return;
            }
            
            
            val = at_int(points, cand);
            res = sdist(val,tgt);
            if (res < best) {
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
        }
        
        if (!changed && step > 0.125) {
            step *= 0.5;
            changed = true;
        }
    }
}


//this works surprisingly well, though some artifacts where original there was a lot of skew
cv::Mat_<cv::Vec3f> smooth_vc_segmentation(const cv::Mat_<cv::Vec3f> &points)
{
    cv::Mat_<cv::Vec3f> out = points.clone();
    cv::Mat_<cv::Vec3f> blur(points.cols, points.rows);
    cv::Mat_<cv::Vec2f> locs(points.size());
    
    cv::Mat trans = out.t();
    
    #pragma omp parallel for
    for(int j=0;j<trans.rows;j++) 
        cv::GaussianBlur(trans({0,j,trans.cols,1}), blur({0,j,trans.cols,1}), {255,1}, 0);
    
    blur = blur.t();
    
    #pragma omp parallel for
    for(int j=1;j<points.rows;j++)
        for(int i=1;i<points.cols-1;i++) {
            cv::Vec2f loc = {i,j};
            min_loc(points, loc, out(j,i), blur(j,i));
        }
        
        return out;
}

void vc_segmentation_scales(cv::Mat_<cv::Vec3f> points, double &sx, double &sy)
{
    //so we get something somewhat meaningful by default
    double sum_x = 0;
    double sum_y = 0;
    int count = 0;
    //NOTE leave out bordes as these contain lots of artifacst if coming from smooth_segmentation() ... would need median or something ...
    int jmin = points.size().height*0.1+1;
    int jmax = points.size().height*0.9;
    int imin = points.size().width*0.1+1;
    int imax = points.size().width*0.9;
    int step = 4;
    if (points.size().height < 20) {
        std::cout << "small array vc scales " << '\n';
        jmin = 1;
        jmax = points.size().height;
        imin = 1;
        imax = points.size().width;
        step = 1;
    }
#pragma omp parallel for
    for(int j=jmin;j<jmax;j+=step) {
        double _sum_x = 0;
        double _sum_y = 0;
        int _count = 0;
        for(int i=imin;i<imax;i+=step) {
            cv::Vec3f v = points(j,i)-points(j,i-1);
            _sum_x += sqrt(v.dot(v));
            v = points(j,i)-points(j-1,i);
            _sum_y += sqrt(v.dot(v));
            _count++;
        }
#pragma omp critical
        {
            sum_x += _sum_x;
            sum_y += _sum_y;
            count += _count;
        }
    }

    sx = count/sum_x;
    sy = count/sum_y;
}
