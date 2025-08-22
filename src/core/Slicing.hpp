#pragma once

#include "xtensor/containers/xarray.hpp"
#include <opencv2/core.hpp>

#include <shared_mutex>
#include <z5/dataset.hxx>


struct vec4i_hash {
    size_t operator()(cv::Vec4i p) const
    {
        const size_t hash1 = std::hash<int>{}(p[0]);
        const size_t hash2 = std::hash<int>{}(p[1]);
        const size_t hash3 = std::hash<int>{}(p[2]);
        const size_t hash4 = std::hash<int>{}(p[3]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash =  hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
        hash =  hash  ^ (hash4 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};

//TODO generation overrun
//TODO groupkey overrun
class ChunkCache
{
public:
    ChunkCache(size_t size) : _size(size) {}
    ~ChunkCache();
    
    //get key for a subvolume - should be uniqueley identified between all groups and volumes that use this cache.
    //for example by using path + group name
    int groupIdx(const std::string& name);
    
    //key should be unique for chunk and contain groupkey (groupkey sets highest 16bits of uint64_t)
    void put(const cv::Vec4i& key, xt::xarray<uint8_t> *ar);
    std::shared_ptr<xt::xarray<uint8_t>> get(const cv::Vec4i& key);
    void reset();
    bool has(const cv::Vec4i& idx);
    void putST(const cv::Vec4i& idx, xt::xarray<uint8_t> *ar);
    std::shared_ptr<xt::xarray<uint8_t>> getST(const cv::Vec4i& idx);
    bool hasST(const cv::Vec4i& idx);
    std::shared_mutex mutex;
private:
    uint64_t _generation = 0;
    size_t _size = 0;
    size_t _stored = 0;
    std::unordered_map<cv::Vec4i,std::shared_ptr<xt::xarray<uint8_t>>,vec4i_hash> _store;
    //store generation number
    std::unordered_map<cv::Vec4i,uint64_t,vec4i_hash> _gen_store;
    //store group keys
    std::unordered_map<std::string,int> _group_store;
};

void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache *cache = nullptr);
void readInterpolated2D(cv::Mat_<uint8_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache *cache);
cv::Mat_<cv::Vec3f> smooth_vc_segmentation(const cv::Mat_<cv::Vec3f> &points);
void vc_segmentation_scales(cv::Mat_<cv::Vec3f> points, double &sx, double &sy);
cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &loc);

template <typename E>
static inline E at_int(const cv::Mat_<E> &points, const cv::Vec2f& p)
{
    int x = p[0];
    int y = p[1];
    float fx = p[0]-x;
    float fy = p[1]-y;
    
    E p00 = points(y,x);
    E p01 = points(y,x+1);
    E p10 = points(y+1,x);
    E p11 = points(y+1,x+1);
    
    E p0 = (1-fx)*p00 + fx*p01;
    E p1 = (1-fx)*p10 + fx*p11;
    
    return (1-fy)*p0 + fy*p1;
}

template<typename T, int C>
//l is [y, x]!
static inline bool loc_valid(const cv::Mat_<cv::Vec<T,C>> &m, const cv::Vec2d &l)
{
    if (l[0] == -1)
        return false;
    
    cv::Rect bounds = {0, 0, m.rows-2,m.cols-2};
    cv::Vec2i li = {floor(l[0]),floor(l[1])};
    
    if (!bounds.contains(cv::Point(li)))
        return false;
    
    if (m(li[0],li[1])[0] == -1)
        return false;
    if (m(li[0]+1,li[1])[0] == -1)
        return false;
    if (m(li[0],li[1]+1)[0] == -1)
        return false;
    if (m(li[0]+1,li[1]+1)[0] == -1)
        return false;
    return true;
}

template<typename T, int C>
//l is [x, y]!
static inline bool loc_valid_xy(const cv::Mat_<cv::Vec<T,C>> &m, const cv::Vec2d &l)
{
    return loc_valid(m, {l[1],l[0]});
}


template<typename T>
static xt::xarray<T> *readChunk(const z5::Dataset & ds, const z5::types::ShapeType& chunkId)
{
    if (!ds.chunkExists(chunkId)) {
        return nullptr;
    }

    if (!ds.isZarr())
        throw std::runtime_error("only zarr datasets supported currently!");
    if (ds.getDtype() != z5::types::Datatype::uint8 && ds.getDtype() != z5::types::Datatype::uint16)
        throw std::runtime_error("only uint8_t/uint16 zarrs supported currently!");

    z5::types::ShapeType chunkShape;
    // size_t chunkSize;
    ds.getChunkShape(chunkId, chunkShape);
    // get the shape of the chunk (as stored it is stored)
    //for ZARR also edge chunks are always full size!
    const auto & maxChunkShape = ds.defaultChunkShape();

    // chunkSize = std::accumulate(chunkShape.begin(), chunkShape.end(), 1, std::multiplies<std::size_t>());

    auto *out = new xt::xarray<T>();
    *out = xt::empty<T>(maxChunkShape);


    // read/decompress & convert data
    if (ds.getDtype() == z5::types::Datatype::uint8) {
        ds.readChunk(chunkId, out->data());
    }
    else if (ds.getDtype() == z5::types::Datatype::uint16) {
        std::cout << "uint16 not supported!\n";
        abort();
    }

    return out;
}