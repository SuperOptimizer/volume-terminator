#pragma once

#include "Slicing.hpp"

#include <opencv2/core.hpp>
#include "z5/dataset.hxx"

#include "xtensor/containers/xtensor.hpp"
#include "xtensor/views/xview.hpp"
#include "z5/multiarray/xtensor_access.hxx"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

#ifndef CCI_TLS_MAX // Max number for ChunkedCachedInterpolator
#define CCI_TLS_MAX 256
#endif

struct vec3i_hash {
    size_t operator()(cv::Vec3i p) const
    {
        size_t hash1 = std::hash<int>{}(p[0]);
        size_t hash2 = std::hash<int>{}(p[1]);
        size_t hash3 = std::hash<int>{}(p[2]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        return hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
    }
};

struct passTroughComputor
{
    enum {BORDER = 0};
    enum {CHUNK_SIZE = 32};
    enum {FILL_V = 0};
    const std::string UNIQUE_ID_STRING = "";
    template <typename T, typename E> void compute(const T &large, T &small, const cv::Vec3i &offset_large)
    {
        int low = int(BORDER);
        int high = int(BORDER)+int(CHUNK_SIZE);
        small = view(large, xt::range(low,high),xt::range(low,high),xt::range(low,high));
    }
};


//algorithm 2: do interpolation on basis of individual chunks
static inline void readArea3D(xt::xtensor<uint8_t,3,xt::layout_type::column_major> &out, const cv::Vec3i& offset, z5::Dataset *ds, ChunkCache *cache)
{
    //FIXME assert dims
    //FIXME based on key math we should check bounds here using volume and chunk size
    int group_idx = cache->groupIdx(ds->path());

    cv::Vec3i size = {out.shape()[0],out.shape()[1],out.shape()[2]};

    auto chunksize = ds->chunking().blockShape();

    cv::Vec3i to = offset+size;
    cv::Vec3i offset_valid = offset;
    for(int i=0;i<3;i++) {
        offset_valid[i] = std::max(0,offset_valid[i]);
        to[i] = std::max(0,to[i]);
        offset_valid[i] = std::min(int(ds->shape(i)),offset_valid[i]);
        to[i] = std::min(int(ds->shape(i)),to[i]);
    }

    {
        cv::Vec4i last_idx = {-1,-1,-1,-1};
        std::shared_ptr<xt::xarray<uint8_t>> chunk_ref;
        xt::xarray<uint8_t> *chunk = nullptr;
        for(size_t z = offset_valid[0];z<to[0];z++)
            for(size_t y = offset_valid[1];y<to[1];y++)
                for(size_t x = offset_valid[2];x<to[2];x++) {

                    int iz = z/chunksize[0];
                    int iy = y/chunksize[1];
                    int ix = x/chunksize[2];

                    cv::Vec4i idx = {group_idx,iz,iy,ix};

                    if (idx != last_idx) {
                        last_idx = idx;
                        cache->mutex.lock();

                        if (!cache->has(idx)) {
                            cache->mutex.unlock();
                            // std::cout << "reading chunk " << cv::Vec3i(ix,iy,iz) << " for " << cv::Vec3i(x,y,z) << chunksize << std::endl;
                            chunk = readChunk<uint8_t>(*ds, {size_t(iz),size_t(iy),size_t(ix)});
                            cache->mutex.lock();
                            cache->put(idx, chunk);
                        }
                        else {
                            chunk_ref = cache->get(idx);
                            chunk = chunk_ref.get();
                        }
                        cache->mutex.unlock();
                    }

                    if (chunk) {
                        int lz = z-iz*chunksize[0];
                        int ly = y-iy*chunksize[1];
                        int lx = x-ix*chunksize[2];
                        out(z-offset[0], y-offset[1], x-offset[2]) = chunk->operator()(lz,ly,lx);
                    }
            }
    }
}

static uint64_t miss = 0;
static uint64_t total = 0;
static uint64_t chunk_compute_collisions = 0;
static uint64_t chunk_compute_total = 0;

template <typename T, typename C> class Chunked3dAccessor;

static inline std::string tmp_name_proc_thread()
{
    std::stringstream ss;
    ss << "tmp_" << getpid() << "_" << std::this_thread::get_id();
    return ss.str();
}

//chunked 3d tensor for on-demand computation from a zarr dataset ... could as some point be file backed ...
template <typename T, typename C>
class Chunked3d {
public:
    using CHUNKT = xt::xtensor<T,3,xt::layout_type::column_major>;

    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache *cache) : _compute_f(compute_f), _ds(ds), _cache(cache)
    {
        _border = compute_f.BORDER;
    };
    ~Chunked3d()
    {
        if (!_persistent)
            remove_all(_cache_dir);
    };
    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache *cache, const std::filesystem::path &cache_root) : _compute_f(compute_f), _ds(ds), _cache(cache)
    {
        _border = compute_f.BORDER;
        
        if (_ds)
            _shape = {_ds->shape()[0],_ds->shape()[1],_ds->shape()[2]};
        
        if (cache_root.empty())
            return;

        if (!_compute_f.UNIQUE_ID_STRING.size())
            throw std::runtime_error("requested std::filesystem cache for compute function without identifier");
        
        std::filesystem::path root = cache_root/_compute_f.UNIQUE_ID_STRING;
        
        std::filesystem::create_directories(root);
        
        if (!_ds)
            _persistent = false;
        
        //create cache dir while others are competing to do the same
        for(int r=0;r<1000 && _cache_dir.empty();r++) {
            std::set<std::string> paths;
            if (_persistent) {
                for (auto const& entry : std::filesystem::directory_iterator(root))
                    if (std::filesystem::is_directory(entry) && std::filesystem::exists(entry.path()/"meta.json") && std::filesystem::is_regular_file(entry.path()/"meta.json")) {
                        paths.insert(entry.path());
                        std::ifstream meta_f(entry.path()/"meta.json");
                        nlohmann::json meta = nlohmann::json::parse(meta_f);
                        std::filesystem::path src = std::filesystem::canonical(meta["dataset_source_path"]);
                        if (src == std::filesystem::canonical(ds->path())) {
                            _cache_dir = entry.path();
                            break;
                        }
                    }
                
                if (!_cache_dir.empty())
                    continue;
            }
            
            //try generating our own cache dir atomically
            std::filesystem::path tmp_dir = cache_root/tmp_name_proc_thread();
            std::filesystem::create_directories(tmp_dir);
            
            if (_persistent) {
                nlohmann::json meta;
                meta["dataset_source_path"] = std::filesystem::canonical(ds->path()).string();
                std::ofstream o(tmp_dir/"meta.json");
                o << std::setw(4) << meta << std::endl;
                
                std::filesystem::path tgt_path;
                for(int i=0;i<1000;i++) {
                    tgt_path = root/std::to_string(i);
                    if (paths.count(tgt_path.string()))
                        continue;
                    try {
                        std::filesystem::rename(tmp_dir, tgt_path);
                    }
                    catch (std::filesystem::filesystem_error){
                        continue;
                    }
                    _cache_dir = tgt_path;
                    break;
                }
            }
            else {
                _cache_dir = tmp_dir;
            }
        }
        
        if (_cache_dir.empty())
            throw std::runtime_error("could not create cache dir - maybe too many caches in cache root (max 1000!)");
        
    };
    size_t calc_off(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        return p[0] + p[1]*s + p[2]*s*s;
    }
    T &operator()(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        cv::Vec3i id = {p[0]/s,p[1]/s,p[2]/s};

        if (!_chunks.count(id))
            cache_chunk(id);

        return _chunks[id][calc_off({p[0]-id[0]*s,p[1]-id[1]*s,p[2]-id[2]*s})];
    }
    T &operator()(int z, int y, int x)
    {
        return operator()({z,y,x});
    }
    T &safe_at(const cv::Vec3i &p)
    {
        const auto s = C::CHUNK_SIZE;
        const cv::Vec3i id{ p[0]/s, p[1]/s, p[2]/s };

        {
            std::shared_lock<std::shared_mutex> rlock(_mutex);
            if (auto it = _chunks.find(id); it != _chunks.end()) {
                T* chunk = it->second;
                return chunk[calc_off({p[0]-id[0]*s, p[1]-id[1]*s, p[2]-id[2]*s})];
            }
        }
        // compute/load outside shared lock
        T* chunk = cache_chunk_safe(id);
        return chunk[calc_off({p[0]-id[0]*s, p[1]-id[1]*s, p[2]-id[2]*s})];
    }
    T &safe_at(int z, int y, int x)
    {
        return safe_at({z,y,x});
    }

    std::filesystem::path id_path(const std::filesystem::path &dir, const cv::Vec3i &id)
    {
        return dir / (std::to_string(id[0]) + "." + std::to_string(id[1]) + "." + std::to_string(id[2]));
    }

    T *cache_chunk_safe_mmap(const cv::Vec3i &id)
    {
        auto s = C::CHUNK_SIZE;

        std::filesystem::path tgt_path = id_path(_cache_dir, id);
        size_t len = s*s*s;
        size_t len_bytes = len*sizeof(T);

        if (std::filesystem::exists(tgt_path)) {
            int fd = open(tgt_path.string().c_str(), O_RDWR);
            T *chunk = (T*)mmap(NULL, len_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);

            _mutex.lock();
            if (!_chunks.count(id)) {
                _chunks[id] = chunk;
            }
            else {
#pragma omp atomic
                chunk_compute_collisions++;
                munmap(chunk, len_bytes);
                chunk = _chunks[id];
            }
#pragma omp atomic
            chunk_compute_total++;
            _mutex.unlock();

            return chunk;
        }

        std::filesystem::path tmp_path;
        _mutex.lock();
        std::stringstream ss;
        ss << this << "_" << std::this_thread::get_id() << "_" << _tmp_counter++;
        tmp_path = std::filesystem::path(_cache_dir) / ss.str();
        _mutex.unlock();
        int fd = open(tmp_path.string().c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
        int ret = ftruncate(fd, len_bytes);
        if (ret != 0)
            throw std::runtime_error("oops ftruncate failed!");
        T *chunk = (T*)mmap(NULL, len_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        
        cv::Vec3i offset =
        {id[0]*s-_border,
            id[1]*s-_border,
            id[2]*s-_border};

        CHUNKT small = xt::empty<T>({s,s,s});
        CHUNKT large;
        if (_ds) {
            large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
            large = xt::full_like(large, C::FILL_V);

            readArea3D(large, offset, _ds, _cache);
        }

        _compute_f.template compute<CHUNKT,T>(large, small, offset);

        for(int i=0;i<len;i++)
            chunk[i] = (&small(0,0,0))[i];

        _mutex.lock();
        if (!_chunks.count(id)) {
            _chunks[id] = chunk;
            int ret = rename(tmp_path.string().c_str(), tgt_path.string().c_str());

            if (ret)
                throw std::runtime_error("oops rename failed!");
        }
        else {
#pragma omp atomic
            chunk_compute_collisions++;
            munmap(chunk, len_bytes);
            unlink(tmp_path.string().c_str());
            chunk = _chunks[id];
        }
#pragma omp atomic
        chunk_compute_total++;
        _mutex.unlock();

        return chunk;
    }


    T *cache_chunk_safe_alloc(const cv::Vec3i &id)
    {
        auto s = C::CHUNK_SIZE;
        CHUNKT small = xt::empty<T>({s,s,s});

        cv::Vec3i offset =
        {id[0]*s-_border,
            id[1]*s-_border,
            id[2]*s-_border};
            
        CHUNKT large;
        if (_ds) {
            large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
            large = xt::full_like(large, C::FILL_V);
            
            readArea3D(large, offset, _ds, _cache);
        }

        _compute_f.template compute<CHUNKT,T>(large, small, offset);

        T *chunk = nullptr;

        _mutex.lock();
        if (!_chunks.count(id)) {
            chunk = (T*)malloc(s*s*s*sizeof(T));
            memcpy(chunk, &small(0,0,0), s*s*s*sizeof(T));
            _chunks[id] = chunk;
        }
        else {
#pragma omp atomic
            chunk_compute_collisions++;
            chunk = _chunks[id];
        }
#pragma omp atomic
        chunk_compute_total++;
        _mutex.unlock();

        return chunk;
    }

    T *cache_chunk_safe(const cv::Vec3i &id)
    {
        if (_cache_dir.empty())
            return cache_chunk_safe_alloc(id);
        else
            return cache_chunk_safe_mmap(id);
    }

    T *cache_chunk(const cv::Vec3i &id) {
        return cache_chunk_safe(id);
    }

    //T *chunk(const cv::Vec3i &id) {
    //    if (!_chunks.count(id))
    //        return cache_chunk(id);
    //    return _chunks[id];
    //}

    T *chunk_safe(const cv::Vec3i &id) {
        T *chunk = nullptr;
        _mutex.lock_shared();
        if (_chunks.count(id)) {
            chunk = _chunks[id];
            _mutex.unlock();
        }
        else {
            _mutex.unlock();
            chunk = cache_chunk_safe(id);
        }

        return chunk;
    }
    
    std::vector<int> shape() {
        return _shape;
    }

    std::unordered_map<cv::Vec3i,T*,vec3i_hash> _chunks;
    z5::Dataset *_ds;
    ChunkCache *_cache;
    size_t _border;
    C &_compute_f;
    std::shared_mutex _mutex;
    uint64_t _tmp_counter = 0;
    std::filesystem::path _cache_dir;
    bool _persistent = true;
    std::vector<int> _shape;
};

template <typename T, typename C>
class Chunked3dAccessor
{
public:
    using CHUNKT = typename Chunked3d<T,C>::CHUNKT;

    Chunked3dAccessor(Chunked3d<T,C> &ar) : _ar(ar) {};

    static Chunked3dAccessor create(Chunked3d<T,C> &ar)
    {
        return Chunked3dAccessor(ar);
    }

    T &operator()(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;

        if (_corner[0] == -1)
            get_chunk(p);
        else {
            bool miss = false;
            for(int i=0;i<3;i++)
                if (p[i] < _corner[i])
                    miss = true;
            for(int i=0;i<3;i++)
                if (p[i] >= _corner[i]+C::CHUNK_SIZE)
                    miss = true;
            if (miss)
                get_chunk(p);
        }

        total++;

        return _chunk[_ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]})];
    }



    T &operator()(int z, int y, int x)
    {
        return operator()({z,y,x});
    }

    T& safe_at(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;

        if (_corner[0] == -1)
            get_chunk_safe(p);
        else {
            bool miss = false;
            for(int i=0;i<3;i++)
                if (p[i] < _corner[i])
                    miss = true;
            for(int i=0;i<3;i++)
                if (p[i] >= _corner[i]+C::CHUNK_SIZE)
                    miss = true;
            if (miss)
                get_chunk_safe(p);
        }

        #pragma omp atomic
        total++;

        return _chunk[_ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]})];
    }

    T& safe_at(int z, int y, int x)
    {
        return safe_at({z,y,x});
    }

    void get_chunk(const cv::Vec3i &p)
    {
        miss++;
        cv::Vec3i id = {p[0]/C::CHUNK_SIZE,p[1]/C::CHUNK_SIZE,p[2]/C::CHUNK_SIZE};
        _chunk = _ar.chunk(id);
        _corner = id*C::CHUNK_SIZE;
    }

    void get_chunk_safe(const cv::Vec3i &p)
    {
        #pragma omp atomic
        miss++;
        cv::Vec3i id = {p[0]/C::CHUNK_SIZE,p[1]/C::CHUNK_SIZE,p[2]/C::CHUNK_SIZE};
        _chunk = _ar.chunk_safe(id);
        _corner = id*C::CHUNK_SIZE;
    }

    Chunked3d<T,C> &_ar;
protected:
    T *_chunk;
    cv::Vec3i _corner = {-1,-1,-1};
};

// ────────────────────────────────────────────────────────────────────────────────
//  CachedChunked3dInterpolator – thread-safe version
// ────────────────────────────────────────────────────────────────────────────────

template <typename T, typename C>
class CachedChunked3dInterpolator
{
public:
    using Acc   = Chunked3dAccessor<T, C>;
    using ArRef = Chunked3d<T, C>&;

    explicit CachedChunked3dInterpolator(ArRef ar)
        : _ar(ar), _shape(ar.shape())
    {}

    CachedChunked3dInterpolator(const CachedChunked3dInterpolator&)            = delete;
    CachedChunked3dInterpolator& operator=(const CachedChunked3dInterpolator&) = delete;

    /** Trilinear interpolation. */
    template <typename V>
    inline void Evaluate(const V& z, const V& y, const V& x, V* out) const
    {
        // ---- 1. get *this* thread’s private accessor ------------------------
        Acc& a = local_accessor();

        // ---- 2. fast trilinear interpolation --------------------
        cv::Vec3d f { val(z), val(y), val(x) };

        cv::Vec3i corner { static_cast<int>(std::floor(f[0])),
                           static_cast<int>(std::floor(f[1])),
                           static_cast<int>(std::floor(f[2])) };

        for (int i=0; i<3; ++i) {
            corner[i] = std::max(corner[i], 0);
            if (!_shape.empty())
                corner[i] = std::min(corner[i], _shape[i]-2);
        }

        const V fx = z - V(corner[0]);
        const V fy = y - V(corner[1]);
        const V fz = x - V(corner[2]);

        // clamp only once – cheaper than three branches per component
        const V cx = std::clamp(fx, V(0), V(1));
        const V cy = std::clamp(fy, V(0), V(1));
        const V cz = std::clamp(fz, V(0), V(1));

        // fetch the eight lattice points
        const V c000 = V(a.safe_at(corner));
        const V c100 = V(a.safe_at(corner + cv::Vec3i(1,0,0)));
        const V c010 = V(a.safe_at(corner + cv::Vec3i(0,1,0)));
        const V c110 = V(a.safe_at(corner + cv::Vec3i(1,1,0)));
        const V c001 = V(a.safe_at(corner + cv::Vec3i(0,0,1)));
        const V c101 = V(a.safe_at(corner + cv::Vec3i(1,0,1)));
        const V c011 = V(a.safe_at(corner + cv::Vec3i(0,1,1)));
        const V c111 = V(a.safe_at(corner + cv::Vec3i(1,1,1)));

        // interpolate
        const V c00 = (V(1)-cz)*c000 + cz*c001;
        const V c01 = (V(1)-cz)*c010 + cz*c011;
        const V c10 = (V(1)-cz)*c100 + cz*c101;
        const V c11 = (V(1)-cz)*c110 + cz*c111;

        const V c0  = (V(1)-cy)*c00 + cy*c01;
        const V c1  = (V(1)-cy)*c10 + cy*c11;

        *out = (V(1)-cx)*c0 + cx*c1;
    }

    // -------------------------------------------------------------------------
    double val(const double& v) const { return v; }
    template <typename JetT> double val(const JetT& v) const { return v.a; }

private:
    /** Return the accessor that is *unique to this combination of
     *  (interpolator instance, thread)*. */
    Acc& local_accessor() const
    {
        // Per-thread, bounded cache keyed by the underlying array address.
        // (Multiple interpolators over the same array share one accessor.)
        struct TLS {
            std::unordered_map<const void*, Acc> map;
            std::deque<const void*> order;          // FIFO ~ LRU-ish
        };
        thread_local TLS tls;

        constexpr std::size_t kMax = CCI_TLS_MAX;

        const void* key = static_cast<const void*>(&_ar);

        if (auto it = tls.map.find(key); it != tls.map.end()) {
            return it->second;
        }

        // Evict oldest if at capacity
        if (tls.map.size() >= kMax && !tls.order.empty()) {
            const void* old = tls.order.front();
            tls.order.pop_front();
            tls.map.erase(old);
        }

        auto [it2, inserted] = tls.map.emplace(key, Acc{_ar});
        if (inserted) tls.order.push_back(key);
        return it2->second;
    }

    ArRef               _ar;
    std::vector<int>    _shape;
};

static inline void print_accessor_stats()
{
    std::cout << "acc miss/total " << miss << " " << total << " " << double(miss)/total << std::endl;
    std::cout << "chunk compute overhead/total " << chunk_compute_collisions << " " << chunk_compute_total << " " << double(chunk_compute_collisions)/chunk_compute_total << std::endl;
}