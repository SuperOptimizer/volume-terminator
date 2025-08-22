#include <nlohmann/json.hpp>

#include "xtensor/io/xio.hpp"
#include "xtensor/views/xview.hpp"
#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/common.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include <opencv2/core.hpp>

#include "../core/Slicing.hpp"
#include "../core/Surface.hpp"

#include <filesystem>

#include "SurfaceHelpers.hpp"

using shape = z5::types::ShapeType;
using namespace xt::placeholders;

using json = nlohmann::json;

std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

class MeasureLife
{
public:
    MeasureLife(std::string msg)
    {
        std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    ~MeasureLife()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};



template <typename T, typename I>
float get_val(I &interp, cv::Vec3d l) {
    T v;
    interp.Evaluate(l[2], l[1], l[0], &v);
    return v;
}

int main(int argc, char *argv[])
{
    if (argc != 6) {
        std::cout << "usage: " << argv[0] << " <zarr-volume> <src-dir> <tgt-dir> <json-params> <src-segment>" << std::endl;
        return EXIT_SUCCESS;
    }

    std::filesystem::path vol_path = argv[1];
    std::filesystem::path src_dir = argv[2];
    std::filesystem::path tgt_dir = argv[3];
    std::filesystem::path params_path = argv[4];
    std::filesystem::path src_path = argv[5];
    while (src_path.filename().empty())
        src_path = src_path.parent_path();

    std::ifstream params_f(params_path);
    json params = json::parse(params_f);
    params["tgt_dir"] = tgt_dir;

    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, "0", json::parse(std::ifstream(vol_path/"0/.zarray")).value<std::string>("dimension_separator","."));
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group 0 " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;

    float voxelsize = json::parse(std::ifstream(vol_path/"meta.json"))["voxelsize"];

    std::string name_prefix = "auto_grown_";
    std::vector<QuadSurface*> surfaces;
    std::filesystem::path meta_fn = src_path / "meta.json";
    std::ifstream meta_f(meta_fn);
    json meta = json::parse(meta_f);
    QuadSurface *src = new QuadSurface(src_path);
    src->readOverlapping();

    for (const auto& entry : std::filesystem::directory_iterator(src_dir))
        if (std::filesystem::is_directory(entry)) {
            std::string name = entry.path().filename();
            if (name.compare(0, name_prefix.size(), name_prefix))
                continue;

            std::filesystem::path meta_fn = entry.path() / "meta.json";
            if (!std::filesystem::exists(meta_fn))
                continue;

            std::ifstream meta_f(meta_fn);
            json meta = json::parse(meta_f);

            if (!meta.count("bbox"))
                continue;

            if (meta.value("format","NONE") != "tifxyz")
                continue;

            QuadSurface *qs;
            if (entry.path().filename() == src->name())
                qs = src;
            else {
                qs = new QuadSurface(entry.path());
                qs->readOverlapping();
            }

            surfaces.push_back(qs);
        }

    QuadSurface *surf = grow_surf_from_surfs(src, surfaces, params, voxelsize);

    if (!surf)
        return EXIT_SUCCESS;

    (*surf->meta)["source"] = "vc_grow_seg_from_segments";
    (*surf->meta)["vc_grow_seg_from_segments_params"] = params;
    std::string uuid = "auto_trace_" + get_surface_time_str();;
    std::filesystem::path seg_dir = tgt_dir / uuid;
    surf->save(seg_dir, uuid);

    delete surf;
    for(auto qs : surfaces) {
        delete qs;
    }

    return EXIT_SUCCESS;
}
