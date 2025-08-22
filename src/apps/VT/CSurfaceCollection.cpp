#include "CSurfaceCollection.hpp"

#include "../../core/Surface.hpp"


CSurfaceCollection::~CSurfaceCollection()
{
    for (auto &val: _surfs | std::views::values) {
        delete val;
    }

    for (auto &val: _pois | std::views::values) {
        delete val;
    }

    for (auto &val: _intersections | std::views::values) {
        delete val;
    }
}

void CSurfaceCollection::setSurface(const std::string &name, Surface* surf, bool noSignalSend)
{
    _surfs[name] = surf;
    if (!noSignalSend) {
        sendSurfaceChanged(name, surf);
    }
}

void CSurfaceCollection::setPOI(const std::string &name, POI *poi)
{
    _pois[name] = poi;
    sendPOIChanged(name, poi);
}

Surface* CSurfaceCollection::surface(const std::string &name)
{
    if (!_surfs.contains(name))
        return nullptr;
    return _surfs[name];
}

POI *CSurfaceCollection::poi(const std::string &name)
{
    if (!_pois.contains(name))
        return nullptr;
    return _pois[name];
}

std::vector<Surface*> CSurfaceCollection::surfaces() const {
    std::vector<Surface*> surfaces;
    surfaces.reserve(_surfs.size());

    for(const auto &val: _surfs | std::views::values) {
        surfaces.push_back(val);
    } 

    return surfaces;
}

std::vector<POI*> CSurfaceCollection::pois() const {
    std::vector<POI*> pois;
    pois.reserve(_pois.size());

    for(const auto &val: _pois | std::views::values) {
        pois.push_back(val);
    } 

    return pois;
}

std::vector<std::string> CSurfaceCollection::surfaceNames() const {
    std::vector<std::string> keys;
    for(const auto &key: _surfs | std::views::keys)
        keys.push_back(key);
    
    return keys;
}

std::vector<std::string> CSurfaceCollection::poiNames() const {
    std::vector<std::string> keys;
    for(const auto &key: _pois | std::views::keys)
        keys.push_back(key);

    return keys;
}

void CSurfaceCollection::setIntersection(const std::string &a, const std::string &b, Intersection *intersect)
{
    _intersections[{a,b}] = intersect;
    sendIntersectionChanged(a, b, intersect);
}

Intersection *CSurfaceCollection::intersection(const std::string &a, const std::string &b)
{
    if (_intersections.contains({a,b}))
        return _intersections[{a,b}];
        
    if (_intersections.contains({b,a}))
        return _intersections[{b,a}];
    
    return nullptr;
}

std::vector<std::pair<std::string,std::string>> CSurfaceCollection::intersections(const std::string &a) const {
    std::vector<std::pair<std::string,std::string>> res;

    if (!a.size()) {
        for(const auto &key: _intersections | std::views::keys)
            res.push_back(key);
    }
    else
        for(const auto &key: _intersections | std::views::keys) {
            if (key.first == a)
                res.push_back(key);
            else if (key.second == a)
                res.push_back(key);
        }
    return res;
}
