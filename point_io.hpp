#ifndef POINTIO_H
#define POINTIO_H

#include <iostream>
#include <fstream>
#include <torch/torch.h>

#ifdef WITH_PDAL
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/io/BufferReader.hpp>
#endif

#include <nanoflann.hpp>

struct XYZ {
    float x;
    float y;
    float z;
};

#define KDTREE_MAX_LEAF 10

#define RELEASE_POINTSET(__POINTER) { if (__POINTER != nullptr) { __POINTER->freeIndex<KdTree>(); delete __POINTER; __POINTER = nullptr; } }

struct PointSet {
    std::vector<std::array<float, 3> > points;
    std::vector<std::array<uint8_t, 3> > colors;

    std::vector<std::array<float, 3> > normals;
    std::vector<uint8_t> views;

    void *kdTree = nullptr;

    #ifdef WITH_PDAL
    pdal::PointViewPtr pointView = nullptr;
    #endif

    template <typename T>
    inline T *getIndex() {
        return kdTree != nullptr ? reinterpret_cast<T *>(kdTree) : buildIndex<T>();
    }

    template <typename T>
    inline T *buildIndex() {
        if (kdTree == nullptr) kdTree = static_cast<void *>(new T(3, *this, { KDTREE_MAX_LEAF }));
        return reinterpret_cast<T *>(kdTree);
    }

    inline size_t count() const { return points.size(); }
    inline size_t kdtree_get_point_count() const { return points.size(); }
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return points[idx][dim];
    };
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const
    {
        return false;
    }

    void appendPoint(PointSet &src, size_t idx) {
        points.push_back(src.points[idx]);
        colors.push_back(src.colors[idx]);
    }

    bool hasNormals() const { return normals.size() > 0; }
    bool hasColors() const { return colors.size() > 0; }
    bool hasViews() const { return views.size() > 0; }

    double spacing(int kNeighbors = 3);

    template <typename T>
    void freeIndex() {
        if (kdTree != nullptr) {
            T *tree = getIndex<T>();
            delete tree;
            kdTree = nullptr;
        }
    }

    inline torch::Tensor colorsTensor(){
        return torch::from_blob(colors.data(), { static_cast<long int>(colors.size()), 3 }, torch::kU8);
    }
    inline torch::Tensor pointsTensor(){
        return torch::from_blob(points.data(), { static_cast<long int>(points.size()), 3 }, torch::kFloat32);
    }
    

    ~PointSet() {
    }
private:
    double m_spacing = -1.0;
};

using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointSet>,
    PointSet, 3, size_t
>;

std::string getVertexLine(std::ifstream &reader);
size_t getVertexCount(const std::string &line);
inline void checkHeader(std::ifstream &reader, const std::string &prop);
inline bool hasHeader(const std::string &line, const std::string &prop);

template <typename T>
inline T readBinary(std::ifstream &s){
    T data;
    s.read(reinterpret_cast<char*>(&data), sizeof(T));
    return data;
}

PointSet *fastPlyReadPointSet(const std::string &filename);
PointSet *pdalReadPointSet(const std::string &filename);
PointSet *colmapReadPointSet(const std::string &filename);
PointSet *readPointSet(const std::string &filename);

void fastPlySavePointSet(PointSet &pSet, const std::string &filename);
void pdalSavePointSet(PointSet &pSet, const std::string &filename);
void savePointSet(PointSet &pSet, const std::string &filename);

bool fileExists(const std::string &path);


#endif
