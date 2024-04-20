#ifndef KDTREE_TENSOR
#define KDTREE_TENSOR

#include <torch/torch.h>
#include <nanoflann.hpp>

struct PointsTensor {
    torch::Tensor tensor;
    torch::TensorAccessor<float, 2> accessor;

    void *kdTree = nullptr;

    template <typename T>
    inline T *getIndex() {
        return kdTree != nullptr ? reinterpret_cast<T *>(kdTree) : buildIndex<T>();
    }

    template <typename T>
    inline T *buildIndex() {
        if (kdTree == nullptr) kdTree = static_cast<void *>(new T(3, *this, { 10 }));
        return reinterpret_cast<T *>(kdTree);
    }

    inline size_t kdtree_get_point_count() const { return tensor.size(0); }
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return accessor[idx][dim];
    };
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const
    {
        return false;
    }

    template <typename T>
    void freeIndex() {
        if (kdTree != nullptr) {
            T *tree = getIndex<T>();
            delete tree;
            kdTree = nullptr;
        }
    }

    PointsTensor(const torch::Tensor &tensor) : tensor(tensor), accessor(tensor.accessor<float, 2>()){
    }
    ~PointsTensor();

    torch::Tensor scales();
};

using KdTreeTensor = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointsTensor>,
    PointsTensor, 3, size_t
>;

#endif