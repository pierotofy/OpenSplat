#include "kdtree_tensor.hpp"


torch::Tensor PointsTensor::scales(){
    // Compute scales by finding the average of the three nearest neighbors for each point
    const auto index = getIndex<KdTreeTensor>();
    torch::Tensor scales = torch::zeros({static_cast<long int>(tensor.size(0)), 1}, torch::kFloat32);
    const int count = 4;

    std::vector<size_t> indices(count);
    std::vector<float> sqr_dists(count);
    for (size_t i = 0; i < tensor.size(0); i++){
        index->knnSearch(reinterpret_cast<float *>(tensor[i].data_ptr()), count, indices.data(), sqr_dists.data());

        float sum = 0.0;
        for (size_t j = 1; j < count; j++) {
            sum += std::sqrt(sqr_dists[j]);
        }
        scales[i] = sum / (count - 1);
    }

    return scales;
}

PointsTensor::~PointsTensor(){
    freeIndex<KdTreeTensor>();
}