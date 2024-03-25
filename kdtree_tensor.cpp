#include "kdtree_tensor.hpp"


torch::Tensor PointsTensor::scales(){
    // Compute scales by finding the average of the three nearest neighbors for each point
    const auto index = getIndex<KdTreeTensor>();
    torch::Tensor scales = torch::zeros({static_cast<long int>(tensor.size(0)), 1}, torch::kFloat32);
    const int count = 4;

    std::vector<size_t> indices(count);
    std::vector<float> sqrtDists(count);
    for (size_t i = 0; i < tensor.size(0); i++){
        index->knnSearch(reinterpret_cast<float *>(tensor[i].data_ptr()), count, indices.data(), sqrtDists.data());

        float sum = 0.0;
        for (size_t j = 1; j < count; j++) {
            sum += std::sqrt(sqrtDists[j]);
        }
        scales[i] = sum / (count - 1);
    }

    freeIndex<KdTreeTensor>();

    return scales;
}

torch::Tensor PointsTensor::outliers(double multiplier, int meanK){
    // Roughly based on https://github.com/PDAL/PDAL/blob/master/filters/OutlierFilter.cpp
    const auto index = getIndex<KdTreeTensor>();

    size_t numPoints = tensor.size(0);
    torch::Tensor outliers = torch::zeros({static_cast<long int>(numPoints), 1}, torch::TensorOptions().dtype(torch::kBool));
    bool *pOutliers = static_cast<bool *>(outliers.data_ptr());

    size_t count = meanK + 1;
    std::vector<double> distances(numPoints, 0.0);
    std::vector<size_t> indices(count);
    std::vector<float> sqrtDists(count);

    // TODO: this can be made parallel
    for (size_t i = 0; i < numPoints; i++){
        index->knnSearch(reinterpret_cast<float *>(tensor[i].data_ptr()), count, indices.data(), sqrtDists.data());

        for (size_t j = 1; j < count; ++j){
            double delta = std::sqrt(sqrtDists[j]) - distances[i];
            distances[i] += (delta / j);
        }
        indices.clear(); 
        indices.resize(count);
        sqrtDists.clear(); 
        sqrtDists.resize(count);
    }

    size_t n = 0;
    double M1 = 0.0;
    double M2 = 0.0;
    for (double const& d : distances){
        size_t n1 = n;
        n++;
        double delta = d - M1;
        double deltaN = delta / n;
        M1 += deltaN;
        M2 += delta * deltaN * n1;
    }

    double mean = M1;
    double variance = M2 / (n - 1.0);
    double stdev = std::sqrt(variance);

    double threshold = mean + multiplier * stdev;

    int pIn = 0;
    int pOut = 0;
    for (size_t i = 0; i < numPoints; i++){
        pOutliers[i] = distances[i] < threshold;

        if (distances[i] < threshold){
            pIn++;
        }else{
            pOut++;
        }
    }

    std::cout << pIn << " " << pOut  << std::endl;

    freeIndex<KdTreeTensor>();

    return outliers;
}

PointsTensor::~PointsTensor(){
    freeIndex<KdTreeTensor>();
}