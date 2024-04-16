#ifndef PROJECT_GAUSSIANS_H
#define PROJECT_GAUSSIANS_H

#include <torch/torch.h>
#include "tile_bounds.hpp"
#include "gsplat.hpp"

using namespace torch::autograd;

#if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)

class ProjectGaussians : public Function<ProjectGaussians>{
public:
    static variable_list forward(AutogradContext *ctx, 
            torch::Tensor means,
            torch::Tensor scales,
            float globScale,
            torch::Tensor quats,
            torch::Tensor viewMat,
            torch::Tensor projMat,
            float fx,
            float fy,
            float cx,
            float cy,
            int imgHeight,
            int imgWidth,
            TileBounds tileBounds,
            float clipThresh = 0.01);
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

#endif

class ProjectGaussiansCPU{
public:
    static variable_list apply( 
            torch::Tensor means,
            torch::Tensor scales,
            float globScale,
            torch::Tensor quats,
            torch::Tensor viewMat,
            torch::Tensor projMat,
            float fx,
            float fy,
            float cx,
            float cy,
            int imgHeight,
            int imgWidth,
            float clipThresh = 0.01);
};


#endif