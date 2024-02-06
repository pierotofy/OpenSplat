#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include "nerfstudio.hpp"
#include "kdtree_tensor.hpp"

namespace ns{

struct Model : torch::nn::Module {
  Model(const Points &points) {
    means = register_parameter("means", points.xyz, true);
    scales = register_parameter("scales", PointsTensor(means).scales(), true);
    std::cout << scales << std::endl;
    exit(1);

  }

  torch::Tensor means;
  torch::Tensor scales;
};

}

#endif