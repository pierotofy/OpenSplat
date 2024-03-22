#ifndef TENSOR_MATH_H
#define TENSOR_MATH_H

#include <torch/torch.h>
#include <tuple>

torch::Tensor quatToRotMat(const torch::Tensor &quat);
std::tuple<torch::Tensor, torch::Tensor, float> autoScaleAndCenterPoses(const torch::Tensor &poses);
torch::Tensor rotationMatrix(const torch::Tensor &a, const torch::Tensor &b);


#endif