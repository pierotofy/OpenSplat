#ifndef TENSOR_MATH_H
#define TENSOR_MATH_H

#include <torch/torch.h>
#include <tuple>
#include "constants.hpp"

torch::Tensor quatToRotMat(const torch::Tensor &quat);
torch::Tensor rotationMatrix(const torch::Tensor &a, const torch::Tensor &b);
torch::Tensor rodriguesToRotation(const torch::Tensor &rodrigues);

#endif
