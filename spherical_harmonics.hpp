#ifndef SPHERICAL_HARMONICS_H
#define SPHERICAL_HARMONICS_H

#include <torch/torch.h>

int numShBases(int degree);
torch::Tensor rgb2sh(const torch::Tensor &rgb);

#endif