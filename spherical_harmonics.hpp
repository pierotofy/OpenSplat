#ifndef SPHERICAL_HARMONICS_H
#define SPHERICAL_HARMONICS_H

#include <torch/torch.h>

using namespace torch::autograd;

int numShBases(int degree);
int degFromSh(int numBases);
torch::Tensor rgb2sh(const torch::Tensor &rgb);

class SphericalHarmonics : public Function<SphericalHarmonics>{
public:
    static torch::Tensor forward(AutogradContext *ctx, 
            int degreesToUse, 
            torch::Tensor viewDirs, 
            torch::Tensor coeffs);
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

#endif