#include "spherical_harmonics.hpp"

int numShBases(int degree){
    switch(degree){
        case 0:
            return 1;
        case 1:
            return 4;
        case 2:
            return 9;
        case 3:
            return 16;
        default:
            return 25;
    }
    return 25;
}

torch::Tensor rgb2sh(const torch::Tensor &rgb){
    // Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    const double C0 = 0.28209479177387814;
    return (rgb - 0.5) / C0;
}