#include "spherical_harmonics.hpp"

int degFromSh(int numBases){
    switch(numBases){
        case 1:
            return 0;
        case 4:
            return 1;
        case 9:
            return 2;
        case 16:
            return 3;
        default:
            return 4;
    }
}

const double C0 = 0.28209479177387814;

torch::Tensor rgb2sh(const torch::Tensor &rgb){
    // Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    return (rgb - 0.5) / C0;
}

torch::Tensor sh2rgb(const torch::Tensor &sh){
    // Converts from 0th spherical harmonic coefficients to RGB values [0,1]
    return torch::clamp((sh * C0) + 0.5, 0.0f, 1.0f);
}

#if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)

torch::Tensor SphericalHarmonics::forward(AutogradContext *ctx, 
            int degreesToUse, 
            torch::Tensor viewDirs, 
            torch::Tensor coeffs){
    long long numPoints = coeffs.size(0);
    int degree = degFromSh(coeffs.size(-2));

    ctx->saved_data["degreesToUse"] = degreesToUse;
    ctx->saved_data["degree"] = degree; 

    ctx->save_for_backward({ viewDirs });

    return compute_sh_forward_tensor(numPoints, degree, degreesToUse, viewDirs, coeffs);
}

tensor_list SphericalHarmonics::backward(AutogradContext *ctx, tensor_list grad_outputs){
    torch::Tensor v_colors = grad_outputs[0];
    int degreesToUse = ctx->saved_data["degreesToUse"].toInt();
    int degree = ctx->saved_data["degree"].toInt();
    variable_list saved = ctx->get_saved_variables();

    torch::Tensor viewDirs = saved[0];
    long long numPoints = v_colors.size(0);
    torch::Tensor none;

    return {
        none,
        none,
        compute_sh_backward_tensor(numPoints, degree, degreesToUse, viewDirs, v_colors)
    };
}

#endif

torch::Tensor SphericalHarmonicsCPU::apply(int degreesToUse, 
            torch::Tensor viewDirs, 
            torch::Tensor coeffs){
    long long numPoints = coeffs.size(0);
    int degree = degFromSh(coeffs.size(-2)); 

    return compute_sh_forward_tensor_cpu(numPoints, degree, degreesToUse, viewDirs, coeffs);
}