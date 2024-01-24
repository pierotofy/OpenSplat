#include "project_gaussians.h"
#include "bindings.h"

tensor_list ProjectGaussians::forward(AutogradContext *ctx, 
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
                float clipThresh
            ){
    
    int numPoints = means.size(0);

    auto r = project_gaussians_forward_tensor(numPoints, means, scales, globScale,
                                              quats, viewMat, projMat, fx, fy,
                                              cx, cy, imgHeight, imgWidth, tileBounds, clipThresh);
    std::cout << std::get<0>(r) << std::endl;
    exit(1);
    // ctx->saved_data["constant"] = constant;
    return { means, means, means, means, means, means };
}

tensor_list ProjectGaussians::backward(AutogradContext *ctx, tensor_list grad_outputs) {
    // We return as many input gradients as there were arguments.
    // Gradients of non-tensor arguments to forward must be `torch::Tensor()`.
    // return {grad_outputs[0] * ctx->saved_data["constant"].toDouble(), torch::Tensor()};
    return {torch::Tensor()};
}