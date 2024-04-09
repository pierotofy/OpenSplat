#include "project_gaussians.hpp"

#if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)

variable_list ProjectGaussians::forward(AutogradContext *ctx, 
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

    auto t = project_gaussians_forward_tensor(numPoints, means, scales, globScale,
                                              quats, viewMat, projMat, fx, fy,
                                              cx, cy, imgHeight, imgWidth, tileBounds, clipThresh);
    torch::Tensor cov3d = std::get<0>(t);
    torch::Tensor xys = std::get<1>(t);
    torch::Tensor depths = std::get<2>(t);
    torch::Tensor radii = std::get<3>(t);
    torch::Tensor conics = std::get<4>(t);
    torch::Tensor numTilesHit = std::get<5>(t);

    ctx->saved_data["imgHeight"] = imgHeight;
    ctx->saved_data["imgWidth"] = imgWidth;
    ctx->saved_data["numPoints"] = numPoints;
    ctx->saved_data["globScale"] = globScale;
    ctx->saved_data["fx"] = fx;
    ctx->saved_data["fy"] = fy;
    ctx->saved_data["cx"] = cx;
    ctx->saved_data["cy"] = cy;
    ctx->save_for_backward({ means, scales, quats, viewMat, projMat, cov3d, radii, conics });

    return { xys, depths, radii, conics, numTilesHit, cov3d };
}

tensor_list ProjectGaussians::backward(AutogradContext *ctx, tensor_list grad_outputs) {
    torch::Tensor v_xys = grad_outputs[0];
    torch::Tensor v_depths = grad_outputs[1];
    torch::Tensor v_radii = grad_outputs[2];
    torch::Tensor v_conics = grad_outputs[3];
    torch::Tensor v_numTiles = grad_outputs[4];
    torch::Tensor v_cov3d = grad_outputs[5];

    variable_list saved = ctx->get_saved_variables();
    torch::Tensor means = saved[0];
    torch::Tensor scales = saved[1];
    torch::Tensor quats = saved[2];
    torch::Tensor viewMat = saved[3];
    torch::Tensor projMat = saved[4];
    torch::Tensor cov3d = saved[5];
    torch::Tensor radii = saved[6];
    torch::Tensor conics = saved[7];
    
    auto t = project_gaussians_backward_tensor(ctx->saved_data["numPoints"].toInt(), 
                                            means, scales, ctx->saved_data["globScale"].toDouble(),
                                            quats, viewMat, projMat, 
                                            ctx->saved_data["fx"].toDouble(), ctx->saved_data["fy"].toDouble(),
                                            ctx->saved_data["cx"].toDouble(), ctx->saved_data["cy"].toDouble(), 
                                            ctx->saved_data["imgHeight"].toInt(), ctx->saved_data["imgWidth"].toInt(), 
                                            cov3d, radii,
                                            conics, v_xys, v_depths, v_conics);
    torch::Tensor none;

    return {std::get<2>(t), // v_mean
            std::get<3>(t), // v_scale
            none, // globScale
            std::get<4>(t), // v_quat
            none, // viewMat
            none, // projMat
            none, // fx
            none, // fy
            none, // cx
            none, // cy
            none, // imgHeight
            none, // imgWidth
            none, // tileBounds
            none // clipThresh
        };
}

#endif

variable_list ProjectGaussiansCPU::apply(
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
                float clipThresh
            ){
    
    int numPoints = means.size(0);

    auto t = project_gaussians_forward_tensor_cpu(numPoints, means, scales, globScale,
                                              quats, viewMat, projMat, fx, fy,
                                              cx, cy, imgHeight, imgWidth, clipThresh);
                                              
    torch::Tensor xys = std::get<0>(t);
    torch::Tensor radii = std::get<1>(t);
    torch::Tensor conics = std::get<2>(t);
    torch::Tensor cov2d = std::get<3>(t);
    torch::Tensor camDepths = std::get<4>(t);

    return { xys, radii, conics, cov2d, camDepths };
}