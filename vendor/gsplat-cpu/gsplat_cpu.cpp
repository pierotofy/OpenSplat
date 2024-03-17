// Originally based on https://github.dev/nerfstudio-project/gsplat
// This implementation is licensed under the AGPLv3

#include "bindings.h"
#include "../gsplat/config.h"

#include <cstdio>
#include <iostream>
#include <math.h>
#include <tuple>

using namespace torch::indexing;

torch::Tensor quatToRotMat(const torch::Tensor &quat){
    auto u = torch::unbind(torch::nn::functional::normalize(quat, torch::nn::functional::NormalizeFuncOptions().dim(-1)), -1);
    torch::Tensor w = u[0];
    torch::Tensor x = u[1];
    torch::Tensor y = u[2];
    torch::Tensor z = u[3];
    return torch::stack({
        torch::stack({
            1.0 - 2.0 * (y.pow(2) + z.pow(2)),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y)
        }, -1),
        torch::stack({
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x.pow(2) + z.pow(2)),
            2.0 * (y * z - w * x)
        }, -1),
        torch::stack({
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x.pow(2) + y.pow(2))
        }, -1)
    }, -2);
    
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor_cpu(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    torch::Tensor &viewmat,
    torch::Tensor &projmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    const float clip_thresh
){
    float fovx = 0.5f * static_cast<float>(img_width) / fx;
    float fovy = 0.5f * static_cast<float>(img_height) / fy;
    
    // TODO: no need to recompute W,p,t below (they are the same)

    // clip_near_plane
    torch::Tensor Rclip = viewmat.index({"...", Slice(None, 3), Slice(None, 3)}); 
    torch::Tensor Tclip = viewmat.index({"...", Slice(None, 3), 3});
    torch::Tensor pView = torch::matmul(Rclip, means3d.index({"...", None})).index({"...", 0}) + Tclip;
    // torch::Tensor isClose = pView.index({"...", 2}) < clip_thresh;

    // scale_rot_to_cov3d
    torch::Tensor R = quatToRotMat(quats);
    torch::Tensor M = R * glob_scale * scales.index({"...", None, Slice()});
    torch::Tensor cov3d = torch::matmul(M, M.transpose(-1, -2));

    // project_cov3d_ewa
    torch::Tensor W = viewmat.index({"...", Slice(None, 3), Slice(None, 3)});
    torch::Tensor p = viewmat.index({"...", Slice(None, 3), 3});
    torch::Tensor t = torch::matmul(W, means3d.index({"...", None})).index({"...", 0}) + p;

    torch::Tensor limX = 1.3f * torch::tensor({fovx}, means3d.device());
    torch::Tensor limY = 1.3f * torch::tensor({fovy}, means3d.device());
    
    torch::Tensor minLimX = t.index({"...", 2}) * torch::min(limX, torch::max(-limX, t.index({"...", 0}) / t.index({"...", 2})));
    torch::Tensor minLimY = t.index({"...", 2}) * torch::min(limY, torch::max(-limY, t.index({"...", 1}) / t.index({"...", 2})));
    
    t = torch::cat({minLimX.index({"...", None}), minLimY.index({"...", None}), t.index({"...", 2, None})}, -1);
    torch::Tensor rz = 1.0f / t.index({"...", 2});
    torch::Tensor rz2 = rz.pow(2);

    torch::Tensor J = torch::stack({
        torch::stack({fx * rz, torch::zeros_like(rz), -fx * t.index({"...", 0}) * rz2}, -1),
        torch::stack({torch::zeros_like(rz), fy * rz, -fy * t.index({"...", 1}) * rz2}, -1)
    }, -2);

    torch::Tensor T = torch::matmul(J, W);
    torch::Tensor cov2d = torch::matmul(T, torch::matmul(cov3d, T.transpose(-1, -2)));

    // Add blur along axes
    cov2d.index_put_({"...", 0, 0}, cov2d.index({"...", 0, 0}) + 0.3f);
    cov2d.index_put_({"...", 1, 1}, cov2d.index({"...", 1, 1}) + 0.3f);
     
    // compute_cov2d_bounds
    float eps = 1e-6f;
    torch::Tensor det = cov2d.index({"...", 0, 0}) * cov2d.index({"...", 1, 1}) - cov2d.index({"...", 0, 1}).pow(2);
    det = torch::clamp_min(det, eps);
    torch::Tensor conic = torch::stack({
            cov2d.index({"...", 1, 1}) / det,
            -cov2d.index({"...", 0, 1}) / det,
            cov2d.index({"...", 0, 0}) / det
        }, -1);

    torch::Tensor b = (cov2d.index({"...", 0, 0}) + cov2d.index({"...", 1, 1})) / 2.0f;
    torch::Tensor sq = torch::sqrt(torch::clamp_min(b.pow(2) - det, 0.1f));
    torch::Tensor v1 = b + sq;
    torch::Tensor v2 = b - sq;
    torch::Tensor radius = torch::ceil(3.0f * torch::sqrt(torch::max(v1, v2)));
    // torch::Tensor detValid = det > eps;

    // project_pix
    torch::Tensor pHom = torch::nn::functional::pad(means3d, torch::nn::functional::PadFuncOptions({0, 1}).mode(torch::kConstant).value(1.0f));
    pHom = torch::einsum("...ij,...j->...i", {projmat, pHom});
    torch::Tensor rw = 1.0f / torch::clamp_min(pHom.index({"...", 3}), eps);
    torch::Tensor pProj = pHom.index({"...", Slice(None, 3)}) * rw.index({"...", None});
    torch::Tensor u = 0.5f * ((pProj.index({"...", 0}) + 1.0f) * static_cast<float>(img_height) - 1.0f);
    torch::Tensor v = 0.5f * ((pProj.index({"...", 1}) + 1.0f) * static_cast<float>(img_width) - 1.0f);
    torch::Tensor xys = torch::stack({u, v}, -1); // center

    torch::Tensor radii = radius.to(torch::kInt32);
    torch::Tensor camDepths = pProj.index({"...", 2});

    return std::make_tuple(xys, radii, conic, cov2d, camDepths);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> rasterize_forward_tensor_cpu(
    const int width,
    const int height,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background,
    const torch::Tensor &cov2d,
    const torch::Tensor &camDepths
){
    torch::NoGradGuard noGrad;

    int channels = colors.size(1);
    int numPoints = xys.size(0);
    float *pDepths = static_cast<float *>(camDepths.data_ptr());

    std::vector< size_t > gIndices( numPoints );
    std::iota( gIndices.begin(), gIndices.end(), 0 );
    std::sort(gIndices.begin(), gIndices.end(), [&pDepths](int a, int b){
        return pDepths[a] < pDepths[b];
    });

    torch::Device device = xys.device();

    torch::Tensor outImg = torch::zeros({width, height, channels}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor finalTs = torch::ones({width, height, channels}, torch::TensorOptions().dtype(torch::kFloat32).device(device));   
    torch::Tensor finalIdx = torch::zeros({width, height, channels}, torch::TensorOptions().dtype(torch::kInt32).device(device));   

    torch::Tensor sqCov2dX = 3.0f * torch::sqrt(cov2d.index({"...", 0, 0}));
    torch::Tensor sqCov2dY = 3.0f * torch::sqrt(cov2d.index({"...", 1, 1}));
    
    float *pConics = static_cast<float *>(conics.data_ptr());
    float *pCenters = static_cast<float *>(xys.data_ptr());
    float *pSqCov2dX = static_cast<float *>(sqCov2dX.data_ptr());
    float *pSqCov2dY = static_cast<float *>(sqCov2dY.data_ptr());
    float *pOpacities = static_cast<float *>(opacities.data_ptr());

    float *pOutImg = static_cast<float *>(outImg.data_ptr());
    float *pFinalTs = static_cast<float *>(finalTs.data_ptr());
    int32_t *pFinalIdx = static_cast<int32_t *>(finalIdx.data_ptr());
    float *pColors = static_cast<float *>(colors.data_ptr());
    
    float bgX = background[0].item<float>();
    float bgY = background[1].item<float>();
    float bgZ = background[2].item<float>();

    const float alphaThresh = 1.0f / 255.0f;
    float T = 1.0f;
    int idx = 0;
    for (; idx < numPoints; idx++){
        int32_t gaussianId = gIndices[idx];

        float A = pConics[gaussianId * 3 + 0];
        float B = pConics[gaussianId * 3 + 1];
        float C = pConics[gaussianId * 3 + 2];

        float gX = pCenters[gaussianId * 2 + 0];
        float gY = pCenters[gaussianId * 2 + 1];

        float sqx = pSqCov2dX[gaussianId];
        float sqy = pSqCov2dY[gaussianId];
        
        int minx = (std::max)(0, static_cast<int>(std::floor(gY - sqy)) - 2);
        int maxx = (std::min)(width, static_cast<int>(std::ceil(gY + sqy)) + 2);
        int miny = (std::max)(0, static_cast<int>(std::floor(gX - sqx)) - 2);
        int maxy = (std::min)(height, static_cast<int>(std::ceil(gX + sqx)) + 2);

        for (int i = minx; i < maxx; i++){
            for (int j = miny; j < maxy; j++){
                float xCam = gX - j;
                float yCam = gY - i;
                float sigma = (
                    0.5f
                    * (A * xCam * xCam + C * yCam * yCam)
                    + B * xCam * yCam
                );

                if (sigma < 0.0f) continue;
                float alpha = (std::min)(0.999f, (pOpacities[gaussianId] * std::exp(-sigma)));
                if (alpha < alphaThresh) continue;

                size_t pixIdx = (i * height + j);
                float T = pFinalTs[pixIdx];
                float nextT = T * (1.0f - alpha);

                float alphaT = alpha * T;

                pOutImg[pixIdx * 3 + 0] += alphaT * (pColors[gaussianId * 3 + 0] + bgX);
                pOutImg[pixIdx * 3 + 1] += alphaT * (pColors[gaussianId * 3 + 1] + bgY);
                pOutImg[pixIdx * 3 + 2] += alphaT * (pColors[gaussianId * 3 + 2] + bgZ);
                
                pFinalTs[pixIdx] = nextT;
                pFinalIdx[pixIdx] = idx;
            }
        }
    }

    return std::make_tuple(outImg, finalTs, finalIdx);
}


std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    rasterize_backward_tensor_cpu(
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha
    ){
        return std::make_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor());
}