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

std::tuple<torch::Tensor, torch::Tensor> getTileBbox(const torch::Tensor &pixCenter, const torch::Tensor &pixRadius, const std::tuple<int, int, int> &tileBounds){
    torch::Tensor tileSize = torch::tensor({BLOCK_X, BLOCK_Y}, torch::TensorOptions().dtype(torch::kFloat32).device(pixCenter.device()));
    torch::Tensor tileCenter = pixCenter / tileSize;
    torch::Tensor tileRadius = pixRadius.index({"...", None}) / tileSize;
    torch::Tensor topLeft = (tileCenter - tileRadius).to(torch::kInt32);
    torch::Tensor bottomRight = (tileCenter + tileRadius).to(torch::kInt32) + 1;
    torch::Tensor tileMin = torch::stack({
        torch::clamp(topLeft.index({"...", 0}), 0, std::get<0>(tileBounds)),
        torch::clamp(topLeft.index({"...", 1}), 0, std::get<1>(tileBounds))
    }, -1);
    torch::Tensor tileMax = torch::stack({
        torch::clamp(bottomRight.index({"...", 0}), 0, std::get<0>(tileBounds)),
        torch::clamp(bottomRight.index({"...", 1}), 0, std::get<1>(tileBounds))
    }, -1);

    return std::make_tuple(tileMin, tileMax);    
}

torch::Tensor compute_sh_forward_tensor(
    unsigned num_points,
    unsigned degree,
    unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &coeffs
){
    return torch::Tensor();
}

torch::Tensor compute_sh_backward_tensor(
    unsigned num_points,
    unsigned degree,
    unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
){
    return torch::Tensor();
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
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
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
    torch::Tensor &cov3d,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &v_xy,
    torch::Tensor &v_depth,
    torch::Tensor &v_conic
){
    return std::make_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor());
}


std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds
){
    torch::Device device = xys.device();
    int numIntersects = cum_tiles_hit[-1].item<int>();
    torch::Tensor isectIds = torch::zeros(numIntersects, torch::TensorOptions().dtype(torch::kInt64).device(device));
    torch::Tensor gaussianIds = torch::zeros(numIntersects, torch::TensorOptions().dtype(torch::kInt32).device(device));
    for (int idx = 0; idx < num_points; idx++){
        if (radii[idx].item<float>() <= 0.0f) break;

        auto bbox = getTileBbox(xys[idx], radii[idx], tile_bounds);
        torch::Tensor tileMin = std::get<0>(bbox);
        torch::Tensor tileMax = std::get<1>(bbox);
        int curIdx;

        if (idx == 0){
            curIdx = 0;
        }else{
            curIdx = cum_tiles_hit[idx - 1].item<int>();
        }

        float depth = depths[idx].item<float>();
        int32_t depthIdN = *(reinterpret_cast<int32_t *>(&depth));

        int iStart = tileMin[1].item<int>();
        int iEnd = tileMax[1].item<int>();
        int jStart = tileMin[0].item<int>();
        int jEnd = tileMax[0].item<int>();
        int b = std::get<0>(tile_bounds);

        for (int i = iStart; i < iEnd; i++){
            for (int j = jStart; j < jEnd; j++){
                int64_t tileId = i * b + j;
                isectIds[curIdx] = static_cast<int64_t>(tileId << 32) | depthIdN;
                gaussianIds[curIdx] = idx;
                curIdx += 1;
            }
        }
    }

    return std::make_tuple(isectIds, gaussianIds); 
}

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects,
    const torch::Tensor &isect_ids_sorted
){
    torch::Tensor tileBins = torch::zeros({num_intersects, 2}, torch::TensorOptions().dtype(torch::kInt32).device(isect_ids_sorted.device()));

    for (int idx = 0; idx < num_intersects; idx++){
        int32_t curTileIdx = static_cast<int32_t>(isect_ids_sorted[idx].item<int64_t>() >> 32);

        if (idx == 0){
            tileBins[curTileIdx][0] = 0;
            continue;
        }

        if (idx == num_intersects - 1){
            tileBins[curTileIdx][1] = num_intersects;
            break;
        }

        int32_t prevTileIdx = static_cast<int32_t>(isect_ids_sorted[idx - 1].item<int64_t>() >> 32);

        if (curTileIdx != prevTileIdx){
            tileBins[prevTileIdx][1] = idx;
            tileBins[curTileIdx][0] = idx;
        }
    }

    return tileBins;
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

    std::cout << pDepths[0] << std::endl;

    std::cout << pDepths[100];

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


/*
    int minx = 99999;
    int miny = 99999;
    int maxx = 0;
    int maxy = 0;
    for (int i = 0; i < width; i++){
        std::cout << i << std::endl;
        for (int j = 0; j < height; j++){
            float T = 1.0f;
            torch::Tensor ji = torch::tensor({j, i}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            
            int idx = 0;
            for (; idx < 1; idx++){
                torch::Tensor gaussianId = gaussian_ids_sorted[idx];
                torch::Tensor conic = conics[gaussianId];
                torch::Tensor center = xys[gaussianId];
                torch::Tensor delta = center - ji;

                torch::Tensor sigma = (
                    0.5f
                    * (conic[0] * delta[0] * delta[0] + conic[2] * delta[1] * delta[1])
                    + conic[1] * delta[0] * delta[1]
                );

                if (sigma.item<float>() < 0.0f) continue;

                float alpha = (std::min)(0.999f, (opacities[gaussianId] * torch::exp(-sigma)).item<float>());

                if (alpha < 1.0f / 255.0f) continue;

                float nextT = T * (1.0f - alpha);

                if (nextT <= 1e-4f){
                    idx -= 1;
                    break;
                }

                float vis = alpha * T;
                // outImg[i][j] = torch::tensor({1.0f, 1.0f, 1.0f}); 
                outImg[i][j] += vis * colors[gaussianId];

                maxx = (std::max)(i, maxx);
                maxy = (std::max)(j, maxy);
                minx = (std::min)(i, minx);
                miny = (std::min)(j, miny);               
                

                T = nextT;
            }

            finalTs[i][j] = T;
            finalIdx[i][j] = idx;
            outImg[i][j] += T * background;
        }
    }

    std::cout << "[" << minx << ", " << miny << "], [" << maxx << ", " << maxy << "]" << std::endl;

    return std::make_tuple(outImg, finalTs, finalIdx);

*/
/*
    int blockX = std::get<0>(block);
    int blockY = std::get<1>(block);
    int tileBoundsX = std::get<0>(tile_bounds);
    
    for (int i = 0; i < width; i++){
        for (int j = 0; j < height; j++){
            int tileId = (i / blockX) * tileBoundsX + (j / blockY);
            int tileBinStart = tile_bins[tileId][0].item<int>();
            int tileBinEnd = tile_bins[tileId][1].item<int>();
            float T = 1.0f;
            torch::Tensor ji = torch::tensor({j, i}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            
            int idx = tileBinStart;
            for (; idx < tileBinEnd; idx++){
                torch::Tensor gaussianId = gaussian_ids_sorted[idx];
                torch::Tensor conic = conics[gaussianId];
                torch::Tensor center = xys[gaussianId];
                torch::Tensor delta = center - ji;
pGaussianIds
                torch::Tensor sigma = (
                    0.5f
                    * (conic[0] * delta[0] * delta[0] + conic[2] * delta[1] * delta[1])
                    + conic[1] * delta[0] * delta[1]
                );

                if (sigma.item<float>() < 0.0f) continue;

                float alpha = (std::min)(0.999f, (opacities[gaussianId] * torch::exp(-sigma)).item<float>());

                if (alpha < 1.0f / 255.0f) continue;

                float nextT = T * (1.0f - alpha);

                if (nextT <= 1e-4f){
                    idx -= 1;
                    break;
                }

                float vis = alpha * T;
                outImg[i][j] += vis * colors[gaussianId];
                T = nextT;
            }

            finalTs[i][j] = T;
            finalIdx[i][j] = idx;
            outImg[i][j] += T * background;
        }
    }

    return std::make_tuple(outImg, finalTs, finalIdx);
*/
}


std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    rasterize_backward_tensor(
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