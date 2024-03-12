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

std::tuple<torch::Tensor, torch::Tensor> getTileBbox(torch::Tensor &pixCenter, const torch::Tensor &pixRadius, const std::tuple<int, int, int> &tileBounds){
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
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor(
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
    const std::tuple<int, int, int> tile_bounds,
    const float clip_thresh
){
    float fovx = 0.5f * static_cast<float>(img_width) / fx;
    float fovy = 0.5f * static_cast<float>(img_height) / fy;

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

    auto bbox = getTileBbox(xys, radius, tile_bounds);
    torch::Tensor tileMin = std::get<0>(bbox);
    torch::Tensor tileMax = std::get<1>(bbox);
    torch::Tensor numTilesHit = (tileMax.index({"...", 0}) - tileMin.index({"...", 0})) * 
                   (tileMax.index({"...", 1}) - tileMin.index({"...", 1}));

    torch::Tensor depths = pView.index({"...", 2});
    torch::Tensor radii = radius.to(torch::kInt32);

    return std::make_tuple(cov3d, xys, depths, radii, conic, numTilesHit );
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

        int32_t depthIdN = static_cast<int32_t>(depths[idx].item<float>());
        int iStart = tileMin[1].item<int>();
        int iEnd = tileMax[1].item<int>();
        int jStart = tileMin[0].item<int>();
        int jEnd = tileMax[0].item<int>();
        int b = std::get<0>(tile_bounds);

        for (int i = iStart; i < iEnd; i++){
            for (int j = jStart; j < jEnd; j++){
                int tileId = i * b + j;
                isectIds[curIdx]
            }
        }
    }
    //     for i in range(tile_min[1], tile_max[1]):
    //         for j in range(tile_min[0], tile_max[0]):
    //             tile_id = i * tile_bounds[0] + j
    //             isect_ids[cur_idx] = (tile_id << 32) | depth_id_n
    //             gaussian_ids[cur_idx] = idx
    //             cur_idx += 1

    // return isect_ids, gaussian_ids

    return std::make_tuple(torch::Tensor(), torch::Tensor());
}

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects,
    const torch::Tensor &isect_ids_sorted
){
    return torch::Tensor();
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &background
){
    int channels = colors.size(1);
    int width = std::get<1>(img_size);
    int height = std::get<0>(img_size);
    torch::Device device = xys.device();

    torch::Tensor outImg = torch::zeros({width, height, channels}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor finalTs = torch::zeros({width, height, channels}, torch::TensorOptions().dtype(torch::kFloat32).device(device));   
    torch::Tensor finalIdx = torch::zeros({width, height, channels}, torch::TensorOptions().dtype(torch::kInt32).device(device));   


    for (int i = 0; i < width; i++){
        for (int j = 0; j < height; j++){
            int tileId = (i / std::get<0>(block)) * std::get<0>(tile_bounds) + (j / std::get<1>(block));
            int tileBinStart = tile_bins[tileId][0].item<int>();
            int tileBinEnd = tile_bins[tileId][1].item<int>();
            float T = 1.0f;

            int idx = tileBinStart;
            for (; idx < tileBinEnd; idx++){
                torch::Tensor gaussianId = gaussian_ids_sorted[idx];
                torch::Tensor conic = conics[gaussianId];
                torch::Tensor center = xys[gaussianId];
                torch::Tensor delta = center - torch::tensor({j, i}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

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