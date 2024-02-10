#include "model.hpp"
#include "constants.hpp"
#include "tile_bounds.hpp"
#include "project_gaussians.hpp"
#include "rasterize_gaussians.hpp"
#include "vendor/gsplat/config.h"

namespace ns{

torch::Tensor randomQuatTensor(long long n){
    torch::Tensor u = torch::rand(n);
    torch::Tensor v = torch::rand(n);
    torch::Tensor w = torch::rand(n);
    return torch::stack({
        torch::sqrt(1 - u) * torch::sin(2 * PI * v),
        torch::sqrt(1 - u) * torch::cos(2 * PI * v),
        torch::sqrt(u) * torch::sin(2 * PI * w),
        torch::sqrt(u) * torch::cos(2 * PI * w)
    }, -1);
}

torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device){
    // OpenGL perspective projection matrix
    float t = zNear * std::tan(0.5f * fovY);
    float b = -t;
    float r = zNear * std::tan(0.5f * fovX);
    float l = -r;
    return torch::tensor({
        {2.0f * zNear / (r - l), 0.0f, (r + l) / (r - l), 0.0f},
        {0.0f, 2 * zNear / (t - b), (t + b) / (t - b), 0.0f},
        {0.0f, 0.0f, (zFar + zNear) / (zFar - zNear), -1.0f * zFar * zNear / (zFar - zNear)},
        {0.0f, 0.0f, 1.0f, 0.0f}
    }, device);
}

torch::Tensor Model::forward(Camera& cam, int step){

    float scaleFactor = 1.0f / static_cast<float>(getDownscaleFactor(step));
    cam.scaleOutputResolution(scaleFactor);

    // TODO: these can be moved to Camera and computed only once?
    torch::Tensor R = cam.camToWorld.index({Slice(None, 3), Slice(None, 3)});
    torch::Tensor T = cam.camToWorld.index({Slice(None, 3), Slice(3,4)});

    // Flip the z and y axes to align with gsplat conventions
    R = torch::matmul(R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f}, R.device())));

    // worldToCam
    torch::Tensor Rinv = R.transpose(0, 1);
    torch::Tensor Tinv = torch::matmul(-Rinv, T);

    torch::Tensor viewMat = torch::eye(4, device);
    viewMat.index_put_({Slice(None, 3), Slice(None, 3)}, Rinv);
    viewMat.index_put_({Slice(None, 3), Slice(3, 4)}, Tinv);
        
    float fovX = 2.0f * std::atan(cam.width / (2.0f * cam.fx));
    float fovY = 2.0f * std::atan(cam.height / (2.0f * cam.fy));

    torch::Tensor projMat = projectionMatrix(0.001f, 1000.0f, fovX, fovY, device);

    TileBounds tileBounds = std::make_tuple((cam.width + BLOCK_X - 1) / BLOCK_X,
                      (cam.height + BLOCK_Y - 1) / BLOCK_Y,
                      1);

    torch::Tensor colors =  torch::cat({featuresDc.index({Slice(), None, Slice()}), featuresRest}, 1);

    auto p = ProjectGaussians::apply(means, 
                    torch::exp(scales), 
                    1, 
                    quats / quats.norm(2, {-1}, true), 
                    viewMat, 
                    torch::matmul(projMat, viewMat),
                    cam.fx, 
                    cam.fy,
                    cam.cx,
                    cam.cy,
                    cam.height,
                    cam.width,
                    tileBounds);
    torch::Tensor xys = p[0];
    torch::Tensor depths = p[1];
    torch::Tensor radii = p[2];
    torch::Tensor conics = p[3];
    torch::Tensor numTilesHit = p[4];
    

    if (radii.sum().item<float>() == 0.0f){
        // Rescale resolution back
        cam.scaleOutputResolution(1.0f / scaleFactor);
        return backgroundColor.repeat({cam.height, cam.width, 1});
    }

    // TODO: is this needed?
    xys.retain_grad();

    torch::Tensor viewDirs = means.detach() - T.transpose(0, 1).to(device);
    viewDirs = viewDirs / viewDirs.norm(2, {-1}, true);
    int degreesToUse = (std::min<int>)(step / shDegreeInterval, shDegree);
    torch::Tensor rgbs = SphericalHarmonics::apply(degreesToUse, viewDirs, colors);
    rgbs = torch::clamp_min(rgbs + 0.5f, 0.0f); 

    
    torch::Tensor rgb = RasterizeGaussians::apply(
            xys,
            depths,
            radii,
            conics,
            numTilesHit,
            rgbs, // TODO: why not sigmod?
            torch::sigmoid(opacities),
            cam.height,
            cam.width,
            backgroundColor);
    
    rgb = torch::clamp_max(rgb, 1.0f);

    // Rescale resolution back
    cam.scaleOutputResolution(1.0f / scaleFactor);
    
    return rgb;
}

int Model::getDownscaleFactor(int step){
    return std::pow(2, (std::max<int>)(numDownscales - step / resolutionSchedule, 0));
}


}