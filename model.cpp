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

torch::Tensor psnr(const torch::Tensor& rendered, const torch::Tensor& gt){
    torch::Tensor mse = (rendered - gt).pow(2).mean();
    return (10.f * torch::log10(1.0 / mse));
}

torch::Tensor l1(const torch::Tensor& rendered, const torch::Tensor& gt){
    return torch::abs(gt - rendered).mean();
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

    lastHeight = cam.height;
    lastWidth = cam.width;

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
    xys = p[0];
    torch::Tensor depths = p[1];
    radii = p[2];
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

void Model::optimizersZeroGrad(){
  meansOpt->zero_grad();
  scalesOpt->zero_grad();
  quatsOpt->zero_grad();
  featuresDcOpt->zero_grad();
  featuresRestOpt->zero_grad();
  opacitiesOpt->zero_grad();
}

void Model::optimizersStep(){
  meansOpt->step();
  scalesOpt->step();
  quatsOpt->step();
  featuresDcOpt->step();
  featuresRestOpt->step();
  opacitiesOpt->step();
}

int Model::getDownscaleFactor(int step){
    return std::pow(2, (std::max<int>)(numDownscales - step / resolutionSchedule, 0));
}

void Model::afterTrain(int step){
    torch::NoGradGuard noGrad;

    if (step < stopSplitAt){
        torch::Tensor visibleMask = (radii > 0).flatten();
        
        torch::Tensor grads = torch::linalg::vector_norm(xys.grad().detach(), 2, { -1 }, false, torch::kFloat32);
        if (!xysGradNorm.numel()){
            xysGradNorm = grads;
            visCounts = torch::ones_like(xysGradNorm);
        }else{
            visCounts.index_put_({visibleMask}, visCounts.index({visibleMask}) + 1);
            xysGradNorm.index_put_({visibleMask}, grads.index({visibleMask}) + xysGradNorm.index({visibleMask}));
        }

        if (!max2DSize.numel()){
            max2DSize = torch::zeros_like(radii, torch::kFloat32);
        }

        torch::Tensor newRadii = radii.detach().index({visibleMask});
        max2DSize.index_put_({visibleMask}, torch::maximum(
                max2DSize.index({visibleMask}), newRadii / static_cast<float>( (std::max)(lastHeight, lastWidth) )
            ));
    }

    if (step % refineEvery == 0 && step > warmupLength){
        

        int resetInterval = resetAlphaEvery * refineEvery;
        bool doDensification = step < stopSplitAt && step % resetInterval > numCameras + refineEvery;
        if (doDensification){
            torch::Tensor avgGradNorm = (xysGradNorm / visCounts) * 0.5f * static_cast<float>( (std::max)(lastWidth, lastHeight) );
            torch::Tensor highGrads = (avgGradNorm > densifyGradThresh).squeeze();
            torch::Tensor splits = (std::get<0>(scales.exp().max(-1)) > densifySizeThresh).squeeze();
            if (step < stopScreenSizeAt){
                splits |= (max2DSize > splitScreenSize).squeeze();
            }

            splits &= highGrads;
            const int nSplitSamples = 2;
            int nSplits = splits.sum().item<int>();
            torch::Tensor centeredSamples = torch::randn({nSplitSamples * nSplits, 3}, device);  // Nx3 of axis-aligned scales
            torch::Tensor scaledSamples = torch::exp(scales.index({splits}).repeat({nSplitSamples, 1})) * centeredSamples;
            torch::Tensor quats = this->quats.index({splits}) / torch::linalg::vector_norm(this->quats.index({splits}), 2, { -1 }, true, torch::kFloat32);

        
        }
    }
}


}