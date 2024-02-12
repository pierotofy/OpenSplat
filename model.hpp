#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include "nerfstudio.hpp"
#include "kdtree_tensor.hpp"
#include "spherical_harmonics.hpp"
#include "ssim.hpp"

using namespace torch::indexing;
using namespace torch::autograd;

namespace ns{

torch::Tensor randomQuatTensor(long long n);
torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device);
torch::Tensor psnr(const torch::Tensor& rendered, const torch::Tensor& gt);
torch::Tensor l1(const torch::Tensor& rendered, const torch::Tensor& gt);

struct Model : torch::nn::Module {
  Model(const Points &points, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval, 
        int refineEvery, int warmupLength, int resetAlphaEvery, int stopSplitAt,
        const torch::Device &device) :
    numCameras(numCameras),
    numDownscales(numDownscales), resolutionSchedule(resolutionSchedule), shDegree(shDegree), shDegreeInterval(shDegreeInterval), 
    refineEvery(refineEvery), warmupLength(warmupLength), resetAlphaEvery(resetAlphaEvery), stopSplitAt(stopSplitAt),
    device(device), ssim(11, 3) {
    long long numPoints = points.xyz.size(0); 
    torch::manual_seed(42);

    means = register_parameter("means", points.xyz, true);
    scales = register_parameter("scales", PointsTensor(means).scales().repeat({1, 3}).log(), true);
    quats = register_parameter("quats", randomQuatTensor(numPoints), true);

    int dimSh = numShBases(shDegree);
    torch::Tensor shs = torch::zeros({numPoints, dimSh, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    shs.index({Slice(), 0, Slice(None, 3)}) = rgb2sh(points.rgb.toType(torch::kFloat64) / 255.0).toType(torch::kFloat32);
    shs.index({Slice(), Slice(1, None), Slice(3, None)}) = 0.0f;

    featuresDc = register_parameter("featuresDc", shs.index({Slice(), 0, Slice()}), true);
    featuresRest = register_parameter("featuresRest", shs.index({Slice(), Slice(1, None), Slice()}), true);
    opacities = register_parameter("opacities", torch::logit(0.1f * torch::ones({numPoints, 1})), true);
    
    // backgroundColor = torch::tensor({0.0f, 0.0f, 0.0f}, device); // Black
    backgroundColor = torch::tensor({0.6130f, 0.0101f, 0.3984f}, device); // Nerf Studio default

    meansOpt = new torch::optim::Adam({means}, torch::optim::AdamOptions(0.00016));
    scalesOpt = new torch::optim::Adam({scales}, torch::optim::AdamOptions(0.005));
    quatsOpt = new torch::optim::Adam({quats}, torch::optim::AdamOptions(0.001));
    featuresDcOpt = new torch::optim::Adam({featuresDc}, torch::optim::AdamOptions(0.0025));
    featuresRestOpt = new torch::optim::Adam({featuresRest}, torch::optim::AdamOptions(0.000125));
    opacitiesOpt = new torch::optim::Adam({opacities}, torch::optim::AdamOptions(0.05));
  }

  ~Model(){
    delete meansOpt;
    delete scalesOpt;
    delete quatsOpt;
    delete featuresDcOpt;
    delete featuresRestOpt;
    delete opacitiesOpt;
  }

  torch::Tensor forward(Camera& cam, int step);
  void optimizersZeroGrad();
  void optimizersStep();
  int getDownscaleFactor(int step);
  void afterTrain(int step);

  torch::Tensor means;
  torch::Tensor scales;
  torch::Tensor quats;
  torch::Tensor featuresDc;
  torch::Tensor featuresRest;
  torch::Tensor opacities;

  torch::optim::Adam *meansOpt;
  torch::optim::Adam *scalesOpt;
  torch::optim::Adam *quatsOpt;
  torch::optim::Adam *featuresDcOpt;
  torch::optim::Adam *featuresRestOpt;
  torch::optim::Adam *opacitiesOpt;

  torch::Tensor radii; // set in forward()
  torch::Tensor xys; // set in forward()
  int lastHeight; // set in forward()
  int lastWidth; // set in forward()

  torch::Tensor xysGradNorm; // set in afterTrain()
  torch::Tensor visCounts; // set in afterTrain()  
  torch::Tensor max2DSize; // set in afterTrain()


  torch::Tensor backgroundColor;
  torch::Device device;
  SSIM ssim;

  int numCameras;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  int shDegreeInterval;
  int refineEvery;
  int warmupLength;
  int resetAlphaEvery;
  int stopSplitAt;
};


}

#endif