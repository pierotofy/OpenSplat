#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/version.h>
#include "nerfstudio.hpp"
#include "kdtree_tensor.hpp"
#include "spherical_harmonics.hpp"
#include "ssim.hpp"
#include "input_data.hpp"
#include "optim_scheduler.hpp"
#include <functional>
#include <span>
#include "trainer_params.hpp"	//	model params

using namespace torch::indexing;
using namespace torch::autograd;

torch::Tensor randomQuatTensor(long long n);
torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device);
torch::Tensor psnr(const torch::Tensor& rendered, const torch::Tensor& gt);
torch::Tensor l1(const torch::Tensor& rendered, const torch::Tensor& gt);



//	store results of a forward render
//	this was previously stored on the model, meaning an arbritary render/forward() would mutate state
//	and put state out of sync 
class ModelForwardResults
{
public:
	torch::Tensor	rgb;	//	rendered image
	torch::Tensor	radii;
	torch::Tensor	xys;	//	splats in view/screen space
	
	//	store other camera intrinsics?
	int				lastHeight;
	int				lastWidth;
};

//	ephemeral post-training data for culling/splitting
class Model2DVisibility
{
public:
	torch::Tensor xysGradNorm;
	torch::Tensor visCounts;  
	torch::Tensor max2DSize;
};


class Model
{
public:
	Model(const InputData &inputData,
		  const ModelParams& modelParams,
		  int maxSteps,
		  std::array<float,3> backgroundColour,
		  const torch::Device &device);
	~Model();
  
  void setupOptimizers();
  void releaseOptimizers();

	ModelForwardResults forward(Camera& cam, int step);
	ModelForwardResults forward(CameraTransform& CameraToWorldTransform,CameraIntrinsics RenderIntrinsics,int step);
  void optimizersZeroGrad();
  void optimizersStep();
  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  void afterTrain(int step,ModelForwardResults& ForwardMeta);
  void findInvalidPoints();
	
  void save(const std::string &filename, int step,bool keepCrs);
  void savePly(const std::string &filename, int step,bool keepCrs);
  void saveSplat(const std::string &filename,bool keepCrs);
  void saveDebugPly(const std::string &filename, int step,bool keepCrs);
  void iteratePoints(std::function<void(std::span<float> xyz,std::span<float> opacity,std::span<float> scale,std::span<float> quaternionwxyz,std::span<float> dcFeatures,std::span<float> restFeatures)> OnFoundPoint);
  int getPointCount();
	
  //	modelPointsNeedToBeNormalised == keepCrs
  //	it means that the PLY we're loading, was saved in it's original space instead of centered and scaled to -1...1
  int loadPly(const std::string &filename,bool modelPointsNeedToBeNormalised);
	
  torch::Tensor mainLoss(torch::Tensor &rgb, torch::Tensor &gt, float ssimWeight);

  void addToOptimizer(torch::optim::Adam& optimizer, const torch::Tensor &newParam, const torch::Tensor &idcs, int nSamples);
  void removeFromOptimizer(torch::optim::Adam& optimizer, const torch::Tensor &newParam, const torch::Tensor &deletedMask);
  torch::Tensor means;
  torch::Tensor scales;
  torch::Tensor quats;
  torch::Tensor featuresDc;
  torch::Tensor featuresRest;
  torch::Tensor opacities;

	std::shared_ptr<torch::optim::Adam> meansOpt;
	std::shared_ptr<torch::optim::Adam> scalesOpt;
	std::shared_ptr<torch::optim::Adam> quatsOpt;
	std::shared_ptr<torch::optim::Adam> featuresDcOpt;
	std::shared_ptr<torch::optim::Adam> featuresRestOpt;
	std::shared_ptr<torch::optim::Adam> opacitiesOpt;

	std::shared_ptr<OptimScheduler> meansOptScheduler;


  torch::Tensor backgroundColor;
  torch::Device device;
  SSIM ssim;

  int numCameras;		//	used only for determining refine parameters (no other camera meta required)
	ModelParams	params;
  int stopSplitAt;	//	maxsteps/2
  int maxSteps;

	//	transform of input data - only used for restoring into original space
  float scale;
  torch::Tensor translation;
};


#endif
