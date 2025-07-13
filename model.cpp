#include <filesystem>
#include "model.hpp"
#include "constants.hpp"
#include "tile_bounds.hpp"
#include "project_gaussians.hpp"
#include "rasterize_gaussians.hpp"
#include "tensor_math.hpp"
#include "gsplat.hpp"
#include "utils.hpp"
#include <span>
#include "ply.hpp"

#ifdef USE_HIP
#include <c10/hip/HIPCachingAllocator.h>
#elif defined(USE_CUDA)
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace fs = std::filesystem;

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

torch::Tensor psnr(const torch::Tensor& rendered, const torch::Tensor& gt){
    torch::Tensor mse = (rendered - gt).pow(2).mean();
    return (10.f * torch::log10(1.0 / mse));
}

torch::Tensor l1(const torch::Tensor& rendered, const torch::Tensor& gt){
    return torch::abs(gt - rendered).mean();
}

Model::Model(const InputData &inputData,
			 const ModelParams& modelParams,
	  int maxSteps,
			 std::array<float,3> backgroundColour,
	  const torch::Device &device) :
	numCameras(inputData.cameras.size()),
	params(modelParams), 
	stopSplitAt(maxSteps / 2), 
	maxSteps(maxSteps),
	device(device), 
	ssim(11, 3)
{
	//	this will fail later with a mod %0, but not the craziest check anyway
	if ( numCameras < 1 )
		throw std::runtime_error("Model requires at least one camera");
	
	long long numPoints = inputData.points.xyz.size(0);
	scale = inputData.scale;
	translation = inputData.translation;
	
	torch::manual_seed(42);
	
	means = inputData.points.xyz.to(device).requires_grad_();
	scales = PointsTensor(inputData.points.xyz).scales().repeat({1, 3}).log().to(device).requires_grad_();
	quats = randomQuatTensor(numPoints).to(device).requires_grad_();
	
	int dimSh = numShBases(params.shDegree);
	torch::Tensor shs = torch::zeros({numPoints, dimSh, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	
	shs.index({Slice(), 0, Slice(None, 3)}) = rgb2sh(inputData.points.rgb.toType(torch::kFloat64) / 255.0).toType(torch::kFloat32);
	shs.index({Slice(), Slice(1, None), Slice(3, None)}) = 0.0f;
	
	featuresDc = shs.index({Slice(), 0, Slice()}).to(device).requires_grad_();
	featuresRest = shs.index({Slice(), Slice(1, None), Slice()}).to(device).requires_grad_();
	opacities = torch::logit(0.1f * torch::ones({numPoints, 1})).to(device).requires_grad_();
	
	backgroundColor = torch::tensor({backgroundColour[0],backgroundColour[1],backgroundColour[2]}, device).requires_grad_();
	
	setupOptimizers();
}

Model::~Model()
{
	releaseOptimizers();
}

void Model::setupOptimizers(){
    releaseOptimizers();

	meansOpt.reset( new torch::optim::Adam({means}, torch::optim::AdamOptions(0.00016)) );
	scalesOpt.reset( new torch::optim::Adam({scales}, torch::optim::AdamOptions(0.005)) );
	quatsOpt.reset( new torch::optim::Adam({quats}, torch::optim::AdamOptions(0.001)) );
	featuresDcOpt.reset( new torch::optim::Adam({featuresDc}, torch::optim::AdamOptions(0.0025)) );
	featuresRestOpt.reset( new torch::optim::Adam({featuresRest}, torch::optim::AdamOptions(0.000125)) );
	opacitiesOpt.reset( new torch::optim::Adam({opacities}, torch::optim::AdamOptions(0.05)) );
	meansOptScheduler.reset( new OptimScheduler(meansOpt, 0.0000016f, maxSteps) );
}

void Model::releaseOptimizers()
{
	//	here in case they need to be released in order...
	meansOpt.reset();
	scalesOpt.reset();
	quatsOpt.reset();
	featuresDcOpt.reset();
	featuresRestOpt.reset();
	opacitiesOpt.reset();
	
	meansOptScheduler.reset();
}



ModelForwardResults Model::forward(Camera& cam,int step)
{
	ModelForwardResults Results;
	
	const float scaleFactor = getDownscaleFactor(step);
	
	CameraIntrinsics renderIntrinsics = cam.intrinsics;
	
	renderIntrinsics.fx /= scaleFactor;
	renderIntrinsics.fy /= scaleFactor;
	renderIntrinsics.cx /= scaleFactor;
	renderIntrinsics.cy /= scaleFactor;
	renderIntrinsics.imageWidth /= scaleFactor;
	renderIntrinsics.imageHeight /= scaleFactor;
	
	return forward( cam.camToWorld, renderIntrinsics, step );
}


ModelForwardResults Model::forward(CameraTransform& CameraTransform,CameraIntrinsics renderIntrinsics,int step)
{
	ModelForwardResults Results;

	//	expecting to already be scaled down now
	//const float scaleFactor = getDownscaleFactor(step);
	
	const float fx = renderIntrinsics.fx;
	const float fy = renderIntrinsics.fy;
	const float cx = renderIntrinsics.cx;
	const float cy = renderIntrinsics.cy;
	const int height = static_cast<int>( renderIntrinsics.imageHeight );
	const int width = static_cast<int>( renderIntrinsics.imageWidth );
	
	auto T = CameraTransform.GetCamToWorldTranslation();
	auto Rinv = CameraTransform.GetWorldToCamRotation();
	auto Tinv = CameraTransform.GetWorldToCamTranslation();

	Results.lastHeight = height;
	Results.lastWidth = width;

	//	camera to world transform
    torch::Tensor viewMat = torch::eye(4, device);
    viewMat.index_put_({Slice(None, 3), Slice(None, 3)}, Rinv);
    viewMat.index_put_({Slice(None, 3), Slice(3, 4)}, Tinv);
        
    float fovX = 2.0f * std::atan(width / (2.0f * fx));
    float fovY = 2.0f * std::atan(height / (2.0f * fy));

	//	todo? should be using same projection matrix from intrinsics?
    torch::Tensor projMat = projectionMatrix(0.001f, 1000.0f, fovX, fovY, device);
    
	torch::Tensor colors =  torch::cat({featuresDc.index({Slice(), None, Slice()}), featuresRest}, 1);
    torch::Tensor conics;
    torch::Tensor depths; // GPU-only
    torch::Tensor numTilesHit; // GPU-only
    torch::Tensor cov2d; // CPU-only
    torch::Tensor camDepths; // CPU-only

	//	project splats into screen
    if (device == torch::kCPU){
        auto p = ProjectGaussiansCPU::apply(means, 
                                torch::exp(scales), 
                                1, 
                                quats / quats.norm(2, {-1}, true), 
                                viewMat, 
                                torch::matmul(projMat, viewMat),
                                fx, 
                                fy,
                                cx,
                                cy,
                                height,
                                width);
        Results.xys = p[0];
		Results.radii = p[1];
        conics = p[2];
        cov2d = p[3];
        camDepths = p[4];
    }else{
        #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)

        TileBounds tileBounds = std::make_tuple((width + BLOCK_X - 1) / BLOCK_X,
                        (height + BLOCK_Y - 1) / BLOCK_Y,
                        1);
        auto p = ProjectGaussians::apply(means, 
                        torch::exp(scales), 
                        1, 
                        quats / quats.norm(2, {-1}, true), 
                        viewMat, 
                        torch::matmul(projMat, viewMat),
                        fx, 
                        fy,
                        cx,
                        cy,
                        height,
                        width,
                        tileBounds);

		Results.xys = p[0];
        depths = p[1];
		Results.radii = p[2];
        conics = p[3];
        numTilesHit = p[4];
        #else
            throw std::runtime_error("GPU support not built, use --cpu");
        #endif
    }
    
	Results.xys.retain_grad();

	//	is this a failure? no splats present??
    if (Results.radii.sum().item<float>() == 0.0f)
	{
		Results.rgb = backgroundColor.repeat({height, width, 1});
		return Results;
	}

	auto shDegreeInterval = params.shDegreeInterval;
	auto shDegree = params.shDegree;
	
	//	get each splat's direction to camera in world space
    torch::Tensor viewDirs = means.detach() - T.transpose(0, 1).to(device);
    viewDirs = viewDirs / viewDirs.norm(2, {-1}, true);
    int degreesToUse = (std::min<int>)(step / shDegreeInterval, shDegree);
    
    if (device == torch::kCPU){
        Results.rgb = SphericalHarmonicsCPU::apply(degreesToUse, viewDirs, colors);
    }else{
        #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
		Results.rgb = SphericalHarmonics::apply(degreesToUse, viewDirs, colors);
        #endif
    }
    
	//	add 0.5 to each colour... no cap?
	Results.rgb = torch::clamp_min(Results.rgb + 0.5f, 0.0f);

    if (device == torch::kCPU){
        Results.rgb = RasterizeGaussiansCPU::apply(
										   Results.xys,
										   Results.radii,
										   conics,
										   Results.rgb,
										   torch::sigmoid(opacities),
										   cov2d,
										   camDepths,
										   height,
										   width,
										   backgroundColor);
    }else{  
        #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
        Results.rgb = RasterizeGaussians::apply(
										Results.xys,
										depths,
										Results.radii,
										conics,
										numTilesHit,
										Results.rgb,
										torch::sigmoid(opacities),
										height,
										width,
										backgroundColor);
        #endif
    }

    Results.rgb = torch::clamp_max(Results.rgb, 1.0f);

    return Results;
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

void Model::schedulersStep(int step){
  meansOptScheduler->step(step);
}

int Model::getDownscaleFactor(int step)
{
	auto numDownscales = params.numDownscales;
	auto resolutionSchedule = params.resolutionSchedule;
    return std::pow(2, (std::max<int>)(numDownscales - step / resolutionSchedule, 0));
}

void Model::addToOptimizer(torch::optim::Adam& optimizer, const torch::Tensor &newParam, const torch::Tensor &idcs, int nSamples){
    torch::Tensor param = optimizer.param_groups()[0].params()[0];
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto pId = param.unsafeGetTensorImpl();
#else
    auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
#endif
    auto paramState = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(*optimizer.state()[pId]));
    
    std::vector<int64_t> repeats;
    repeats.push_back(nSamples);
    for (long int i = 0; i < paramState->exp_avg().dim() - 1; i++){
        repeats.push_back(1);
    }

    paramState->exp_avg(torch::cat({
        paramState->exp_avg(), 
        torch::zeros_like(paramState->exp_avg().index({idcs.squeeze()})).repeat(repeats)
    }, 0));
    
    paramState->exp_avg_sq(torch::cat({
        paramState->exp_avg_sq(), 
        torch::zeros_like(paramState->exp_avg_sq().index({idcs.squeeze()})).repeat(repeats)
    }, 0));

    optimizer.state().erase(pId);

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto newPId = newParam.unsafeGetTensorImpl();
#else
    auto newPId = c10::guts::to_string(newParam.unsafeGetTensorImpl());
#endif    
    optimizer.state()[newPId] = std::move(paramState);
    optimizer.param_groups()[0].params()[0] = newParam;
}

void Model::removeFromOptimizer(torch::optim::Adam& optimizer, const torch::Tensor &newParam, const torch::Tensor &deletedMask){
    torch::Tensor param = optimizer.param_groups()[0].params()[0];
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto pId = param.unsafeGetTensorImpl();
#else
    auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
#endif
    auto paramState = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(*optimizer.state()[pId]));

    paramState->exp_avg(paramState->exp_avg().index({~deletedMask}));
    paramState->exp_avg_sq(paramState->exp_avg_sq().index({~deletedMask}));

    optimizer.state().erase(pId);
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto newPId = newParam.unsafeGetTensorImpl();
#else
    auto newPId = c10::guts::to_string(newParam.unsafeGetTensorImpl());
#endif
    optimizer.param_groups()[0].params()[0] = newParam;
    optimizer.state()[newPId] = std::move(paramState);
}

void Model::afterTrain(int step,ModelForwardResults& ForwardMeta){
    torch::NoGradGuard noGrad;

    // When radii.sum() == 0
    if (!ForwardMeta.xys.grad().defined()) return;

    if (step < stopSplitAt){
        torch::Tensor visibleMask = (ForwardMeta.radii > 0).flatten();
        
        torch::Tensor grads = torch::linalg_vector_norm(ForwardMeta.xys.grad().detach(), 2, { -1 }, false, torch::kFloat32);
        if (!xysGradNorm.numel()){
            xysGradNorm = grads;
            visCounts = torch::ones_like(xysGradNorm);
        }else{
            visCounts.index_put_({visibleMask}, visCounts.index({visibleMask}) + 1);
            xysGradNorm.index_put_({visibleMask}, grads.index({visibleMask}) + xysGradNorm.index({visibleMask}));
        }

        if (!max2DSize.numel()){
            max2DSize = torch::zeros_like(ForwardMeta.radii, torch::kFloat32);
        }

        torch::Tensor newRadii = ForwardMeta.radii.detach().index({visibleMask});
        max2DSize.index_put_({visibleMask}, torch::maximum(
                max2DSize.index({visibleMask}), newRadii / static_cast<float>( std::max(ForwardMeta.lastHeight, ForwardMeta.lastWidth) )
            ));
    }

	auto refineEvery = params.refineEvery;
	auto warmupLength = params.warmupLength;
	auto resetAlphaEvery = params.resetAlphaEvery;
	auto densifyGradThresh = params.densifyGradThresh;
	auto densifySizeThresh = params.densifySizeThresh;
	auto stopScreenSizeAt = params.stopScreenSizeAt;
	auto splitScreenSize = params.splitScreenSize;
	
    if (step % refineEvery == 0 && step > warmupLength){
        int resetInterval = resetAlphaEvery * refineEvery;
        bool doDensification = step < stopSplitAt && step % resetInterval > numCameras + refineEvery;
        torch::Tensor splitsMask;
        const float cullAlphaThresh = 0.1f;

        if (doDensification){
            int numPointsBefore = means.size(0);
            torch::Tensor avgGradNorm = (xysGradNorm / visCounts) * 0.5f * static_cast<float>( std::max(ForwardMeta.lastWidth, ForwardMeta.lastHeight) );
            torch::Tensor highGrads = (avgGradNorm > densifyGradThresh).squeeze();

            // Split gaussians that are too large
            torch::Tensor splits = (std::get<0>(scales.exp().max(-1)) > densifySizeThresh).squeeze();
            if (step < stopScreenSizeAt){
                splits |= (max2DSize > splitScreenSize).squeeze();
            }

            splits &= highGrads;
            const int nSplitSamples = 2;
            int nSplits = splits.sum().item<int>();

            torch::Tensor centeredSamples = torch::randn({nSplitSamples * nSplits, 3}, device);  // Nx3 of axis-aligned scales
            torch::Tensor scaledSamples = torch::exp(scales.index({splits}).repeat({nSplitSamples, 1})) * centeredSamples;
            torch::Tensor qs = quats.index({splits}) / torch::linalg_vector_norm(quats.index({splits}), 2, { -1 }, true, torch::kFloat32);
            torch::Tensor rots = quatToRotMat(qs.repeat({nSplitSamples, 1}));
            torch::Tensor rotatedSamples = torch::bmm(rots, scaledSamples.index({"...", None})).squeeze();
            torch::Tensor splitMeans = rotatedSamples + means.index({splits}).repeat({nSplitSamples, 1});
            
            torch::Tensor splitFeaturesDc = featuresDc.index({splits}).repeat({nSplitSamples, 1});
            torch::Tensor splitFeaturesRest = featuresRest.index({splits}).repeat({nSplitSamples, 1, 1});
            
            torch::Tensor splitOpacities = opacities.index({splits}).repeat({nSplitSamples, 1});
        
            const float sizeFac = 1.6f;
            torch::Tensor splitScales = torch::log(torch::exp(scales.index({splits})) / sizeFac).repeat({nSplitSamples, 1});
            scales.index({splits}) = torch::log(torch::exp(scales.index({splits})) / sizeFac);
            torch::Tensor splitQuats = quats.index({splits}).repeat({nSplitSamples, 1});

            // Duplicate gaussians that are too small
            torch::Tensor dups = (std::get<0>(scales.exp().max(-1)) <= densifySizeThresh).squeeze();
            dups &= highGrads;
            torch::Tensor dupMeans = means.index({dups});
            torch::Tensor dupFeaturesDc = featuresDc.index({dups});
            torch::Tensor dupFeaturesRest = featuresRest.index({dups});
            torch::Tensor dupOpacities = opacities.index({dups});
            torch::Tensor dupScales = scales.index({dups});
            torch::Tensor dupQuats = quats.index({dups});

            means = torch::cat({means.detach(), splitMeans, dupMeans}, 0).requires_grad_();
            featuresDc = torch::cat({featuresDc.detach(), splitFeaturesDc, dupFeaturesDc}, 0).requires_grad_();
            featuresRest = torch::cat({featuresRest.detach(), splitFeaturesRest, dupFeaturesRest}, 0).requires_grad_();
            opacities = torch::cat({opacities.detach(), splitOpacities, dupOpacities}, 0).requires_grad_();
            scales = torch::cat({scales.detach(), splitScales, dupScales}, 0).requires_grad_();
            quats = torch::cat({quats.detach(), splitQuats, dupQuats}, 0).requires_grad_();
            
            max2DSize = torch::cat({
                max2DSize,
                torch::zeros_like(splitScales.index({Slice(), 0})),
                torch::zeros_like(dupScales.index({Slice(), 0}))
            }, 0);

            torch::Tensor splitIdcs = torch::where(splits)[0];

			addToOptimizer(*meansOpt, means, splitIdcs, nSplitSamples);
            addToOptimizer(*scalesOpt, scales, splitIdcs, nSplitSamples);
            addToOptimizer(*quatsOpt, quats, splitIdcs, nSplitSamples);
            addToOptimizer(*featuresDcOpt, featuresDc, splitIdcs, nSplitSamples);
            addToOptimizer(*featuresRestOpt, featuresRest, splitIdcs, nSplitSamples);
            addToOptimizer(*opacitiesOpt, opacities, splitIdcs, nSplitSamples);
            
            torch::Tensor dupIdcs = torch::where(dups)[0];
            addToOptimizer(*meansOpt, means, dupIdcs, 1);
            addToOptimizer(*scalesOpt, scales, dupIdcs, 1);
            addToOptimizer(*quatsOpt, quats, dupIdcs, 1);
            addToOptimizer(*featuresDcOpt, featuresDc, dupIdcs, 1);
            addToOptimizer(*featuresRestOpt, featuresRest, dupIdcs, 1);
            addToOptimizer(*opacitiesOpt, opacities, dupIdcs, 1);

            splitsMask = torch::cat({
                splits,
                torch::full({nSplitSamples * splits.sum().item<int>() + dups.sum().item<int>()}, false, torch::TensorOptions().dtype(torch::kBool).device(device))
            }, 0);

            std::cout << "Added " << (means.size(0) - numPointsBefore) << " gaussians, new count " << means.size(0) << std::endl;
        }

        if (doDensification){
            // Cull
            int numPointsBefore = means.size(0);

            torch::Tensor culls = (torch::sigmoid(opacities) < cullAlphaThresh).squeeze();
            if (splitsMask.numel()){
                culls |= splitsMask;
            }

            if (step > refineEvery * resetAlphaEvery){
                const float cullScaleThresh = 0.5f; // cull huge gaussians
                const float cullScreenSize = 0.15f; // % of screen space
                torch::Tensor huge = std::get<0>(torch::exp(scales).max(-1)) > cullScaleThresh;
                if (step < stopScreenSizeAt){
                    huge |= max2DSize > cullScreenSize;
                }
                culls |= huge;
            }

            int cullCount = torch::sum(culls).item<int>();
            if (cullCount > 0){
                means = means.index({~culls}).detach().requires_grad_();
                scales = scales.index({~culls}).detach().requires_grad_();
                quats = quats.index({~culls}).detach().requires_grad_();
                featuresDc = featuresDc.index({~culls}).detach().requires_grad_();
                featuresRest = featuresRest.index({~culls}).detach().requires_grad_();
                opacities = opacities.index({~culls}).detach().requires_grad_();

                removeFromOptimizer(*meansOpt, means, culls);
                removeFromOptimizer(*scalesOpt, scales, culls);
                removeFromOptimizer(*quatsOpt, quats, culls);
                removeFromOptimizer(*featuresDcOpt, featuresDc, culls);
                removeFromOptimizer(*featuresRestOpt, featuresRest, culls);
                removeFromOptimizer(*opacitiesOpt, opacities, culls);
                
                std::cout << "Culled " << (numPointsBefore - means.size(0)) << " gaussians, remaining " << means.size(0) << std::endl;
            }
        }

        if (step < stopSplitAt && step % resetInterval == refineEvery){
            float resetValue = cullAlphaThresh * 2.0f;
            opacities = torch::clamp_max(opacities, torch::logit(torch::tensor(resetValue)).item<float>());

            // Reset optimizer
            torch::Tensor param = opacitiesOpt->param_groups()[0].params()[0];
            #if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
                auto pId = param.unsafeGetTensorImpl();
            #else
                auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
            #endif    
            auto paramState = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(*opacitiesOpt->state()[pId]));
            paramState->exp_avg(torch::zeros_like(paramState->exp_avg()));
            paramState->exp_avg_sq(torch::zeros_like(paramState->exp_avg_sq()));
            std::cout << "Alpha reset" << std::endl;
        }

        // Clear
        xysGradNorm = torch::Tensor();
        visCounts = torch::Tensor();
        max2DSize = torch::Tensor();

        if (device != torch::kCPU){
            #ifdef USE_HIP
                    c10::hip::HIPCachingAllocator::emptyCache();
            #elif defined(USE_CUDA)
                    c10::cuda::CUDACachingAllocator::emptyCache();
            #endif
        }
    }
}

void Model::findInvalidPoints()
{
	auto FindNans = [](std::span<float> Values,std::string_view Context)
	{
		auto NanCount = 0;
		auto InfCount = 0;
		for ( auto Value : Values )
		{
			if ( std::isnan(Value) )
				NanCount++;
			if ( std::isinf(Value) )
				InfCount++;
			//	todo: other invalid values; eg, 0,0,0 should never really be expected?
		}
		if ( NanCount > 0 || InfCount > 0 )
		{
			std::stringstream Error;
			Error << "Found " << NanCount << " NaNs & " << InfCount << " infinitys in " << Context << " data";
			throw std::runtime_error(Error.str());
		}
	};

	auto CheckPoint = [&](std::span<float> xyz,
						  std::span<float> opacity,
						  std::span<float> scale,
						  std::span<float> quaternionwxyz,
						  std::span<float> dcFeatures,
						  std::span<float> restFeatures)
	{
		FindNans(xyz,"means");
		FindNans(dcFeatures,"dcFeatures");
		FindNans(restFeatures,"restFeatures");
		FindNans(opacity,"opacity");
		FindNans(scale,"scale");
		FindNans(quaternionwxyz,"quaternion");
	};
	
	iteratePoints( CheckPoint );
}

void Model::iteratePoints(std::function<void(std::span<float> xyz,std::span<float> opacity,std::span<float> scale,std::span<float> quaternionwxyz,std::span<float> dcfeatures,std::span<float> restfeatures)> OnFoundPoint)
{
	int numPoints = means.size(0);
	
	torch::Tensor meansCpu = means.cpu();
	torch::Tensor featuresDcCpu = featuresDc.cpu();
	//torch::Tensor featuresRestCpu = featuresRest.cpu().transpose(1, 2).reshape({numPoints, -1});
	torch::Tensor featuresRestCpu = featuresRest.cpu();
	torch::Tensor opacitiesCpu = opacities.cpu();
	torch::Tensor scalesCpu = scales.cpu();
	torch::Tensor quatsCpu = quats.cpu();
	
		
	for (size_t i = 0; i < numPoints; i++) 
	{
		auto DcFeaturesCount = featuresDcCpu.size(1);
		auto RestFeaturesCount = featuresRestCpu.size(1);
		
		std::span xyz( reinterpret_cast<float*>( meansCpu[i].data_ptr() ), 3 );
		std::span dcFeatures( reinterpret_cast<float*>( featuresDcCpu[i].data_ptr() ), DcFeaturesCount );
		std::span restFeatures( reinterpret_cast<float*>(featuresRestCpu[i].data_ptr()), RestFeaturesCount );
		std::span opacity( reinterpret_cast<float*>(opacitiesCpu[i].data_ptr()), 2 );
		std::span scale( reinterpret_cast<float*>(scalesCpu[i].data_ptr()), 3 );
		std::span quaternion( reinterpret_cast<float*>(quatsCpu[i].data_ptr()), 4 );
		
		OnFoundPoint( xyz, opacity, scale, quaternion, dcFeatures, restFeatures );
	}
}

void Model::save(const std::string &filename, int step,bool KeepCrs){
    if (fs::path(filename).extension().string() == ".splat"){
        saveSplat(filename,KeepCrs);
    }else{
        savePly(filename, step,KeepCrs);
    }
    std::cout << "Wrote " << filename << std::endl;
}

void Model::savePly(const std::string &filename, int step,bool keepCrs){
    std::ofstream o(filename, std::ios::binary);
	if ( !o.is_open() )
		throw std::runtime_error(std::string("Failed to open file") + filename + " to write PLY");
	
	auto PointCount = means.size(0);

	torch::Tensor featuresRestCpu = featuresRest.cpu().transpose(1, 2).reshape({PointCount, -1});
	torch::Tensor meansCpu = keepCrs ? (means.cpu() / scale) + translation : means.cpu();
	torch::Tensor featuresDcCpu = featuresDc.cpu();
	torch::Tensor opacitiesCpu = opacities.cpu();
	torch::Tensor scalesCpu = keepCrs ? torch::log((torch::exp(scales.cpu()) / scale)) : scales.cpu();
	torch::Tensor quatsCpu = quats.cpu();

	Ply::WriteParams WriteParams;
	WriteParams.PointCount = PointCount;

	auto FeatureDcCount = featuresDc.size(1);
	auto FeatureRestCount = featuresRestCpu.size(1);

	auto GetPoint = [&](int Index)
	{
		SplatElement Point;
		
		std::span xyz( reinterpret_cast<float*>( meansCpu[Index].data_ptr() ), 3 );
		std::span dcFeatures( reinterpret_cast<float*>( featuresDcCpu[Index].data_ptr() ), FeatureDcCount );
		std::span restFeatures( reinterpret_cast<float*>(featuresRestCpu[Index].data_ptr()), FeatureRestCount );
		std::span opacity( reinterpret_cast<float*>(opacitiesCpu[Index].data_ptr()), 2 );
		std::span scale( reinterpret_cast<float*>(scalesCpu[Index].data_ptr()), 3 );
		std::span quaternion( reinterpret_cast<float*>(quatsCpu[Index].data_ptr()), 4 );
		
		Point.Splat.x = xyz[0];
		Point.Splat.y = xyz[1];
		Point.Splat.z = xyz[2];
		Point.Splat.scalex = scale[0];
		Point.Splat.scaley = scale[1];
		Point.Splat.scalez = scale[2];
		Point.Splat.rotx = quaternion[1];
		Point.Splat.roty = quaternion[2];
		Point.Splat.rotz = quaternion[3];
		Point.Splat.rotw = quaternion[0];
		Point.Splat.opacity = opacity[0];
		
		Point.OverrideFeatureDcs = dcFeatures;
		Point.FeatureRests = restFeatures;
		
		return Point;
	};
	
	
	
	std::stringstream Comment;
	Comment << "Generated by opensplat at iteration " << step;
	WriteParams.Comment = Comment.str();
	
	Ply::Write( o, WriteParams, GetPoint ); 

	o.close();
}

void Model::saveSplat(const std::string &filename,bool keepCrs){
    std::ofstream o(filename, std::ios::binary);
	if ( !o.is_open() )
		throw std::runtime_error(std::string("Failed to open file") + filename + " to write PLY");

	int numPoints = means.size(0);

    torch::Tensor meansCpu = keepCrs ? (means.cpu() / scale) + translation : means.cpu();
    torch::Tensor scalesCpu = keepCrs ? (torch::exp(scales.cpu()) / scale) : torch::exp(scales.cpu());
    torch::Tensor rgbsCpu = (sh2rgb(featuresDc.cpu()) * 255.0f).toType(torch::kUInt8);
    torch::Tensor opac = (1.0f + torch::exp(-opacities.cpu()));
    torch::Tensor opacitiesCpu = torch::clamp(((1.0f / opac) * 255.0f), 0.0f, 255.0f).toType(torch::kUInt8);
    torch::Tensor quatsCpu = torch::clamp(quats.cpu() * 128.0f + 128.0f, 0.0f, 255.0f).toType(torch::kUInt8);

    std::vector< size_t > splatIndices( numPoints );
    std::iota( splatIndices.begin(), splatIndices.end(), 0 );
    torch::Tensor order = (scalesCpu.index({"...", 0}) + 
                            scalesCpu.index({"...", 1}) + 
                            scalesCpu.index({"...", 2})) / 
                            opac.index({"...", 0});
    float *orderPtr = reinterpret_cast<float *>(order.data_ptr());

    std::sort(splatIndices.begin(), splatIndices.end(), 
        [&orderPtr](size_t const &a, size_t const &b) {
            return orderPtr[a] > orderPtr[b];
        });

    for (int i = 0; i < numPoints; i++){
        size_t idx = splatIndices[i];

        o.write(reinterpret_cast<const char *>(meansCpu[idx].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(scalesCpu[idx].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(rgbsCpu[idx].data_ptr()), sizeof(uint8_t) * 3);
        o.write(reinterpret_cast<const char *>(opacitiesCpu[idx].data_ptr()), sizeof(uint8_t) * 1);
        o.write(reinterpret_cast<const char *>(quatsCpu[idx].data_ptr()), sizeof(uint8_t) * 4);
    }
    o.close();
}

void Model::saveDebugPly(const std::string &filename, int step,bool keepCrs){
    // A standard PLY
    std::ofstream o(filename, std::ios::binary);
	if ( !o.is_open() )
		throw std::runtime_error(std::string("Failed to open file") + filename + " to write debug PLY");

	int numPoints = means.size(0);

    o << "ply" << std::endl;
    o << "format binary_little_endian 1.0" << std::endl;
    o << "comment Generated by opensplat at iteration " << step << std::endl;
    o << "element vertex " << numPoints << std::endl;
    o << "property float x" << std::endl;
    o << "property float y" << std::endl;
    o << "property float z" << std::endl;
    o << "property uchar red" << std::endl;
    o << "property uchar green" << std::endl;
    o << "property uchar blue" << std::endl;
    o << "end_header" << std::endl;

    torch::Tensor meansCpu = keepCrs ? (means.cpu() / scale) + translation : means.cpu();
    torch::Tensor rgbsCpu = (sh2rgb(featuresDc.cpu()) * 255.0f).toType(torch::kUInt8);

    for (size_t i = 0; i < numPoints; i++) {
        o.write(reinterpret_cast<const char *>(meansCpu[i].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(rgbsCpu[i].data_ptr()), sizeof(uint8_t) * 3);
    }

    o.close();
    std::cout << "Wrote " << filename << std::endl;
}

//	modelPointsNeedToBeNormalised == keepCrs
//	it means that the PLY we're loading, was saved in it's original space instead of centered and scaled to -1...1
int Model::loadPly(const std::string &filename,bool modelPointsNeedToBeNormalised){
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Invalid PLY file");

    // Ensure we have a valid ply file
    std::string line;
	int step = 0;
    size_t bytesRead = 0;

    std::getline(f, line);
    bytesRead += f.gcount();

    if (line == "ply"){
        std::getline(f, line);
        bytesRead += f.gcount();
        if (line == "format binary_little_endian 1.0"){
            std::getline(f, line);
            bytesRead += f.gcount();
            const std::string pattern = "comment Generated by opensplat at iteration ";

            if (line.rfind(pattern, 0) == 0){
                step = std::stoi(line.substr(pattern.length()));
                if (step >= 0){
                    std::getline(f, line);
                    bytesRead += f.gcount();
                    const std::string pattern = "element vertex ";

                    if (line.rfind(pattern, 0) == 0){
                        const int numPoints = std::stoi(line.substr(pattern.length()));
                        
                        const char *requiredProps[] = {
                            "property float x",
                            "property float y",
                            "property float z",
                            "property float nx",
                            "property float ny",
                            "property float nz",
                            "property float f_dc_"
                            "property float f_rest_",
                            "property float opacity",
                            "property float scale_0",
                            "property float scale_1",
                            "property float scale_2",
                            "property float rot_0",
                            "property float rot_1",
                            "property float rot_2",
                            "property float rot_3",
                            "end_header"
                        };

                        for (int i = 0; i < 6; i++){
                            std::getline(f, line);
                            bytesRead += f.gcount();
                            if (line != requiredProps[i]){
                                throw std::runtime_error(std::string("PLY file's header does not contain required property: ") + requiredProps[i]);
                            }
                        }
                        std::getline(f, line);
                        bytesRead += f.gcount();

                        auto countPrefixes = [&f, &line](const char *prefix){
                                int n = 0;
                                while(true){
                                    if (line.rfind(prefix, 0) == 0){
                                        ++n;
                                        std::getline(f, line);
                                    } else {
                                        break;
                                    }
                                }
                                return n;
                        };
                        int featuresDcSize = countPrefixes("property float f_dc_");
                        int featuresRestSize = countPrefixes("property float f_rest_");
                        
                        bool foundEnd = false;
                        for (int i = 8; i < std::size(requiredProps); i++){
                            std::getline(f, line);
                            bytesRead += f.gcount();

                            if (line != requiredProps[i]){
                                throw std::runtime_error(std::string("PLY file's header does not contain required property: ") + requiredProps[i]);
                            }

                            if (line == "end_header"){
                                foundEnd = true;
                                break;
                            }
                        }

                        if (!foundEnd){
                            throw std::runtime_error("PLY file header does not contain header end");
                        } 

                        const size_t bytesPerPoint = sizeof(float) * (14 + featuresDcSize + featuresRestSize);
                        const size_t remainingFileSize = fs::file_size(filename) - bytesRead;
                        if (remainingFileSize != bytesPerPoint * numPoints){
                            std::cout << "Loading PLY..." << std::endl;
                            
                            float zeros[3];

                            torch::Tensor meansCpu = torch::zeros({numPoints, 3}, torch::TensorOptions().dtype(torch::kFloat32));
                            torch::Tensor featuresDcCpu = torch::zeros({numPoints, featuresDcSize}, torch::TensorOptions().dtype(torch::kFloat32));
                            torch::Tensor featuresRestCpu = torch::zeros({numPoints, featuresRestSize}, torch::TensorOptions().dtype(torch::kFloat32));
                            torch::Tensor opacitiesCpu = torch::zeros({numPoints, 1}, torch::TensorOptions().dtype(torch::kFloat32));
                            torch::Tensor scalesCpu = torch::zeros({numPoints, 3}, torch::TensorOptions().dtype(torch::kFloat32));
                            torch::Tensor quatsCpu = torch::zeros({numPoints, 4}, torch::TensorOptions().dtype(torch::kFloat32));

                            for (size_t i = 0; i < numPoints; i++){
                                f.read(reinterpret_cast<char *>(meansCpu[i].data_ptr()), sizeof(float) * 3);
                                f.read(reinterpret_cast<char *>(&zeros[0]), sizeof(float) * 3);
                                f.read(reinterpret_cast<char *>(featuresDcCpu[i].data_ptr()), sizeof(float) * featuresDcSize);
                                f.read(reinterpret_cast<char *>(featuresRestCpu[i].data_ptr()), sizeof(float) * featuresRestSize);
                                f.read(reinterpret_cast<char *>(opacitiesCpu[i].data_ptr()), sizeof(float) * 1);
                                f.read(reinterpret_cast<char *>(scalesCpu[i].data_ptr()), sizeof(float) * 3);
                                f.read(reinterpret_cast<char *>(quatsCpu[i].data_ptr()), sizeof(float) * 4);
                            }
							if (modelPointsNeedToBeNormalised){
                                meansCpu = (meansCpu - translation) * scale;
                                scalesCpu = torch::log(scale * torch::exp(scalesCpu));
                            }
                            
                            means = meansCpu.to(device).requires_grad_();
                            featuresDc = featuresDcCpu.to(device).requires_grad_();
                            featuresRest = featuresRestCpu.reshape({numPoints, 3, featuresRestSize/3}).transpose(2, 1).to(device).requires_grad_();
                            opacities = opacitiesCpu.to(device).requires_grad_();
                            scales = scalesCpu.to(device).requires_grad_();
                            quats = quatsCpu.to(device).requires_grad_();
                            
                            std::cerr << "Loaded " << means.size(0) << " gaussians" << std::endl;
                            
                            setupOptimizers();
                            
                            f.close();
                            return step;
                        } else {
                            throw std::runtime_error("PLY file's data section is wrong size");
                        }
                    }
                } else {
                    throw std::runtime_error("PLY file failed sanity check: iteration count should not begin at 0");
                }
            } else if (line.rfind("comment Generated by opensplat")){
                throw std::runtime_error("PLY file does not contain iteration count metadata. You can edit the file to add this metadata manually, by changing \"comment Generated by opensplat\" to \"comment Generated by opensplat at iteration 12345\", changing 12345 to the desired value.");
            }
        }
    }
    throw std::runtime_error("Invalid PLY file");
}

torch::Tensor Model::mainLoss(torch::Tensor &rgb, torch::Tensor &gt, float ssimWeight)
{
	//	catch error before torch and give a much clearer error
	auto RgbHeight = rgb.size(0);
	auto RgbWidth = rgb.size(1);
	auto RgbComponents = rgb.size(2);
	auto GroundTruthHeight = gt.size(0);
	auto GroundTruthWidth = gt.size(1);
	auto GroundTruthComponents = gt.size(2);
	if ( RgbWidth != GroundTruthWidth || RgbHeight != GroundTruthHeight || RgbComponents != GroundTruthComponents )
	{
		std::stringstream Error;
		Error << "Cannot calculate loss between image(" << RgbWidth << "x" << RgbHeight << "x" << RgbComponents << ") and ground truth (" << GroundTruthWidth << "x" << GroundTruthHeight << "x" << GroundTruthComponents << ") of different sizes";
		throw std::runtime_error(Error.str());
	}
	
    torch::Tensor ssimLoss = 1.0f - ssim.eval(rgb, gt);
    torch::Tensor l1Loss = l1(rgb, gt);
    return (1.0f - ssimWeight) * l1Loss + ssimWeight * ssimLoss;
}
