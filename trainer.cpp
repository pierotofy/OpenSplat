#include "trainer.hpp"

#include <nlohmann/json.hpp>
#include "opensplat.hpp"
#include "input_data.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"
#include "constants.hpp"
#include "trainer_params.hpp"

#include <torch/torch.h>


ImagePixels::ImagePixels(const torch::Tensor& tensor)
{
	//	from tensorToImage()
	torch::Tensor Tensor8 = tensor.cpu();

	//	todo: allow float image to make this extraction faster
	Tensor8 = (Tensor8 * 255.0);
	Tensor8 = Tensor8.toType(torch::kU8);

	mHeight = Tensor8.size(0);
	mWidth = Tensor8.size(1);
	mFormat = Rgb;
	int components = Tensor8.size(2);
	if ( components != 3 )
		throw std::runtime_error("Only images with 3 channels are supported");

	//	gr: is the data garunteed to be contiguious?	
	auto TensorSize = mWidth * mHeight * components;
	uint8_t* TensorData = static_cast<uint8_t*>(Tensor8.data_ptr());
	std::span TensorView( TensorData, TensorSize );
	std::copy( TensorView.begin(), TensorView.end(), std::back_inserter(mPixels) );
}

//	callback so we can use pixels in place - mat is only valid for lifetime of callback
void ImagePixels::GetCvImage(std::function<void(cv::Mat&)> OnImage)
{
	if ( mFormat != Rgb )
		throw std::runtime_error("ImagePixels::GetCvImage only supports RGB");
	
	int type = CV_8UC3;
	cv::Mat image( mHeight, mWidth, type, mPixels.data() );
	OnImage(image);
}

	

Trainer::Trainer(const TrainerParams& Params) :
	mParams		( Params )
{
	mIterationRandomCameraIndex.seed( Params.iterationRandomCameraIndexSeed );
}

torch::Device Trainer::GetDevice()
{
	auto ForceCpuDevice = mParams.mForceCpuDevice;
	
	if (torch::hasCUDA() && !ForceCpuDevice )
		return torch::kCUDA;

	if (torch::hasMPS() && !ForceCpuDevice)
		return torch::kMPS;

	return torch::kCPU;	
}

void Trainer::Run(std::function<void(TrainerIterationMeta,Camera*)> OnIterationFinished,std::function<void(int,Camera*)> OnRunFinished)
{
	InputData& inputData = GetInputData();
	
	//	temp during refactor
	auto& Params = mParams;
	auto& resume = Params.resumeFromPlyFilename;
	auto& validate = Params.validate;
	auto& valImage = Params.valImage;
	auto& numIters = Params.numIters;
	auto& numDownscales = Params.numDownscales;
	auto& resolutionSchedule = Params.resolutionSchedule;
	auto& shDegree = Params.shDegree;
	auto& shDegreeInterval = Params.shDegreeInterval;
	auto& ssimWeight = Params.ssimWeight;
	auto& refineEvery = Params.refineEvery;
	auto& warmupLength = Params.warmupLength;
	auto& resetAlphaEvery = Params.resetAlphaEvery;
	auto& densifyGradThresh = Params.densifyGradThresh;
	auto& densifySizeThresh = Params.densifySizeThresh;
	auto& stopScreenSizeAt = Params.stopScreenSizeAt;
	auto& splitScreenSize = Params.splitScreenSize;
	auto ForceCpuDevice = Params.mForceCpuDevice;
	
	auto device = GetDevice();
	if ( device == torch::kCUDA )
	{
		std::cout << "Using CUDA" << std::endl;
	}
	else if ( device == torch::kMPS )
	{
		std::cout << "Using MPS" << std::endl;
	}
	else
	{
		std::cout << "Using CPU" << std::endl;
	}
	
	// Withhold a validation camera if necessary
	auto t = inputData.getCameras(validate, valImage);
	std::vector<Camera> cams = std::get<0>(t);
	Camera *valCam = std::get<1>(t);
	
	mModel = std::make_shared<Model>(inputData,
				cams.size(),
				numDownscales, resolutionSchedule, shDegree, shDegreeInterval, 
				refineEvery, warmupLength, resetAlphaEvery, densifyGradThresh, densifySizeThresh, stopScreenSizeAt, splitScreenSize,
				numIters, 
				mParams.BackgroundRgb,
				device);
	auto& model = *mModel;
	/*
	std::vector<size_t> camIndices( cams.size() );
	std::iota( camIndices.begin(), camIndices.end(), 0 );
	InfiniteRandomIterator<size_t> camsIter( camIndices );
	*/
	size_t step = 1;
	
	if (!resume.empty())
	{
		step = model.loadPly(resume,Params.resumeFromPlyNeedsNormalising) + 1;
	}
	
	for (; step <= numIters; step++)
	{
		auto IterationMeta = Iteration(step);

		OnIterationFinished( IterationMeta, valCam );
	}
	
	auto CompletedSteps = step;	//	num_iters
	OnRunFinished( CompletedSteps, valCam );
}

TrainerIterationMeta Trainer::Iteration(int step)
{
	auto& model = GetModel();
	auto& inputData = GetInputData();
	auto& validate = mParams.validate;
	auto& valImage = mParams.valImage;
	
	auto device = GetDevice();
	
	auto t = inputData.getCameras(validate, valImage);
	std::vector<Camera> cams = std::get<0>(t);
	Camera *valCam = std::get<1>(t);

	TrainerIterationMeta IterationMeta;
	IterationMeta.mStep = step;
	IterationMeta.mCameraIndex = mIterationRandomCameraIndex() % cams.size();
	
	std::cout << "Step #" << step << " training with camera " << IterationMeta.mCameraIndex << std::endl;
	Camera& cam = cams[IterationMeta.mCameraIndex];
	
	//	look for nans before a step
	try
	{
		model.findInvalidPoints();
	}
	catch(std::exception& e)
	{
		std::cerr << "Warning; " << e.what() << std::endl;
	}
	
	model.optimizersZeroGrad();
	
	//	rgb is a render of the scene
	auto ForwardResults = model.forward(cam, step);
	torch::Tensor groundTruth = cam.getImage(model.getDownscaleFactor(step));
	groundTruth = groundTruth.to(device);
	
	//	calculate loss from render to ground truth
	auto ssimWeight = mParams.ssimWeight;
	torch::Tensor mainLoss = model.mainLoss(ForwardResults.rgb, groundTruth, ssimWeight);
	mainLoss.backward();

	IterationMeta.mLoss = mainLoss.item<float>();
	
	
	model.optimizersStep();
	model.schedulersStep(step);
	model.afterTrain(step,ForwardResults);
	
	return IterationMeta;
}

ImagePixels Trainer::GetForwardImage(Camera& Camera,int step)
{
	auto& Model = GetModel();
	auto ForwardResults = Model.forward( Camera, step );
	
	//	old code detached, but we're not going to modify it... do we need to?
	auto rgbCpu = ForwardResults.rgb.detach().cpu();
	
	ImagePixels Image(rgbCpu);
	return Image;
}

ImagePixels Trainer::GetForwardImage(int CameraIndex,int RenderWidth,int RenderHeight)
{
	auto& Model = GetModel();
	if ( CameraIndex < 0 || CameraIndex >= mInputData->cameras.size() )
	{
		std::stringstream Error;
		Error << "Camera index " << CameraIndex << "/" << mInputData->cameras.size() << " out of range"; 
		throw std::runtime_error(Error.str());
	}
	
	int Step = 0;
	auto& Camera = mInputData->cameras[CameraIndex];
	auto CameraTransform = Camera.camToWorld;
	auto CameraIntrinsics = Camera.intrinsics;
	CameraIntrinsics.RemoveDistortionParameters();
	CameraIntrinsics.ScaleTo( RenderWidth, RenderHeight );
	
	auto ForwardResults = Model.forward( CameraTransform, CameraIntrinsics, Step );
	
	ImagePixels ForwardImage(ForwardResults.rgb);
	return ForwardImage;
}
