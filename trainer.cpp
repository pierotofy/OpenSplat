#include "trainer.hpp"

#include <nlohmann/json.hpp>
#include "opensplat.hpp"
#include "input_data.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"
#include "constants.hpp"
#include "trainer_params.hpp"
#include "utils.hpp"

#include <torch/torch.h>



OpenSplat::NoCameraException::NoCameraException(int CameraIndex,int CameraCount)
{
	std::stringstream Error;
	//	with 1 camera, this will show 1/1 when 1 is out of range...
	//	todo: better message
	Error << "Camera index " << CameraIndex << "/" << CameraCount << " out of range";
	mMessage = Error.str();
}


ImagePixels::ImagePixels(std::span<uint8_t> Pixels,int Width,int Height,OpenSplat_PixelFormat Format)
{
	mWidth = Width;
	mHeight = Height;
	mFormat = Format;
	
	//	verify size
	auto ExpectedSize = 3 * mWidth * mHeight;
	if ( Pixels.size() != ExpectedSize )
	{
		throw std::runtime_error("Wrong pixel count");
	}
	std::copy( Pixels.begin(), Pixels.end(), std::back_inserter(mPixels) );
}


ImagePixels::ImagePixels(const torch::Tensor& tensor,OpenSplat_PixelFormat TensorPixelFormat)
{
	//	from tensorToImage()
	torch::Tensor Tensor8 = tensor.cpu();

	//	todo: allow float image to make this extraction faster
	Tensor8 = (Tensor8 * 255.0);
	Tensor8 = Tensor8.toType(torch::kU8);

	mHeight = Tensor8.size(0);
	mWidth = Tensor8.size(1);
	mFormat = TensorPixelFormat;
	int components = Tensor8.size(2);
	if ( components != 3 )
		throw std::runtime_error("Only images with 3 channels are supported");

	//	gr: is the data garunteed to be contiguious?	
	auto TensorSize = mWidth * mHeight * components;
	uint8_t* TensorData = static_cast<uint8_t*>(Tensor8.data_ptr());
	std::span TensorView( TensorData, TensorSize );
	std::copy( TensorView.begin(), TensorView.end(), std::back_inserter(mPixels) );
}


ImagePixels::ImagePixels(const cv::Mat& OpencvImage,OpenSplat_PixelFormat OpencvImagePixelFormat)
{
	auto OpencvPixels = std::span( OpencvImage.data, OpencvImage.total() * OpencvImage.elemSize() );
	
	auto PixelSize = OpencvImage.elemSize();
	if ( PixelSize != 3*sizeof(uint8_t) )
	{
		std::stringstream Error;
		Error << "Opencv image has pixel size " << PixelSize << "bytes, expected 3(bgr)"; 
		throw std::runtime_error(Error.str());
	}
	
	mWidth = OpencvImage.cols;
	mHeight = OpencvImage.rows;
	mFormat = OpencvImagePixelFormat;

	std::copy( OpencvPixels.begin(), OpencvPixels.end(), std::back_inserter(mPixels) );
}

void ImagePixels::ConvertTo(OpenSplat_PixelFormat NewFormat)
{
	if ( mFormat == NewFormat )
		return;
	
	auto SwapBgrRgb = [](cv::Mat& Image)
	{
		cv::cvtColor(Image, Image, cv::COLOR_RGB2BGR);
	};
	
	//	checks in case of future changes
	if ( mFormat == OpenSplat_PixelFormat_Bgr && NewFormat == OpenSplat_PixelFormat_Rgb )
	{
		GetOpenCvImage( SwapBgrRgb, false );
		mFormat = NewFormat;
		return;
	}
	
	if ( mFormat == OpenSplat_PixelFormat_Rgb && NewFormat == OpenSplat_PixelFormat_Bgr )
	{
		GetOpenCvImage( SwapBgrRgb, false );
		mFormat = NewFormat;
		return;
	}
	
	std::stringstream Error;
	Error << "ImagePixels conversion from " << mFormat << " to " << NewFormat << " unhandled";
	throw std::runtime_error(Error.str());
}

//	callback so we can use pixels in place - mat is only valid for lifetime of callback
void ImagePixels::GetOpenCvImage(std::function<void(cv::Mat&)> OnImage,bool AllowConversion)
{
	if ( mFormat == OpenSplat_PixelFormat_Bgr )
	{
		GetOpenCvImage( mPixels, mWidth, mHeight, OnImage );
		return;
	}

	int type = CV_8UC3;
	cv::Mat ImageInPlace( mHeight, mWidth, type, mPixels.data() );

	//	slow path to convert to Rgb
	if ( !AllowConversion )
		throw std::runtime_error("ImagePixels::GetOpencvImage cannot convert to bgr");

	cv::Mat BgrCopy;
	cv::cvtColor( ImageInPlace, BgrCopy, cv::COLOR_RGB2BGR);
	OnImage(BgrCopy);
}


void ImagePixels::GetOpenCvImage(std::span<uint8_t> Pixels,int Width,int Height,std::function<void(cv::Mat&)> OnImage)
{
	int type = CV_8UC3;
	cv::Mat ImageInPlace( Height, Width, type, Pixels.data() );
	OnImage(ImageInPlace);
}

	

Trainer::Trainer(const TrainerParams& Params) :
	mParams		( Params )
{
	mIterationRandomCameraIndex.seed( Params.iterationRandomCameraIndexSeed );
}

Trainer::~Trainer()
{
	//	lock when setting free'd state
	{
		std::lock_guard Lock(mModelLock);
		mRunning = false;
	}
	//	any locks occurring now should bail out if !mRunning
	
	//	todo: use a mutex to make sure the blocking Run() is done
	//		wont use that in future when api controls iterations 
	{
		std::lock_guard Lock(mModelLock);
	}
}

torch::Device Trainer::GetDevice()
{
	auto ForceCpuDevice = mParams.ForceCpuDevice;
	
	if (torch::hasCUDA() && !ForceCpuDevice )
		return torch::kCUDA;

	if (torch::hasMPS() && !ForceCpuDevice)
		return torch::kMPS;

	return torch::kCPU;	
}

void Trainer::Run(std::function<void(TrainerIterationMeta)> OnIterationFinished,std::function<void(int)> OnRunFinished)
{
	InputData& inputData = GetInputData();
	
	//	temp during refactor
	auto& Params = mParams;
	auto& resume = Params.resumeFromPlyFilename;
	auto& numIters = Params.numIters;
	
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

	{
		std::lock_guard Lock(mModelLock);
		if ( !mRunning )
			throw OpenSplat::InstanceFreedException();
		mModel = std::make_shared<Model>(inputData, Params, numIters, mParams.BackgroundRgb, device );
	}
	size_t step = 1;
	
	if (!resume.empty())
	{
		std::lock_guard Lock(mModelLock);
		if ( !mRunning )
			throw OpenSplat::InstanceFreedException();
		auto& model = *mModel;
		step = model.loadPly(resume,Params.resumeFromPlyNeedsNormalising) + 1;
	}
	
	for (; step <= numIters; step++)
	{
		auto IterationMeta = Iteration(step);

		//	update meta/cache
		//	this can get a bit racy with std::map...
		mIterationsCompleted = step;
		mSplatCountCache = IterationMeta.mSplatCount;
		if ( mCameraIterations.count(IterationMeta.mCameraIndex) == 0 )
			mCameraIterations[IterationMeta.mCameraIndex] = 1;
		else
			mCameraIterations.at(IterationMeta.mCameraIndex)++;

		//	callback
		OnIterationFinished( IterationMeta );
	}
	
	auto CompletedSteps = step;	//	num_iters
	OnRunFinished( CompletedSteps );
}

TrainerIterationMeta Trainer::Iteration(int step)
{
	std::lock_guard Lock(mModelLock);
	if ( !mRunning )
		throw OpenSplat::InstanceFreedException();

	auto& model = GetModel();
	auto& inputData = GetInputData();
	
	auto device = GetDevice();
	
	auto& cams = inputData.cameras;

	TrainerIterationMeta IterationMeta;
	IterationMeta.mStep = step;
	IterationMeta.mCameraIndex = mIterationRandomCameraIndex() % cams.size();
	
	Camera& cam = cams[IterationMeta.mCameraIndex];
	std::cout << "Step #" << step << " training with camera " << cam.getName() << "(#" << IterationMeta.mCameraIndex << ")" << std::endl;
	
	//	look for nans before a step
	if ( mParams.CheckForInvalidPoints )
	{
		try
		{
			model.findInvalidPoints();
		}
		catch(std::exception& e)
		{
			std::cerr << "Warning; " << e.what() << std::endl;
		}
	}
	
	model.optimizersZeroGrad();
	
	//	rgb is a render of the scene
	//	todo: cache this to allow a fast fetch from the API
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
	
	IterationMeta.mSplatCount = model.getPointCount();
	
	return IterationMeta;
}

ImagePixels Trainer::GetForwardImage(Camera& Camera,int step)
{
	std::lock_guard Lock(mModelLock);
	
	auto& Model = GetModel();
	auto ForwardResults = Model.forward( Camera, step );
	
	//	old code detached, but we're not going to modify it... do we need to?
	auto rgbCpu = ForwardResults.rgb.detach().cpu();
	
	ImagePixels Image(rgbCpu);
	return Image;
}

ImagePixels Trainer::GetForwardImage(int CameraIndex,int RenderWidth,int RenderHeight)
{
	std::lock_guard Lock(mModelLock);
	
	auto& Model = GetModel();
	auto& Camera = GetInputData().GetCamera(CameraIndex);
	
	int Step = 0;
	auto CameraTransform = Camera.camToWorld;
	auto CameraIntrinsics = Camera.intrinsics;
	CameraIntrinsics.RemoveDistortionParameters();
	CameraIntrinsics.ScaleTo( RenderWidth, RenderHeight );
	
	auto ForwardResults = Model.forward( CameraTransform, CameraIntrinsics, Step );
	
	ImagePixels ForwardImage(ForwardResults.rgb);
	return ForwardImage;
}

ImagePixels Trainer::GetCameraImage(int CameraIndex,int OutputWidth,int OutputHeight)
{
	std::lock_guard Lock(mModelLock);
	
	auto& Camera = GetInputData().GetCamera(CameraIndex);

	//	todo: if we keep this function, we can pass in the pixel buffer to opencv and rescale in-place and avoid all these allocs & copies
	cv::Mat ImageScaled = Camera.getOpencvRgbImageStretched( OutputWidth, OutputHeight );
	ImagePixels Pixels(ImageScaled,OpenSplat_PixelFormat_Rgb);
	Pixels.ConvertTo(OpenSplat_PixelFormat_Rgb);
	return Pixels;
}

std::vector<OpenSplat_Splat> Trainer::GetModelSplats()
{
	std::lock_guard Lock(mModelLock);
	if ( !mModel )
		return {};
	
	auto& Model = GetModel();

	std::vector<OpenSplat_Splat> Splats;

	auto OnSplat = [&](std::span<float> xyz,std::span<float> opacity,std::span<float> scale,std::span<float> quaternionwxyz,std::span<float> dcFeatures,std::span<float> restFeatures)
	{
		OpenSplat_Splat Splat;
		Splat.x = xyz[0];
		Splat.y = xyz[1];
		Splat.z = xyz[2];
		Splat.scalex = scale[0];
		Splat.scaley = scale[1];
		Splat.scalez = scale[2];
		//	note quaternion order
		Splat.rotx = quaternionwxyz[1];
		Splat.roty = quaternionwxyz[2];
		Splat.rotz = quaternionwxyz[3];
		Splat.rotw = quaternionwxyz[0];
		Splat.opacity = opacity[0];
		
		Splat.dc0 = 0;
		Splat.dc1 = 0;
		Splat.dc2 = 0;
		if ( dcFeatures.size() > 0 )
			Splat.dc0 = dcFeatures[0];
		if ( dcFeatures.size() > 1 )
			Splat.dc1 = dcFeatures[1];
		if ( dcFeatures.size() > 2 )
			Splat.dc2 = dcFeatures[2];
		
		Splats.emplace_back(Splat);
	};

	Model.iteratePoints(OnSplat);
	
	return Splats;
}

int Trainer::GetIterationsForCamera(int CameraIndex)
{
	//	verify index
	auto& Camera = GetInputData().GetCamera(CameraIndex);
	
	if ( mCameraIterations.count(CameraIndex) == 0 )
		return 0;
	
	auto Count = mCameraIterations[CameraIndex];
	return Count;
}

OpenSplat_TrainerState Trainer::GetState()
{
	OpenSplat_TrainerState Meta;
	
	Meta.IterationsCompleted = mIterationsCompleted;
	Meta.CameraCount = GetInputData().cameras.size();
	Meta.SplatCount = mSplatCountCache;
	
	return Meta;
}

OpenSplat_CameraMeta Trainer::GetCameraMeta(int CameraIndex)
{
	auto& Camera = GetInputData().GetCamera(CameraIndex);
	
	OpenSplat_CameraMeta Meta;
	
	CopyStringToBuffer( Camera.getName(), Meta.Name, std::size(Meta.Name) );
	Meta.LocalToWorld = Camera.camToWorld.GetCamToWorldMatrix();
	Meta.TrainedIterations = GetIterationsForCamera(CameraIndex);
	
	return Meta;
}

void Trainer::LoadCamera(const OpenSplat_CameraMeta& CameraMeta,std::span<uint8_t> PixelBuffer,OpenSplat_PixelFormat PixelFormat)
{
	auto& InputData = GetInputData();
	
	//	find matching camera
	std::string CameraName(CameraMeta.Name);
	auto& Camera = InputData.GetCamera(CameraName);
	
	
	//	save meta
	//	todo: verify is good pose!
	Camera.camToWorld = CameraTransform(CameraMeta.LocalToWorld);
	
	
	//	load pixels
	auto Load = [&](cv::Mat& Pixels)
	{
		Camera.loadImage(Pixels,1);
	};

	if ( PixelFormat == OpenSplat_PixelFormat_Bgr )
	{
		ImagePixels::GetOpenCvImage( PixelBuffer, CameraMeta.Intrinsics.Width, CameraMeta.Intrinsics.Height, Load );
	}
	else
	{
		//	a copy here is extremely slow!
		ImagePixels Pixels( PixelBuffer, CameraMeta.Intrinsics.Width, CameraMeta.Intrinsics.Height, PixelFormat );
		Pixels.GetOpenCvImage(Load);
	}
}
