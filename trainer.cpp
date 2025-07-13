#include "trainer.hpp"

#include <nlohmann/json.hpp>
#include "opensplat.hpp"
#include "input_data.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"
#include "constants.hpp"
#include "trainer_params.hpp"

#include <torch/torch.h>


ImagePixels::ImagePixels(const torch::Tensor& tensor,Format TensorPixelFormat)
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


ImagePixels::ImagePixels(const cv::Mat& OpencvImage,Format OpencvImagePixelFormat)
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

void ImagePixels::ConvertTo(Format NewFormat)
{
	if ( mFormat == NewFormat )
		return;
	
	auto SwapBgrRgb = [](cv::Mat& Image)
	{
		cv::cvtColor(Image, Image, cv::COLOR_RGB2BGR);
	};
	
	//	checks in case of future changes
	if ( mFormat == Bgr && NewFormat == Rgb )
	{
		GetOpenCvImage( SwapBgrRgb, false );
		mFormat = NewFormat;
		return;
	}
	
	if ( mFormat == Rgb && NewFormat == Bgr )
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
	int type = CV_8UC3;
	cv::Mat ImageInPlace( mHeight, mWidth, type, mPixels.data() );

	//	slow path to convert to Rgb
	if ( mFormat != Bgr )
	{
		if ( AllowConversion )
		{
			cv::Mat BgrCopy;
			cv::cvtColor( ImageInPlace, BgrCopy, cv::COLOR_RGB2BGR);
			OnImage(BgrCopy);
			return;
		}
		
		throw std::runtime_error("ImagePixels::GetOpencvImage cannot convert to bgr");
	}
	
	OnImage(ImageInPlace);
}

	

Trainer::Trainer(const TrainerParams& Params) :
	mParams		( Params )
{
	mIterationRandomCameraIndex.seed( Params.iterationRandomCameraIndexSeed );
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
	
	
	mModel = std::make_shared<Model>(inputData, Params, numIters, mParams.BackgroundRgb, device );
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

		OnIterationFinished( IterationMeta );
	}
	
	auto CompletedSteps = step;	//	num_iters
	OnRunFinished( CompletedSteps );
}

TrainerIterationMeta Trainer::Iteration(int step)
{
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
	auto& Camera = GetInputData().GetCamera(CameraIndex);

	//	todo: if we keep this function, we can pass in the pixel buffer to opencv and rescale in-place and avoid all these allocs & copies
	cv::Mat ImageScaled = Camera.getOpencvRgbImageStretched( OutputWidth, OutputHeight );
	ImagePixels Pixels(ImageScaled,ImagePixels::Rgb);
	Pixels.ConvertTo(ImagePixels::Rgb);
	return Pixels;
}

std::vector<OpenSplat_Splat> Trainer::GetModelSplats()
{
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

