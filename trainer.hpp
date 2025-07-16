#pragma once

#include <string_view>
#include <functional>
#include "trainer_params.hpp"
#include <random>
#include "trainer_api.h"

class Model;
class Camera;
class InputData;


//	cannot forward declare torch device...
#include "model.hpp"
/*
namespace torch
{
	struct Device;
}
*/

namespace cv
{
	class Mat;
}

class TrainerIterationMeta
{
public:
	int		mStep = -1;
	int		mCameraIndex = -1;
	float	mLoss = -1.0f;
	int		mSplatCount = -1;
};


namespace OpenSplat
{
	class ApiException : public std::exception
	{
	public:
		virtual OpenSplat_Error	GetApiError()=0;
	};
	
	class NoCameraException : public ApiException
	{
	public:
		NoCameraException(int CameraIndex,int CameraCount);
		
		virtual OpenSplat_Error	GetApiError() override	{	return OpenSplat_Error_NoCamera;	}
		virtual const char*		what() const noexcept override	{	return mMessage.c_str();	}

		std::string		mMessage;
	};
	
	class NoInstanceException : public ApiException
	{
	public:
		virtual OpenSplat_Error	GetApiError() override	{	return OpenSplat_Error_NoInstance;	}
	};
	
	
	class InstanceFreedException : public ApiException
	{
	public:
		virtual OpenSplat_Error	GetApiError() override	{	return OpenSplat_Error_InstanceFreed;	}
	};
}


class ImagePixels
{
public:
	ImagePixels(std::span<uint8_t> Pixels,int Width,int Height,OpenSplat_PixelFormat Format);
	ImagePixels(const torch::Tensor& Tensor,OpenSplat_PixelFormat TensorPixelFormat=OpenSplat_PixelFormat_Rgb);
	ImagePixels(const cv::Mat& OpencvImage,OpenSplat_PixelFormat OpencvImagePixelFormat=OpenSplat_PixelFormat_Bgr);
	
	//	callback so we can use pixels in place - mat is only valid for lifetime of callback
	//	if not-AllowConversion rgb will be passed in-place.
	void					GetOpenCvImage(std::function<void(cv::Mat&)> OnImage,bool AllowConversion=true);
	//	assumes format is BGR
	static void				GetOpenCvImage(std::span<uint8_t> Pixels,int Width,int Height,std::function<void(cv::Mat&)> OnImage);
	
	void					ConvertTo(OpenSplat_PixelFormat NewFormat);
	
	int						mWidth = 0;
	int						mHeight = 0;
	OpenSplat_PixelFormat	mFormat = OpenSplat_PixelFormat_Rgb;
	std::vector<uint8_t>	mPixels;
};

/*
 
	Class to self-contain a trainer with a simple API, with the purpose of moving towards
	a more instance based library.
 
*/
class Trainer
{
public:
	//	callbacks to events during refactor
	//	later iterations will be manually called and this class will become more pure/modularised
	Trainer(const TrainerParams& Params);
	~Trainer();

	//	this blocking call will be repalced with manually called init(), iterate()
	void				Run(std::function<void(TrainerIterationMeta)> OnIterationFinished,std::function<void(int)> OnRunFinished);

	torch::Device		GetDevice();
	Model&				GetModel()		
	{
		if ( !mModel )
			throw std::runtime_error("Model not initialised");
		return *mModel;	
	}
	InputData&			GetInputData()	
	{
		if ( !mInputData )
			throw std::runtime_error("Input data not setup");
		return *mInputData;	
	}
	
	void							LoadCameraImage(const OpenSplat_CameraMeta& CameraMeta,std::span<uint8_t> PixelBuffer,OpenSplat_PixelFormat PixelFormat);
	
	std::vector<OpenSplat_Splat>	GetModelSplats();
	ImagePixels						GetForwardImage(Camera& Camera,int Step);
	ImagePixels						GetForwardImage(int CameraIndex,int RenderWidth,int RenderHeight);
	ImagePixels						GetCameraImage(int CameraIndex,int OutputWidth,int OutputHeight);
	OpenSplat_TrainerState			GetState();
	OpenSplat_CameraMeta			GetCameraMeta(int CameraIndex);
	
	//	public when ready
protected:
	TrainerIterationMeta	Iteration(int step);
	
	int							GetIterationsForCamera(int CameraIndex);
	
public:
	TrainerParams				mParams;

	//	these should really have a shared_lock
	int							mIterationsCompleted = 0;	//	could calculate this from mCameraIterations
	std::map<int,int>			mCameraIterations;			//	[camera] = IterationsForCamera	
	int							mSplatCountCache = 0;		//	to allow non-model-locking access to splat count, cache the splat count after iterations
	
	std::recursive_mutex		mModelLock;
	std::shared_ptr<Model>		mModel;
	std::shared_ptr<InputData>	mInputData;
	bool						mRunning = true;		//	false when trying to free
	
	//	random index generator. default_random_engine is implementation defined, so different on different platforms
	//	if this wants to be deterministic... make a new one
	std::default_random_engine	mIterationRandomCameraIndex;
};
