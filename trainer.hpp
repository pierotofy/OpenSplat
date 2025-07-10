#pragma once

#include <string_view>
#include <functional>
#include "trainer_params.hpp"
#include <random>

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
};

	

class ImagePixels
{
public:
	enum Format
	{
		Rgb = 0
	};
public:
	ImagePixels(const torch::Tensor& tensor);
	
	//	callback so we can use pixels in place - mat is only valid for lifetime of callback
	void					GetCvImage(std::function<void(cv::Mat&)> OnImage);
	
	int						mWidth = 0;
	int						mHeight = 0;
	Format					mFormat = Rgb;
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

	//	this blocking call will be repalced with manually called init(), iterate()
	void				Run(std::function<void(TrainerIterationMeta,Camera*)> OnIterationFinished,std::function<void(int,Camera*)> OnRunFinished);

	torch::Device		GetDevice();
	Model&				GetModel()		{	return *mModel;	}
	InputData&			GetInputData()	
	{
		if ( !mInputData )
			throw std::runtime_error("Input data not setup");
		return *mInputData;	
	}
	
	ImagePixels			GetForwardImage(Camera& Camera,int Step);

	//	public when ready
protected:
	TrainerIterationMeta	Iteration(int step); 
	
public:
	TrainerParams				mParams;
	std::shared_ptr<Model>		mModel;
	std::shared_ptr<InputData>	mInputData;
	
	//	random index generator. default_random_engine is implementation defined, so different on different platforms
	//	if this wants to be deterministic... make a new one
	std::default_random_engine	mIterationRandomCameraIndex;
};
