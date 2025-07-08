#pragma once

#include <string_view>
#include <functional>
#include "trainer_params.hpp"

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
	void		Run(std::function<void(int,float,Model&,Camera*)> OnIterationFinished,std::function<void(int,Model&,InputData&,Camera*,torch::Device&)> OnRunFinished);
	
	Model&		GetModel()	{	return *mModel;	}
	
private:
	//std::function<void(int)>	mOnIterationFinished;
	TrainerParams		mParams;
	std::shared_ptr<Model>	mModel;
	InputData			mInputData;
};
