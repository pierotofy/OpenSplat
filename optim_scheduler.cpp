#include "optim_scheduler.hpp"



OptimScheduler::OptimScheduler(std::shared_ptr<torch::optim::Adam> optimiser, float learningRateFinal, int maxSteps) :
	optimiser(optimiser), 
	learningRateInitial( static_cast<torch::optim::AdamOptions&>(optimiser->param_groups()[0].options()).get_lr() ), 
	learningRateFinal(learningRateFinal), 
	maxSteps(maxSteps) 
{
	if ( maxSteps <= 0 )
	{
		std::stringstream Error;
		Error << "Invalid maxSteps(" << maxSteps << ") for OptimScheduler. Expected > 0";
		throw std::runtime_error(Error.str());
	}
}


float OptimScheduler::getLearningRate(int step)
{
	float t = static_cast<float>(step) / static_cast<float>(maxSteps);
	t = std::clamp( t, 0.f, 1.f );
    return std::exp(std::log(learningRateInitial) * (1.0f - t) + std::log(learningRateFinal) * t);
}

void OptimScheduler::step(int step)
{
	float lr = getLearningRate(step);
	auto& options = static_cast<torch::optim::AdamOptions&>(optimiser->param_groups()[0].options());
	options.set_lr(lr);
}
