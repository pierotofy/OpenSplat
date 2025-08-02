#ifndef OPTIM_SCHEDULER
#define OPTIM_SCHEDULER

#include <iostream>
#include <torch/torch.h>

class OptimScheduler
{
public:
	OptimScheduler(std::shared_ptr<torch::optim::Adam> optimiser, float learningRateFinal, int maxSteps);
	void	step(int step);
	float	getLearningRate(int step);
	
private:
	std::shared_ptr<torch::optim::Adam> optimiser;
	float	learningRateInitial;
	float	learningRateFinal;
	int		maxSteps;
};

#endif
