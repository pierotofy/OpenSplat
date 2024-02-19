#ifndef OPTIM_SCHEDULER
#define OPTIM_SCHEDULER

#include <iostream>
#include <torch/torch.h>

class OptimScheduler{
public:
    OptimScheduler(torch::optim::Adam *opt, float lrFinal, int maxSteps) :
        opt(opt), lrInit(
            static_cast<torch::optim::AdamOptions&>(opt->param_groups()[0].options()).get_lr()
        ), lrFinal(lrFinal), maxSteps(maxSteps) {};
    void step(int step);
    float getLearningRate(int step);

private:
    torch::optim::Adam *opt;
    float lrInit;
    float lrFinal;
    int maxSteps;
};

#endif