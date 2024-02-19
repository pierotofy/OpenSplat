#include "optim_scheduler.hpp"


float OptimScheduler::getLearningRate(int step){
    float t = (std::max)((std::min)(static_cast<float>(step) / static_cast<float>(maxSteps), 1.0f), 0.0f);
    return std::exp(std::log(lrInit) * (1.0f - t) + std::log(lrFinal) * t);
}

void OptimScheduler::step(int step){
    float lr = getLearningRate(step);
    static_cast<torch::optim::AdamOptions&>(opt->param_groups()[0].options()).set_lr(lr);
}