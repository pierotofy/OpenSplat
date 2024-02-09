#include <filesystem>
#include <json.hpp>
#include "opensplat.hpp"
#include "point_io.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;
using namespace torch::indexing;

int main(int argc, char *argv[]){
    std::string projectRoot = "banana";
    const float downScaleFactor = 2.0f;
    const int numIters = 1000;
    const int numDownscales = 3;
    const int resolutionSchedule = 250;

    torch::Device device = torch::kCPU;

    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA" << std::endl;
        device = torch::kCUDA;
    }else{
        std::cout << "Using CPU" << std::endl;
    }

    ns::InputData inputData = ns::inputDataFromNerfStudio(projectRoot);
    
    ns::Model model(inputData.points, numDownscales, resolutionSchedule, device);
    model.to(device);

    // TODO: uncomment
    // for (ns::Camera &cam : inputData.cameras){
    //     cam.loadImage(downScaleFactor);
    // }

    InfiniteRandomIterator<ns::Camera> cams(inputData.cameras);

    for (size_t step = 0; step < numIters; step++){
        // ns::Camera cam = cams.next();
        ns::Camera cam = inputData.cameras[6];
        
        // TODO: remove
        cam.loadImage(downScaleFactor);

        std::cout << model.forward(cam, step);

        exit(1);
    }
    // inputData.cameras[0].loadImage(downScaleFactor);  
    
}