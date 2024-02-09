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
    torch::Device device = torch::kCPU;

    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA" << std::endl;
        device = torch::kCUDA;
    }else{
        std::cout << "Using CPU" << std::endl;
    }

    ns::InputData inputData = ns::inputDataFromNerfStudio(projectRoot);
    
    ns::Model m(inputData.points, device);
    m.to(device);

    std::vector<int> items = {1,2,3,4,5};
    InfiniteRandomIterator<int> iter(items);

    for (size_t i = 0; i < 100; i++){
        std::cout << iter.next() << std::endl;
    }
    // inputData.cameras[0].loadImage(downScaleFactor);  
    
}