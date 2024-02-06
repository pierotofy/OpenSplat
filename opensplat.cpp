#include <filesystem>
#include <json.hpp>
#include "opensplat.hpp"
#include "point_io.hpp"

namespace fs = std::filesystem;
using namespace torch::indexing;

int main(int argc, char *argv[]){
    std::string projectRoot = "banana";
    const float downScaleFactor = 2.0f;

    ns::InputData inputData = ns::inputDataFromNerfStudio(projectRoot);
    ns::rescaleOutputResolution(inputData.cameras, 1.0f / downScaleFactor);
    
    ns::Model m(inputData.points);

}