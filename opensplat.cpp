#include <filesystem>
#include <json.hpp>
#include "opensplat.hpp"
#include "point_io.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"
#include "vendor/cxxopts.hpp"

namespace fs = std::filesystem;
using namespace torch::indexing;

int main(int argc, char *argv[]){
    cxxopts::Options options("opensplat", "Open Source 3D Gaussian Splats generator");
    options.add_options()
        ("i,input", "Path to nerfstudio project", cxxopts::value<std::string>())
        ("n,num-iters", "Number of iterations to run", cxxopts::value<int>()->default_value("30000"))
        // ("r,resolution", "Resolution of the first scale (-1 = estimate automatically)", cxxopts::value<double>()->default_value("-1"))
        // ("s,scales", "Number of scales to compute", cxxopts::value<int>()->default_value(MKSTR(NUM_SCALES)))
        // ("t,trees", "Number of trees in the forest", cxxopts::value<int>()->default_value(MKSTR(N_TREES)))
        // ("d,depth", "Maximum depth of trees", cxxopts::value<int>()->default_value(MKSTR(MAX_DEPTH)))
        // ("m,max-samples", "Approximate maximum number of samples for each input point cloud", cxxopts::value<int>()->default_value("100000"))
        // ("radius", "Radius size to use for neighbor search (meters)", cxxopts::value<double>()->default_value(MKSTR(RADIUS)))
        // ("e,eval", "Labeled point cloud to use for model accuracy evaluation", cxxopts::value<std::string>()->default_value(""))
        // ("eval-result", "Path where to store evaluation results (PLY)", cxxopts::value<std::string>()->default_value(""))
        // ("stats", "Path where to store evaluation statistics (JSON)", cxxopts::value<std::string>()->default_value(""))
        // ("c,classifier", "Which classifier type to use (rf = Random Forest, gbt = Gradient Boosted Trees)", cxxopts::value<std::string>()->default_value("rf"))
        // ("classes", "Train only these classification classes (comma separated IDs)", cxxopts::value<std::vector<int>>())
        ("h,help", "Print usage")
        ;
    options.parse_positional({ "input" });
    options.positional_help("[labeled point cloud(s)]");
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help") || !result.count("input")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    std::string projectRoot = result["input"].as<std::string>();
    const float downScaleFactor = 2.0f;
    const int numIters = result["num-iters"].as<int>();
    const int numDownscales = 3;
    const int resolutionSchedule = 250;
    const int shDegree = 3;
    const int shDegreeInterval = 1000;
    const float ssimLambda = 0.2f;
    const int refineEvery = 100;
    const int warmupLength = 500;

    const int resetAlphaEvery = 30;
    const int stopSplitAt = 15000;
    const float densifyGradThresh = 0.0002f;
    const float densifySizeThresh = 0.01f;
    const int stopScreenSizeAt = 4000;
    const float splitScreenSize = 0.05f;

    torch::Device device = torch::kCPU;

    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA" << std::endl;
        device = torch::kCUDA;
    }else{
        std::cout << "Using CPU" << std::endl;
    }

    try{

        ns::InputData inputData = ns::inputDataFromNerfStudio(projectRoot);
        
        ns::Model model(inputData.points, 
                        inputData.cameras.size(),
                        numDownscales, resolutionSchedule, shDegree, shDegreeInterval, 
                        refineEvery, warmupLength, resetAlphaEvery, stopSplitAt, densifyGradThresh, densifySizeThresh, stopScreenSizeAt, splitScreenSize,
                        device);

        // TODO: uncomment
        for (ns::Camera &cam : inputData.cameras){
            cam.loadImage(downScaleFactor);
        }

        InfiniteRandomIterator<ns::Camera> cams(inputData.cameras);

        for (size_t step = 0; step < numIters; step++){
            ns::Camera cam = cams.next();

            // TODO: remove
            // ns::Camera cam = inputData.cameras[6];
            
            // TODO: remove
            // cam.loadImage(downScaleFactor);

            model.optimizersZeroGrad();

            torch::Tensor rgb = model.forward(cam, step);
            torch::Tensor gt = cam.getImage(model.getDownscaleFactor(step));
            gt = gt.to(device);

            torch::Tensor ssimLoss = 1.0f - model.ssim.eval(rgb, gt);
            torch::Tensor l1Loss = ns::l1(rgb, gt);
            torch::Tensor mainLoss = (1.0f - ssimLambda) * l1Loss + ssimLambda * ssimLoss;
            mainLoss.backward();

            model.optimizersStep();
            //model.optimizersScheduleStep(); // TODO
            model.afterTrain(step);

            if (step % 10 == 0) std::cout << "Step " << step << ": " << mainLoss.item<float>() << std::endl;
        }

        model.savePlySplat("splat.ply");
    }catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}