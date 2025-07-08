#include <filesystem>
#include <nlohmann/json.hpp>
#include "opensplat.hpp"
#include "input_data.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"
#include "constants.hpp"
#include <cxxopts.hpp>
#include "trainer_params.hpp"
#include "trainer.hpp"

#ifdef USE_VISUALIZATION
#include "visualizer.hpp"
#endif



namespace fs = std::filesystem;
using namespace torch::indexing;

int main(int argc, char *argv[]){
    cxxopts::Options options("opensplat", "Open Source 3D Gaussian Splats generator - " APP_VERSION);
    options.add_options()
        ("i,input", "Path to nerfstudio project", cxxopts::value<std::string>())
        ("o,output", "Path where to save output scene", cxxopts::value<std::string>()->default_value("splat.ply"))
        ("s,save-every", "Save output scene every these many steps (set to -1 to disable)", cxxopts::value<int>()->default_value("-1"))
        ("resume", "Resume training from this PLY file", cxxopts::value<std::string>()->default_value(""))
        ("val", "Withhold a camera shot for validating the scene loss")
        ("val-image", "Filename of the image to withhold for validating scene loss", cxxopts::value<std::string>()->default_value("random"))
        ("val-render", "Path of the directory where to render validation images", cxxopts::value<std::string>()->default_value(""))
        ("keep-crs", "Retain the project input's coordinate reference system")
        ("cpu", "Force CPU execution")
        
        ("n,num-iters", "Number of iterations to run", cxxopts::value<int>()->default_value("30000"))
        ("d,downscale-factor", "Scale input images by this factor.", cxxopts::value<float>()->default_value("1"))
        ("num-downscales", "Number of images downscales to use. After being scaled by [downscale-factor], images are initially scaled by a further (2^[num-downscales]) and the scale is increased every [resolution-schedule]", cxxopts::value<int>()->default_value("2"))
        ("resolution-schedule", "Double the image resolution every these many steps", cxxopts::value<int>()->default_value("3000"))
        ("sh-degree", "Maximum spherical harmonics degree (must be > 0)", cxxopts::value<int>()->default_value("3"))
        ("sh-degree-interval", "Increase the number of spherical harmonics degree after these many steps (will not exceed [sh-degree])", cxxopts::value<int>()->default_value("1000"))
        ("ssim-weight", "Weight to apply to the structural similarity loss. Set to zero to use least absolute deviation (L1) loss only", cxxopts::value<float>()->default_value("0.2"))
        ("refine-every", "Split/duplicate/prune gaussians every these many steps", cxxopts::value<int>()->default_value("100"))
        ("warmup-length", "Split/duplicate/prune gaussians only after these many steps", cxxopts::value<int>()->default_value("500"))
        ("reset-alpha-every", "Reset the opacity values of gaussians after these many refinements (not steps)", cxxopts::value<int>()->default_value("30"))
        ("densify-grad-thresh", "Threshold of the positional gradient norm (magnitude of the loss function) which when exceeded leads to a gaussian split/duplication", cxxopts::value<float>()->default_value("0.0002"))
        ("densify-size-thresh", "Gaussians' scales below this threshold are duplicated, otherwise split", cxxopts::value<float>()->default_value("0.01"))
        ("stop-screen-size-at", "Stop splitting gaussians that are larger than [split-screen-size] after these many steps", cxxopts::value<int>()->default_value("4000"))
        ("split-screen-size", "Split gaussians that are larger than this percentage of screen space", cxxopts::value<float>()->default_value("0.05"))
        ("colmap-image-path", "Override the default image path for COLMAP-based input", cxxopts::value<std::string>()->default_value(""))

        ("h,help", "Print usage")
        ("version", "Print version")
        ;
    options.parse_positional({ "input" });
    options.positional_help("[colmap/nerfstudio/opensfm/odm/openmvg project path]");
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("version")){
        std::cout << APP_VERSION << std::endl;
        return EXIT_SUCCESS;
    }
    if (result.count("help") || !result.count("input")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }


	TrainerParams Params(result);
	
	//	temp during refactor
	auto& projectRoot = Params.projectRoot;
	auto& validate = Params.validate;
	auto& valImage = Params.valImage;
	auto& valRender = Params.valRender;
	auto& keepCrs = Params.keepCrs;
	auto& downScaleFactor = Params.downScaleFactor;
	auto& numIters = Params.numIters;
	auto& numDownscales = Params.numDownscales;
	auto& resolutionSchedule = Params.resolutionSchedule;
	auto& shDegree = Params.shDegree;
	auto& shDegreeInterval = Params.shDegreeInterval;
	auto& ssimWeight = Params.ssimWeight;
	auto& refineEvery = Params.refineEvery;
	auto& warmupLength = Params.warmupLength;
	auto& resetAlphaEvery = Params.resetAlphaEvery;
	auto& densifyGradThresh = Params.densifyGradThresh;
	auto& densifySizeThresh = Params.densifySizeThresh;
	auto& stopScreenSizeAt = Params.stopScreenSizeAt;
	auto& splitScreenSize = Params.splitScreenSize;
	auto& colmapImageSourcePath = Params.colmapImageSourcePath;

	
	if (!Params.valRender.empty() && !fs::exists(Params.valRender)) 
		fs::create_directories(Params.valRender);
	
#ifdef USE_VISUALIZATION
    Visualizer visualizer;
    visualizer.Initialize(numIters);
#endif

	//	if the output path is a directory, and not a filename, we'll get confusing file-write errors
	if ( std::filesystem::is_directory(Params.outputScene) )
	{
		std::stringstream Error;
		Error << "OutputScene argument " << Params.outputScene << " is a directory, expecting a filename (.ply or .splat)";
		throw std::runtime_error(Error.str());
	}
	
	try
	{
		Trainer trainer(Params);
		
		auto OnIterationFinished = [&](int step,float Loss,Model& model,Camera* ValidationCamera)
		{
			//	old code made this every step if using CPU
			if (Params.printDebugEvery > 0 && step % Params.printDebugEvery == 0)
			{
				const float percentage = static_cast<float>(step) / numIters;
				std::cout << "Step " << step << ": " << Loss << " (" << floor(percentage * 100) << "%)" <<  std::endl;
			}			
			
			if (Params.saveModelEvery > 0 && step % Params.saveModelEvery == 0)
			{
				auto Suffix = std::string("_") + std::to_string(step);
				auto OutputFilename = Params.GetOutputModelFilenameWithSuffix(Suffix);
				model.save( OutputFilename, step );
			}
			
			
			if ( ValidationCamera && !valRender.empty() && Params.saveValidationRenderEvery > 0 && step % Params.saveValidationRenderEvery == 0)
			{
				torch::Tensor rgb = model.forward(*ValidationCamera, step);
				cv::Mat image = tensorToImage(rgb.detach().cpu());
				cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
				cv::imwrite((fs::path(valRender) / (std::to_string(step) + ".png")).string(), image);
			}
			
			
#ifdef USE_VISUALIZATION
			visualizer.SetInitialGaussianNum(inputData.points.xyz.size(0));
			visualizer.SetLoss(step, mainLoss.item<float>());
			visualizer.SetGaussians(model.means, model.scales, model.featuresDc,
									model.opacities);
			visualizer.SetImage(rgb, gt);
			visualizer.Draw();
#endif
		};
		
		auto OnRunFinished = [&](int numIters,Model& model,InputData& inputData,Camera* ValidationCamera,torch::Device& device)
		{
			auto CamerasJsonFilename = Params.GetOutputFilePath("cameras.json");
			auto ModelFilename = Params.GetOutputModelFilename();
			inputData.saveCameras( CamerasJsonFilename.string(), keepCrs);
			model.save(ModelFilename, numIters);
			// model.saveDebugPly("debug.ply", numIters);
			
			// Validate
			if (ValidationCamera != nullptr)
			{
				auto& valCam = *ValidationCamera;
				torch::Tensor rgb = model.forward(valCam, numIters);
				torch::Tensor gt = valCam.getImage(model.getDownscaleFactor(numIters)).to(device);
				auto FinalLoss = model.mainLoss(rgb, gt, ssimWeight).item<float>();
				std::cout << valCam.filePath << " validation loss: " << FinalLoss << std::endl; 
			}
		};
		
		trainer.Run( OnIterationFinished, OnRunFinished );
        
    }catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}
