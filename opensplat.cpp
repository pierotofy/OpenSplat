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
#include "ply.hpp"

#ifdef USE_VISUALIZATION
#include "visualizer.hpp"
#endif

#include "trainer_api.h"
//	temp exposure from api
namespace OpenSplat
{
	Trainer&	GetInstance(int Instance);
}


template <typename T>
std::basic_string<T> lowercase(const std::basic_string<T>& s)
{
	std::basic_string<T> s2 = s;
	std::transform(s2.begin(), s2.end(), s2.begin(),
				   [](const T v){ return static_cast<T>(std::tolower(v)); });
	return s2;
}


void WriteTrainerPly(std::filesystem::path Filename,int TrainerInstance,int Step)
{
	std::vector<OpenSplat_Splat> Splats( 1000 );
	auto GetSplat = [&](int Index)
	{
		SplatElement Element;
		Element.Splat = Splats[Index];
		return Element;
	};
	
	auto SplatCount = OpenSplat_GetSnapshot(TrainerInstance, Splats.data(), Splats.size() );
	SplatCount = std::min<int>( SplatCount, Splats.size() );
	
	std::ofstream File(Filename,std::ios::binary);
	Ply::WriteParams Params;
	Params.PointCount = SplatCount;
	Params.Comment = std::string("OpenSplat at step ") + std::to_string(Step);
	
	Ply::Write( File, Params, GetSplat );
	File.close();
}


namespace fs = std::filesystem;
using namespace torch::indexing;

int main(int argc, char *argv[])
{
    cxxopts::Options options("opensplat", "Open Source 3D Gaussian Splats generator - " APP_VERSION);
	{
		TrainerParams DefaultTrainerParams;
		AppParams DefaultAppParams;
		options.add_options()
		("i,input", "Path to nerfstudio project", cxxopts::value<std::string>())
		("o,output", "Path where to save output scene", cxxopts::value<std::string>()->default_value(DefaultAppParams.outputScene))
		("s,save-every", "Save output scene every these many steps (set to -1 to disable)", cxxopts::value<int>()->default_value(std::to_string(DefaultAppParams.saveModelEvery)))
		("resume", "Resume training from this PLY file", cxxopts::value<std::string>()->default_value(""))
		("val", "Withhold a camera shot for validating the scene loss")
		("val-image", "Filename of the image to withhold for validating scene loss", cxxopts::value<std::string>()->default_value(DefaultTrainerParams.valImage))
		("val-render", "Path of the directory where to render validation images", cxxopts::value<std::string>()->default_value(DefaultAppParams.valRender))
		("keep-crs", "Retain the project input's coordinate reference system")
		("cpu", "Force CPU execution")
		
		("n,num-iters", "Number of iterations to run", cxxopts::value<int>()->default_value(std::to_string(DefaultTrainerParams.numIters)))
		("d,downscale-factor", "Scale input images by this factor.", cxxopts::value<float>()->default_value(std::to_string(DefaultAppParams.downScaleFactor)))
		("num-downscales", "Number of images downscales to use. After being scaled by [downscale-factor], images are initially scaled by a further (2^[num-downscales]) and the scale is increased every [resolution-schedule]", cxxopts::value<int>()->default_value(std::to_string(DefaultTrainerParams.numDownscales)))
		("resolution-schedule", "Double the image resolution every these many steps", cxxopts::value<int>()->default_value(std::to_string(DefaultTrainerParams.resolutionSchedule)))
		("sh-degree", "Maximum spherical harmonics degree (must be > 0)", cxxopts::value<int>()->default_value(std::to_string(DefaultTrainerParams.shDegree)))
		("sh-degree-interval", "Increase the number of spherical harmonics degree after these many steps (will not exceed [sh-degree])", cxxopts::value<int>()->default_value(std::to_string(DefaultTrainerParams.shDegreeInterval)))
		("ssim-weight", "Weight to apply to the structural similarity loss. Set to zero to use least absolute deviation (L1) loss only", cxxopts::value<float>()->default_value(std::to_string(DefaultTrainerParams.ssimWeight)))
		("refine-every", "Split/duplicate/prune gaussians every these many steps", cxxopts::value<int>()->default_value(std::to_string(DefaultTrainerParams.refineEvery)))
		("warmup-length", "Split/duplicate/prune gaussians only after these many steps", cxxopts::value<int>()->default_value(std::to_string(DefaultTrainerParams.warmupLength)))
		("reset-alpha-every", "Reset the opacity values of gaussians after these many refinements (not steps)", cxxopts::value<int>()->default_value(std::to_string(DefaultTrainerParams.resetAlphaEvery)))
		("densify-grad-thresh", "Threshold of the positional gradient norm (magnitude of the loss function) which when exceeded leads to a gaussian split/duplication", cxxopts::value<float>()->default_value(std::to_string(DefaultTrainerParams.densifyGradThresh)))
		("densify-size-thresh", "Gaussians' scales below this threshold are duplicated, otherwise split", cxxopts::value<float>()->default_value(std::to_string(DefaultTrainerParams.densifySizeThresh)))
		("stop-screen-size-at", "Stop splitting gaussians that are larger than [split-screen-size] after these many steps", cxxopts::value<int>()->default_value(std::to_string(DefaultTrainerParams.stopScreenSizeAt)))
		("split-screen-size", "Split gaussians that are larger than this percentage of screen space", cxxopts::value<float>()->default_value(std::to_string(DefaultTrainerParams.splitScreenSize)))
		("colmap-image-path", "Override the default image path for COLMAP-based input", cxxopts::value<std::string>()->default_value(DefaultAppParams.colmapImageSourcePath))
		
		("h,help", "Print usage")
		("version", "Print version")
		;
	}
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


	AppParams AppParams(result);
	TrainerParams TrainerParams(result,AppParams.keepCrs);
	
	//	temp during refactor
	auto& keepCrs = AppParams.keepCrs;
	auto& numIters = TrainerParams.numIters;
	auto& ssimWeight = TrainerParams.ssimWeight;

	
	if (!AppParams.valRender.empty() && !fs::exists(AppParams.valRender)) 
		fs::create_directories(AppParams.valRender);
	
#ifdef USE_VISUALIZATION
    Visualizer visualizer;
    visualizer.Initialize(numIters);
#endif

	//	if the output path is a directory, and not a filename, we'll get confusing file-write errors
	if ( std::filesystem::is_directory(AppParams.outputScene) )
	{
		std::stringstream Error;
		Error << "OutputScene argument " << AppParams.outputScene << " is a directory, expecting a filename (.ply or .splat)";
		throw std::runtime_error(Error.str());
	}
	
	try
	{
		auto TrainerInstance = OpenSplat_AllocateInstanceFromPath( AppParams.projectRoot.c_str() );
		auto& trainer = OpenSplat::GetInstance( TrainerInstance );
		trainer.mParams = TrainerParams;
		
		auto OnIterationFinished = [&](TrainerIterationMeta IterationMeta,Camera* ValidationCamera)
		{
			auto step = IterationMeta.mStep;
			auto& model = trainer.GetModel();
			
			//	old code made this every step if using CPU
			if (AppParams.printDebugEvery > 0 && step % AppParams.printDebugEvery == 0)
			{
				const float percentage = static_cast<float>(step) / numIters;
				std::cout << "Step " << step << ": " << IterationMeta.mLoss << " (" << floor(percentage * 100) << "%)" <<  std::endl;
			}			
			
			if (AppParams.saveModelEvery > 0 && step % AppParams.saveModelEvery == 0)
			{
				auto Suffix = std::string("_") + std::to_string(step);
				auto OutputFilename = AppParams.GetOutputModelFilenameWithSuffix(Suffix);

				//	use api and explicitly write ply
				auto ext = lowercase( OutputFilename.extension().string() );
				if ( ext == ".ply" )
				{
					WriteTrainerPly( OutputFilename, TrainerInstance, step );
				}
				else
				{
					model.save( OutputFilename, step, AppParams.keepCrs );
				}
			}
			
			
			if ( ValidationCamera && !AppParams.valRender.empty() && AppParams.saveValidationRenderEvery > 0 && step % AppParams.saveValidationRenderEvery == 0)
			{
				//	render some-image from some-camera using api (not using step, so not exact as training!)
				if ( false )
				{
					cv::Mat image( 1200, 2100, CV_8UC3 );
					int CameraIndex = 0;
					auto rgb = std::span( image.data, image.total() * image.elemSize() );
					std::for_each( rgb.begin(), rgb.end(), [](uint8_t& p){	p=255;	} );
					auto Error = OpenSplat_RenderCamera( TrainerInstance, CameraIndex, rgb.data(), rgb.size_bytes(), image.cols, image.rows );
					//	opencv expects BGR
					cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
					cv::imwrite((fs::path(AppParams.valRender) / (std::to_string(step) + "_ApiRender.png")).string(), image);
				}
			
				{
					//	render exact step-image
					auto image = trainer.GetForwardImage( *ValidationCamera, step );
					auto WriteImage = [&](cv::Mat& image)
					{
					//	tensor image is RGB, opencv expects BGR
					cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
						cv::imwrite((fs::path(AppParams.valRender) / (std::to_string(step) + ".png")).string(), image);
					};
					image.GetCvImage(WriteImage);
				}
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
		
		auto OnRunFinished = [&](int numIters,Camera* ValidationCamera)
		{
			auto& model = trainer.GetModel();
			auto& inputData = trainer.GetInputData();
			
			auto CamerasJsonFilename = AppParams.GetOutputFilePath("cameras.json");
			auto ModelFilename = AppParams.GetOutputModelFilename();
			inputData.saveCameras( CamerasJsonFilename.string(), keepCrs);
			model.save(ModelFilename, numIters, AppParams.keepCrs);
			// model.saveDebugPly("debug.ply", numIters);
			
			// Validate
			if (ValidationCamera != nullptr)
			{
				auto device = trainer.GetDevice();
				auto& valCam = *ValidationCamera;
				auto ForwardResults = model.forward(valCam, numIters);
				torch::Tensor gt = valCam.getImage(model.getDownscaleFactor(numIters)).to(device);
				auto FinalLoss = model.mainLoss(ForwardResults.rgb, gt, ssimWeight).item<float>();
				std::cout << "Camera " << valCam.getName() << " validation loss: " << FinalLoss << std::endl; 
			}
		};
		
		trainer.Run( OnIterationFinished, OnRunFinished );
		
		OpenSplat_FreeInstance( TrainerInstance );
		
		
		return EXIT_SUCCESS;
        
    }catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
    }
}
