#include "trainer_params.hpp"
#include "cxxopts.hpp"


TrainerParams::TrainerParams(cxxopts::ParseResult& Arguments)
{
	projectRoot = Arguments["input"].as<std::string>();
	outputScene = Arguments["output"].as<std::string>();
	saveEvery = Arguments["save-every"].as<int>(); 
	resume = Arguments["resume"].as<std::string>();
	validate = Arguments.count("val") > 0 || Arguments.count("val-render") > 0;
	valImage = Arguments["val-image"].as<std::string>();
	valRender = Arguments["val-render"].as<std::string>();
	keepCrs = Arguments.count("keep-crs") > 0;
	downScaleFactor = (std::max)(Arguments["downscale-factor"].as<float>(), 1.0f);
	numIters = Arguments["num-iters"].as<int>();
	numDownscales = Arguments["num-downscales"].as<int>();
	resolutionSchedule = Arguments["resolution-schedule"].as<int>();
	shDegree = Arguments["sh-degree"].as<int>();
	shDegreeInterval = Arguments["sh-degree-interval"].as<int>();
	ssimWeight = Arguments["ssim-weight"].as<float>();
	refineEvery = Arguments["refine-every"].as<int>();
	warmupLength = Arguments["warmup-length"].as<int>();
	resetAlphaEvery = Arguments["reset-alpha-every"].as<int>();
	densifyGradThresh = Arguments["densify-grad-thresh"].as<float>();
	densifySizeThresh = Arguments["densify-size-thresh"].as<float>();
	stopScreenSizeAt = Arguments["stop-screen-size-at"].as<int>();
	splitScreenSize = Arguments["split-screen-size"].as<float>();
	colmapImageSourcePath = Arguments["colmap-image-path"].as<std::string>();
}
