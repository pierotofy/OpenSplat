#include "trainer_params.hpp"
#include "cxxopts.hpp"


TrainerParams::TrainerParams(cxxopts::ParseResult& Arguments)
{
	projectRoot = Arguments["input"].as<std::string>();
	outputScene = Arguments["output"].as<std::string>();
	saveModelEvery = Arguments["save-every"].as<int>(); 
	resumeFromPlyFilename = Arguments["resume"].as<std::string>();
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
	
	//	user didn't supply any force-cpu argument[s]
	mForceCpuDevice = Arguments.count("cpu") > 0;
}

std::filesystem::path TrainerParams::GetOutputFilePath(const std::string& Filename)
{
	std::filesystem::path p( outputScene );
	auto CamerasJsonFilename = p.parent_path() / "cameras.json";

	return CamerasJsonFilename;
}

std::filesystem::path TrainerParams::GetOutputModelFilename()
{
	std::filesystem::path p( outputScene );
	return p;
}

std::filesystem::path TrainerParams::GetOutputModelFilenameWithSuffix(const std::string& Suffix)
{
	std::filesystem::path p( outputScene );
	auto ModelFilename = p.stem().string() + Suffix + p.extension().string();
	std::filesystem::path ModelFilePath( ModelFilename );
	auto OutputFilename = p.replace_filename( ModelFilePath.string() );
	return OutputFilename;
}

