#pragma once

#include <string>
#include <filesystem>



class ModelParams
{
public:
	int numDownscales = 2;
	int resolutionSchedule = 3000;
	int shDegree = 3;
	int shDegreeInterval = 1000;
	int refineEvery = 100;
	int warmupLength = 500;
	int resetAlphaEvery = 30;
	float densifyGradThresh = 0.0002;
	float densifySizeThresh = 0.01;
	int stopScreenSizeAt = 4000;
	float splitScreenSize = 0.05;
};


namespace OpenSplat
{
	static constexpr std::string_view	randomValidationImageName = "random";
}

namespace cxxopts
{
	class ParseResult;
}

class AppParams
{
public:
	AppParams(){};
	AppParams(cxxopts::ParseResult& Arguments);
	
	//	these are for the app, rather than training
	//	refactor to split this distinction
	std::filesystem::path	GetOutputFilePath(const std::string& Filename);
	std::filesystem::path	GetOutputModelFilename();
	std::filesystem::path	GetOutputModelFilenameWithSuffix(const std::string& Suffix);
	
	//	output
	std::string valImage = std::string(OpenSplat::randomValidationImageName);
	bool validate = false;			//	this is to ex
	int saveValidationRenderEvery = 10;
	std::string valRender = "";		//	path to render validation camera to (rename this!)
	std::string outputScene = "splat.ply";
	int saveModelEvery = -1;
	int printDebugEvery = 10;
	bool keepCrs = false;	//	output in original position & scale

	//	input
	std::string projectRoot;
	float downScaleFactor = 1;	//	initial camera image downscaling
	std::string colmapImageSourcePath = "";
};


class TrainerParams : public ModelParams
{

public:
	TrainerParams(){};
	//	keepCrs is temporarily here
	TrainerParams(cxxopts::ParseResult& Arguments,bool KeepCrs);
	
	int numIters = 30000;
	float ssimWeight = 0.2;
	int iterationRandomCameraIndexSeed = 42;

	//	todo: move into app and load resuming points into InputData
	std::string resumeFromPlyFilename;
	bool resumeFromPlyNeedsNormalising = false;	//	keepCrs

	std::array<float,3> BackgroundRgb = {0.6130f, 0.0101f, 0.3984f};	// Nerf Studio default
	
	
	//	refactored params
	bool		mForceCpuDevice = false;
};
