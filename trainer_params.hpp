#pragma once

#include <string>
#include <filesystem>

namespace cxxopts
{
	class ParseResult;
}

class TrainerParams
{
public:
	TrainerParams(){};
	TrainerParams(cxxopts::ParseResult& Arguments);
	
	//	these are for the app, rather than training
	//	refactor to split this distinction
	std::filesystem::path	GetOutputFilePath(const std::string& Filename);
	std::filesystem::path	GetOutputModelFilename();
	std::filesystem::path	GetOutputModelFilenameWithSuffix(const std::string& Suffix);
	
	std::string outputScene = "splat.ply";
	int saveModelEvery = -1;
	int saveValidationRenderEvery = 10;
	int printDebugEvery = 10;
	
	//	todo: assign sensible defaults in initialisation here
	std::string projectRoot;
	std::string resumeFromPlyFilename;
	bool validate;
	std::string valImage = "random";
	std::string valRender = "";
	bool keepCrs;
	float downScaleFactor = 1;
	int numIters = 30000;
	int numDownscales = 2;
	int resolutionSchedule = 3000;
	int shDegree = 3;
	int shDegreeInterval = 1000;
	float ssimWeight = 0.2;
	int refineEvery = 100;
	int warmupLength = 500;
	int resetAlphaEvery = 30;
	float densifyGradThresh = 0.0002;
	float densifySizeThresh = 0.01;
	int stopScreenSizeAt = 4000;
	float splitScreenSize = 0.05;
	std::string colmapImageSourcePath = "";
	
	//	refactored params
	bool		mForceCpuDevice = false;
};
