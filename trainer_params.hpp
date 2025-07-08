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
	TrainerParams(cxxopts::ParseResult& Arguments);
	
	//	these are for the app, rather than training
	//	refactor to split this distinction
	std::filesystem::path	GetOutputFilePath(const std::string& Filename);
	std::filesystem::path	GetOutputModelFilename();
	std::filesystem::path	GetOutputModelFilenameWithSuffix(const std::string& Suffix);
	
	std::string outputScene;
	int saveModelEvery;
	int saveValidationRenderEvery = 10;
	
	//	todo: assign sensible defaults in initialisation here
	std::string projectRoot;
	std::string resumeFromPlyFilename;
	bool validate;
	std::string valImage;
	std::string valRender;
	bool keepCrs;
	float downScaleFactor;
	int numIters;
	int numDownscales;
	int resolutionSchedule;
	int shDegree;
	int shDegreeInterval;
	float ssimWeight;
	int refineEvery;
	int warmupLength;
	int resetAlphaEvery;
	float densifyGradThresh;
	float densifySizeThresh;
	int stopScreenSizeAt;
	float splitScreenSize;
	std::string colmapImageSourcePath;
	
	//	refactored params
	bool		mForceCpuDevice = false;
};
