#pragma once

#include <string>

namespace cxxopts
{
	class ParseResult;
}

class TrainerParams
{
public:
	TrainerParams(cxxopts::ParseResult& Arguments);
	
	//	todo: assign sensible defaults in initialisation here
	std::string projectRoot;
	std::string outputScene;
	int saveEvery; 
	std::string resume;
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
};
