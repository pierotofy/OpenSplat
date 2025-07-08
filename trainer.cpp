#include "trainer.hpp"

#include <filesystem>
#include <nlohmann/json.hpp>
#include "opensplat.hpp"
#include "input_data.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"
#include "constants.hpp"
#include "trainer_params.hpp"


Trainer::Trainer(const TrainerParams& Params) :
	mParams		( Params )
{
	
}

void Trainer::Run(InputData& inputData,std::function<void(int,float,Model&,Camera*)> OnIterationFinished,std::function<void(int,Model&,InputData&,Camera*,torch::Device&)> OnRunFinished)
{
	//	remove any use of the filesystem from this func/class
	namespace fs = std::filesystem;
	
	//	temp during refactor
	auto& Params = mParams;
	auto& resume = Params.resumeFromPlyFilename;
	auto& validate = Params.validate;
	auto& valImage = Params.valImage;
	auto& keepCrs = Params.keepCrs;
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
	auto ForceCpuDevice = Params.mForceCpuDevice;
	
	torch::Device device = torch::kCPU;


	if (torch::hasCUDA() && !ForceCpuDevice ) {
		std::cout << "Using CUDA" << std::endl;
		device = torch::kCUDA;
	} else if (torch::hasMPS() && !ForceCpuDevice) {
		std::cout << "Using MPS" << std::endl;
		device = torch::kMPS;
	}else{
		std::cout << "Using CPU" << std::endl;
	}

	
	// Withhold a validation camera if necessary
	auto t = inputData.getCameras(validate, valImage);
	std::vector<Camera> cams = std::get<0>(t);
	Camera *valCam = std::get<1>(t);
	
	mModel = std::make_shared<Model>(inputData,
				cams.size(),
				numDownscales, resolutionSchedule, shDegree, shDegreeInterval, 
				refineEvery, warmupLength, resetAlphaEvery, densifyGradThresh, densifySizeThresh, stopScreenSizeAt, splitScreenSize,
				numIters, keepCrs,
				device);
	auto& model = *mModel;
	
	std::vector<size_t> camIndices( cams.size() );
	std::iota( camIndices.begin(), camIndices.end(), 0 );
	InfiniteRandomIterator<size_t> camsIter( camIndices );
	
	size_t step = 1;
	
	if (!resume.empty())
	{
		step = model.loadPly(resume) + 1;
	}
	
	for (; step <= numIters; step++){
		Camera& cam = cams[ camsIter.next() ];
		
		model.optimizersZeroGrad();
		
		//	rgb is a render of the scene
		torch::Tensor rgb = model.forward(cam, step);
		torch::Tensor groundTruth = cam.getImage(model.getDownscaleFactor(step));
		groundTruth = groundTruth.to(device);
		
		//	calculate loss from render to ground truth
		torch::Tensor mainLoss = model.mainLoss(rgb, groundTruth, ssimWeight);
		mainLoss.backward();
		auto mainLossValue = mainLoss.item<float>();
		
		
		model.optimizersStep();
		model.schedulersStep(step);
		model.afterTrain(step);
		
		
		OnIterationFinished( step, mainLossValue, model, valCam );
	}
	
	auto CompletedSteps = step;	//	num_iters
	OnRunFinished( CompletedSteps, model, inputData, valCam, device );
}
