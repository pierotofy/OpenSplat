#include "trainer_api.h"
#include "trainer.hpp"
#include "trainer_params.hpp"
#include "input_data.hpp"
#include "utils.hpp"



//	todo: actual instance manager & locks
namespace OpenSplat
{
	std::shared_ptr<Trainer> gSingleInstance;
	int gSingleInstanceId = OpenSplat_NullInstance;
	
	int			AllocateInstance(TrainerParams Params,const std::string& InputPath);
	void		FreeInstance(int Instance);
	Trainer&	GetInstance(int Instance);
}

__export int	OpenSplat_AllocateInstanceFromPath(const char* InputDataPath)
{
	try
	{
		TrainerParams Params;
		std::string InputPath( InputDataPath ? InputDataPath : "" );
		auto Instance = OpenSplat::AllocateInstance( Params, InputPath );
		return Instance;
	}
	catch(std::exception& e)
	{
		std::cerr << __FUNCTION__ << ": " << e.what() << std::endl;
		return OpenSplat_NullInstance;
	}
}

__export void	OpenSplat_FreeInstance(int Instance)
{
	try
	{
		OpenSplat::FreeInstance(Instance);
	}
	catch(std::exception& e)
	{
		std::cerr << __FUNCTION__ << ": " << e.what() << std::endl;
	}
}


//	this wants to be simpler and initialise data with a different call
int OpenSplat::AllocateInstance(TrainerParams Params,const std::string& InputPath)
{
	if ( gSingleInstance )
	{
		throw std::runtime_error("Currently only supporting one instance, which is already allocated");
	}
	
	auto DownscaleFactor = 1;
	auto colmapImageSourcePath = "";
	InputData inputData = inputDataFromX( InputPath, colmapImageSourcePath );
	parallel_for(inputData.cameras.begin(), inputData.cameras.end(), [&](Camera &cam)
	{
		cam.loadImageFromFilename(DownscaleFactor);
	});

	gSingleInstance = std::make_shared<Trainer>( Params );
	if ( gSingleInstanceId == OpenSplat_NullInstance )
	{
		gSingleInstanceId = 1000;
	}
	gSingleInstanceId++;
	
	//	copy in input data
	//	todo: setup input data via other API calls
	auto pInputData = std::make_shared<InputData>(inputData);
	gSingleInstance->mInputData = pInputData;
	
	return gSingleInstanceId;
}

void OpenSplat::FreeInstance(int Instance)
{
	if ( gSingleInstanceId != Instance )
	{
		std::stringstream Error;
		Error << "No such instance " << Instance;
		throw std::runtime_error(Error.str());
	}
	
	gSingleInstance.reset();
}

Trainer& OpenSplat::GetInstance(int Instance)
{
	if ( gSingleInstanceId != Instance || gSingleInstance == nullptr )
	{
		std::stringstream Error;
		Error << "No such instance " << Instance;
		throw std::runtime_error(Error.str());
	}
	
	return *gSingleInstance;
}


