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



//	todo: flip this around and return image meta into a byte buffer, to do a faster image-copy library side 
//		and force app to do (faster) image conversion
//	todo: provide arbritary camera extrinscs & intrinsics so we dont rely on blind camera indexes
//	returns OpenSplat_Error_XXX
__export enum OpenSplat_Error	OpenSplat_RenderCamera(int TrainerInstance,int CameraIndex,uint8_t* ImageRgbBuffer,int ImageRgbBufferSize,int ImageRgbWidth,int ImageRgbHeight)
{
	try
	{
		auto& Trainer = OpenSplat::GetInstance(TrainerInstance);
		auto Image = Trainer.GetForwardImage( CameraIndex, ImageRgbWidth, ImageRgbHeight );
		
		//	just don't write anything, but succeed if no buffer supplied
		if ( !ImageRgbBuffer )
			return OpenSplat_Error_Success;
		
		std::span RgbPixels( ImageRgbBuffer, ImageRgbBufferSize );
		if ( RgbPixels.size() != ImageRgbWidth * ImageRgbHeight * 3 )
		{
			throw std::runtime_error("Wrong size rgb buffer provided");
		}
		
		std::copy( Image.mPixels.begin(), Image.mPixels.end(), RgbPixels.begin() );
		return OpenSplat_Error_Success;
	}
	catch(std::exception& e)
	{
		std::cerr << __FUNCTION__ << ": " << e.what() << std::endl;
		return OpenSplat_Error_Unknown;
	}
}

//	returns number of points in model (which can be more or less than buffer size)
__export int OpenSplat_GetSnapshot(int TrainerInstance,struct OpenSplat_Splat* SplatBuffer,int SplatBufferCount)
{
	try
	{
		auto& Trainer = OpenSplat::GetInstance(TrainerInstance);
		auto Splats = Trainer.GetModelSplats();
		
		//	dont fail, but still return info
		if ( !SplatBuffer )
			return static_cast<int>(Splats.size());
		
		int CopyCount = std::min<int>(SplatBufferCount,Splats.size());
		for ( int i=0;	i<CopyCount;	i++ )
		{
			auto& Src = Splats[i];
			SplatBuffer[i] = Src;
		}
		
		return static_cast<int>(Splats.size());
	}
	catch(std::exception& e)
	{
		std::cerr << __FUNCTION__ << ": " << e.what() << std::endl;
		return 0;
	}
}


//	copy ground truth [camera] image into an rgb buffer.
//	todo: provide a byte buffer and json-meta buffer to copy directly without any library side resize/conversion etc (and copy other camera meta, extrinsics, intrisincs, number of iterations with this camera etc)
__export enum OpenSplat_Error	OpenSplat_GetGroundTruthCameraImage(int TrainerInstance,int CameraIndex,uint8_t* ImageRgbBuffer,int ImageRgbBufferSize,int ImageRgbWidth,int ImageRgbHeight)
{
	try
	{
		auto& Trainer = OpenSplat::GetInstance(TrainerInstance);
		auto Image = Trainer.GetCameraImage( CameraIndex, ImageRgbWidth, ImageRgbHeight );
		
		//	just don't write anything, but succeed if no buffer supplied
		if ( !ImageRgbBuffer )
			return OpenSplat_Error_Success;
		
		std::span RgbPixels( ImageRgbBuffer, ImageRgbBufferSize );
		if ( RgbPixels.size() != ImageRgbWidth * ImageRgbHeight * 3 )
		{
			throw std::runtime_error("Wrong size rgb buffer provided");
		}
		
		std::copy( Image.mPixels.begin(), Image.mPixels.end(), RgbPixels.begin() );
		
		return OpenSplat_Error_Success;
	}
	catch(std::exception& e)
	{
		std::cerr << __FUNCTION__ << ": " << e.what() << std::endl;
		return OpenSplat_Error_Unknown;
	}
}


__export OpenSplat_Error	OpenSplat_InstanceRunBlocking(int Instance)
{
	try
	{
		auto& Trainer = OpenSplat::GetInstance(Instance);
		
		auto OnIterationFinished = [&](TrainerIterationMeta IterationMeta)
		{
			auto step = IterationMeta.mStep;
			std::cout << "Done step " << step << std::endl;
		};
		
		auto OnRunFinished = [&](int numIters)
		{
		};
		
		Trainer.Run( OnIterationFinished, OnRunFinished );
		
		return OpenSplat_Error_Success;
	}
	catch(std::exception& e)
	{
		std::cerr << __FUNCTION__ << ": " << e.what() << std::endl;
		return OpenSplat_Error_Unknown;
	}
}
