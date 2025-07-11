/*
	C99 api to a/the trainer
*/
#pragma once

#include <stdbool.h>
#include <stdint.h>

#if !defined(__export)

#if defined(_MSC_VER)	//	windows compiler
#define __export			extern "C" __declspec(dllexport)
#else
#define __export			extern "C"
#endif

#endif


//	constant for invalid instance numbers, to avoid use of magic-number 0 around code bases
enum { OpenSplat_NullInstance=0 };

enum OpenSplat_Error 
{
	OpenSplat_Error_Success		= 0,
	OpenSplat_Error_Unknown		= 1,
	//	todo: turn c++ exceptions into specific errors
	//OpenSplat_Error_NoInstance	= 2,
	//OpenSplat_Error_NoCamera	= 3
};


//	exported splat
struct OpenSplat_Splat
{
	float x,y,z;
	float opacity;
	float scalex,scaley,scalez;
	float rotw,rotx,roty,rotz;
	
	//	first 3 spherical harmonics values, enough to get colour as a v1
	float dc0,dc1,dc2;
};

//	deprecate this in future for pushing data - app should be repsonsible for i/o
__export int	OpenSplat_AllocateInstanceFromPath(const char* InputDataPath);
__export void	OpenSplat_FreeInstance(int Instance);


//	returns number of points in model (which can be more or less than buffer size)
__export int					OpenSplat_GetSnapshot(int TrainerInstance,struct OpenSplat_Splat* SplatBuffer,int SplatBufferCount);

//	Render a camera into an rgb buffer
//	todo: flip this around and return image meta into a byte buffer, to do a faster image-copy library side 
//		and force app to do (faster) image conversion
//	todo: provide arbritary camera extrinscs & intrinsics so we dont rely on blind camera indexes
//	returns OpenSplat_Error_XXX
__export enum OpenSplat_Error	OpenSplat_RenderCamera(int TrainerInstance,int CameraIndex,uint8_t* ImageRgbBuffer,int ImageRgbBufferSize,int ImageRgbWidth,int ImageRgbHeight);


//	copy ground truth [camera] image into an rgb buffer.
//	todo: provide a byte buffer and json-meta buffer to copy directly without any library side resize/conversion etc (and copy other camera meta, extrinsics, intrisincs, number of iterations with this camera etc)
__export enum OpenSplat_Error	OpenSplat_GetGroundTruthCameraImage(int TrainerInstance,int CameraIndex,uint8_t* ImageRgbBuffer,int ImageRgbBufferSize,int ImageRgbWidth,int ImageRgbHeight);
