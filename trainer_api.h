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


//	exported splat
struct OpenSplat_Splat
{
	float x,y,z;
	float opacity;
	float scalex,scaley,scalez;
	float rotx,roty,rotz,rotw;
	
	//	first 3 spherical harmonics values, enough to get colour as a v1
	float dc0,dc1,dc2;
};

//	deprecate this in future for pushing data - app should be repsonsible for i/o
__export int	OpenSplat_AllocateInstanceFromPath(const char* InputDataPath);
__export void	OpenSplat_FreeInstance(int Instance);
