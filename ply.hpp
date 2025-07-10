#pragma once

#include <ostream>
#include <span>
#include <array>

#include "trainer_api.h"	//	OpenSplat_Splat


class SplatElement
{
public:
	//	reuse api splat with most members
	OpenSplat_Splat		Splat;
	std::span<float>	FeatureDcs;
	std::span<float>	FeatureRests;
	std::array<float,3>	Normalxyz{0,0,0};
	
	//	hacky pointers!
	std::span<float>	Position3()			{	return std::span( &Splat.x, 3 );	}
	std::span<float>	Normal3()			{	return std::span( Normalxyz );	} 
	std::span<float>	Scale3()			{	return std::span( &Splat.scalex, 3 );	}
	std::span<float>	QuaternionWXYZ()	{	return std::span( &Splat.rotw, 4 );	}
	std::span<float>	Opacity()			{	return std::span( &Splat.opacity, 1 );	}
};

namespace Ply
{
	class WriteParams
	{
	public:
		std::string	Comment;
		bool		WriteNormals = true;
		int			PointCount = 0;
		int			FeatureDcCount = 0;
		int 		FeatureRestCount = 0;
	};
	
	void		Write(std::ostream& Output,WriteParams Params,std::function<SplatElement(int)> GetSplatByIndex);
}
