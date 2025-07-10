#include "ply.hpp"
#include <fstream>

void Ply::Write(std::ostream& Output,WriteParams Params,std::function<SplatElement(int)> GetSplatByIndex)
{
	//	if input is a file, check it's open
	{
		auto* OutputFile = dynamic_cast<std::ofstream*>(&Output);
		if ( OutputFile )
		{
			if ( !OutputFile->is_open() )
				throw std::runtime_error("Ply::Write with stream that's not open");
		}
	}
	
	auto& o = Output;
	
	//	todo: sanitise Comment to remove line feeds
	
	o << "ply" << std::endl;
	o << "format binary_little_endian 1.0" << std::endl;
	o << "comment " << Params.Comment << std::endl;
	o << "element vertex " << Params.PointCount << std::endl;
	o << "property float x" << std::endl;
	o << "property float y" << std::endl;
	o << "property float z" << std::endl;
	
	if ( Params.WriteNormals )
	{
		o << "property float nx" << std::endl;
		o << "property float ny" << std::endl;
		o << "property float nz" << std::endl;
	}
	
	for ( int i=0;	i<Params.FeatureDcCount;	i++)
	{
		o << "property float f_dc_" << i << std::endl;
	}
	
	// Match Inria's version
	for (int i = 0; i <Params.FeatureRestCount; i++)
	{
		o << "property float f_rest_" << i << std::endl;
	}
		
	o << "property float opacity" << std::endl;
	
	o << "property float scale_0" << std::endl;
	o << "property float scale_1" << std::endl;
	o << "property float scale_2" << std::endl;
	
	o << "property float rot_0" << std::endl;
	o << "property float rot_1" << std::endl;
	o << "property float rot_2" << std::endl;
	o << "property float rot_3" << std::endl;
	
	o << "end_header" << std::endl;
	
	auto Write = [&o](std::span<float> Values)
	{
		auto* Bytes = reinterpret_cast<const char *>(Values.data());
		o.write( Bytes, Values.size_bytes() );
	};
	
	
	for ( int p=0;	p<Params.PointCount;	p++ )
	{
		auto Element = GetSplatByIndex(p);
		
		//	remember to strictly match header order!
		Write( Element.Position3() );
		if ( Params.WriteNormals )
		{
			Write( Element.Normal3() );
		}
		
		if ( Element.FeatureDcs.size() != Params.FeatureDcCount )
			throw std::runtime_error("DC Feature count mismatch");
		Write( Element.FeatureDcs );
		
		if ( Element.FeatureRests.size() != Params.FeatureRestCount )
			throw std::runtime_error("Rest Feature count mismatch");
		Write( Element.FeatureRests );
		
		Write( Element.Opacity() );
		Write( Element.Scale3() );
		
		//	note this order is supposed to be WXYZ not XYZW in splat viewers
		Write( Element.QuaternionWXYZ() );
	}
	
	//	leave this to caller
	//o.close();
}
