#include "utils.hpp"

void CopyStringToBuffer(std::string_view input, char *dst, size_t dst_size)
{
	//	gr: throw?
	if ( !dst )
		return;
	auto CopyLength = std::min( dst_size, input.length() );
	for ( auto i=0;	i<CopyLength;	i++ )
		dst[i] = input[i];
	
	//	input.length has no terminator, thus [length]=0 in output
	auto TerminatorPos = std::min( dst_size-1, input.length() );
	dst[TerminatorPos] = '\0';
}
