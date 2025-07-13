import CoreGraphics
import Accelerate

public func rgbBufferToCGImage(_ rgbBuffer:inout[UInt8],width:Int,height:Int) throws -> CGImage
{
	var convert = 
	{
		(rgb:inout vImage_Buffer,rgba:inout vImage_Buffer) in
		let alphaBuffer : UnsafePointer<vImage_Buffer>? = nil
		let alpha = Pixel_8(255)
		let premultiply = false	//	true performs {r = (a * r + 127) / 255}
		let flags = vImage_Flags(kvImageDoNotTile)
		let error = vImageConvert_RGB888toRGBA8888( &rgb, alphaBuffer, alpha, &rgba, premultiply, flags )
		if error != kvImageNoError
		{
			throw OpenSplatError("Some RGB to RGBA error")
		}
	}
	
	
	//	need to convert to RGBA for coregraphics
	var rgbaBuffer = [UInt8](repeating: 0, count: width*height*4)
	try rgbBuffer.withUnsafeMutableBytes
	{
		rgbBufferPointer in 
		try rgbaBuffer.withUnsafeMutableBytes
		{
			rgbaBufferPointer in
			var rgbImage = vImage_Buffer( data:rgbBufferPointer.baseAddress!, height:vImagePixelCount(height), width:vImagePixelCount(width), rowBytes: 3*width )
			var rgbaImage = vImage_Buffer( data:rgbaBufferPointer.baseAddress!, height:vImagePixelCount(height), width:vImagePixelCount(width), rowBytes: 4*width )
			
			try convert( &rgbImage, &rgbaImage )
		}
	}
	
	
	let bytesPerPixel = 4
	let bytesPerRow = bytesPerPixel * width
	let bitsPerComponent = 8
	
	
	let colorSpace = CGColorSpaceCreateDeviceRGB()
	let alpha = CGImageAlphaInfo.premultipliedLast.rawValue
	
	let cgimage = try rgbaBuffer.withUnsafeMutableBytes
	{
		rgbaBufferPointer in
		guard let context = CGContext(data: rgbaBufferPointer.baseAddress!, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: alpha) else
		{
			throw OpenSplatError("Failed to create cg context")
		}
		
		guard let cgImage = context.makeImage() else
		{
			throw OpenSplatError("Failed to create cg image")
		}
		return cgImage
	}
	
	return cgimage
}
