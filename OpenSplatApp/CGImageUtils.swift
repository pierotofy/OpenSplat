import CoreGraphics
import Accelerate
import AppKit

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



//	very basic image type - consider using CGImage...
public struct ImagePixels
{
	var pixels : [UInt8]
	var width : Int
	var height: Int
	var components : Int
}


public extension ImagePixels
{
	init(image:NSImage) throws
	{
		//	extract pixels
		guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else
		{
			throw OpenSplatError("Failed to get cgimage from NSImage")
		}
		
		//	gr: this causes a copy - can we access the data in-place with a callback?
		guard let pixelData = cgImage.dataProvider?.data else
		{
			throw OpenSplatError("Failed to get data out of cgimage")
		}
		let sourcePointer : UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
		let sourcePointerSize = CFDataGetLength(pixelData)
		let sourceBuffer = UnsafeRawBufferPointer(start: sourcePointer,count: sourcePointerSize)
		
		self.components = cgImage.bitsPerPixel / cgImage.bitsPerComponent
		self.width = Int(image.size.width)
		self.height = Int(image.size.height)
		self.pixels = Array(repeating: 0, count: width*height*components)
		
		if sourcePointerSize != self.pixels.count
		{
			throw OpenSplatError("CGImage probably not aligned")
		}			
		
		pixels.withUnsafeMutableBytes
		{
			//(dest:UnsafePointer<UInt8>) in
			dest in
			//dest.copyBytes(from: sourceBuffer)
			dest.copyMemory(from: sourceBuffer)
		}
		/*
		 self.pixels = []
		 
		 for y in 0..<height {
		 for x in 0..<width {
		 let pos = CGPoint(x: x, y: y)
		 
		 let pixelIndex : Int = ((width * Int(pos.y) * 4) + Int(pos.x) * 4)
		 
		 let r = dataPointer[pixelIndex + 0]
		 let g = dataPointer[pixelIndex + 1]
		 let b = dataPointer[pixelIndex + 2]
		 let a = dataPointer[pixelIndex + 3]
		 pixels.append(r)
		 pixels.append(g)
		 pixels.append(b)
		 }
		 }
		 */
	}
}
