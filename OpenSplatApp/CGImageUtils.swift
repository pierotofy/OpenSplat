import CoreGraphics
import Accelerate
import AppKit
import PopCommon


/*
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

	}
}
*/
