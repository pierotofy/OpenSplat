import OpenSplat
import CoreGraphics
import Accelerate

public struct OpenSplatError : LocalizedError
{
	var message : String
	
	public init(_ message:String)
	{
		self.message = message
	}
	public init(apiError:OpenSplat_Error)
	{
		switch apiError
		{
			case OpenSplat_Error_Unknown:	message = "Unknown OpenSplat API error"
			case OpenSplat_Error_Success:	message = "OpenSplat success(not an error)"
			case OpenSplat_Error_NoCamera:	message = "No such camera"
			case OpenSplat_Error_NoInstance:	message = "No such instance"
			default:
				message = "OpenSplat unknown-error [\(apiError)]"
		}
	}
	
	public var errorDescription: String?	{	message	}
}



public protocol SplatTrainer
{
	var trainingError : Error?	{	get	}
	var isTraining : Bool		{	get	}

	init(projectPath:String)

	func Run() async throws
	func GetState() throws -> OpenSplat_TrainerState
	func GetCameraMeta() throws -> [OpenSplat_CameraMeta]	//	todo: cache this
	func GetSplats() async throws -> [OpenSplat_Splat]
	func RenderCamera(cameraIndex:Int,width:Int,height:Int) async throws -> CGImage
	func GetCameraGroundTruthImage(cameraIndex:Int,width:Int,height:Int) async throws -> CGImage
}



public class DummySplatTrainer : SplatTrainer
{
	public func GetState() throws -> OpenSplat_TrainerState 
	{
		return OpenSplat_TrainerState(IterationsCompleted:999,CameraCount:3, SplatCount:1000)
	}
	
	public func GetCameraMeta() throws -> [OpenSplat_CameraMeta]
	{
		return []
	}
	
	public var trainingError: Error?	{	nil	}
	public var isTraining: Bool			{	false	}
	
	public required init(projectPath: String) 
	{
	}
	
	public func Run() async throws 
	{
		throw OpenSplatError("Run not implemented")
	}
	
	public func GetSplats() async throws -> [OpenSplat_Splat] 
	{
		throw OpenSplatError("GetSplats not implemented")
	}
	
	public func RenderCamera(cameraIndex: Int, width: Int, height: Int) async throws -> CGImage 
	{
		throw OpenSplatError("RenderCamera not implemented")
	}
	
	public func GetCameraGroundTruthImage(cameraIndex: Int, width: Int, height: Int) async throws -> CGImage 
	{
		throw OpenSplatError("GetCameraGroundTruthImage not implemented")
	}
}



public class OpenSplatTrainer : ObservableObject, SplatTrainer
{
	private var instance : Int32
	var trainingTask : Task<Void,Error>!
	@Published public var trainingError : Error? = nil
	@Published public var trainingThreadFinished : Bool = false
	public var isTraining : Bool	{	(trainingError == nil) && !trainingThreadFinished	}
	
	required public init(projectPath:String)
	{
		instance = OpenSplat_AllocateInstanceFromPath(projectPath)
		trainingTask = Task.detached(priority: .background)
		{
			do
			{
				try await self.Run()
				DispatchQueue.main.async
				{
					self.trainingThreadFinished = true
				}
			}
			catch
			{
				DispatchQueue.main.async
				{
					self.trainingError = error
				}
			}
		}
	}
	
	deinit
	{
		OpenSplat_FreeInstance(instance)
	}
	
	public func GetState() throws -> OpenSplat_TrainerState 
	{
		var state = OpenSplat_TrainerState()
		let error = OpenSplat_GetState( instance, &state )
		if error != OpenSplat_Error_Success
		{
			throw OpenSplatError(apiError: error)
		}
		return state
	}
	
	public func GetCameraMeta() throws -> [OpenSplat_CameraMeta] 
	{
		let cameraCount = try GetState().CameraCount
		let cameras = try (0..<cameraCount).map
		{
			cameraIndex in
			var cameraMeta = OpenSplat_CameraMeta()
			let error = OpenSplat_GetCameraMeta(instance, cameraIndex, &cameraMeta )
			if error != OpenSplat_Error_Success
			{
				throw OpenSplatError(apiError: error)
			}
			return cameraMeta
		}
		return cameras
	}
	
	
	public func Run() async throws 
	{
		let error = OpenSplat_InstanceRunBlocking( instance )
		if error != OpenSplat_Error_Success
		{
			throw OpenSplatError(apiError: error)
		}
	}
	
	public func GetSplats() async throws -> [OpenSplat_Splat]
	{
		let task = Task.detached(priority: .background)
		{
			try self.GetSplatsBlocking()
		}
		let result = try await task.result.get()
		return result
	}
	
	func GetSplatsBlocking() throws -> [OpenSplat_Splat]
	{
		let splatCount = OpenSplat_GetSnapshot( instance, nil, 0 )
		
		var splatBuffer = [OpenSplat_Splat](repeating: OpenSplat_Splat(), count: Int(splatCount) )
		let outputSplatCount = try splatBuffer.withUnsafeMutableBufferPointer
		{
			(buffer:inout UnsafeMutableBufferPointer<OpenSplat_Splat>) in
			guard let bufferAddress = buffer.baseAddress else
			{
				throw OpenSplatError("Failed to get buffer address for \(splatCount) splats")
			}
			let outputSplatCount = OpenSplat_GetSnapshot( instance, bufferAddress, splatCount )
			return outputSplatCount
		}
		
		return splatBuffer
	}
	
	public func RenderCamera(cameraIndex:Int,width:Int,height:Int) async throws -> CGImage
	{
		let task = Task.detached(priority: .background)
		{
			try self.RenderCameraBlocking(cameraIndex: cameraIndex, width: width, height: height)
		}
		let result = try await task.result.get()
		return result
	}
	
	//	blocking, so best not to call on mainactor
	func RenderCameraBlocking(cameraIndex:Int,width:Int,height:Int) throws -> CGImage
	{
		let rgbBufferSize = width*height*3
		var rgbBuffer = [UInt8](repeating: 0, count: rgbBufferSize )
		try rgbBuffer.withUnsafeMutableBufferPointer
		{
			(buffer:inout UnsafeMutableBufferPointer<UInt8>) in
			guard let bufferAddress = buffer.baseAddress else
			{
				throw OpenSplatError("Failed to get buffer address for rgb buffer")
			}
			let error = OpenSplat_RenderCamera( instance, Int32(cameraIndex), bufferAddress, Int32(rgbBufferSize), Int32(width), Int32(height) )
			if error != OpenSplat_Error_Success
			{
				throw OpenSplatError(apiError: error)
			}
		}
		
		return try OpenSplatTrainer.rgbBufferToCGImage(&rgbBuffer,width:width,height: height)
	}
	
	public func GetCameraGroundTruthImage(cameraIndex:Int,width:Int,height:Int) async throws -> CGImage
	{
		let task = Task.detached(priority: .background)
		{
			try self.GetCameraGroundTruthImageBlocking(cameraIndex: cameraIndex, width: width, height: height)
		}
		let result = try await task.result.get()
		return result
	}
	
	func GetCameraGroundTruthImageBlocking(cameraIndex:Int,width:Int,height:Int) throws -> CGImage
	{
		let rgbBufferSize = width*height*3
		var rgbBuffer = [UInt8](repeating: 0, count: rgbBufferSize )
		try rgbBuffer.withUnsafeMutableBufferPointer
		{
			(buffer:inout UnsafeMutableBufferPointer<UInt8>) in
			guard let bufferAddress = buffer.baseAddress else
			{
				throw OpenSplatError("Failed to get buffer address for rgb buffer")
			}
			let error = OpenSplat_GetGroundTruthCameraImage( instance, Int32(cameraIndex), bufferAddress, Int32(rgbBufferSize), Int32(width), Int32(height) )
			if error != OpenSplat_Error_Success
			{
				throw OpenSplatError(apiError: error)
			}
		}
		//__export enum OpenSplat_Error	OpenSplat_RenderCamera(int TrainerInstance,int CameraIndex,uint8_t* ImageRgbBuffer,int ImageRgbBufferSize,int ImageRgbWidth,int ImageRgbHeight);
		return try OpenSplatTrainer.rgbBufferToCGImage(&rgbBuffer,width:width,height: height)
	}

	static func rgbBufferToCGImage(_ rgbBuffer:inout[UInt8],width:Int,height:Int) throws -> CGImage
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
	
}
