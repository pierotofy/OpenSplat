import OpenSplat
import AppKit
import CoreGraphics
import Accelerate
import simd
import PopCommon

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

extension OpenSplat_Matrix4x4
{
	var row0 : simd_float4	{	simd_float4(m00,m01,m02,m03)	}
	var row1 : simd_float4	{	simd_float4(m10,m11,m12,m13)	}
	var row2 : simd_float4	{	simd_float4(m20,m21,m22,m23)	}
	var row3 : simd_float4	{	simd_float4(m30,m31,m32,m33)	}
	var float4x4 : simd_float4x4	
	{
		simd_float4x4(rows:[row0,row1,row2,row3])
	}
}

extension UInt8 
{
	var char: Character 
	{
		return Character(UnicodeScalar(self))
	}
}

extension OpenSplat_CameraMeta
{
	var localToWorld : simd_float4x4	{	LocalToWorld.float4x4	}

	//	convert Name tuple to string
	var name : String
	{	
		get
		{
			return withUnsafeBytes(of: self.Name)
			{
				namePtr in
				var out = String()
				for i in namePtr.indices
				{
					if namePtr[i] == 0
					{
						break
					}
					out.append(namePtr[i].char)
				}
				//	gr; this is giving a null string... because of junk after terminator?
				//let nameString = String(bytes:namePtr,encoding: .utf8) 
				//return nameString ?? out
				return out
			}
		}
		set
		{
			let chars = newValue.utf8CString
			withUnsafeMutableBytes(of: &self.Name)
			{
				namePtr in
				for i in chars.indices
				{
					let c = UInt8(chars[i])
					if c == 0
					{
						break
					}
					namePtr[i] = c
				}
			}
		}
	}

	//	construct a new instance but make a name from the file path
	//	this needs to be the reverse of c++ Camera::getName
	public init(nameFromFilename:String)
	{
		//	strip path 
		let parts = nameFromFilename.components(separatedBy: "/")
		let filename = parts.last ?? ""
		
		self.init()
		self.name = filename
	}
}

public protocol SplatTrainer
{
	var trainingError : Error?	{	get	}
	//var isTraining : Bool		{	get	}
	var status : String			{	get	}

	init(projectPath:String)

	func Run() async throws
	func GetState() throws -> OpenSplat_TrainerState
	func GetCameraMeta(cameraIndex:Int) throws -> OpenSplat_CameraMeta
	func GetSplats() async throws -> [OpenSplat_Splat]
	func RenderCamera(cameraIndex:Int,width:Int,height:Int) async throws -> CGImage
	func GetCameraGroundTruthImage(cameraIndex:Int,width:Int,height:Int) async throws -> CGImage
}



public class DummySplatTrainer : SplatTrainer
{
	public var status : String			{	"Dummy Status"	}

	public func GetState() throws -> OpenSplat_TrainerState 
	{
		return OpenSplat_TrainerState(IterationsCompleted:999,CameraCount:3, SplatCount:1000)
	}
	
	public func GetCameraMeta(cameraIndex:Int) throws -> OpenSplat_CameraMeta
	{
		throw OpenSplatError("No such camera \(cameraIndex)")
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
	//public var isTraining : Bool	{	(trainingError == nil) && !trainingThreadFinished	}

	@Published public var status : String = "init"
	
	required public init(projectPath:String)
	{
		let loadCameraImagesInApi = false
		let centerAndNormalisePoints = false
		instance = OpenSplat_AllocateInstanceFromPath(projectPath,loadCameraImagesInApi,centerAndNormalisePoints)
		
		trainingTask = Task
		{
			do
			{
				try await self.Thread(projectPath: projectPath,loadCameraImages: !loadCameraImagesInApi)
				self.trainingThreadFinished = true
			}
			catch
			{
				self.trainingError = error
			}
		}
				
		
	}
	
	deinit
	{
		OpenSplat_FreeInstance(instance)
	}
	
	func Thread(projectPath:String,loadCameraImages:Bool) async throws
	{
		//	load these at high priority
		let loadTask = Task.detached(priority: .high)
		{
			if loadCameraImages
			{
				try await self.LoadCameras(projectPath:projectPath)
			}
		}
		try await loadTask.value

		//	high priority blocks other task-calls to the API... (but runs noticably faster)
		//	is this because of locks or task scheduling?
		let runTask = Task.detached(priority: .high)
		{
			try await self.Run()
		}
		try await runTask.value
		print("Run finished")
	}
	
	
	func LoadCameras(projectPath:String) async throws
	{
		let nerfStudioData = try NerfStudioData(projectRoot: projectPath)

		try await withThrowingTaskGroup(of: Void.self) 
		{
			taskGroup in
			//	load each camera
			for camera in nerfStudioData.transforms.frames
			{
				taskGroup.addTask
				{ 
					let intrinsics = try nerfStudioData.transforms.GetCameraIntrinsics(frame: camera)
					try self.LoadCamera(projectPath: projectPath, camera: camera)
				}
			}
			try await taskGroup.waitForAll()
		}
		
	}
	
	func LoadCamera(projectPath:String,camera:NerfStudioFrame) throws
	{
		DispatchQueue.main.async
		{
			self.status = "Loading camera \(camera.file_path)..."
		}
		
		print("Loading camera \(camera.file_path)...")
		
		//let intrinsics = try nerfStudioData.transforms.GetCameraIntrinsics(frame: camera)
		let cameraImagePath = URL(fileURLWithPath: "\(projectPath)/\(camera.file_path)" )
		guard let image = NSImage(contentsOf:cameraImagePath) else
		{
			throw OpenSplatError("Failed to load image at \(cameraImagePath)")
		}
		
		//	use pixels in-place
		try image.withUnsafePixels
		{
			u8pointer,width,height,rowStride,isArgb in
			
			var meta = OpenSplat_CameraMeta(nameFromFilename:camera.file_path)
			meta.Intrinsics.Width = Int32(width)
			meta.Intrinsics.Height = Int32(height)
			let rgbBufferSize = rowStride * height
			
			let format = OpenSplat_PixelFormat_Bgr
			//	expense here is cv::undistort.... implement in a shader and pass in pre-undistorted image & intrinsics
			let error = OpenSplat_AddCamera(instance, &meta, u8pointer, Int32(rgbBufferSize), format )
			if error != OpenSplat_Error_Success
			{
				throw OpenSplatError(apiError: error)
			}
		}
		
		print("Loaded camera \(camera.file_path).")
		
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
	
	public func GetCameraMeta(cameraIndex:Int) throws -> OpenSplat_CameraMeta
	{
		var cameraMeta = OpenSplat_CameraMeta()
		let error = OpenSplat_GetCameraMeta(instance, Int32(cameraIndex), &cameraMeta )
		if error != OpenSplat_Error_Success
		{
			throw OpenSplatError(apiError: error)
		}
		return cameraMeta
	}
	
	
	public func Run() async throws 
	{
		DispatchQueue.main.async
		{
			self.status = "Training..."
		}
		
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
