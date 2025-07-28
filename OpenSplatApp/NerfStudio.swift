import Foundation
import simd
import PopPly
import PopCommon

public struct NerfDataError : LocalizedError
{
	var message : String
	
	public init(_ message:String)
	{
		self.message = message
	}
	
	public var errorDescription: String?	{	message	}
}


struct CameraIntrinsics : Decodable
{
	var w : Float
	var h : Float
	var fx : Float
	var fy : Float
	var cx : Float
	var cy : Float
	var k1 : Float
	var k2 : Float
	var k3 : Float
	var p1 : Float
	var p2 : Float
}

extension simd_float4
{
	var array : [Float]
	{
		return [x,y,z,w]
	}
}

public struct NerfStudioFrame : Decodable
{
	var file_path : String
	var transform_matrix : [[Float]]
	var localToWorld : simd_float4x4
	{
		get
		{
			//	todo: check if column or row major
			let row0 = simd_float4( transform_matrix[0] )
			let row1 = simd_float4( transform_matrix[1] )
			let row2 = simd_float4( transform_matrix[2] )
			let row3 = simd_float4( transform_matrix[3] )
			return simd_float4x4(rows: [row0,row1,row2,row3] )
		}
		set
		{
			let rowMajor = newValue.transpose
			let row0 = rowMajor.columns.0.array 
			let row1 = rowMajor.columns.1.array 
			let row2 = rowMajor.columns.2.array 
			let row3 = rowMajor.columns.3.array 
			transform_matrix = [row0,row1,row2,row3]
		}
	}
	
	//	optional intrinsics
	var w : Float?
	var h : Float?
	var fl_x : Float?
	var fl_y : Float?
	var fx : Float?	{	fl_x	}
	var fy : Float?	{	fl_y	}
	var cx : Float?
	var cy : Float?
	var k1 : Float?
	var k2 : Float?
	var k3 : Float?
	var p1 : Float?
	var p2 : Float?
}

public struct NerfStudioTransforms : Decodable
{
	var camera_model : String
	var frames : [NerfStudioFrame]
	var ply_file_path : String?
	
	//	optional global values for intrinsics
	var w : Float?
	var h : Float?
	var fl_x : Float?
	var fl_y : Float?
	var fx : Float?	{	fl_x	}
	var fy : Float?	{	fl_y	}
	var cx : Float?
	var cy : Float?
	var k1 : Float?
	var k2 : Float?
	var k3 : Float?
	var p1 : Float?
	var p2 : Float?
	
	func GetCameraIntrinsics(frame:NerfStudioFrame) throws -> CameraIntrinsics
	{
		func getValue(_ v:Float?,_ context:String) throws -> Float
		{
			if let v
			{
				return v
			}
			throw NerfDataError("Missing \(context)")
		}
		return CameraIntrinsics(
			w: try getValue(frame.w ?? w, "w"), 
			h: try getValue(frame.h ?? h, "h"), 
			fx: try getValue(frame.fx ?? fx, "fx"), 
			fy: try getValue(frame.fy ?? fy, "fy"), 
			cx: try getValue(frame.cx ?? cx, "cx"), 
			cy: try getValue(frame.cy ?? cy, "cy"), 
			k1: try getValue(frame.k1 ?? k1, "k1"), 
			k2: try getValue(frame.k2 ?? k2, "k2"), 
			k3: try getValue(frame.k3 ?? k3, "k3"), 
			p1: try getValue(frame.p1 ?? p1, "p1"), 
			p2: try getValue(frame.p2 ?? p2, "p2")
		)
	}
}


public extension NerfStudioTransforms
{
	public static func Load(path:URL) throws -> NerfStudioTransforms
	{
		let jsonData = try Data(contentsOf: path)
		let transforms = try JSONDecoder().decode(NerfStudioTransforms.self, from: jsonData)
		return transforms
	}
}

public class PlySink : PLYReaderDelegate
{
	var xyzs = [Float]()
	var rgbs = [Float]()
	
	var applyTransform : simd_float4x4?=nil
	
	public init(applyTransform:simd_float4x4?=nil)
	{
		self.applyTransform = applyTransform
	}
	
	public func didStartReading(withHeader header: PopPly.PLYHeader) throws
	{
		//	cache headers
	}
	
	public func didRead(element: PopPly.PLYElement, typeIndex: Int, withHeader elementHeader: PopPly.PLYHeader.Element) throws
	{
		//	read each element
		if elementHeader.name != "vertex"
		{
			return
		}
		
		let x = try elementHeader.index(forPropertyNamed: "x").map{ try element.float32Value(forPropertyIndex: $0) }
		let y = try elementHeader.index(forPropertyNamed: "y").map{ try element.float32Value(forPropertyIndex: $0) }
		let z = try elementHeader.index(forPropertyNamed: "z").map{ try element.float32Value(forPropertyIndex: $0) }
		let r = try elementHeader.index(forPropertyNamed: "red").map{ try element.float32Value(forPropertyIndex: $0, normalise8: 255) }
		let g = try elementHeader.index(forPropertyNamed: "green").map{ try element.float32Value(forPropertyIndex: $0, normalise8: 255) }
		let b = try elementHeader.index(forPropertyNamed: "blue").map{ try element.float32Value(forPropertyIndex: $0, normalise8: 255) }
		
		guard var x,var y,var z,let r,let g,let b else
		{
			throw PLYTypeError("Missing x/y/z/red/green/blue from seed points")
		}
		
		if let applyTransform
		{
			let xyzw = applyTransform * simd_float4(x,y,z,1)
			x = xyzw.x * xyzw.w
			y = xyzw.y * xyzw.w
			z = xyzw.z * xyzw.w
		}
		
		xyzs.append(x)
		xyzs.append(y)
		xyzs.append(z)
		rgbs.append(r)
		rgbs.append(g)
		rgbs.append(b)
	}
	
}

public struct NerfStudioData
{
	public var transforms : NerfStudioTransforms
	var pointsXyz : [Float]
	var pointsRgb : [Float]
	var pointCount : Int		{	pointsXyz.count / 3	}
	
	public init(projectRoot:String,applyTransform:simd_float4x4=simd_float4x4.identity) throws
	{
		//	load json
		let jsonPath = URL(fileURLWithPath: projectRoot + "/transforms.json" )
		transforms = try NerfStudioTransforms.Load(path: jsonPath)
		
		//	modify camera transforms
		transforms.frames.mutateEach
		{
			frame in
			frame.localToWorld = applyTransform * frame.localToWorld
		}
		
		(pointsXyz,pointsRgb) = try NerfStudioData.LoadPoints(projectRoot:projectRoot,plyFilePath: transforms.ply_file_path, applyTransform:applyTransform)
	}
	
	static func LoadPoints(projectRoot:String,plyFilePath:String?,applyTransform:simd_float4x4) throws -> ([Float],[Float])
	{
		guard let pointFilename = plyFilePath else
		{
			throw NerfDataError("No point/ply/bin filename in meta")
		}
		let pointFilenameFileUrl = URL(fileURLWithPath: projectRoot + "/" + pointFilename )

		if pointFilename.hasSuffix(".ply")
		{
			var sink = PlySink(applyTransform:applyTransform)
			try PopPly.PLYReader.read(url: pointFilenameFileUrl, to: sink)
			return (sink.xyzs,sink.rgbs)
		}
		
		throw NerfDataError("Dont know how to load \(pointFilename)")
	}
	
	
}

