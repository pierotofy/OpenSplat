import Foundation
import simd
import PopPly


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

public struct NerfStudioFrame : Decodable
{
	var file_path : String
	var transform_matrix : [[Float]]
	var localToWorld : simd_float4x4
	{
		//	todo: check if column or row major
		let row0 = simd_float4( transform_matrix[0] )
		let row1 = simd_float4( transform_matrix[1] )
		let row2 = simd_float4( transform_matrix[2] )
		let row3 = simd_float4( transform_matrix[3] )
		return simd_float4x4(rows: [row0,row1,row2,row3] )
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

class PlySink : PLYReaderDelegate
{
	var xyzs = [Float]()
	var rgbs = [Float]()
	
	func didStartReading(withHeader header: PopPly.PLYHeader) throws
	{
		//	cache headers
	}
	
	func didRead(element: PopPly.PLYElement, typeIndex: Int, withHeader elementHeader: PopPly.PLYHeader.Element) throws
	{
		//	read each element
		if elementHeader.name != "vertex"
		{
			return
		}
		
		let x = try elementHeader.index(forPropertyNamed: "x").map{ try element.float32Value(forPropertyIndex: $0) }
		let y = try elementHeader.index(forPropertyNamed: "y").map{ try element.float32Value(forPropertyIndex: $0) }
		let z = try elementHeader.index(forPropertyNamed: "z").map{ try element.float32Value(forPropertyIndex: $0) }
		let r = try elementHeader.index(forPropertyNamed: "red").map{ try element.uint8Value(forPropertyIndex: $0) }
		let g = try elementHeader.index(forPropertyNamed: "green").map{ try element.uint8Value(forPropertyIndex: $0) }
		let b = try elementHeader.index(forPropertyNamed: "blue").map{ try element.uint8Value(forPropertyIndex: $0) }
		
		guard let x,let y,let z,let r,let g,let b else
		{
			throw PLYTypeError("Missing x/y/z/red/green/blue from seed points")
		}
		
		xyzs.append(x)
		xyzs.append(y)
		xyzs.append(z)
		rgbs.append( Float(r) / 255.0 )
		rgbs.append( Float(g) / 255.0 )
		rgbs.append( Float(b) / 255.0 )
	}
	
}

public struct NerfStudioData
{
	public var transforms : NerfStudioTransforms
	var pointsXyz : [Float]
	var pointsRgb : [Float]
	var pointCount : Int		{	pointsXyz.count / 3	}
	
	public init(projectRoot:String) throws
	{
		//	load json
		let jsonPath = URL(fileURLWithPath: projectRoot + "/transforms.json" )
		transforms = try NerfStudioTransforms.Load(path: jsonPath)
		
		(pointsXyz,pointsRgb) = try NerfStudioData.LoadPoints(projectRoot:projectRoot,plyFilePath: transforms.ply_file_path)
	}
	
	static func LoadPoints(projectRoot:String,plyFilePath:String?) throws -> ([Float],[Float])
	{
		guard let pointFilename = plyFilePath else
		{
			throw NerfDataError("No point/ply/bin filename in meta")
		}
		let pointFilenameFileUrl = URL(fileURLWithPath: projectRoot + "/" + pointFilename )

		if pointFilename.hasSuffix(".ply")
		{
			var sink = PlySink()
			try PopPly.PLYReader.read(url: pointFilenameFileUrl, to: sink)
			return (sink.xyzs,sink.rgbs)
		}
		
		throw NerfDataError("Dont know how to load \(pointFilename)")
	}
	
	
}

