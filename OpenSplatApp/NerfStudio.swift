import Foundation



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

public struct NerfStudioData
{
	public var transforms : NerfStudioTransforms
	
	public init(projectRoot:String) throws
	{
		//	load json
		let jsonPath = URL(fileURLWithPath: projectRoot + "/transforms.json" )
		transforms = try NerfStudioTransforms.Load(path: jsonPath)
	}
	
	
}

