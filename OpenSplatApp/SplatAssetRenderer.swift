import Accelerate
import PopMetalView
import Foundation
import Metal
import MetalKit


public struct SplatRenderParams
{
	var minAlpha : Float = 0.0
	var clipMaxAlpha : Float = 0.0
}

struct PackedHalf3 {
	var x: Float16
	var y: Float16
	var z: Float16
}

struct PackedRGBHalf4 {
	var r: Float16
	var g: Float16
	var b: Float16
	var a: Float16
}

//	matching shader struct
struct SplatElement
{
	var position: MTLPackedFloat3
	var color: PackedRGBHalf4
	var covA: PackedHalf3
	var covB: PackedHalf3
}

public enum SplatColor 
{
	case sphericalHarmonic(Float, Float, Float, [Float])
	case firstOrderSphericalHarmonic(Float, Float, Float)
	case linearFloat(Float, Float, Float)
	case linearUInt8(UInt8, UInt8, UInt8)
	case none
	
	var nonFirstOrderSphericalHarmonics: [Float]? {
		switch self {
			case .sphericalHarmonic(_, _, _, let nonFirstOrderSphericalHarmonics):
				nonFirstOrderSphericalHarmonics
			case .firstOrderSphericalHarmonic, .linearFloat, .linearUInt8:
				nil
			case .none:
				nil
		}
	}
}

extension SplatElement 
{
	init(position: MTLPackedFloat3,
		 color: SIMD4<Float>,
		 scale: SIMD3<Float>,
		 rotation: simd_quatf) 
	{
		let transform = simd_float3x3(rotation) * simd_float3x3(diagonal: scale)
		let cov3D = transform * transform.transpose
		self.init(position: position,
				  color: PackedRGBHalf4(r: Float16(color.x), g: Float16(color.y), b: Float16(color.z), a: Float16(color.w)),
				  covA: PackedHalf3(x: Float16(cov3D[0, 0]), y: Float16(cov3D[0, 1]), z: Float16(cov3D[0, 2])),
				  covB: PackedHalf3(x: Float16(cov3D[1, 1]), y: Float16(cov3D[1, 2]), z: Float16(cov3D[2, 2])))
	}
	
	
	static func FixOpacity(_ opacity:Float) -> Float
	{
		return 1.0 / (1.0 + exp(-opacity) )
	}
	
	static func FixScale(_ scale:[Float]) -> SIMD3<Float>
	{
		let x = scale[0]
		let y = scale[1]
		let z = scale[2]
		return SIMD3<Float>( exp(x), exp(y), exp(z) )
	}
	
	static func FirstOrderSphericalHarmonicsToRgb(_ shValues:[Float]) -> [Float]
	{
		//	gr: example code didn't use the rest! need to use that in shader
		let sh_r = shValues[0]
		let sh_g = shValues[1]
		let sh_b = shValues[2]
		
		let SH_C0: Float = 0.28209479177387814
		let r = max(0, min(1, 0.5 + SH_C0 * sh_r))
		let g = max(0, min(1, 0.5 + SH_C0 * sh_g))
		let b = max(0, min(1, 0.5 + SH_C0 * sh_b))
		
		let rgb3 = [r,g,b].map
		{
			//	gr: we're rendering to non-srgb metal buffer, but this produces colours
			//		that are too dark - without is correct.
			//		Find out where in the pipeline this is wrong (export? shader?)
			//	convert srgb to linear
			//pow( $0, 2.2 )
			$0
		}
		return rgb3
	}
	
}



class MetalBuffer<T> {
	enum Error: LocalizedError {
		case capacityGreatedThanMaxCapacity(requested: Int, max: Int)
		case bufferCreationFailed
		
		var errorDescription: String? {
			switch self {
				case .capacityGreatedThanMaxCapacity(let requested, let max):
					"Requested metal buffer size (\(requested)) exceeds device maximum (\(max))"
				case .bufferCreationFailed:
					"Failed to create metal buffer"
			}
		}
	}
	
	let device: MTLDevice
	
	var capacity: Int = 0
	var count: Int = 0
	var buffer: MTLBuffer
	var values: UnsafeMutablePointer<T>
	
	init(device: MTLDevice, capacity: Int = 1) throws {
		let capacity = max(capacity, 1)
		guard capacity <= Self.maxCapacity(for: device) else {
			throw Error.capacityGreatedThanMaxCapacity(requested: capacity, max: Self.maxCapacity(for: device))
		}
		
		self.device = device
		
		self.capacity = capacity
		self.count = 0
		guard let buffer = device.makeBuffer(length: MemoryLayout<T>.stride * self.capacity,
											 options: .storageModeShared) else {
			throw Error.bufferCreationFailed
		}
		self.buffer = buffer
		self.values = UnsafeMutableRawPointer(self.buffer.contents()).bindMemory(to: T.self, capacity: self.capacity)
	}
	
	static func maxCapacity(for device: MTLDevice) -> Int {
		device.maxBufferLength / MemoryLayout<T>.stride
	}
	
	var maxCapacity: Int {
		device.maxBufferLength / MemoryLayout<T>.stride
	}
	
	func setCapacity(_ newCapacity: Int) throws {
		let newCapacity = max(newCapacity, 1)
		guard newCapacity != capacity else { return }
		guard capacity <= maxCapacity else {
			throw Error.capacityGreatedThanMaxCapacity(requested: capacity, max: maxCapacity)
		}
		
		// log.info("Allocating a new buffer of size \(MemoryLayout<T>.stride) * \(newCapacity) = \(Float(MemoryLayout<T>.stride * newCapacity) / (1024.0 * 1024.0))mb")
		guard let newBuffer = device.makeBuffer(length: MemoryLayout<T>.stride * newCapacity,
												options: .storageModeShared) else {
			throw Error.bufferCreationFailed
		}
		let newValues = UnsafeMutableRawPointer(newBuffer.contents()).bindMemory(to: T.self, capacity: newCapacity)
		let newCount = min(count, newCapacity)
		if newCount > 0 {
			memcpy(newValues, values, MemoryLayout<T>.stride * newCount)
		}
		
		self.capacity = newCapacity
		self.count = newCount
		self.buffer = newBuffer
		self.values = newValues
	}
	
	func ensureCapacity(_ minimumCapacity: Int) throws {
		guard capacity < minimumCapacity else { return }
		try setCapacity(minimumCapacity)
	}
	
	/// Assumes capacity is available
	/// Returns the index of the value
	@discardableResult
	func append(_ element: T) -> Int {
		(values + count).pointee = element
		defer { count += 1 }
		return count
	}
	
	/// Assumes capacity is available.
	/// Returns the index of the first values.
	@discardableResult
	func append(_ elements: [T]) -> Int {
		(values + count).update(from: elements, count: elements.count)
		defer { count += elements.count }
		return count
	}
}



//	move to Scene code
public struct CameraDescriptor 
{
	public var projectionMatrix: simd_float4x4
	public var viewMatrix: simd_float4x4
	public var screenSize: SIMD2<Int>
	public var worldForward : simd_float3	{	(viewMatrix.inverse * SIMD4<Float>(x: 0, y: 0, z: -1, w: 0)).xyz	}
	public var worldPositon : simd_float3	{	(viewMatrix.inverse * SIMD4<Float>(x: 0, y: 0, z: 0, w: 1)).xyz	}
	
	
	public init(projectionMatrix: simd_float4x4, viewMatrix: simd_float4x4, screenSize: SIMD2<Int>) 
	{
		self.projectionMatrix = projectionMatrix
		self.viewMatrix = viewMatrix
		self.screenSize = screenSize
	}
}

public class SplatAssetRenderer 
{
	enum Constants {

		// Sort by euclidian distance squared from camera position (true), or along the "forward" vector (false)
		// TODO: compare the behaviour and performance of sortByDistance
		// notes: sortByDistance introduces unstable artifacts when you get close to an object; whereas !sortByDistance introduces artifacts are you turn -- but they're a little subtler maybe?
		static let sortByDistance = true
		// TODO: compare the performance of useAccelerateForSort, both for small and large scenes
		static let useAccelerateForSort = false
		static let renderFrontToBack = false
	}
	
	
	
	// Keep in sync with Shaders.metal : BufferIndex
	enum BufferIndex: NSInteger {
		case uniforms = 0
		case splat    = 1
		case order    = 2
	}
	
	// Keep in sync with Shaders.metal : Uniforms
	struct Uniforms 
	{
		var projectionMatrix: matrix_float4x4
		var viewMatrix: matrix_float4x4
		var screenSize: SIMD2<UInt32> // Size of screen in pixels
		var pad0 = UInt32(123) 
		var pad1 = UInt32(456)
	}
	
	
	struct PackedHalf3 {
		var x: Float16
		var y: Float16
		var z: Float16
	}
	
	struct PackedRGBHalf4 {
		var r: Float16
		var g: Float16
		var b: Float16
		var a: Float16
	}
	
	
	struct SplatIndexAndDepth {
		var index: UInt32
		var depth: Float
	}
	
	var pipelineState: MTLRenderPipelineState
	var depthState: MTLDepthStencilState
	
	
	typealias IndexType = UInt32
	// splatBuffer contains one entry for each gaussian splat
	var splatBuffer: MetalBuffer<SplatElement>
	// orderBuffer indexes into splatBuffer, and is sorted by distance
	var orderBuffer: MetalBuffer<IndexType>
	
	public var splatCount: Int { splatBuffer.count }
	
	var sortPerSecond : CGFloat = 0.0
	//var sortCounter : FrameCounterModel!
	var sorting = false	//	todo: remove this hack (deferred cpu sorting)
	
	// orderBufferPrime is a copy of orderBuffer, which is not currenly in use for rendering.
	// We use this for sorting, and when we're done, swap it with orderBuffer.
	// There's a good chance that we'll sometimes end up sorting an orderBuffer still in use for
	// rendering;.
	// TODO: Replace this with a more robust multiple-buffer scheme to guarantee we're never actively sorting a buffer still in use for rendering
	var orderBufferPrime: MetalBuffer<IndexType>
	
	// Sorting via Accelerate
	// While not sorting, we guarantee that orderBufferTempSort remains valid: the count may not match splatCount, but for every i in 0..<orderBufferTempSort.count, orderBufferTempSort should contain exactly one element equal to i
	var orderBufferTempSort: MetalBuffer<UInt>
	// depthBufferTempSort is ordered by vertex index; so depthBufferTempSort[0] -> splatBuffer[0], *not* orderBufferTempSort[0]
	var depthBufferTempSort: MetalBuffer<Float>
	
	// Sorting on CPU
	// While not sorting, we guarantee that orderAndDepthTempSort remains valid: the count may not match splatCount, but the array should contain all indices.
	// So for every i in 0..<orderAndDepthTempSort.count, orderAndDepthTempSort should contain exactly one element with .index = i
	var orderAndDepthTempSort: [SplatIndexAndDepth] = []
	
	
	public init(device: MTLDevice,
				colorFormat: MTLPixelFormat,
				depthFormat: MTLPixelFormat,
				stencilFormat: MTLPixelFormat,
				sampleCount: Int
				) throws 
	{
		self.splatBuffer = try MetalBuffer(device: device)
		self.orderBuffer = try MetalBuffer(device: device)
		self.orderBufferPrime = try MetalBuffer(device: device)
		self.orderBufferTempSort = try MetalBuffer(device: device)
		self.depthBufferTempSort = try MetalBuffer(device: device)

		pipelineState = try Self.buildRenderPipelineWithDevice(device: device,
																   colorFormat: colorFormat,
																   depthFormat: depthFormat,
																   stencilFormat: stencilFormat,
																   sampleCount: sampleCount)
	
		let depthStateDescriptor = MTLDepthStencilDescriptor()
		depthStateDescriptor.depthCompareFunction = MTLCompareFunction.lessEqual
		depthStateDescriptor.isDepthWriteEnabled = true
		self.depthState = device.makeDepthStencilState(descriptor:depthStateDescriptor)!

		//self.sortCounter = FrameCounterModel(OnLap:self.OnSortCounterLap)
	}
	
	func OnSortCounterLap(_ sortPerSec:CGFloat)
	{
		self.sortPerSecond = sortPerSec
		print("Sort Per Sec \(sortPerSec)")
	}
	
	
	private class func buildRenderPipelineWithDevice(device: MTLDevice,
													 colorFormat: MTLPixelFormat,
													 depthFormat: MTLPixelFormat,
													 stencilFormat: MTLPixelFormat,
													 sampleCount: Int) throws -> MTLRenderPipelineState {
		let library = try device.makeDefaultLibrary(bundle: Bundle.main)
		
		let vertexFunction = library.makeFunction(name: "splatVertexShader")
		let fragmentFunction = library.makeFunction(name: "splatFragmentShader")
		
		let pipelineDescriptor = MTLRenderPipelineDescriptor()
		pipelineDescriptor.label = "RenderPipeline"
		pipelineDescriptor.vertexFunction = vertexFunction
		pipelineDescriptor.fragmentFunction = fragmentFunction
		
		pipelineDescriptor.rasterSampleCount = sampleCount
		
		let colorAttachment = pipelineDescriptor.colorAttachments[0]!
		colorAttachment.pixelFormat = colorFormat
		colorAttachment.isBlendingEnabled = true
		colorAttachment.rgbBlendOperation = .add
		colorAttachment.alphaBlendOperation = .add
		if Constants.renderFrontToBack {
			colorAttachment.sourceRGBBlendFactor = .oneMinusDestinationAlpha
			colorAttachment.sourceAlphaBlendFactor = .oneMinusDestinationAlpha
			colorAttachment.destinationRGBBlendFactor = .one
			colorAttachment.destinationAlphaBlendFactor = .one
		} else {
			colorAttachment.sourceRGBBlendFactor = .one
			colorAttachment.sourceAlphaBlendFactor = .one
			colorAttachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
			colorAttachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
		}
		pipelineDescriptor.colorAttachments[0] = colorAttachment
		
		pipelineDescriptor.depthAttachmentPixelFormat = depthFormat
		pipelineDescriptor.stencilAttachmentPixelFormat = stencilFormat
		
		//pipelineDescriptor.maxVertexAmplificationCount = maxViewCount
		
		return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
	}
	
	func clearSplats()
	{
		splatBuffer.count = 0
	}
	
	func loadSplats(splatElements:[SplatElement]) throws 
	{
		try splatBuffer.ensureCapacity(splatBuffer.count + splatElements.count)
		splatBuffer.append(splatElements)
	}
	
	
	public func render(camera: CameraDescriptor, to renderEncoder: MTLRenderCommandEncoder,params:SplatRenderParams)
	{
		guard splatBuffer.count != 0 else { return }
		
		var uniforms = Uniforms(projectionMatrix: camera.projectionMatrix,
								viewMatrix: camera.viewMatrix,
								screenSize: SIMD2(x: UInt32(camera.screenSize.x), y: UInt32(camera.screenSize.y)))
			
		if !sorting 
		{
			resortIndices(camera:camera)
		}
		
		//	gr: move this to scene/actor render
		renderEncoder.pushDebugGroup("Draw Splat")
		
		renderEncoder.setRenderPipelineState(pipelineState)
		
		renderEncoder.setDepthStencilState(depthState)
		
		renderEncoder.setVertexBytes( &uniforms, length: MemoryLayout<Uniforms>.size, index: BufferIndex.uniforms.rawValue)
		renderEncoder.setVertexBuffer(splatBuffer.buffer, offset: 0, index: BufferIndex.splat.rawValue)
		renderEncoder.setVertexBuffer(orderBuffer.buffer, offset: 0, index: BufferIndex.order.rawValue)
		
		let renderParamsFragBufferIndex = 0
		var paramsMutable = params
		renderEncoder.setFragmentBytes( &paramsMutable, length: MemoryLayout<SplatRenderParams>.size, index: renderParamsFragBufferIndex)
		
		
		renderEncoder.drawPrimitives(type: .triangleStrip,
									 vertexStart: 0,
									 vertexCount: 4,
									 instanceCount: splatBuffer.count)
		
		renderEncoder.popDebugGroup()
	}
	
	// Set indicesPrime to a depth-sorted version of indices, then swap indices and indicesPrime
	public func resortIndices(camera:CameraDescriptor) 
	{
		if Constants.useAccelerateForSort 
		{
			resortIndicesViaAccelerate(camera: camera)
		}
		else 
		{
			resortIndicesOnCPU(camera: camera)
		}
	}
	
	public func resortIndicesOnCPU(camera:CameraDescriptor) 
	{
		guard !sorting else { return }
		sorting = true
		
		let splatCount = splatBuffer.count
		
		if orderAndDepthTempSort.count != splatCount {
			orderAndDepthTempSort = Array(repeating: SplatIndexAndDepth(index: .max, depth: 0), count: splatCount)
			for i in 0..<splatCount {
				orderAndDepthTempSort[i].index = UInt32(i)
			}
		}
		
		let cameraWorldForward = camera.worldForward
		let cameraWorldPosition = camera.worldPositon
		
		Task(priority: .high) {
			defer {
				//sortCounter.Add()
				sorting = false
			}
			
			// We maintain the old order in indicesAndDepthTempSort in order to provide the opportunity to optimize the sort performance
			for i in 0..<splatCount {
				let index = orderAndDepthTempSort[i].index
				let splatPosition = splatBuffer.values[Int(index)].position
				let splatPositionUnpacked = SIMD3<Float>(splatPosition.x, splatPosition.y, splatPosition.z)
				if Constants.sortByDistance {
					orderAndDepthTempSort[i].depth = (splatPositionUnpacked - cameraWorldPosition).lengthSquared
				} else {
					orderAndDepthTempSort[i].depth = dot(splatPositionUnpacked, cameraWorldForward)
				}
			}
			
			if Constants.renderFrontToBack {
				orderAndDepthTempSort.sort { $0.depth < $1.depth }
			} else {
				orderAndDepthTempSort.sort { $0.depth > $1.depth }
			}
			
			do {
				orderBufferPrime.count = 0
				try orderBufferPrime.ensureCapacity(splatCount)
				for i in 0..<splatCount {
					orderBufferPrime.append(orderAndDepthTempSort[i].index)
				}
				
				swap(&orderBuffer, &orderBufferPrime)
			} catch {
				// TODO: report error
			}
		}
	}
	
	public func resortIndicesViaAccelerate(camera:CameraDescriptor)
	{
		guard !sorting else { return }
		sorting = true
		let splatCount = splatBuffer.count
		
		if orderBufferTempSort.count != splatCount {
			do {
				try orderBufferTempSort.ensureCapacity(splatCount)
				orderBufferTempSort.count = splatCount
				try depthBufferTempSort.ensureCapacity(splatCount)
				depthBufferTempSort.count = splatCount
				
				for i in 0..<splatCount {
					orderBufferTempSort.values[i] = UInt(i)
				}
			} catch {
				// TODO: report error
				sorting = false
				
				return
			}
		}
		
		let cameraWorldForward = camera.worldForward
		let cameraWorldPosition = camera.worldPositon
		
		Task(priority: .high) {
			defer {
				sorting = false
				//sortCounter.Add()
			}
			
			// TODO: use Accelerate to calculate the depth
			// We maintain the old order in indicesTempSort in order to provide the opportunity to optimize the sort performance
			for index in 0..<splatCount {
				let splatPosition = splatBuffer.values[Int(index)].position
				let splatPositionUnpacked = SIMD3<Float>(splatPosition.x, splatPosition.y, splatPosition.z)
				if Constants.sortByDistance {
					depthBufferTempSort.values[index] = (splatPositionUnpacked - cameraWorldPosition).lengthSquared
				} else {
					depthBufferTempSort.values[index] = dot(splatPositionUnpacked, cameraWorldForward)
				}
			}
			
			vDSP_vsorti(depthBufferTempSort.values,
						orderBufferTempSort.values,
						nil,
						vDSP_Length(splatCount),
						Constants.renderFrontToBack ? 1 : -1)
			
			do {
				orderBufferPrime.count = 0
				try orderBufferPrime.ensureCapacity(splatCount)
				for i in 0..<splatCount {
					orderBufferPrime.append(UInt32(orderBufferTempSort.values[i]))
				}
				
				swap(&orderBuffer, &orderBufferPrime)
			} catch {
				// TODO: report error
			}
		}
	}
}


protocol MTLIndexTypeProvider {
	static var asMTLIndexType: MTLIndexType { get }
}

extension UInt32: MTLIndexTypeProvider {
	static var asMTLIndexType: MTLIndexType { .uint32 }
}
extension UInt16: MTLIndexTypeProvider {
	static var asMTLIndexType: MTLIndexType { .uint16 }
}


private extension SIMD3 where Scalar: BinaryFloatingPoint, Scalar.RawSignificand: FixedWidthInteger {
	var normalized: SIMD3<Scalar> {
		self / Scalar(sqrt(lengthSquared))
	}
	
	var lengthSquared: Scalar {
		x*x + y*y + z*z
	}
	
	func vector4(w: Scalar) -> SIMD4<Scalar> {
		SIMD4<Scalar>(x: x, y: y, z: z, w: w)
	}
	
	static func random(in range: Range<Scalar>) -> SIMD3<Scalar> {
		Self(x: Scalar.random(in: range), y: .random(in: range), z: .random(in: range))
	}
}

private extension SIMD3<Float> {
	var sRGBToLinear: SIMD3<Float> {
		SIMD3(x: pow(x, 2.2), y: pow(y, 2.2), z: pow(z, 2.2))
	}
}

private extension SIMD4 where Scalar: BinaryFloatingPoint {
	var xyz: SIMD3<Scalar> {
		.init(x: x, y: y, z: z)
	}
}
