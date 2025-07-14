//
//  ContentView.swift
//  OpenSplatApp
//
//  Created by Graham Reeves on 13/07/2025.
//

import SwiftUI
import PopMetalView
import OpenSplat
import MetalKit


struct CameraImageCache
{
	//	cache
	var render : Image?
	var error : Error?
	var groundTruth : Image?
}

struct SplatScene : PopScene
{
	var actors: [any PopActor]
	{
		let splatActors : [any PopActor] = [splatAsset].compactMap{ $0 }
		return splatActors
	}
	
	var splatAsset : OpenSplatSplatAsset?
	//var cameras : [TrainingCameraActor]
}

class OpenSplatSplatAsset : PopActor
{
	var id = UUID()
	
	var translation = simd_float3(0,0,0)
	var rotationPitch = Angle(degrees: 0)
	var rotationYaw = Angle(degrees: 0)
	
	var splatAssetRenderer : SplatAssetRenderer?
	var newSplats : [SplatElement]? = nil

	var splats : [OpenSplat_Splat]
	{
		get	{	[]	}
		set
		{
			newSplats = newValue.map
			{
				oss in
				let pos = MTLPackedFloat3Make(oss.x,oss.y,oss.z)
				let scale = SplatElement.FixScale( [oss.scalex, oss.scaley, oss.scalez] )
				let rotation = simd_quatf( ix: oss.rotx, iy: oss.roty, iz: oss.rotz, r: oss.rotw )
				let alpha = SplatElement.FixOpacity(oss.opacity)
				let sh = [oss.dc0,oss.dc1,oss.dc2]
				let rgb = SplatElement.FirstOrderSphericalHarmonicsToRgb(sh)
				let colour = SIMD4<Float>( rgb[0], rgb[1], rgb[2], alpha )
				return SplatElement(position: pos, color: colour, scale: scale, rotation: rotation)
			}
		}
	}
	
	
	func Render(camera: PopMetalView.PopRenderCamera, metalView: MTKView, commandEncoder: any MTLRenderCommandEncoder) throws 
	{
		//metalView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
		metalView.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
		//metalView.clearDepth = 0.0
		
		let fovy = Angle(degrees: 65)
		let projectionMatrix = matrix_perspective_right_hand(fovyRadians: Float(fovy.radians),
															 aspectRatio: Float(camera.viewportPixelSize.width / camera.viewportPixelSize.height),
															 nearZ: 0.1,
															 farZ: 100.0)
		
		
		let camDescription = CameraDescriptor(projectionMatrix: projectionMatrix, viewMatrix: self.localToWorldTransform, screenSize:camera.viewportPixelSizeSimd)
		
		let renderer = try GetSplatAssetRenderer(metalKitView: metalView)
		renderer.render(camera: camDescription, to: commandEncoder)
	}
		
	func GetSplatAssetRenderer(metalKitView:MTKView) throws -> SplatAssetRenderer
	{
		if splatAssetRenderer == nil
		{
			self.splatAssetRenderer = try SplatAssetRenderer(device: metalKitView.device!,
															 colorFormat: metalKitView.colorPixelFormat,
															 depthFormat: metalKitView.depthStencilPixelFormat,
															 stencilFormat: metalKitView.depthStencilPixelFormat,
															 sampleCount: metalKitView.sampleCount)
			
		}
		guard let splatAssetRenderer else
		{
			throw OpenSplatError("Failed to create splat asset renderer")
		}
		
		//	load new data
		if let newSplats
		{
			splatAssetRenderer.clearSplats()
			try splatAssetRenderer.loadSplats(splatElements: newSplats)
		}
		newSplats = nil
		
		return splatAssetRenderer
	}
}

struct TrainerView : View
{
	enum CameraUserView
	{
		case Render, GroundTruth
		
		func icon() -> Image
		{  
			switch self 
			{
				case .Render:		return Image(systemName: "sparkles.tv")
				case .GroundTruth:	return Image(systemName: "photo")
			}
		}
	}

	@StateObject var splatAsset = OpenSplatSplatAsset()
	var scene : PopScene
	{
		return SplatScene(splatAsset: splatAsset)
	}
	
	@State var someError : Error?
	@StateObject var trainer : OpenSplatTrainer
	@State var trainerState = OpenSplat_TrainerState()
	@State var cameraRender = [Int:CameraImageCache]()
	@State var cameraUserView = [Int:CameraUserView]()
	var noImageImage = Image(systemName: "questionmark.square.dashed")
	var cameraUserViewDefault : CameraUserView = .Render
	
	@State var renderImageSize = CGSize(width: 400, height: 400)
	
	var body: some View 
	{
		HSplitView
		{
			TrainingView()
			CameraGridView()
		}
		.task(AutoUpdateThread)
	}
	
	@ViewBuilder func CameraGridView() -> some View
	{
		let cameraCount = 16
		let rowCount = Int(sqrt(Double(cameraCount)))
		let rows = 0..<rowCount
		let colCount = rowCount
		let cols = 0..<colCount
		Grid 
		{
			ForEach(rows, id: \.self)
			{
				rowIndex in
				
				GridRow 
				{
					ForEach(cols, id: \.self)
					{
						colIndex in
						let cameraIndex = (rowIndex * colCount) + colIndex
						CameraView(cameraIndex: cameraIndex)
							.onGeometryChange(for: CGSize.self) 
							{
								proxy in
								proxy.size
							} 
							action: 
							{
								self.renderImageSize = $0
							}
							.onAppear
							{
								UpdateGroundTruthImage(cameraIndex: cameraIndex)
							}
					}
				}
			}
		}
	}
	
	@ViewBuilder func CameraView(cameraIndex:Int) -> some View
	{
		let viewOption = cameraUserView[cameraIndex] ?? cameraUserViewDefault
		let renderImage = (cameraRender[cameraIndex].map{ $0.render } ?? noImageImage) ?? noImageImage
		let groundTruthImage = (cameraRender[cameraIndex].map{ $0.groundTruth } ?? noImageImage ) ?? noImageImage
		let drawImage = viewOption == .GroundTruth ? groundTruthImage : renderImage
		
		drawImage
			.resizable()
			.scaledToFit()
			.foregroundStyle(.white)
			.frame(minWidth: 50,minHeight: 50)
			.frame(maxWidth: .infinity,maxHeight: .infinity)
			.background
		{
			Rectangle()
				.fill(.blue)
		}
		.overlay
		{
			VStack
			{
				Text("Camera \(cameraIndex)")
					.padding(4)
					.background(.black.opacity(0.5))
					.foregroundStyle(.white)
				if let error = cameraRender[cameraIndex]?.error
				{
					Text("Error: \(error.localizedDescription)")
						.padding(5)
						.background(.red)
						.foregroundStyle(.white)
				}
				Spacer()
				HStack(alignment: .bottom)
				{
					Spacer()
					Button(action:{ cameraUserView[cameraIndex] = .GroundTruth })
					{
						CameraUserView.GroundTruth.icon()
							.resizable()
							.scaledToFit()
							.foregroundStyle( viewOption == .GroundTruth ? .white : .black )
					}
					.buttonStyle(PlainButtonStyle())
					Button(action:{ cameraUserView[cameraIndex] = .Render })
					{
						CameraUserView.Render.icon()
							.resizable()
							.scaledToFit()
							.foregroundStyle( viewOption == .Render ? .white : .black )
					}
					.buttonStyle(PlainButtonStyle())
				}
				.frame(maxHeight:20)
			}
			.padding(10)
		}
		.onTapGesture 
		{
			UpdateRenderImage(cameraIndex: cameraIndex)
			cameraUserView[cameraIndex] = .Render
		}
	}
	
	@ViewBuilder func TrainingView() -> some View
	{
		MetalSceneView(scene: scene, showGizmosOnActors: [splatAsset.id])
			.background
			{
				Rectangle()
					.foregroundStyle(.image(Image("TransparentBackground")))
			}
			.overlay
			{
				VStack(alignment: .leading)
				{
					TrainingStateView()
					Spacer()
					TrainingViewControls()
				}
				.frame(maxWidth: .infinity,maxHeight: .infinity,alignment: .topLeading)
			}

	}
	
	func OnClickedUpdateSplats()
	{
		Task
		{
			do
			{
				let splats = try await trainer.GetSplats()
				splatAsset.splats = splats
			}
			catch
			{
				self.someError = error
			}
		}
	}
	
	func OnClickedUpdateState()
	{
		do
		{
			trainerState = try trainer.GetState()
		}
		catch
		{
			self.someError = error
		}
	}
	
	@ViewBuilder func TrainingViewControls() -> some View
	{
		Button(action:OnClickedUpdateSplats)
		{
			Text("Update splats")
		}
	}
	
	@ViewBuilder func TrainingStateView() -> some View
	{
		if let error = someError
		{
			Text("Error: \(error.localizedDescription)")
				.padding(5)
				.background(.red)
				.foregroundStyle(.white)
				.onTapGesture {
					someError = nil
				}
		}
		
		if let error = trainer.trainingError
		{
			Text("Error: \(error.localizedDescription)")
				.padding(5)
				.background(.red)
				.foregroundStyle(.white)
				.onTapGesture {
					trainer.trainingError = nil
				}
		}
		else if trainer.isTraining
		{
			Text("Training...")
				.padding(5)
				.background(.green)
				.foregroundStyle(.white)
		}
		else if trainer.isTraining
		{
			Text("Training Finished.")
				.padding(5)
				.background(.black)
				.foregroundStyle(.white)
		}
		
		Text("\(trainerState.IterationsCompleted) Steps Completed (\(trainerState.SplatCount) splats)")
			.padding(5)
			.background(.black)
			.foregroundStyle(.white)
	}
	
	
	func UpdateRenderImage(cameraIndex:Int)
	{
		Task
		{
			do
			{
				let renderWidth = Int(renderImageSize.width)
				let renderHeight = Int(renderImageSize.height)
				
				
				let image = try await trainer.RenderCamera(cameraIndex: cameraIndex, width:renderWidth, height: renderHeight)
				let imageNs = NSImage(cgImage:image, size: .zero)
				var cameraCache = cameraRender[cameraIndex] ?? CameraImageCache()
				cameraCache.render = Image(nsImage: imageNs)
				cameraCache.error = nil
				cameraRender[cameraIndex] = cameraCache
			}
			catch
			{
				var cameraCache = cameraRender[cameraIndex] ?? CameraImageCache()
				cameraCache.error = error
				cameraRender[cameraIndex] = cameraCache
			}	
		}
	}
	
	func UpdateGroundTruthImage(cameraIndex:Int)
	{
		Task
		{
			do
			{
				let renderWidth = Int(renderImageSize.width)
				let renderHeight = Int(renderImageSize.height)
				
				
				let image = try await trainer.GetCameraGroundTruthImage(cameraIndex: cameraIndex, width:renderWidth, height: renderHeight)
				let imageNs = NSImage(cgImage:image, size: .zero)
				var cameraCache = cameraRender[cameraIndex] ?? CameraImageCache()
				cameraCache.groundTruth = Image(nsImage: imageNs)
				cameraCache.error = nil
				cameraRender[cameraIndex] = cameraCache
			}
			catch
			{
				var cameraCache = cameraRender[cameraIndex] ?? CameraImageCache()
				cameraCache.error = error
				cameraRender[cameraIndex] = cameraCache
			}	
		}
	}
	
	@MainActor
	func AutoUpdateThread() async
	{
		while ( !Task.isCancelled )
		{
			OnClickedUpdateState()
			OnClickedUpdateSplats()
			let ms = 200
			try? await Task.sleep(nanoseconds: UInt64(ms * 1_000_000))
		}
	}
}

struct AppView: View 
{
	@State var trainer = OpenSplatTrainer(projectPath: "/Users/graham/Downloads/banana")

	var body: some View 
	{
		TrainerView(trainer: trainer)
	}
}

