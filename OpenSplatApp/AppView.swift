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
import PopCommon


struct CameraImageCache
{
	//	cache
	var render : Image?
	var iterationsAtRender : Int?
	var error : Error?
	var groundTruth : Image?
	var cameraMeta : OpenSplat_CameraMeta?
	var cameraActor : PopCamera?	//	cached from cameraMeta
}

struct SplatScene : PopScene
{
	var actors: [any PopActor]
	{
		let splatActors : [any PopActor] = [splatAsset].compactMap{ $0 }
		return otherActors + splatActors 
	}
	
	var splatAsset : OpenSplatSplatAsset?
	var otherActors : [any PopActor] = []
	//var cameras : [TrainingCameraActor]
}

class OpenSplatSplatAsset : PopActor
{
	var id = UUID()
	
	@Published var translation = simd_float3(0,0,0)
	@Published var rotationPitch = Angle(degrees: 0)
	@Published var rotationYaw = Angle(degrees: 0)
	
	var splatAssetRenderer : SplatAssetRenderer?
	var newSplats : [SplatElement]? = nil
	@Published var renderParams = SplatRenderParams()

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
		
		let localToViewTransform = camera.worldToCameraTransform * self.localToWorldTransform
		
		let camDescription = CameraDescriptor(projectionMatrix: projectionMatrix, viewMatrix: localToViewTransform, screenSize:camera.viewportPixelSizeSimd)
		
		let renderer = try GetSplatAssetRenderer(metalKitView: metalView)
		renderer.render(camera: camDescription, to: commandEncoder, params:renderParams)
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
	@StateObject var floorAsset = FloorPlaneActor()
	func GetScene() -> Binding<SplatScene>
	{
		Binding<SplatScene>( 
			get:
				{
					let trainerCameras = showTrainerCameras ? self.cameraRender.values.compactMap{$0.cameraActor} : []
					let inputCameras = showInputCameras ? self.inputDataCameras : []
					let others = showInputPoints ? [self.trainer.inputPointsActor].compactMap{$0} : []
					return SplatScene(splatAsset: splatAsset, otherActors: [renderCamera,floorAsset]+trainerCameras+inputCameras+others )
				},
			set:{_ in}
		)
	}
	
	@State var someError : Error?
	@State var renderCamera = PopCamera()
	@State var renderThroughCameraIndex : Int?
	@StateObject var trainer : OpenSplatTrainer
	@State var trainerState = OpenSplat_TrainerState()
	@State var cameraRender = [Int:CameraImageCache]()
	@State var cameraUserView = [Int:CameraUserView]()
	var noImageImage = Image(systemName: "questionmark.square.dashed")
	var cameraUserViewDefault : CameraUserView = .Render
	
	@State var renderImageSize = CGSize(width: 400, height: 400)
	@State var showInputCameras : Bool = false
	@State var showInputImages : Bool = false
	@State var showTrainerCameras : Bool = true
	@State var showInputPoints : Bool = false
	
	
	//	temp to verify input vs API
	var inputDataCameras : [PopCamera]
	{
		return trainer.inputCameras.map
		{
			return PopCamera(localToWorldTransform: $0.localToWorld )
		}
	}
	
	var body: some View 
	{
		HSplitView
		{
			TrainingView()
			VStack
			{
				VSplitView
				{
					ScrollView(.horizontal)
					{
						HStack
						{
							ForEach(Array(trainer.inputCameraImages), id:\.key)
							{
								let image = $0.value
								Image(nsImage:image)
									.resizable()
									.scaledToFit()
							}
						}
					}
					.frame(minHeight: 100)
				
					CameraGridView()
				}
			}
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
								UpdateCameraMeta(cameraIndex: cameraIndex)
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
				let name = cameraRender[cameraIndex]?.cameraMeta?.name ?? "Camera \(cameraIndex)"
				let iterations = cameraRender[cameraIndex]?.cameraMeta?.TrainedIterations ?? 0
				let renderIterations = cameraRender[cameraIndex]?.iterationsAtRender ?? 0
				
				Text(name)
					.padding(4)
					.background(.black.opacity(0.5))
					.foregroundStyle(.white)
				Text("\(iterations) iterations (viewing #\(renderIterations))")
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
					
					Button(action:{ ToggleRenderThroughCamrea(cameraIndex:cameraIndex) })
					{
						Image(systemName: "eye")
							.resizable()
							.scaledToFit()
							.foregroundStyle( renderThroughCameraIndex == cameraIndex ? .white : .black )
					}
					.buttonStyle(PlainButtonStyle())
					
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
			UpdateCameraMeta(cameraIndex: cameraIndex)
			UpdateCameraRenderImage(cameraIndex: cameraIndex)
			
			//	if there's no ground truth (eg, fetched before they were ready) try and reload it now
			if cameraRender[cameraIndex]?.groundTruth == nil
			{
				UpdateGroundTruthImage(cameraIndex:cameraIndex)
			}
		}
	}
	
	func GetRenderCamera() -> Binding<PopCamera>
	{
		return Binding(
			get:
				{
					var cameraEntry = renderThroughCameraIndex.map{ cameraRender[$0] }
					var useCamera : PopCamera? = cameraEntry??.cameraActor
					return useCamera ?? renderCamera
				},
				set:{_ in}
		   )
	}
	
	@ViewBuilder func TrainingView() -> some View
	{
		MetalSceneView(scene: GetScene(), camera:GetRenderCamera(), showGizmosOnActors: [])
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
					VStack(alignment: .leading)
					{
						TrainingViewControls()
					}
					.padding(10)
					.background(.black.opacity(0.5))
				}
				.frame(maxWidth: .infinity,maxHeight: .infinity,alignment: .topLeading)
			}

	}
	
	func ToggleRenderThroughCamrea(cameraIndex:Int)
	{
		if renderThroughCameraIndex == cameraIndex
		{
			renderThroughCameraIndex = nil
			return
		}
		
		renderThroughCameraIndex = cameraIndex
		
		
		//	use this once we can extract rotation from transform in PopActor
		guard let cameraTransform = cameraRender[cameraIndex]?.cameraMeta?.localToWorld else
		{
			PlaySystemBeep()
			return
		}
		self.renderCamera.localToWorldTransform = cameraTransform
		self.renderCamera.override_localToWorldTransform = nil
		
	}
	
	func OnClickedUpdateSplats()
	{
		Task
		{
			await UpdateSplats()
		}
	}
	
	func UpdateSplats() async
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
		
		Toggle(isOn:$showInputCameras){	Text("Show Input Cameras")	}
		Toggle(isOn:$showInputImages){	Text("Show Input Images")	}
		Toggle(isOn:$showTrainerCameras){	Text("Show trainer Cameras")	}
		Toggle(isOn:$showInputPoints){	Text("Show input points")	}
		
		Slider(value: $splatAsset.renderParams.minAlpha, in:0...1)
		{
			Text("Min-Alpha")
				.foregroundStyle(.white)
				.frame(width: 100)
		}
		
		Slider(value: $splatAsset.renderParams.clipMaxAlpha, in:0...1)
		{
			Text("Clip-Max-Alpha")
				.foregroundStyle(.white)
				.frame(width: 100)
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
		else
		{
			Text( trainer.status )
				.padding(5)
				.background(.green)
				.foregroundStyle(.white)
		}
		/*
		else if trainer.isTraining
		{
			Text("Training...")
				.padding(5)
				.background(.green)
				.foregroundStyle(.white)
		}
		else if !trainer.isTraining
		{
			Text("Training Finished.")
				.padding(5)
				.background(.black)
				.foregroundStyle(.white)
		}
		*/
		Text("\(trainerState.IterationsCompleted) Steps Completed (\(trainerState.SplatCount) splats)")
			.padding(5)
			.background(.black)
			.foregroundStyle(.white)
	}
	
	
	func UpdateCameraRenderImage(cameraIndex:Int)
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

				//	update the iteration counter at this render
				let meta = try? trainer.GetCameraMeta(cameraIndex: cameraIndex)
				cameraCache.iterationsAtRender = meta.map{ Int($0.TrainedIterations) }

				cameraRender[cameraIndex] = cameraCache
				
				//	switch whenever new data appears
				cameraUserView[cameraIndex] = .Render
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
				
				//	switch whenever new data appears
				cameraUserView[cameraIndex] = .GroundTruth

			}
			catch
			{
				var cameraCache = cameraRender[cameraIndex] ?? CameraImageCache()
				cameraCache.error = error
				cameraRender[cameraIndex] = cameraCache
			}	
		}
	}
	
	
	func UpdateCameraMeta(cameraIndex:Int)
	{
		Task
		{
			await UpdateCameraMetaAsync(cameraIndex: cameraIndex)	
		}
	}
	
	func UpdateCameraMetaAsync(cameraIndex:Int) async
	{
		do
		{
			let meta = try await trainer.GetCameraMeta(cameraIndex: cameraIndex)
			var cameraCache = cameraRender[cameraIndex] ?? CameraImageCache()
			cameraCache.cameraMeta = meta
			cameraCache.cameraActor = cameraCache.cameraActor ?? PopCamera(localToWorldTransform: meta.localToWorld )
			cameraRender[cameraIndex] = cameraCache
		}
		catch
		{
			if var cameraCache = cameraRender[cameraIndex]
			{
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
			await UpdateSplats()
			
			//	update state of cameras (iteration counts)
			for c in 0..<Int(self.trainerState.CameraCount)
			{
				await UpdateCameraMetaAsync(cameraIndex: c)
			}
			
			let ms = 500
			try? await Task.sleep(nanoseconds: UInt64(ms * 1_000_000))
		}
	}
}

struct AppView: View 
{
	//@State var trainer = OpenSplatTrainer(projectPath: "/Users/graham/Downloads/banana")
	@State var trainer = OpenSplatTrainer(projectPath: "/Users/graham/Downloads/iphonecap")
	

	var body: some View 
	{
		TrainerView(trainer: trainer)
	}
}

