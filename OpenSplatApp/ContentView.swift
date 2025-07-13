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


class OpenSplatSplatRenderer : ContentRenderer, ObservableObject
{
	//	cache splats
	var splats = [OpenSplat_Splat]()
	
	public func Draw(metalView: MTKView, size: CGSize, commandEncoder: any MTLRenderCommandEncoder) throws 
	{
		metalView.clearColor = MTLClearColor(red: 0,green: 1,blue: 1,alpha: 1)
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
				case .Render:		return Image(systemName: "photo.circle")
				case .GroundTruth:	return Image(systemName: "photo.circle.fill")
			}
		}
	}
	
	@StateObject var trainer : OpenSplatTrainer
	@StateObject var renderer = OpenSplatSplatRenderer()
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
				.overlay
			{
				VStack(alignment: .leading)
				{
					TrainingStateView()
					Spacer()
				}
				.frame(maxWidth: .infinity,maxHeight: .infinity,alignment: .topLeading)
			}
			CameraGridView()
		}
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
		}
		.onTapGesture 
		{
			UpdateRenderImage(cameraIndex: cameraIndex)
			cameraUserView[cameraIndex] = .Render
		}
	}
	
	@ViewBuilder func TrainingView() -> some View
	{
		MetalView(contentRenderer: renderer)
	}
	
	@ViewBuilder func TrainingStateView() -> some View
	{
		if let error = trainer.trainingError
		{
			Text("Error: \(error.localizedDescription)")
				.padding(5)
				.background(.red)
				.foregroundStyle(.white)
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
}

struct ContentView: View 
{
	@State var trainer = OpenSplatTrainer(projectPath: "/Users/graham/Downloads/banana")

	var body: some View 
	{
		TrainerView(trainer: trainer)
	}
}

