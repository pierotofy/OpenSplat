//
//  ContentView.swift
//  OpenSplatApp
//
//  Created by Graham Reeves on 13/07/2025.
//

import SwiftUI




struct TrainerView : View 
{
	@StateObject var trainer : OpenSplatTrainer
	@State var lastRenderImage : Image = Image(systemName: "clock.circle.fill")
	@State var cameraIndex = 0
	@State var renderImageSize = CGSize(width: 400, height: 400)
	@State var someError : Error?
	
	var body: some View 
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
		.onGeometryChange(for: CGSize.self) 
		{
			proxy in
			proxy.size
		} 
		action: 
		{
			self.renderImageSize = $0
		}
		
		
		HStack
		{
			var cameraIndexFloat = Binding<Float>( get: {Float(cameraIndex)}, set:{cameraIndex = Int($0)
				print("new camera \(cameraIndex)")
				OnClickedUpdateImage()
			} )
			Slider(value: cameraIndexFloat, in: 0...10)
			{
				editing in
				//print("changed \($0)")
				OnClickedUpdateImage()
			}
			
			Button(action:OnClickedUpdateImage)
			{
				let w = Int(renderImageSize.width)
				let h = Int(renderImageSize.height)
				Text("Capture new image from camera \(cameraIndex) (at \(w)x\(h))")
			}
		}
	}
	
	@ViewBuilder func TrainingView() -> some View
	{
		lastRenderImage
			.resizable()
			.scaledToFit()
			.frame(maxWidth:.infinity,maxHeight: .infinity)
			.background
		{
			Rectangle()
				.fill(.blue)
		}
		.foregroundStyle(.white)
		
	}
	
	@ViewBuilder func TrainingStateView() -> some View
	{
		if let error = someError
		{
			Text("Error: \(error.localizedDescription)")
				.padding(5)
				.background(.red)
				.foregroundStyle(.white)
		}
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
	
	func OnClickedUpdateImage()
	{
		UpdateImage(cameraIndex: cameraIndex)
	}
	
	func UpdateImage(cameraIndex:Int)
	{
		//	todo: make a queue!
		Task
		{
			do
			{
				let renderWidth = Int(renderImageSize.width)
				let renderHeight = Int(renderImageSize.height)
				
				
				let image = try await trainer.RenderCamera(cameraIndex: cameraIndex, width:renderWidth, height: renderHeight)
				let imageNs = NSImage(cgImage:image, size: .zero)
				lastRenderImage = Image(nsImage: imageNs)
			}
			catch
			{
				someError = error
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

