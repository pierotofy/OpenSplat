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
	var renderImageSize = CGSize(width: 800, height: 600)
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
		Button(action:OnClickedUpdateImage)
		{
			Text("Capture new image from camera \(cameraIndex)")
		}
	}
	
	@ViewBuilder func TrainingView() -> some View
	{
		lastRenderImage
			//.resizable()
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
		do
		{
			let renderWidth = Int(renderImageSize.width)
			let renderHeight = Int(renderImageSize.height)
			let image = try trainer.RenderCamera(cameraIndex: cameraIndex, width:renderWidth, height: renderHeight)
			let imageNs = NSImage(cgImage:image, size: .zero)
			lastRenderImage = Image(nsImage: imageNs)
		}
		catch
		{
			someError = error
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

