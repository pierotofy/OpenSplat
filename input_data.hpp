#ifndef INPUTDATA_H
#define INPUTDATA_H

#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <opencv2/calib3d.hpp>
#include <torch/torch.h>
#include "trainer_params.hpp"

struct CameraIntrinsics
{
	//	focal length & center; in pixels, relative to image size
	float imageWidth = 0;
	float imageHeight = 0;
	float fx = 0;
	float fy = 0;
	float cx = 0;
	float cy = 0;
	
	//	distortion parameters
	float k1 = 0;
	float k2 = 0;
	float k3 = 0;
	float p1 = 0;
	float p2 = 0;
	
	bool				HasDistortionParameters();
	//	todo: return std::array<8>
	std::vector<float>	GetOpencvUndistortionParameters();	//	opencv distortion coefficients
	void				RemoveDistortionParameters();		//	remove the _values_, but without undistorting the image
	void				ScaleTo(int Width,int Height);
	torch::Tensor		GetProjectionMatrix() const;
	
};

class CameraTransform
{
public:
	torch::Tensor	camToWorld;		//	todo: initialise to identity!
	
	torch::Tensor	GetCamToWorldRotation() const;
	torch::Tensor	GetCamToWorldTranslation() const;
	torch::Tensor	GetWorldToCamRotation() const;
	torch::Tensor	GetWorldToCamTranslation() const;
};

enum CameraType { Perspective };
struct Camera
{
	Camera(){};
	Camera(CameraIntrinsics intrinsics,
		   const torch::Tensor &camToWorld,	//	extrinsics 
		   std::filesystem::path cameraImageFilename	//	path's filename also serves as name
		   );	

	int id = -1;
	CameraIntrinsics intrinsics;
   
	CameraTransform camToWorld;
	std::filesystem::path cameraImagePath;
    CameraType cameraType = CameraType::Perspective;

    torch::Tensor projectionMatrix;	//	formerly "K". Only here as a cache

	std::string			getName() const;	//	name is filename part of path
	torch::Tensor		getImage(int downscaleFactor);
	cv::Mat				getOpencvRgbImageStretched(int Width,int Height);
	void				loadImageFromFilename(float downscaleFactor);	//	refactor this; dont make Camera responsible for i/o
	void				loadImage(cv::Mat& RgbPixels,float downscaleFactor);	//	loads pixels and resizes intrinsics to fit image


private:
	torch::Tensor image;			//	rgb
	std::unordered_map<int, torch::Tensor> imagePyramids;
};

struct Points{
    torch::Tensor xyz;
    torch::Tensor rgb;
};
struct InputData
{
    std::vector<Camera> cameras;
    float scale;
    torch::Tensor translation;
    Points points;

	//	remove camera from the training data (typically for application to use for validation)
	std::shared_ptr<Camera>	PopCamera(std::string_view CameraImageName=OpenSplat::randomValidationImageName);
	Camera&					GetCamera(int CameraIndex);
	Camera&					GetCamera(std::string_view CameraName);
	
    void saveCamerasJson(const std::string &filename, bool keepCrs);
	
	//	this finds the center & bounds of the camera poses and moves all
	//	points to be centered in the middle. It then normalises all points to be -1...1
	//	transform is saved to scale&translation for future restoration
	void NormalisePoints();
	
	std::vector<std::string>	GetCameraNames();
};
// The colmapImageSourcePath is only used in Colmap. In other methods, this path is ignored.
InputData inputDataFromX(const std::string& projectRoot, const std::string& colmapImageSourcePath = "");

#endif
