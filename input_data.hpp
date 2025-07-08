#ifndef INPUTDATA_H
#define INPUTDATA_H

#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <opencv2/calib3d.hpp>
#include <torch/torch.h>

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
	
	torch::Tensor	GetProjectionMatrix() const;
};

enum CameraType { Perspective };
struct Camera
{
	Camera(){};
	Camera(CameraIntrinsics intrinsics,
		   const torch::Tensor &camToWorld,	//	extrinsics 
		   const std::string &filePath);

	int id = -1;
	CameraIntrinsics intrinsics;
   
    torch::Tensor camToWorld;
    std::string filePath = "";
    CameraType cameraType = CameraType::Perspective;

    torch::Tensor projectionMatrix;	//	formerly "K". Only here as a cache
    torch::Tensor image;

    std::unordered_map<int, torch::Tensor> imagePyramids;
	
	torch::Tensor		getIntrinsicsMatrix();
	bool				hasDistortionParameters();
	std::vector<float>	undistortionParameters();	//	opencv distortion coefficients
	
	torch::Tensor		GetCamToWorldRotation();
	torch::Tensor		GetCamToWorldTranslation();
	torch::Tensor		GetWorldToCamRotation();
	torch::Tensor		GetWorldToCamTranslation();
	
	torch::Tensor		getImage(int downscaleFactor);
	void				loadImageFromFilename(float downscaleFactor);	//	refactor this; dont make Camera responsible for i/o
	void				loadImage(cv::Mat& RgbPixels,float downscaleFactor);	//	loads pixels and resizes intrinsics to fit image

};

struct Points{
    torch::Tensor xyz;
    torch::Tensor rgb;
};
struct InputData{
    std::vector<Camera> cameras;
    float scale;
    torch::Tensor translation;
    Points points;

    std::tuple<std::vector<Camera>, Camera *> getCameras(bool validate, const std::string &valImage = "random");

    void saveCameras(const std::string &filename, bool keepCrs);
};
// The colmapImageSourcePath is only used in Colmap. In other methods, this path is ignored.
InputData inputDataFromX(const std::string& projectRoot, const std::string& colmapImageSourcePath = "");

#endif
