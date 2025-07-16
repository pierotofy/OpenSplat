#include <filesystem>
#include <nlohmann/json.hpp>
#include "input_data.hpp"
#include "cv_utils.hpp"
#include "trainer.hpp"	//	NoCameraException

namespace fs = std::filesystem;
using namespace torch::indexing;
using json = nlohmann::json;

namespace ns{ InputData inputDataFromNerfStudio(const std::string &projectRoot); }
namespace cm{ InputData inputDataFromColmap(const std::string &projectRoot, const std::string& imageSourcePath); }
namespace osfm { InputData inputDataFromOpenSfM(const std::string &projectRoot); }
namespace omvg { InputData inputDataFromOpenMVG(const std::string &projectRoot); }

InputData inputDataFromX(const std::string &projectRoot, const std::string& colmapImageSourcePath){
    fs::path root(projectRoot);

    if (fs::exists(root / "transforms.json")){
        return ns::inputDataFromNerfStudio(projectRoot);
    }else if (fs::exists(root / "sparse") || fs::exists(root / "cameras.bin")){
        return cm::inputDataFromColmap(projectRoot, colmapImageSourcePath);
    }else if (fs::exists(root / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM(projectRoot);
    }else if (fs::exists(root / "opensfm" / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM((root / "opensfm").string());
    }else if (fs::exists(root / "sfm_data.json")){
        return omvg::inputDataFromOpenMVG((root).string());
    }
    else{
        throw std::runtime_error("Invalid project folder (must be either a colmap or nerfstudio or openmvg project folder)");
    }
}

Camera::Camera(CameraIntrinsics intrinsics,const torch::Tensor &camToWorld,std::filesystem::path cameraImageFilename) : 
	intrinsics(intrinsics),
	camToWorld(camToWorld), 
	cameraImagePath(cameraImageFilename)
{
}

void CameraIntrinsics::RemoveDistortionParameters()
{
	k1 = 0;
	k2 = 0;
	k3 = 0;
	p1 = 0;
	p2 = 0;
}

void CameraIntrinsics::ScaleTo(int Width,int Height)
{
	//	allowing x & y scale is going to mess up distortion parameters
	//	this would be fine for forward rendering (which doesnt distort)
	//	if there are distortion values here for record of the original camera
	//	they should really have been zeroed out already if they no longer apply
	//	to the image they're attached to
	if ( HasDistortionParameters() )
		throw std::runtime_error("Cannot scale CameraIntrinsics which have distortion parameters");
	
	if ( Width <= 0 || Height <= 0 )
	{
		std::stringstream Error;
		Error << "Cannot scale CameraIntrinsics to " << Width << "x" << Height;
		throw std::runtime_error(Error.str());
	}

	float scalex = static_cast<float>(Width) / imageWidth; 
	float scaley = static_cast<float>(Height) / imageHeight; 
	
	fx *= scalex;
	fy *= scaley;
	cx *= scalex;
	cy *= scaley;
	imageWidth *= scalex;
	imageHeight *= scaley;
	
	imageWidth = Width;
	imageHeight = Height;
}

torch::Tensor CameraIntrinsics::GetProjectionMatrix() const
{
    return torch::tensor({{fx, 0.0f, cx},
                          {0.0f, fy, cy},
                          {0.0f, 0.0f, 1.0f}}, torch::kFloat32);
}


torch::Tensor CameraTransform::GetCamToWorldRotation() const
{
	torch::Tensor R = camToWorld.index({Slice(None, 3), Slice(None, 3)});
	
	// Flip the z and y axes to align with gsplat conventions
	R = torch::matmul(R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f}, R.device())));
	
	return R;
}


torch::Tensor CameraTransform::GetWorldToCamRotation() const
{
	torch::Tensor R = GetCamToWorldRotation();
	
	// worldToCam
	torch::Tensor Rinv = R.transpose(0, 1);
	return Rinv;
}


torch::Tensor CameraTransform::GetWorldToCamTranslation() const
{
	auto Rinv = GetWorldToCamRotation();
	
	torch::Tensor T = GetCamToWorldTranslation();
	torch::Tensor Tinv = torch::matmul(-Rinv, T);
	return Tinv;
}

torch::Tensor CameraTransform::GetCamToWorldTranslation() const
{
	torch::Tensor T = camToWorld.index({Slice(None, 3), Slice(3,4)});
	return T;
}


void Camera::loadImageFromFilename(float downscaleFactor)
{
	auto PathString = cameraImagePath.string();
	std::cout << "Camera Loading Image " << PathString << "..." << std::endl;
	
	cv::Mat cImg = imreadRGB(PathString);
	loadImage( cImg, downscaleFactor );
}

void Camera::loadImage(cv::Mat& RgbPixels,float downscaleFactor)
{
	auto& cImg = RgbPixels;
	
    // Populates image and K, then updates the camera parameters
    // Caution: this function has destructive behaviors
    // and should be called only once
    if (image.numel()) 
		throw std::runtime_error("loadImage already called");
    	
    float rescaleF = 1.0f;
	
    // If camera intrinsics don't match the image dimensions, rescale intrinsics to match pixels
    if (cImg.rows != intrinsics.imageHeight || cImg.cols != intrinsics.imageWidth )
	{
        rescaleF = static_cast<float>(cImg.rows) / static_cast<float>(intrinsics.imageHeight);
		intrinsics.fx *= rescaleF;
		intrinsics.fy *= rescaleF;
		intrinsics.cx *= rescaleF;
		intrinsics.cy *= rescaleF;
		//	gr: should be changing imageWidth/Height in intrinsics here! - currently data is out of sync
    }

	//	user-specified scale regardless (only scales down)
    if (downscaleFactor > 1.0f)
	{
        float scaleFactor = 1.0f / downscaleFactor;
        cv::resize(cImg, cImg, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
		intrinsics.fx *= scaleFactor;
		intrinsics.fy *= scaleFactor;
		intrinsics.cx *= scaleFactor;
		intrinsics.cy *= scaleFactor;
    }

	//	cache projection matrix
	projectionMatrix = intrinsics.GetProjectionMatrix();
    cv::Rect roi;

    if (intrinsics.HasDistortionParameters()){
        // Undistort with opencv
        std::vector<float> distCoeffs = intrinsics.GetOpencvUndistortionParameters();
        cv::Mat cK = floatNxNtensorToMat(projectionMatrix);
        cv::Mat newK = cv::getOptimalNewCameraMatrix(cK, distCoeffs, cv::Size(cImg.cols, cImg.rows), 0, cv::Size(), &roi);

        cv::Mat undistorted = cv::Mat::zeros(cImg.rows, cImg.cols, cImg.type());
        cv::undistort(cImg, undistorted, cK, distCoeffs, newK);
        
        image = imageToTensor(undistorted);
		projectionMatrix = floatNxNMatToTensor(newK);
    }else{
        roi = cv::Rect(0, 0, cImg.cols, cImg.rows);
        image = imageToTensor(cImg);
    }

    // Crop to ROI
    image = image.index({Slice(roi.y, roi.y + roi.height), Slice(roi.x, roi.x + roi.width), Slice()});

    // Update intrinsics to newly scaled parameters & image pixels
	intrinsics.imageHeight = image.size(0);
	intrinsics.imageWidth = image.size(1);
    intrinsics.fx = projectionMatrix[0][0].item<float>();
	intrinsics.fy = projectionMatrix[1][1].item<float>();
	intrinsics.cx = projectionMatrix[0][2].item<float>();
	intrinsics.cy = projectionMatrix[1][2].item<float>();
}

std::string Camera::getName() const
{
	auto Filename = cameraImagePath.filename().string();
	return Filename;
}


torch::Tensor Camera::getImage(int downscaleFactor){
    if (downscaleFactor <= 1) return image;
    else{

        // torch::jit::script::Module container = torch::jit::load("gt.pt");
        // return container.attr("val").toTensor();

        if (imagePyramids.find(downscaleFactor) != imagePyramids.end()){
            return imagePyramids[downscaleFactor];
        }

        // Rescale, store and return
		int NewHeight = image.size(0);
		int NewWidth = image.size(1);
		NewWidth /= downscaleFactor;
		NewHeight /= downscaleFactor;
		cv::Mat cImg = getOpencvRgbImageStretched(NewWidth,NewHeight);
        torch::Tensor t = imageToTensor(cImg);
        imagePyramids[downscaleFactor] = t;
        return t;
    }
}

cv::Mat Camera::getOpencvRgbImageStretched(int Width,int Height)
{
	cv::Mat cImg = tensorToImage(image);
	cv::resize(cImg, cImg, cv::Size(Width,Height), 0.0, 0.0, cv::INTER_AREA);
	return cImg;
}

bool CameraIntrinsics::HasDistortionParameters(){
    return k1 != 0.0f || k2 != 0.0f || k3 != 0.0f || p1 != 0.0f || p2 != 0.0f;
}

std::vector<float> CameraIntrinsics::GetOpencvUndistortionParameters(){
    std::vector<float> p = { k1, k2, p1, p2, k3, 0.0f, 0.0f, 0.0f };
    return p;
}

std::vector<std::string> InputData::GetCameraNames()
{
	std::vector<std::string> Names;
	for ( auto& Camera : cameras )
	{
		auto Name = Camera.getName();
		Names.emplace_back( Name );
	}
	return Names;
}

//	remove camera from the training data (typically for application to use for validation)
std::shared_ptr<Camera>	InputData::PopCamera(std::string_view CameraImageName)
{
	auto FindMatchingCameraIndex = [this,CameraImageName]() -> int
	{
		//	todo: put seed into params, and use a RNG, rather than changing global (random) state
		std::srand(42);
		
		//	pick a random camera index
		if ( cameras.size() > 0 && CameraImageName == OpenSplat::randomValidationImageName )
		{
			auto Index = std::rand() % cameras.size();
			return Index;
		}
		
		//	find match
		for ( auto i=0;	i<cameras.size();	i++ )
		{
			auto CameraName = cameras[i].getName();
			if ( CameraName != CameraImageName )
				continue;
			return i;
		}

		//	helpful error listing all possibilites for the user
		std::stringstream Error;
		Error << "Input data has no camera matching " << CameraImageName << " (";
		for ( auto& CameraName : GetCameraNames() )
			Error << CameraName << ",";
		Error << ")";
		throw std::runtime_error(Error.str());
	};
	
	auto MatchingCameraIndex = FindMatchingCameraIndex();
	auto& MatchingCamera = cameras[MatchingCameraIndex];
	
	//	make copy of camera before removing it
	auto PoppedCameraCopy = std::make_shared<Camera>(MatchingCamera);
	
	cameras.erase( cameras.begin() + MatchingCameraIndex );
	
	return PoppedCameraCopy;
}


void InputData::saveCamerasJson(const std::string &filename, bool keepCrs){
    json j = json::array();
    
    for (size_t i = 0; i < cameras.size(); i++){
        Camera &cam = cameras[i];

        json camera = json::object();
        camera["id"] = i;
		camera["img_name"] = cam.getName();
        camera["width"] = cam.intrinsics.imageWidth;
        camera["height"] = cam.intrinsics.imageHeight;
        camera["fx"] = cam.intrinsics.fx;
        camera["fy"] = cam.intrinsics.fy;

		torch::Tensor R = cam.camToWorld.GetCamToWorldRotation();
		//	gr: what is squeeze()? 
        torch::Tensor T = cam.camToWorld.GetCamToWorldTranslation().squeeze();
        
        //	undo centering transform applied when loading data
        if (keepCrs) 
			T = (T / scale) + translation;

        std::vector<float> position(3);
        std::vector<std::vector<float>> rotation(3, std::vector<float>(3));
        for (int i = 0; i < 3; i++) {
            position[i] = T[i].item<float>();
            for (int j = 0; j < 3; j++) {
                rotation[i][j] = R[i][j].item<float>();
            }
        }

        camera["position"] = position;
        camera["rotation"] = rotation;
 
        j.push_back(camera);
    }
    
    std::ofstream of(filename);
	if ( !of.is_open() )
		throw std::runtime_error(std::string("Failed to open file") + filename + " to write cameras meta");
    
	of << j;
    of.close();

    std::cout << "Wrote " << filename << std::endl;
}

Camera& InputData::GetCamera(int CameraIndex)
{
	if ( CameraIndex < 0 || CameraIndex >= cameras.size() )
	{
		throw OpenSplat::NoCameraException(CameraIndex,cameras.size());
	}

	return cameras[CameraIndex];
}

Camera& InputData::GetCamera(std::string_view CameraName)
{
	for ( auto& Camera : cameras )
	{
		if ( Camera.getName() != CameraName )
			continue;
		return Camera;
	}
	throw std::runtime_error( std::string("No camera named ") + std::string(CameraName) );
}
