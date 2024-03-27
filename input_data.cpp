#include <filesystem>
#include "input_data.hpp"
#include "cv_utils.hpp"

namespace fs = std::filesystem;
using namespace torch::indexing;

namespace ns{ InputData inputDataFromNerfStudio(const std::string &projectRoot); }
namespace cm{ InputData inputDataFromColmap(const std::string &projectRoot); }
namespace osfm { InputData inputDataFromOpenSfM(const std::string &projectRoot); }

InputData inputDataFromX(const std::string &projectRoot){
    fs::path root(projectRoot);

    if (fs::exists(root / "transforms.json")){
        return ns::inputDataFromNerfStudio(projectRoot);
    }else if (fs::exists(root / "sparse") || fs::exists(root / "cameras.bin")){
        return cm::inputDataFromColmap(projectRoot);
    }else if (fs::exists(root / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM(projectRoot);
    }else if (fs::exists(root / "opensfm" / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM((root / "opensfm").string());
    }else{
        throw std::runtime_error("Invalid project folder (must be either a colmap or nerfstudio project folder)");
    }
}

torch::Tensor Camera::getIntrinsicsMatrix(){
    return torch::tensor({{fx, 0.0f, cx},
                          {0.0f, fy, cy},
                          {0.0f, 0.0f, 1.0f}}, torch::kFloat32);
}

void Camera::loadImage(float downscaleFactor){
    // Populates image and K, then updates the camera parameters
    // Caution: this function has destructive behaviors
    // and should be called only once
    if (image.numel()) std::runtime_error("loadImage already called");
    std::cout << "Loading " << filePath << std::endl;

    float scaleFactor = 1.0f / downscaleFactor;
    cv::Mat cImg = imreadRGB(filePath);
    
    float rescaleF = 1.0f;
    // If camera intrinsics don't match the image dimensions 
    if (cImg.rows != height || cImg.cols != width){
        rescaleF = static_cast<float>(cImg.rows) / static_cast<float>(height);
    }
    fx *= scaleFactor * rescaleF;
    fy *= scaleFactor * rescaleF;
    cx *= scaleFactor * rescaleF;
    cy *= scaleFactor * rescaleF;

    if (downscaleFactor > 1.0f){
        float f = 1.0f / downscaleFactor;
        cv::resize(cImg, cImg, cv::Size(), f, f, cv::INTER_AREA);
    }

    K = getIntrinsicsMatrix();
    cv::Rect roi;

    if (hasDistortionParameters()){
        // Undistort
        std::vector<float> distCoeffs = undistortionParameters();
        cv::Mat cK = floatNxNtensorToMat(K);
        cv::Mat newK = cv::getOptimalNewCameraMatrix(cK, distCoeffs, cv::Size(cImg.cols, cImg.rows), 0, cv::Size(), &roi);

        cv::Mat undistorted = cv::Mat::zeros(cImg.rows, cImg.cols, cImg.type());
        cv::undistort(cImg, undistorted, cK, distCoeffs, newK);
        
        image = imageToTensor(undistorted);
        K = floatNxNMatToTensor(newK);
    }else{
        roi = cv::Rect(0, 0, cImg.cols, cImg.rows);
        image = imageToTensor(cImg);
    }

    // Crop to ROI
    image = image.index({Slice(roi.y, roi.y + roi.height), Slice(roi.x, roi.x + roi.width), Slice()});

    // Update parameters
    height = image.size(0);
    width = image.size(1);
    fx = K[0][0].item<float>();
    fy = K[1][1].item<float>();
    cx = K[0][2].item<float>();
    cy = K[1][2].item<float>();
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
        cv::Mat cImg = tensorToImage(image);
        cv::resize(cImg, cImg, cv::Size(cImg.cols / downscaleFactor, cImg.rows / downscaleFactor), 0.0, 0.0, cv::INTER_AREA);
        torch::Tensor t = imageToTensor(cImg);
        imagePyramids[downscaleFactor] = t;
        return t;
    }
}

bool Camera::hasDistortionParameters(){
    return k1 != 0.0f || k2 != 0.0f || k3 != 0.0f || p1 != 0.0f || p2 != 0.0f;
}

std::vector<float> Camera::undistortionParameters(){
    std::vector<float> p = { k1, k2, p1, p2, k3, 0.0f, 0.0f, 0.0f };
    return p;
}

std::tuple<std::vector<Camera>, Camera *> InputData::getCameras(bool validate, const std::string &valImage){
    if (!validate) return std::make_tuple(cameras, nullptr);
    else{
        size_t valIdx = -1;
        std::srand(42);

        if (valImage == "random"){
            valIdx = std::rand() % cameras.size();
        }else{
            for (size_t i = 0; i < cameras.size(); i++){
                if (fs::path(cameras[i].filePath).filename().string() == valImage){
                    valIdx = i;
                    break;
                }
            }
            if (valIdx == -1) throw std::runtime_error(valImage + " not in the list of cameras");
        }

        std::vector<Camera> cams;
        Camera *valCam = nullptr;

        for (size_t i = 0; i < cameras.size(); i++){
            if (i != valIdx) cams.push_back(cameras[i]);
            else valCam = &cameras[i];
        }

        return std::make_tuple(cams, valCam);
    }
}
