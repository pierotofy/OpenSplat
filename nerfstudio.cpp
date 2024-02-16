#include <filesystem>
#include <json.hpp>
#include "nerfstudio.hpp"
#include "point_io.hpp"
#include "cv_utils.hpp"

#include <torch/script.h> // TODO REMOVE

namespace fs = std::filesystem;

using json = nlohmann::json;
using namespace torch::indexing;

namespace ns{

void to_json(json &j, const Frame &f){
    j = json{ {"file_path", f.filePath }, 
                {"w", f.width }, 
                {"h", f.height },
                {"fl_x", f.fx },
                {"fl_y", f.fy },
                {"cx", f.cx },
                {"cy", f.cy },
                {"k1", f.k1 },
                {"k2", f.k2 },
                {"p1", f.p1 },
                {"p2", f.p2 },
                {"k3", f.k3 },
                {"transform_matrix", f.transformMatrix },
                
            };
}

void from_json(const json& j, Frame &f){
    j.at("file_path").get_to(f.filePath);
    j.at("w").get_to(f.width);
    j.at("h").get_to(f.height);
    j.at("fl_x").get_to(f.fx);
    j.at("fl_y").get_to(f.fy);
    j.at("cx").get_to(f.cx);
    j.at("cy").get_to(f.cy);
    j.at("k1").get_to(f.k1);
    j.at("k2").get_to(f.k2);
    j.at("p1").get_to(f.p1);
    j.at("p2").get_to(f.p2);
    j.at("k3").get_to(f.k3);
    j.at("transform_matrix").get_to(f.transformMatrix);
    
}

void to_json(json &j, const Transforms &t){
    j = json{ {"camera_model", t.cameraModel }, 
                {"frames", t.frames },
                {"ply_file_path", t.plyFilePath },
            };
}

void from_json(const json& j, Transforms &t){
    j.at("camera_model").get_to(t.cameraModel);
    j.at("frames").get_to(t.frames);
    if (j.contains("ply_file_path")) j.at("ply_file_path").get_to(t.plyFilePath);

    std::sort(t.frames.begin(), t.frames.end(), 
        [](Frame const &a, Frame const &b) {
            return a.filePath < b.filePath; 
        });
}    

Transforms readTransforms(const std::string &filename){
    std::ifstream f(filename);
    json data = json::parse(f);
    return data.template get<Transforms>();
}

torch::Tensor posesFromTransforms(const Transforms &t){
    torch::Tensor poses = torch::zeros({static_cast<long int>(t.frames.size()), 4, 4}, torch::kFloat32);
    for (size_t c = 0; c < t.frames.size(); c++){
        for (size_t i = 0; i < 4; i++){
            for (size_t j = 0; j < 4; j++){
                poses[c][i][j] = t.frames[c].transformMatrix[i][j];
            }
        }
    }
    return poses;
}

std::tuple<torch::Tensor, torch::Tensor> autoOrientAndCenterPoses(const torch::Tensor &poses){
    // Center at mean and orient up
    torch::Tensor origins = poses.index({"...", Slice(None, 3), 3});
    torch::Tensor translation = torch::mean(origins, 0);
    torch::Tensor up = torch::mean(poses.index({Slice(), Slice(None, 3), 1}), 0);
    up = up / up.norm();
    
    torch::Tensor rotation = rotationMatrix(up, torch::tensor({0, 0, 1}, torch::kFloat32));
    torch::Tensor transform = torch::cat({rotation, torch::matmul(rotation, -translation.index({"...", None}))}, -1);
    torch::Tensor orientedPoses = torch::matmul(transform, poses);
    return std::make_tuple(orientedPoses, transform);
}


torch::Tensor rotationMatrix(const torch::Tensor &a, const torch::Tensor &b){
    // Rotation matrix that rotates vector a to vector b
    torch::Tensor a1 = a / a.norm();
    torch::Tensor b1 = b / b.norm();
    torch::Tensor v = torch::cross(a1, b1);
    torch::Tensor c = torch::dot(a1, b1);
    const float EPS = 1e-8;
    if (c.item<float>() < -1 + EPS){
        torch::Tensor eps = (torch::rand(3) - 0.5f) * 0.01f;
        return rotationMatrix(a1 + eps, b1);
    }
    torch::Tensor s = v.norm();
    torch::Tensor skew = torch::zeros({3, 3}, torch::kFloat32);
    skew[0][1] = -v[2];
    skew[0][2] = v[1];
    skew[1][0] = v[2];
    skew[1][2] = -v[0];
    skew[2][0] = -v[1];
    skew[2][1] = v[0];

    return torch::eye(3) + skew + torch::matmul(skew, skew * ((1 - c) / (s.pow(2) + EPS)));
}


InputData inputDataFromNerfStudio(const std::string &projectRoot){
    InputData ret;
    fs::path nsRoot(projectRoot);
    fs::path transformsPath = nsRoot / "transforms.json";
    if (!fs::exists(transformsPath)) throw std::runtime_error(transformsPath.string() + " does not exist");

    Transforms t = readTransforms(transformsPath.string());
    if (t.plyFilePath.empty()) throw std::runtime_error("ply_file_path is empty");
    PointSet *pSet = readPointSet((nsRoot / t.plyFilePath).string());

    torch::Tensor unorientedPoses = posesFromTransforms(t);

    auto r = autoOrientAndCenterPoses(unorientedPoses);
    torch::Tensor poses = std::get<0>(r);
    ret.transformMatrix = std::get<1>(r);

    ret.scaleFactor = 1.0f / torch::max(torch::abs(poses.index({Slice(), Slice(None, 3), 3}))).item<float>();
    poses.index({Slice(), Slice(None, 3), 3}) *= ret.scaleFactor;

    // aabbScale = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]

    for (size_t i = 0; i < t.frames.size(); i++){
        // TODO: remove this (emulates the eval/split behavior which we don't need)
        // if (i == 12) continue;

        Frame f = t.frames[i];

        ret.cameras.emplace_back(Camera(f.width, f.height, 
                            static_cast<float>(f.fx), static_cast<float>(f.fy), 
                            static_cast<float>(f.cx), static_cast<float>(f.cy), 
                            static_cast<float>(f.k1), static_cast<float>(f.k2), static_cast<float>(f.k3), 
                            static_cast<float>(f.p1), static_cast<float>(f.p2),  
                            
                            poses[i], (nsRoot / f.filePath).string()));
    }

    torch::Tensor points = pSet->pointsTensor().clone();

    ret.points.xyz = torch::matmul(torch::cat({points, torch::ones_like(points.index({"...", Slice(None, 1)}))}, -1), 
                    ret.transformMatrix.transpose(0, 1));
    ret.points.xyz *= ret.scaleFactor;
    ret.points.rgb = pSet->colorsTensor().clone();

    RELEASE_POINTSET(pSet);

    return ret;
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
    fx *= scaleFactor;
    fy *= scaleFactor;
    cx *= scaleFactor;
    cy *= scaleFactor;
    
    cv::Mat cImg = imreadRGB(filePath);

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

    // TODO: REMOVE
    // if (filePath == "banana/images/frame_00008.JPG"){
    //     std::cout << "Override read of " << filePath << std::endl;
    //     image = imageToTensor(imreadRGB("banana/frame_00008_ns.PNG"));
    // }

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

        // TODO: remove
        // torch::jit::script::Module container = torch::jit::load("banana/gt.pt");
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

void Camera::scaleOutputResolution(float scaleFactor){
    fx = fx * scaleFactor;
    fy = fy * scaleFactor;
    cx = cx * scaleFactor;
    cy = cy * scaleFactor;
    height = static_cast<int>(static_cast<float>(height) * scaleFactor);
    width = static_cast<int>(static_cast<float>(width) * scaleFactor);
}

}