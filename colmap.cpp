#include <filesystem>
#include "colmap.hpp"
#include "point_io.hpp"
#include "tensor_math.hpp"

namespace fs = std::filesystem;
using namespace torch::indexing;

namespace cm{

InputData inputDataFromColmap(const std::string &projectRoot){
    InputData ret;
    fs::path cmRoot(projectRoot);

    if (!fs::exists(cmRoot / "cameras.bin") && fs::exists(cmRoot / "sparse" / "0" / "cameras.bin")){
        cmRoot = cmRoot / "sparse" / "0";
    }

    fs::path camerasPath = cmRoot / "cameras.bin";
    fs::path imagesPath = cmRoot / "images.bin";
    fs::path pointsPath = cmRoot / "points3D.bin";
    
    if (!fs::exists(camerasPath)) throw std::runtime_error(camerasPath.string() + " does not exist");
    if (!fs::exists(imagesPath)) throw std::runtime_error(imagesPath.string() + " does not exist");
    if (!fs::exists(pointsPath)) throw std::runtime_error(pointsPath.string() + " does not exist");

    std::ifstream camf(camerasPath.string(), std::ios::binary);
    if (!camf.is_open()) throw std::runtime_error("Cannot open " + camerasPath.string());
    std::ifstream imgf(imagesPath.string(), std::ios::binary);
    if (!imgf.is_open()) throw std::runtime_error("Cannot open " + imagesPath.string());
    
    size_t numCameras = readBinary<uint64_t>(camf);
    std::vector<Camera> cameras(numCameras);

    std::unordered_map<uint32_t, Camera *> camMap;
    
    for (size_t i = 0; i < numCameras; i++) {
        Camera *cam = &cameras[i];

        cam->id = readBinary<uint32_t>(camf);

        CameraModel model = static_cast<CameraModel>(readBinary<int>(camf)); // model ID
        cam->width = readBinary<uint64_t>(camf);
        cam->height = readBinary<uint64_t>(camf);
        
        if (model == SimplePinhole){
            cam->fx = readBinary<double>(camf);
            cam->fy = cam->fx;
            cam->cx = readBinary<double>(camf);
            cam->cy = readBinary<double>(camf);
        }else if (model == Pinhole){
            cam->fx = readBinary<double>(camf);
            cam->fy = readBinary<double>(camf);
            cam->cx = readBinary<double>(camf);
            cam->cy = readBinary<double>(camf);
        }else if (model == OpenCV){
            cam->fx = readBinary<double>(camf);
            cam->fy = readBinary<double>(camf);
            cam->cx = readBinary<double>(camf);
            cam->cy = readBinary<double>(camf);
            cam->k1 = readBinary<double>(camf);
            cam->k2 = readBinary<double>(camf);
            cam->p1 = readBinary<double>(camf);
            cam->p2 = readBinary<double>(camf);
        }else{
            throw std::runtime_error("Unsupported camera model: " + std::to_string(model));
        }

        camMap[cam->id] = cam;
    }

    camf.close();


    size_t numImages = readBinary<uint64_t>(imgf);
    torch::Tensor unorientedPoses = torch::zeros({static_cast<long int>(numImages), 4, 4}, torch::kFloat32);

    for (size_t i = 0; i < numImages; i++){
        readBinary<uint32_t>(imgf); // imageId
        
        torch::Tensor qVec = torch::tensor({
            readBinary<double>(imgf),
            readBinary<double>(imgf),
            readBinary<double>(imgf),
            readBinary<double>(imgf)
        }, torch::kFloat32);
        torch::Tensor R = quatToRotMat(qVec).transpose(0, 1);
        torch::Tensor T = torch::tensor({
            { readBinary<double>(imgf) },
            { readBinary<double>(imgf) },
            { readBinary<double>(imgf) }
        }, torch::kFloat32);

        // TODO: check cam2world matrix
        // torch::Tensor Rinv = R.transpose(0, 1);
        // torch::Tensor Tinv = torch::matmul(-Rinv, T);

        uint32_t camId = readBinary<uint32_t>(imgf);

        Camera cam = *camMap[camId];

        char ch = '\0';
        std::string filePath = "";
        while(true){
            imgf.read(&ch, 1);
            if (ch == '\0') break;
            filePath += ch;
        }

        // TODO: should "images" be an option?
        cam.filePath = (fs::path(projectRoot) / "images" / filePath).string();

        unorientedPoses.index_put_({Slice(i), Slice(None, 3), Slice(None, 3)}, R);
        unorientedPoses.index_put_({Slice(i), Slice(None, 3), Slice(3, 4)}, T);
        
        size_t numPoints2D = readBinary<uint64_t>(imgf);
        for (size_t j = 0; j < numPoints2D; j++){
            readBinary<double>(imgf); // x
            readBinary<double>(imgf); // y
            readBinary<uint64_t>(imgf); // point3D ID
        }

        ret.cameras.push_back(cam);
    }

    imgf.close();
    
    auto r = autoOrientAndCenterPoses(unorientedPoses);
    torch::Tensor poses = std::get<0>(r);
    ret.transformMatrix = std::get<1>(r);
    ret.scaleFactor = 1.0f / torch::max(torch::abs(poses.index({Slice(), Slice(None, 3), 3}))).item<float>();
    poses.index({Slice(), Slice(None, 3), 3}) *= ret.scaleFactor;

    for (size_t i = 0; i < ret.cameras.size(); i++){
        ret.cameras[i].camToWorld = poses[i];
    }

    PointSet *pSet = readPointSet(pointsPath.string());
    torch::Tensor points = pSet->pointsTensor().clone();

    ret.points.xyz = torch::matmul(torch::cat({points, torch::ones_like(points.index({"...", Slice(None, 1)}))}, -1), 
                    ret.transformMatrix.transpose(0, 1));
    ret.points.xyz *= ret.scaleFactor;
    ret.points.rgb = pSet->colorsTensor().clone();

    RELEASE_POINTSET(pSet);

    return ret;
}

}