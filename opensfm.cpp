#include <filesystem>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "opensfm.hpp"
#include "point_io.hpp"
#include "cv_utils.hpp"
#include "tensor_math.hpp"

namespace fs = std::filesystem;

using json = nlohmann::json;
using namespace torch::indexing;

namespace osfm{

void from_json(const json& j, Cam &c){
    j.at("projection_type").get_to(c.projectionType);
    if (j.contains("width")) j.at("width").get_to(c.width);
    if (j.contains("height")) j.at("height").get_to(c.height);
    if (j.contains("focal_x")) j.at("focal_x").get_to(c.fx);
    if (j.contains("focal_y")) j.at("focal_y").get_to(c.fy);
    if (j.contains("focal")){
        j.at("focal").get_to(c.fx);
        j.at("focal").get_to(c.fy);
    }
    if (j.contains("c_x")) j.at("c_x").get_to(c.cx);
    if (j.contains("c_y")) j.at("c_y").get_to(c.cy);
    if (j.contains("k1")) j.at("k1").get_to(c.k1);
    if (j.contains("k2")) j.at("k2").get_to(c.k2);
    if (j.contains("p1")) j.at("p1").get_to(c.p1);
    if (j.contains("p2")) j.at("p2").get_to(c.p2);
    if (j.contains("k3")) j.at("k3").get_to(c.k3);
}

void from_json(const json& j, Shot &s){
    j.at("rotation").get_to(s.rotation);
    j.at("translation").get_to(s.translation);
    j.at("camera").get_to(s.camera);
}    

void from_json(const json& j, Point &p){
    j.at("coordinates").get_to(p.coordinates);
    j.at("color").get_to(p.color);
}

void from_json(const json& j, Reconstruction &r){
    j.at("cameras").get_to(r.cameras);
    j.at("shots").get_to(r.shots);
    j.at("points").get_to(r.points);
    
}

InputData inputDataFromOpenSfM(const std::string &projectRoot){
    InputData ret;
    fs::path nsRoot(projectRoot);
    fs::path reconstructionPath = nsRoot / "reconstruction.json";
    fs::path imageListPath = nsRoot / "image_list.txt";
    
    if (!fs::exists(reconstructionPath)) throw std::runtime_error(reconstructionPath.string() + " does not exist");
    if (!fs::exists(imageListPath)) throw std::runtime_error(imageListPath.string() + " does not exist");

    std::ifstream f(reconstructionPath.string());
    json data = json::parse(f);
    f.close();

    std::unordered_map<std::string, std::string> images;
    f.open(imageListPath.string());
    std::string line;
    while(std::getline(f, line)){
        fs::path p(line);
        if (p.is_absolute()) images[p.filename().string()] = line;
        else images[p.filename().string()] = fs::absolute(nsRoot / p).string();
    }
    f.close();
    auto reconstructions = data.template get<std::vector<Reconstruction>>();

    if (reconstructions.size() == 0) throw std::runtime_error("No reconstructions found");
    if (reconstructions.size() > 1) std::cout << "Warning: multiple OpenSfM reconstructions found, choosing the first" << std::endl;

    auto reconstruction = reconstructions[0];
    auto shots = reconstruction.shots;
    auto cameras = reconstruction.cameras;
    auto points = reconstruction.points;

    torch::Tensor unorientedPoses = torch::zeros({static_cast<long int>(shots.size()), 4, 4}, torch::kFloat32);
    size_t i = 0;
    for (const auto &s : shots){
        Shot shot = s.second;

        torch::Tensor rotation = rodriguesToRotation(torch::from_blob(shot.rotation.data(), {static_cast<long>(shot.rotation.size())}, torch::kFloat32));
        torch::Tensor translation = torch::from_blob(shot.translation.data(), {static_cast<long>(shot.translation.size())}, torch::kFloat32);
        torch::Tensor w2c = torch::eye(4, torch::kFloat32);
        w2c.index_put_({Slice(None, 3), Slice(None, 3)}, rotation);
        w2c.index_put_({Slice(None, 3), Slice(3,4)}, translation.reshape({3, 1}));

        unorientedPoses[i] = torch::linalg_inv(w2c);

        // Convert OpenSfM's camera CRS (OpenCV) to OpenGL
        unorientedPoses[i].index_put_({Slice(0, 3), Slice(1,3)}, unorientedPoses[i].index({Slice(0, 3), Slice(1,3)}) * -1.0f);
        i++;
    }

    auto r = autoScaleAndCenterPoses(unorientedPoses);
    torch::Tensor poses = std::get<0>(r);
    ret.translation = std::get<1>(r);
    ret.scale = std::get<2>(r);

    i = 0;
    for (const auto &s : shots){
        std::string filename = s.first;
        Shot shot = s.second;
        
        Cam &c = cameras[shot.camera];
        if (c.projectionType != "perspective" && c.projectionType != "brown"){
            throw std::runtime_error("Camera projection type " + c.projectionType + " is not supported");
        }

        float normalizer = static_cast<float>((std::max)(c.width, c.height));
        ret.cameras.emplace_back(Camera(c.width, c.height, 
                            static_cast<float>(c.fx * normalizer), static_cast<float>(c.fy * normalizer), 
                            static_cast<float>(static_cast<float>(c.width) / 2.0f + normalizer * c.cx), static_cast<float>(static_cast<float>(c.height) / 2.0f + normalizer * c.cy), 
                            static_cast<float>(c.k1), static_cast<float>(c.k2), static_cast<float>(c.k3), 
                            static_cast<float>(c.p1), static_cast<float>(c.p2),  
                            
                            poses[i++], images[filename]));
    }

    size_t numPoints = points.size();
    torch::Tensor xyz = torch::zeros({static_cast<long>(numPoints), 3}, torch::kFloat32);
    torch::Tensor rgb = torch::zeros({static_cast<long>(numPoints), 3}, torch::kUInt8);

    i = 0;
    for (const auto &pt: points){
        Point p = pt.second;

        xyz[i][0] = p.coordinates[0];
        xyz[i][1] = p.coordinates[1];
        xyz[i][2] = p.coordinates[2];
        
        rgb[i][0] = static_cast<uint8_t>(p.color[0]);
        rgb[i][1] = static_cast<uint8_t>(p.color[1]);
        rgb[i][2] = static_cast<uint8_t>(p.color[2]);

        i++;
    }

    ret.points.xyz = (xyz - ret.translation) * ret.scale;
    ret.points.rgb = rgb;

    return ret;
}

}