#include "nerfstudio.hpp"
#include <json.hpp>

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
}