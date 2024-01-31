#include "nerfstudio.hpp"
#include <json.hpp>

using json = nlohmann::json;

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
}