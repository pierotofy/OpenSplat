#include <iostream>
#include <string>
#include <fstream>
#include <json_fwd.hpp>

using json = nlohmann::json;

namespace nf{
    struct Frame{
        std::string filePath;
        int width;
        int height;
        double fx;
        double fy;
        double cx;
        double cy;
        double k1;
        double k2;
        double p1;
        double p2;
        double k3;
        std::vector<std::vector<double>> transformMatrix;
    };
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

    struct Transforms{
        std::string cameraModel;
        std::vector<Frame> frames;
        std::string plyFilePath;
    };

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
        
    }


    Transforms readTransforms(const std::string &filename){
        std::ifstream f(filename);
        json data = json::parse(f);
        return data.template get<Transforms>();
    }
}