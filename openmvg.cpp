#include <filesystem>
#include <nlohmann/json.hpp>
#include "openmvg.hpp"
#include "point_io.hpp"
#include "tensor_math.hpp"
#include <stdexcept>
#include <string>
#include <bits/stdc++.h>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace torch::indexing;

namespace omvg{

bool read_intrinsics(const json& data, std::unordered_map<uint32_t, Intrinsic> &intrinsics){
    /* Example Intrinsic
        {
            "key": 0,
            "value": {
                "polymorphic_id": 2147483650,
                "polymorphic_name": "pinhole_brown_t2",
                "ptr_wrapper": {
                    "id": 2147483731,
                    "data": {
                        "width": 4000,
                        "height": 3000,
                        "ccdw": 6.16,
                        "focal_length": 2344.1557760362504,
                        "principal_point": [
                            2000.0,
                            1500.0
                        ],
                        "disto_t2": [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0
                        ]
                    }
                }
            }
        }
    */

    for(auto intrin: data["intrinsics"]){
        //std::cout << intrin << std::endl;

        Intrinsic intrinsic;

        intrinsic.projectionType = intrin["value"]["polymorphic_name"];

        std::uint32_t id = intrin["key"].get<uint32_t>();

        intrinsic.width = intrin["value"]["ptr_wrapper"]["data"]["width"].get<uint32_t>();
        intrinsic.height = intrin["value"]["ptr_wrapper"]["data"]["height"].get<uint32_t>();
        
        intrinsic.fx = intrin["value"]["ptr_wrapper"]["data"]["focal_length"].get<double>();
        intrinsic.fy = intrin["value"]["ptr_wrapper"]["data"]["focal_length"].get<double>();

        intrinsic.cx = intrin["value"]["ptr_wrapper"]["data"]["principal_point"][0].get<double>();
        intrinsic.cy = intrin["value"]["ptr_wrapper"]["data"]["principal_point"][1].get<double>();

        // find the key needed
        std::string key;
        for(auto item: intrin["value"]["ptr_wrapper"]["data"].items()){
            std::string key_ = item.key();
            int res = key_.find("dis");
            if(res != std::string::npos){
                key = key_;
                break;
            } 
        }

        int counter = 0;
        for(auto param: intrin["value"]["ptr_wrapper"]["data"][key]){
            switch(counter){
                case 0:{ intrinsic.k1 = param.get<double>(); break;}
                case 1:{ intrinsic.k2 = param.get<double>(); break;}
                case 2:{ intrinsic.p1 = param.get<double>(); break;}
                case 3:{ intrinsic.p2 = param.get<double>(); break;}
                case 4:{ intrinsic.k3 = param.get<double>(); break;}
                default:{break;}
            }
            counter++;
        }


        intrinsics[id] = intrinsic;
    }

    return true;
}

bool read_views(const json& data, std::unordered_map<uint32_t, View> &views){
    /*Example entry for View
        {
            "key": 0,
            "value": {
                "polymorphic_id": 2147483649,
                "polymorphic_name": "view_priors",
                "ptr_wrapper": {
                    "id": 2147483649,
                    "data": {
                        "local_path": "",
                        "filename": "DJI_0001.JPG",
                        "width": 4000,
                        "height": 3000,
                        "id_view": 0,
                        "id_intrinsic": 0,
                        "id_pose": 0,
                        "use_pose_center_prior": true,
                        "center_weight": [
                            5.0,
                            5.0,
                            10.0
                        ],
                        "center": [
                            246834.2629212376,
                            4309982.276090393,
                            1993.892
                        ]
                    }
                }
            }
        }
    */
    for(auto item: data["views"]){
        auto value = item["value"];

        auto image_view = value["ptr_wrapper"]["data"];
        View view;
        view.s_Img_path = image_view["filename"].get<std::string>();

        std::uint32_t id = image_view["id_view"].get<std::uint32_t>();

        view.id_intrinsic = image_view["id_intrinsic"].get<std::uint32_t>();
        view.id_pose = image_view["id_pose"].get<std::uint32_t>();

        view.ui_width = image_view["width"].get<std::uint32_t>();
        view.ui_height = image_view["height"].get<std::uint32_t>();

        views[id] = view;
    }

    return true;
}

bool read_poses(const json& data, std::unordered_map<uint32_t, Pose> &poses){
    /*Example entry for View
        {
            "key": 0,
            "value": {
                "rotation": [
                    [
                        0.9994698938677475,
                        -0.03213424850692021,
                        0.005226980475224173
                    ],
                    [
                        -0.03231189074567745,
                        -0.9987371007397999,
                        0.038472656813614867
                    ],
                    [
                        0.003984089410678775,
                        -0.03862115584435917,
                        -0.9992459824051401
                    ]
                ],
                "center": [
                    246833.93383001933,
                    4309984.296235806,
                    1993.5002170875148
                ]
            }
        }
    */
    auto scene_poses = data["extrinsics"];
    for(auto item: scene_poses){
        
        std::uint32_t id = item["key"].get<std::uint32_t>();
        auto value = item["value"];

        Pose pose;
        for(auto row: value["rotation"]){
            for(auto r: row){
                pose.rotation.push_back( r.get<double>() );
            }
        }
        for(auto c: value["center"]){
            pose.center.push_back( c.get<double>() );
        }

        poses[id] = pose;
    }

    return true;
}

InputData inputDataFromOpenMVG(const std::string &projectRoot){
    InputData ret;
    fs::path cmRoot(projectRoot);
    fs::path reconstructionPath = cmRoot / "sfm_data.json";
    fs::path colorPointCloud = cmRoot / "colorized.ply";

    if (fs::exists(cmRoot / "sfm_data.bin") && !fs::exists(reconstructionPath)){
        throw std::runtime_error("No json found, please use openMVG_main_ConvertSfM_DataFormat with the bin to create the json file");
    }
    if (!fs::exists(cmRoot / "sfm_data.bin") && !fs::exists(reconstructionPath)){
        throw std::runtime_error("No project files found, please check the file path for sfm_data.json or sfm_data.bin");
    }

    if (fs::exists(cmRoot / "cloud_and_poses.ply") && !fs::exists(colorPointCloud)){
        throw std::runtime_error("No colorized.ply found, cloud_and_poses found, please run openMVG_main_ComputeSfM_DataColor and name the output colorized.ply");
    }
    if (!fs::exists(cmRoot / "cloud_and_poses.ply") && !fs::exists(colorPointCloud)){
        throw std::runtime_error("No project files found, please check the file path for sfm_data.json or sfm_data.bin");
    }

    std::ifstream f(reconstructionPath.string());
    json data = json::parse(f);
    f.close();

    std::string image_root_path = data["root_path"];

    std::cout << "Images should be in :" << image_root_path << std::endl;
    
    std::unordered_map<uint32_t, Intrinsic> intrinsics;
    std::unordered_map<uint32_t, View> views;
    std::unordered_map<uint32_t, Pose> poses;

    bool intrinsics_ok = read_intrinsics(data, intrinsics);
    bool views_ok = read_views(data, views);
    bool poses_ok = read_poses(data, poses);

    if(!intrinsics_ok){
        std::cerr << "Intrinsics didn't read properly" << std::endl;
    }

    std::cout << "Found " << intrinsics.size() << " intrinsics" << std::endl;
    std::cout << "Found " << views.size() << " views" << std::endl;
    std::cout << "Found " << poses.size() << " poses" << std::endl;

    // start putting the information into the tensors
    


    PointSet *pSet = readPointSet(colorPointCloud.string());
    torch::Tensor points = pSet->pointsTensor().clone();



    return ret;
}

}