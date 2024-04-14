#ifndef OPENSFM_H
#define OPENSFM_H

#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <torch/torch.h>
#include <nlohmann/json_fwd.hpp>
#include "input_data.hpp"

using json = nlohmann::json;

namespace osfm{
    struct Cam{
        std::string projectionType = "";
        int width = 0;
        int height = 0;
        double fx = 0;
        double fy = 0;
        double cx = 0;
        double cy = 0;
        double k1 = 0;
        double k2 = 0;
        double p1 = 0;
        double p2 = 0;
        double k3 = 0;
    };
    void from_json(const json& j, Cam &c);

    struct Shot{
        std::vector<float> rotation = {0.0f, 0.0f, 0.0f};
        std::vector<float> translation = {0.0f, 0.0f, 0.0f};
        std::string camera = "";
    };
    void from_json(const json& j, Shot &s);

    struct Point{
        std::vector<float> color;
        std::vector<float> coordinates;
    };
    void from_json(const json& j, Point &p);

    struct Reconstruction{
        std::unordered_map<std::string, Cam> cameras;
        std::unordered_map<std::string, Shot> shots;
        std::unordered_map<std::string, Point> points;
    };
    void from_json(const json& j, Reconstruction &r);

    InputData inputDataFromOpenSfM(const std::string &projectRoot);
}   

#endif