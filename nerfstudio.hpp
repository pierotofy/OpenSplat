#ifndef NERFSTUDIO_H
#define NERFSTUDIO_H

#include <iostream>
#include <string>
#include <fstream>
#include <json_fwd.hpp>

using json = nlohmann::json;

namespace ns{
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
    void to_json(json &j, const Frame &f);
    void from_json(const json& j, Frame &f);

    struct Transforms{
        std::string cameraModel;
        std::vector<Frame> frames;
        std::string plyFilePath;
    };
    void to_json(json &j, const Transforms &t);
    void from_json(const json& j, Transforms &t);


    Transforms readTransforms(const std::string &filename);
}

#endif