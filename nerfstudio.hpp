#ifndef NERFSTUDIO_H
#define NERFSTUDIO_H

#include <iostream>
#include <string>
#include <fstream>
#include <json_fwd.hpp>
#include <torch/torch.h>

using json = nlohmann::json;

namespace ns{
    typedef std::vector<std::vector<float>> Mat4;

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
        Mat4 transformMatrix;
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

    enum CameraType { Perspective };
    struct Camera{
        int width;
        int height;
        float fx;
        float fy;
        float cx;
        float cy;
        // double k1;
        // double k2;
        // double p1;
        // double p2;
        // double k3;
        torch::Tensor camToWorld;
        std::string filePath;
        CameraType cameraType = CameraType::Perspective;

        Camera(int width, int height, float fx, float fy, float cx, float cy, const torch::Tensor &camToWorld, const std::string &filePath) : 
            width(width), height(height), fx(fx), fy(fy), cx(cx), cy(cy), camToWorld(camToWorld), filePath(filePath) {}
        
        void scaleOutputResolution(float scaleFactor);
        
    };

    Transforms readTransforms(const std::string &filename);

    torch::Tensor posesFromTransforms(const Transforms &t);
    std::tuple<torch::Tensor, torch::Tensor> autoOrientAndCenterPoses(const torch::Tensor &poses);

    torch::Tensor rotationMatrix(const torch::Tensor &a, const torch::Tensor &b);

    struct Points{
        torch::Tensor xyz;
        torch::Tensor rgb;
    };
    struct InputData{
        std::vector<Camera> cameras;
        float scaleFactor;
        torch::Tensor transformMatrix;
        Points points;
    };
    InputData inputDataFromNerfStudio(const std::string &projectRoot);
    void rescaleOutputResolution(std::vector<Camera> &cameras, float scaleFactor);
}   



#endif