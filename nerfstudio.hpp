#ifndef NERFSTUDIO_H
#define NERFSTUDIO_H

#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <json_fwd.hpp>
#include <torch/torch.h>
#include <opencv2/calib3d.hpp>

using json = nlohmann::json;

namespace ns{
    typedef std::vector<std::vector<float>> Mat4;

    struct Frame{
        std::string filePath = "";
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
        int width = 0;
        int height = 0;
        float fx = 0;
        float fy = 0;
        float cx = 0;
        float cy = 0;
        float k1 = 0;
        float k2 = 0;
        float k3 = 0;
        float p1 = 0;
        float p2 = 0;
        torch::Tensor camToWorld;
        std::string filePath = "";
        CameraType cameraType = CameraType::Perspective;

        Camera(int width, int height, float fx, float fy, float cx, float cy, 
            float k1, float k2, float k3, float p1, float p2,
            const torch::Tensor &camToWorld, const std::string &filePath) : 
            width(width), height(height), fx(fx), fy(fy), cx(cx), cy(cy), 
            k1(k1), k2(k2), k3(k3), p1(p1), p2(p2),
            camToWorld(camToWorld), filePath(filePath) {}
        
        torch::Tensor getIntrinsicsMatrix();
        bool hasDistortionParameters();
        std::vector<float> undistortionParameters();
        void scaleOutputResolution(float scaleFactor);
        torch::Tensor getImage(int downscaleFactor);

        void loadImage(float downscaleFactor);
        torch::Tensor K;
        torch::Tensor image;

        std::unordered_map<int, torch::Tensor> imagePyramids;
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
}   



#endif