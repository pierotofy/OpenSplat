#ifndef INPUTDATA_H
#define INPUTDATA_H

#include <cstdint>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>

using CameraKey = std::uint32_t;
struct Camera {
    CameraKey key = 0;
    int id = -1;
    int width = 0;
    int height = 0;
    float fx = 0;
    float fy = 0;
    float cx = 0;
    float cy = 0;
    float k1 = 0;
    float k2 = 0;
    float k3 = 0;
    float k4 = 0;
    float k5 = 0;
    float k6 = 0;
    float p1 = 0;
    float p2 = 0;
    torch::Tensor camToWorld;
    std::string filePath, maskPath;

    Camera() = default;
    Camera(int width, int height, float fx, float fy, float cx, float cy, float k1, float k2, float k3,
           float p1, float p2, const torch::Tensor &camToWorld, const std::string &filePath)
        : width(width), height(height), fx(fx), fy(fy), cx(cx), cy(cy), k1(k1), k2(k2), k3(k3), p1(p1), p2(p2),
          camToWorld(camToWorld), filePath(filePath) {}
};

struct CameraSplit {
    std::vector<CameraKey> trainKeys;
    std::optional<CameraKey> validationKey;
};

struct Points {
    torch::Tensor xyz, rgb;
};

struct InputData {
    std::vector<Camera> cameras;
    float scale = 1.0f;
    torch::Tensor translation;
    Points points;
    void assignCameraKeys();
    CameraSplit splitCameras(bool validate, const std::string &valImage = "random") const;
};

InputData inputDataFromX(const std::string &projectRoot);
std::string findMaskPath(const std::string &imagePath, const std::string &projectRoot);

#endif
