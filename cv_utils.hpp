#ifndef CV_UTILS
#define CV_UTILS

#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat imreadRGB(const std::string &filename);
// Accepts uint8 [H,W,3] in [0,255] or float [H,W,3] in [0,1]
cv::Mat tensorToImage(const torch::Tensor &t);
// [H,W,3] uint8 tensor owning a copy of the image
torch::Tensor imageToTensor(const cv::Mat &image);
// float [0,1] view of an image tensor (no-op for float input)
torch::Tensor toUnitFloat(const torch::Tensor &img);

#endif
