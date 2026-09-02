#ifndef CV_UTILS
#define CV_UTILS

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

cv::Mat imreadRGB(const std::string &filename);
cv::Size imageDimensions(const std::string &filename);
cv::Mat tensorToImage(const torch::Tensor &t);
torch::Tensor imageToTensor(const cv::Mat &image);

#endif
