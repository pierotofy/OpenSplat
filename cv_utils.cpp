#include "cv_utils.hpp"

cv::Mat imreadRGB(const std::string &filename){
    cv::Mat cImg = cv::imread(filename);
    cv::cvtColor(cImg, cImg, cv::COLOR_BGR2RGB);
    return cImg;
}

void imwriteRGB(const std::string &filename, const cv::Mat &image){
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_RGB2BGR);
    cv::imwrite(filename, rgb);
}

cv::Mat floatNxNtensorToMat(const torch::Tensor &t){
    return cv::Mat(t.size(0), t.size(1), CV_32F, t.data_ptr());
}

torch::Tensor floatNxNMatToTensor(const cv::Mat &m){
    return torch::from_blob(m.data, { m.rows, m.cols }, torch::kFloat32);
}