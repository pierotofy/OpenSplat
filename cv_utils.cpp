#include "cv_utils.hpp"

cv::Mat imreadRGB(const std::string &filename){
    cv::Mat cImg = cv::imread(filename);

    if (cImg.empty())
	{
		std::stringstream Error;
		Error << "Cannot read " << filename << std::endl
                  << "Make sure the path to your images is correct" << std::endl;
		throw std::runtime_error(Error.str());
    }

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
    return torch::from_blob(m.data, { m.rows, m.cols }, torch::kFloat32).clone();
}

cv::Mat tensorToImage(const torch::Tensor &t)
{
	//if ( t.dim() <= 1 )
    int h = t.sizes()[0];
    int w = t.sizes()[1];
    int c = t.sizes()[2];

    int type = CV_8UC3;
    if (c != 3) 
	{
		std::stringstream Error;
		Error << __FUNCTION__ << " Only images with 3 channels are supported (this: " << w << "x" << h << "x" << c << ")";
		throw std::runtime_error(Error.str());
	}

    cv::Mat image(h, w, type);
    torch::Tensor scaledTensor = (t * 255.0).toType(torch::kU8);
    uint8_t* dataPtr = static_cast<uint8_t*>(scaledTensor.data_ptr());
    std::copy(dataPtr, dataPtr + (w * h * c), image.data);

    return image;
}

void tensorToImage(const torch::Tensor &t,std::function<void(const cv::Mat&)> OnImage)
{
	int h = t.sizes()[0];
	int w = t.sizes()[1];
	int c = t.sizes()[2];
	
	int type = CV_8UC3;
	if (c != 3) 
	{
		std::stringstream Error;
		Error << __FUNCTION__ << " Only images with 3 channels are supported (this: " << w << "x" << h << "x" << c << ")";
		throw std::runtime_error(Error.str());
	}
	
	torch::Tensor scaledTensor = (t * 255.0).toType(torch::kU8);
	uint8_t* dataPtr = static_cast<uint8_t*>(scaledTensor.data_ptr());

	cv::Mat image(h, w, type, dataPtr);
	//std::copy(dataPtr, dataPtr + (w * h * c), image.data);
	OnImage(image);
}

torch::Tensor imageToTensor(const cv::Mat &image)
{
    torch::Tensor img = torch::from_blob(image.data, { image.rows, image.cols, image.dims + 1 }, torch::kU8);
    return (img.toType(torch::kFloat32) / 255.0f);
}

