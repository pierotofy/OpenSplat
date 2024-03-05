#ifndef COLMAP_H
#define COLMAP_H

#include <fstream>
#include "input_data.hpp"

namespace cm{
    InputData inputDataFromColmap(const std::string &projectRoot);

    enum CameraModel{
        SimplePinhole = 0, Pinhole, SimpleRadial, Radial,
        OpenCV, OpenCVFisheye, FullOpenCV, FOV, 
        SimpleRadialFisheye, RadialFisheye, ThinPrismFisheye
    };
}

#endif