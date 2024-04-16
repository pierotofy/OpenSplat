#ifndef OPENMVS_H
#define OPENMVS_H

#include <fstream>
#include <torch/torch.h>

namespace omvs{
    struct DepthHeader {
        static const uint16_t MAGIC = 0x5244; // 'DR'
        enum {
            DEPTH = 1,
            NORMAL = 2,
            CONF = 4,
            VIEWS = 8,
        };
        uint16_t magic;
        uint8_t type;
        uint8_t padding;
        uint32_t imageWidth, imageHeight;
        uint32_t depthWidth, depthHeight;
        float dMin, dMax;
        inline DepthHeader() : magic(0), type(0), padding(0) {}
    };

    struct Depthmap{
        DepthHeader header;

        std::string filename;
        torch::Tensor depth;
    };

    Depthmap readDepthmap(const std::string &dmap, bool readHeaderOnly = false);
}

#endif