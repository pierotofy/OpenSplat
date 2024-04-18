#include <filesystem>
#include <fstream>
#include "openmvs.hpp"

namespace fs = std::filesystem;

namespace omvs{

Depthmap readDepthmap(const std::string &dmap, bool readHeaderOnly){
    Depthmap d;
    std::ifstream fin(dmap, std::ios_base::binary);
    if (!fin.is_open()) throw std::runtime_error("Cannot open " + dmap);

    // Header check
    fin.read(reinterpret_cast<char *>(&d.header), sizeof(DepthHeader));
    if ((d.header.magic != DepthHeader::MAGIC) || 
        (d.header.type & DepthHeader::DEPTH == 0)){
        throw std::runtime_error("Invalid depthmap file: " + dmap);
    }

    // Filename
    uint16_t fnameSize;
    fin.read(reinterpret_cast<char *>(&fnameSize), sizeof(uint16_t));
    std::string fnamePath;
    fnamePath.resize(fnameSize);
    fin.read(reinterpret_cast<char *>(fnamePath.data()), sizeof(char) * fnameSize);
    d.filename = fs::path(fnamePath).filename().string();

    if (readHeaderOnly) return d;

    // neighbor IDs (skip)
    uint32_t numNeighbors;
    fin.read(reinterpret_cast<char *>(&numNeighbors), sizeof(uint32_t));
    fin.ignore(sizeof(uint32_t) * numNeighbors);
    
    // poses (skip)
    fin.ignore(sizeof(double) * 21);

    // Depthmap
    d.depth = torch::zeros({d.header.depthHeight, d.header.depthWidth}, torch::kFloat32);
    fin.read(reinterpret_cast<char *>(d.depth.data_ptr()), sizeof(float) * d.header.depthHeight * d.header.depthWidth);

    fin.close();

    return d;
}

}