#include <filesystem>
#include <json.hpp>
#include "opensplat.hpp"
#include "point_io.hpp"

namespace fs = std::filesystem;

int main(int argc, char *argv[]){
    fs::path nfProjectRoot("banana");
    nf::Transforms t = nf::readTransforms((nfProjectRoot / "transforms.json").string());
    if (t.plyFilePath.empty()) throw std::runtime_error("ply_file_path is empty");

    PointSet *pSet = readPointSet((nfProjectRoot / t.plyFilePath).string());

    std::cout << pSet->points.size();
}