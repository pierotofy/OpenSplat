#include "input_data.hpp"
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;
namespace ns {
InputData inputDataFromNerfStudio(const std::string &projectRoot);
}
namespace cm {
InputData inputDataFromColmap(const std::string &projectRoot);
}
namespace osfm {
InputData inputDataFromOpenSfM(const std::string &projectRoot);
}
namespace omvg {
InputData inputDataFromOpenMVG(const std::string &projectRoot);
}

InputData inputDataFromX(const std::string &projectRoot) {
    fs::path root(projectRoot);
    InputData input;
    if (fs::exists(root / "transforms.json"))
        input = ns::inputDataFromNerfStudio(projectRoot);
    else if (fs::exists(root / "sparse") || fs::exists(root / "cameras.bin"))
        input = cm::inputDataFromColmap(projectRoot);
    else if (fs::exists(root / "reconstruction.json"))
        input = osfm::inputDataFromOpenSfM(projectRoot);
    else if (fs::exists(root / "opensfm" / "reconstruction.json"))
        input = osfm::inputDataFromOpenSfM((root / "opensfm").string());
    else if (fs::exists(root / "sfm_data.json"))
        input = omvg::inputDataFromOpenMVG(projectRoot);
    else
        throw std::runtime_error("Invalid project folder (must be either a colmap "
                                 "or nerfstudio or openmvg project folder)");
    input.assignCameraKeys();
    return input;
}

void InputData::assignCameraKeys() {
    if (cameras.size() > static_cast<size_t>(std::numeric_limits<CameraKey>::max()))
        throw std::overflow_error("Too many cameras for CameraKey");
    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].key = static_cast<CameraKey>(i);
}

CameraSplit InputData::splitCameras(bool validate, const std::string &valImage) const {
    CameraSplit split;
    std::optional<size_t> validationIndex;
    if (validate) {
        if (cameras.empty())
            throw std::runtime_error("Cannot select validation camera from an empty scene");
        if (valImage == "random") {
            std::srand(42);
            validationIndex = static_cast<size_t>(std::rand()) % cameras.size();
        } else {
            for (size_t i = 0; i < cameras.size(); ++i)
                if (fs::path(cameras[i].filePath).filename() == valImage) {
                    validationIndex = i;
                    break;
                }
            if (!validationIndex)
                throw std::runtime_error(valImage + " not in the list of cameras");
        }
    }
    for (size_t i = 0; i < cameras.size(); ++i) {
        if (validationIndex && i == *validationIndex)
            split.validationKey = cameras[i].key;
        else
            split.trainKeys.push_back(cameras[i].key);
    }
    return split;
}

std::string findMaskPath(const std::string &imagePath, const std::string &projectRoot) {
    static const char *folders[] = {"masks", "mask", "segmentation", "dynamic_masks"};
    static const char *extensions[] = {".png", ".jpg", ".jpeg", ".mask.png"};
    fs::path image(imagePath);
    for (const char *folder : folders) {
        fs::path directory = fs::path(projectRoot) / folder;
        if (!fs::is_directory(directory))
            continue;
        for (const char *extension : extensions) {
            fs::path candidate = directory / (image.stem().string() + extension);
            if (fs::exists(candidate))
                return candidate.string();
            candidate = directory / (image.filename().string() + extension);
            if (fs::exists(candidate))
                return candidate.string();
        }
    }
    return "";
}
