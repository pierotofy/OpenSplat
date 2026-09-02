#include "camera_image_store.hpp"
#include "cv_utils.hpp"
#include "input_data.hpp"
#include "opensplat.hpp"
#include "constants.hpp"
#include "utils.hpp"
#include "zip_utils.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef USE_VISUALIZATION
#include "visualizer.hpp"
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace torch::indexing;

static float sceneExtent(const std::vector<Camera> &cameras) {
    if (cameras.empty())
        return 1.0f;
    torch::Tensor centers = torch::zeros({static_cast<long long>(cameras.size()), 3});
    for (size_t i = 0; i < cameras.size(); ++i)
        centers[i] = cameras[i].camToWorld.index({Slice(None, 3), 3});
    torch::Tensor average = centers.mean(0, true);
    const float extent = (centers - average).norm(2, 1).max().item<float>() * 1.1f;
    return extent <= 0 ? 1.0f : extent;
}

static void saveCameras(const std::string &filename, CameraImageStore &store, const InputData &input, bool keepCrs) {
    json output = json::array();
    for (size_t i = 0; i < store.size(); ++i) {
        const PreparedCamera &camera = store.metadata(static_cast<CameraKey>(i));
        torch::Tensor rotation = camera.camToWorld.index({Slice(None, 3), Slice(None, 3)});
        torch::Tensor translation = camera.camToWorld.index({Slice(None, 3), Slice(3, 4)}).squeeze();
        rotation = torch::matmul(rotation, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f})));
        if (keepCrs)
            translation = (translation / input.scale) + input.translation;
        std::vector<float> position(3);
        std::vector<std::vector<float>> rows(3, std::vector<float>(3));
        for (int r = 0; r < 3; ++r) {
            position[r] = translation[r].item<float>();
            for (int c = 0; c < 3; ++c)
                rows[r][c] = rotation[r][c].item<float>();
        }
        output.push_back({{"id", i},
                          {"img_name", fs::path(camera.filePath).filename().string()},
                          {"width", camera.width},
                          {"height", camera.height},
                          {"fx", camera.fx},
                          {"fy", camera.fy},
                          {"position", position},
                          {"rotation", rows}});
    }
    std::ofstream stream(filename);
    stream << output;
    std::cout << "Wrote " << filename << std::endl;
}

int main(int argc, char *argv[]) {
    cxxopts::Options options("opensplat", "Open Source 3D Gaussian Splats generator - " APP_VERSION);
    options.add_options()
        ("i,input", "Path to nerfstudio project", cxxopts::value<std::string>())
        ("o,output", "Path where to save output scene (default: splat.ply next to the input)", cxxopts::value<std::string>()->default_value("splat.ply"))
        ("output-cameras", "Path where to save a cameras JSON file", cxxopts::value<std::string>()->default_value(""))
        ("s,save-every", "Save output scene every these many steps (set to -1 to disable)", cxxopts::value<int>()->default_value("-1"))
        ("resume", "Resume training from this PLY file", cxxopts::value<std::string>()->default_value(""))
        ("val", "Withhold a camera shot for validating the scene loss")
        ("val-image", "Filename of the image to withhold for validating scene loss", cxxopts::value<std::string>()->default_value("random"))
        ("val-render", "Path of the directory where to render validation images", cxxopts::value<std::string>()->default_value(""))
        ("center", "Center the model at the origin")
        ("cpu", "Force CPU execution")
        ("n,num-iters", "Number of iterations to run", cxxopts::value<int>()->default_value("30000"))
        ("d,downscale-factor", "Scale input images by this factor.", cxxopts::value<float>()->default_value("1"))
        ("num-downscales", "Number of images downscales to use. After being scaled by [downscale-factor], images are initially scaled by a further (2^[num-downscales]) and the scale is increased every [resolution-schedule]", cxxopts::value<int>()->default_value("0"))
        ("resolution-schedule", "Double the image resolution every these many steps", cxxopts::value<int>()->default_value("3000"))
        ("sh-degree", "Maximum spherical harmonics degree (must be > 0)", cxxopts::value<int>()->default_value("3"))
        ("sh-degree-interval", "Increase the number of spherical harmonics degree after these many steps (will not exceed [sh-degree])", cxxopts::value<int>()->default_value("1000"))
        ("ssim-weight", "Weight to apply to the structural similarity loss. Set to zero to use least absolute deviation (L1) loss only", cxxopts::value<float>()->default_value("0.2"))
        ("refine-every", "Densify/prune gaussians every these many steps", cxxopts::value<int>()->default_value("500"))
        ("densify-from", "Start densifying gaussians after these many steps", cxxopts::value<int>()->default_value("500"))
        ("densify-until", "Stop densifying gaussians after these many steps (-1 = min(15000, half of num-iters))", cxxopts::value<int>()->default_value("-1"))
        ("loss-thresh", "High-error pixel threshold on the normalized L1 map for multi-view scoring", cxxopts::value<float>()->default_value("0.1"))
        ("no-edge-guidance", "Disable Canny edge weighting of the densification importance", cxxopts::value<bool>()->default_value("false"))
        ("max-gaussians", "Maximum number of gaussians (0 = unlimited)", cxxopts::value<int>()->default_value("5000000"))
        ("no-masks", "Ignore image masks even when present", cxxopts::value<bool>()->default_value("false"))
        ("no-gpu-cache", "Do not cache images/masks on the GPU (reduces VRAM usage, slower)", cxxopts::value<bool>()->default_value("false"))
        ("host-cache-mb", "Decoded host-image cache budget in MiB (0 is automatic)",
         cxxopts::value<unsigned long long>()->default_value("0"))
#ifdef USE_VISUALIZATION
        ("has-visualization", "Show the visualization steps of training", cxxopts::value<bool>()->default_value("0"))
#endif
        ("h,help", "Print usage")
        ("version", "Print version");
    options.parse_positional({"input"});
    options.positional_help("[colmap/nerfstudio/opensfm/odx/openmvg project path or .zip archive]");
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n' << options.help() << std::endl;
        return EXIT_FAILURE;
    }
    if (result.count("version")) {
        std::cout << APP_VERSION << std::endl;
        return EXIT_SUCCESS;
    }
    if (result.count("help") || !result.count("input")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    const std::string projectRoot = result["input"].as<std::string>();
    std::string outputScene = result["output"].as<std::string>();
    if (result.count("output") == 0){
        // Output next to the input scene, for a .zip, next to the archive
        fs::path in = fs::absolute(fs::path(projectRoot));
        if (in.filename().empty()) in = in.parent_path();
        outputScene = (in.parent_path() / outputScene).string();
    }
    const std::string outputCameras = result["output-cameras"].as<std::string>();
    const int saveEvery = result["save-every"].as<int>();
    const std::string resume = result["resume"].as<std::string>();
    const std::string valImage = result["val-image"].as<std::string>();
    const std::string valRender = result["val-render"].as<std::string>();
    const bool validate = result.count("val") || !valRender.empty();
    const bool keepCrs = !result.count("center");
    const int numIters = result["num-iters"].as<int>();
    const float downscaleFactor = std::max(1.0f, result["downscale-factor"].as<float>());
    const int numDownscales = result["num-downscales"].as<int>();
    const int resolutionSchedule = result["resolution-schedule"].as<int>();
    const int shDegree = result["sh-degree"].as<int>();
    const int shDegreeInterval = result["sh-degree-interval"].as<int>();
    const float ssimWeight = result["ssim-weight"].as<float>();
    const int refineEvery = result["refine-every"].as<int>();
    const int densifyFrom = result["densify-from"].as<int>();
    int densifyUntil = result["densify-until"].as<int>();
    if (densifyUntil < 0)
        densifyUntil = std::min(15000, numIters / 2);
    const float lossThresh = result["loss-thresh"].as<float>();
    const int maxGaussians = result["max-gaussians"].as<int>();
    const bool noMasks = result["no-masks"].as<bool>();
    const bool gpuCacheEnabled = !result["no-gpu-cache"].as<bool>();
    const bool edgeGuidance = !result["no-edge-guidance"].as<bool>();
    const unsigned long long cacheMb = result["host-cache-mb"].as<unsigned long long>();
    if (cacheMb > std::numeric_limits<std::uint64_t>::max() / (1ULL << 20)) {
        std::cerr << "--host-cache-mb is too large" << std::endl;
        return EXIT_FAILURE;
    }
    const std::uint64_t hostCacheBudget = resolveHostCacheBudget(cacheMb ? cacheMb << 20 : 0);
    std::cout << "Host image cache: " << (cacheMb ? "explicit " : "automatic ")
              << hostCacheBudget / (1ULL << 20) << " MiB" << std::endl;
    if (!valRender.empty())
        fs::create_directories(valRender);

    torch::Device device = torch::kCPU;
    int displayStep = 10;
    if (torch::hasCUDA() && !result.count("cpu")) {
        std::cout << "Using CUDA" << std::endl;
        device = torch::kCUDA;
    } else if (torch::hasMPS() && !result.count("cpu")) {
        std::cout << "Using MPS" << std::endl;
        device = torch::kMPS;
    } else {
        std::cout << "Using CPU" << std::endl;
        displayStep = 1;
    }

#ifdef USE_VISUALIZATION
    const bool hasVisualization = result["has-visualization"].as<bool>();
    Visualizer visualizer;
    if (hasVisualization)
        visualizer.Initialize(numIters);
#endif

    try {
        const std::string projectPath = isZipArchive(projectRoot) ? extractZipToCache(projectRoot) : projectRoot;
        InputData input = inputDataFromX(projectPath);
        if (!noMasks) {
            int found = 0;
            for (auto &camera : input.cameras) {
                camera.maskPath = findMaskPath(camera.filePath, projectPath);
                if (!camera.maskPath.empty())
                    ++found;
            }
            if (found)
                std::cout << "Found " << found << " masks" << std::endl;
        }
        CameraSplit split = input.splitCameras(validate, valImage);
        if (split.trainKeys.empty())
            throw std::runtime_error("Training requires at least one camera after validation splitting");
        const float extent = sceneExtent(input.cameras);
        CameraImageStore store(std::move(input.cameras), hostCacheBudget, gpuCacheEnabled, downscaleFactor,
                               device == torch::kCPU ? HostImageStorage::Float32 : HostImageStorage::UInt8);
        if (store.primeResidentSetIfFits(split.trainKeys))
            std::cout << "Primed " << split.trainKeys.size() << " camera images in the bounded host cache" << std::endl;
        Model model(input, store, split.trainKeys, extent, numDownscales, resolutionSchedule, shDegree,
                    shDegreeInterval, refineEvery, densifyFrom, densifyUntil, maxGaussians, lossThresh, numIters,
                    keepCrs, device);
        model.edgeGuidance = edgeGuidance;
        std::vector<size_t> indices(split.trainKeys.size());
        std::iota(indices.begin(), indices.end(), 0);
        InfiniteRandomIterator<size_t> iterator(indices);
        size_t step = resume.empty() ? 1 : static_cast<size_t>(model.loadPly(resume) + 1);
        for (; step <= static_cast<size_t>(numIters); ++step) {
            CameraKey currentKey = split.trainKeys[iterator.next()];
            const int downscale = model.getDownscaleFactor(static_cast<int>(step));
            auto lease = store.acquire(currentKey, FrameRequest{downscale, device, true, false});
            torch::Tensor rgb = model.forward(lease.camera(), static_cast<int>(step));
            torch::Tensor loss = model.mainLoss(rgb, lease.image(), lease.mask(), ssimWeight);
            loss.backward();
            if (step % displayStep == 0)
                std::cout << "Step " << step << ": " << loss.item<float>() << " ["
                          << static_cast<int>(100 * step / numIters) << "%]" << std::endl;
            model.afterTrain(static_cast<int>(step));
            model.optimizerStepCadence(static_cast<int>(step));
            model.schedulersStep(static_cast<int>(step));
            if (saveEvery > 0 && step % saveEvery == 0) {
                fs::path checkpoint(outputScene);
                checkpoint.replace_filename(checkpoint.stem().string() + "_" + std::to_string(step) +
                                            checkpoint.extension().string());
                model.save(checkpoint.string(), static_cast<int>(step));
            }
            if (!valRender.empty() && step % 10 == 0 && split.validationKey) {
                auto validation = store.acquire(*split.validationKey, FrameRequest{downscale, device, false, false});
                cv::Mat image =
                    tensorToImage(model.forward(validation.camera(), static_cast<int>(step)).detach().cpu());
                cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                cv::imwrite((fs::path(valRender) / (std::to_string(step) + ".png")).string(), image);
            }
#ifdef USE_VISUALIZATION
            if (hasVisualization) {
                visualizer.SetInitialGaussianNum(static_cast<int>(model.means.size(0)));
                visualizer.SetLoss(static_cast<int>(step), loss.item<float>());
                visualizer.SetGaussians(model.means, model.scales, model.featuresDc, model.opacities);
                visualizer.SetImage(rgb, lease.image());
                if (visualizer.QuitApp())
                    step = static_cast<size_t>(numIters) + 1;
                visualizer.Draw();
            }
#endif
        }
        if (!outputCameras.empty())
            saveCameras(outputCameras, store, input, keepCrs);
        const int finalStep = static_cast<int>(std::min(step, static_cast<size_t>(numIters)));
        model.save(outputScene, finalStep);
        if (split.validationKey) {
            const int downscale = model.getDownscaleFactor(finalStep);
            auto validation = store.acquire(*split.validationKey, FrameRequest{downscale, device, true, false});
            torch::Tensor rgb = model.forward(validation.camera(), finalStep);
            torch::Tensor loss = model.mainLoss(rgb, validation.image(), validation.mask(), ssimWeight);
            std::cout << validation.camera().filePath << " validation loss: " << loss.item<float>() << std::endl;
            torch::Tensor mse = validation.mask().defined() && validation.mask().numel()
                                    ? (validation.mask().unsqueeze(-1) * (rgb - validation.image()).pow(2)).sum() /
                                          (validation.mask().sum() * validation.image().size(2) + 1e-8f)
                                    : (rgb - validation.image()).pow(2).mean();
            std::cout << validation.camera().filePath << " validation PSNR: "
                      << (10.0f * torch::log10(1.0f / mse)).item<float>() << std::endl;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
