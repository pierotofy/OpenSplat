#include "camera_image_store.hpp"
#include "cv_utils.hpp"
#include "undistort.hpp"
#ifdef CAMERA_IMAGE_STORE_TESTING
#include "tests/camera_image_store_test_adapter.hpp"
#endif
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime_api.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sched.h>
#include <sys/sysinfo.h>
#endif

namespace {
#ifdef CAMERA_IMAGE_STORE_TESTING
std::atomic<CameraImageStoreTestHook> cameraImageStoreTestHook{nullptr};
void notifyCameraImageStoreTestHook(CameraKey key, CameraImageStoreTestEvent event) noexcept {
    if (auto hook = cameraImageStoreTestHook.load())
        hook(key, event);
}
#endif
std::uint64_t tensorBytes(const torch::Tensor &tensor) {
    if (!tensor.defined())
        return 0;
    const auto count = static_cast<std::uint64_t>(tensor.numel());
    const auto size = static_cast<std::uint64_t>(tensor.element_size());
    if (size && count > std::numeric_limits<std::uint64_t>::max() / size)
        throw std::overflow_error("Tensor byte count overflow");
    return count * size;
}
torch::Tensor packedImageTensor(const cv::Mat &image) {
    const cv::Mat contiguous = image.isContinuous() ? image : image.clone();
    return torch::from_blob(contiguous.data, {contiguous.rows, contiguous.cols, contiguous.channels()}, torch::kU8)
        .clone();
}
std::uint64_t floatImageBytes(const torch::Tensor &tensor) {
    const auto count = static_cast<std::uint64_t>(tensor.numel());
    if (count > std::numeric_limits<std::uint64_t>::max() / sizeof(float))
        throw std::overflow_error("Materialized image byte count overflow");
    return count * sizeof(float);
}
bool distorted(const Camera &c) {
    return c.k1 != 0.0f || c.k2 != 0.0f || c.k3 != 0.0f || c.k4 != 0.0f || c.k5 != 0.0f || c.k6 != 0.0f ||
           c.p1 != 0.0f || c.p2 != 0.0f;
}
std::string deviceKey(const torch::Device &device, char kind, int downscale) {
    std::ostringstream out;
    out << device << ':' << kind << ':' << downscale;
    return out.str();
}
std::uint64_t automaticDeviceCacheBudget() {
#ifdef USE_CUDA
    std::size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) == cudaSuccess && free)
        return static_cast<std::uint64_t>(free / 2);
#elif defined(USE_HIP)
    std::size_t free = 0, total = 0;
    if (hipMemGetInfo(&free, &total) == hipSuccess && free)
        return static_cast<std::uint64_t>(free / 2);
#elif defined(__APPLE__)
    std::uint64_t ram = 0;
    size_t size = sizeof(ram);
    if (sysctlbyname("hw.memsize", &ram, &size, nullptr, 0) == 0 && ram)
        return ram / 4;
#endif
    return 1ULL << 30;
}

std::uint64_t systemMemoryLimitBytes() {
    std::uint64_t ram = 0;
#ifdef __APPLE__
    std::uint64_t value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname("hw.memsize", &value, &size, nullptr, 0) == 0)
        ram = value;
#elif defined(__linux__)
    struct sysinfo info {};
    if (sysinfo(&info) == 0)
        ram = static_cast<std::uint64_t>(info.totalram) * info.mem_unit;
    for (const char *path : {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"}) {
        std::ifstream input(path);
        std::string value;
        if (input >> value && value != "max") {
            try {
                const std::uint64_t limit = std::stoull(value);
                if (limit && (!ram || limit < ram))
                    ram = limit;
            } catch (const std::exception &) {
            }
        }
    }
#endif
    return ram ? ram : 2ULL << 30;
}

std::size_t availableProcessorCount() {
    std::size_t count = std::thread::hardware_concurrency();
    if (!count)
        count = 1;
#ifdef __linux__
    cpu_set_t processors;
    CPU_ZERO(&processors);
    if (sched_getaffinity(0, sizeof(processors), &processors) == 0) {
        const int affinityCount = CPU_COUNT(&processors);
        if (affinityCount > 0)
            count = (std::min)(count, static_cast<std::size_t>(affinityCount));
    }
    std::ifstream cpuMax("/sys/fs/cgroup/cpu.max");
    std::string quota;
    std::uint64_t period = 0;
    if (cpuMax >> quota >> period && quota != "max" && period) {
        try {
            const std::uint64_t quotaValue = std::stoull(quota);
            count = (std::min)(count, static_cast<std::size_t>((quotaValue + period - 1) / period));
        } catch (const std::exception &) {
        }
    } else {
        std::ifstream quotaInput("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
        std::ifstream periodInput("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
        std::int64_t quotaValue = -1;
        if (quotaInput >> quotaValue && periodInput >> period && quotaValue > 0 && period)
            count = (std::min)(count, static_cast<std::size_t>((quotaValue + period - 1) / period));
    }
#endif
    return count ? count : 1;
}
} // namespace

#ifdef CAMERA_IMAGE_STORE_TESTING
void setCameraImageStoreTestHook(CameraImageStoreTestHook hook) noexcept { cameraImageStoreTestHook = hook; }
#endif

struct PreparedSourceMetadata {
    std::shared_ptr<PreparedCamera> prepared;
    std::optional<UndistortParams> undistort;
    int sourceWidth = 0;
    int sourceHeight = 0;
};

struct CameraPayloadGeneration {
    std::shared_ptr<PreparedCamera> prepared;
    std::shared_ptr<const PreparedSourceMetadata> sourceMetadata;
    torch::Tensor image, mask;
    std::unordered_map<int, torch::Tensor> imagePyramids, maskPyramids, edgePyramids;
    std::unordered_map<std::string, torch::Tensor> deviceTensors;
    std::uint64_t hostBytes = 0, deviceBytes = 0;
    std::size_t pins = 0;
    bool oversize = false;
};

struct StoreEntry {
    Camera descriptor;
    std::shared_ptr<PreparedCamera> prepared;
    std::shared_ptr<const PreparedSourceMetadata> sourceMetadata;
    std::shared_ptr<CameraPayloadGeneration> resident;
    bool loading = false;
    bool hasPublishedGeneration = false;
    std::condition_variable changed;
};

struct CameraStoreState {
    mutable std::mutex mutex;
    std::vector<std::unique_ptr<StoreEntry>> entries;
    std::vector<CameraKey> lru;
    std::uint64_t budget = 0;
    std::uint64_t deviceBudget = 0;
    bool gpuCache = true;
    HostImageStorage imageStorage = HostImageStorage::Float32;
    float initialDownscale = 1.0f;
    CacheStats stats;

    void assertLruLocked() const noexcept {
#ifndef NDEBUG
        assert(lru.size() <= entries.size());
        for (std::size_t i = 0; i < lru.size(); ++i) {
            assert(lru[i] < entries.size());
            assert(entries[lru[i]]->resident);
            for (std::size_t j = i + 1; j < lru.size(); ++j)
                assert(lru[i] != lru[j]);
        }
        for (CameraKey key = 0; key < entries.size(); ++key) {
            const auto count = static_cast<std::size_t>(std::count(lru.begin(), lru.end(), key));
            assert(count == (entries[key]->resident ? 1 : 0));
        }
#endif
    }
    void promoteLocked(CameraKey key) {
        lru.erase(std::remove(lru.begin(), lru.end(), key), lru.end());
        assert(lru.size() < lru.capacity());
        lru.push_back(key);
        assertLruLocked();
    }
    void settleLocked() {
        while (stats.residentHostBytes > budget) {
            auto victim = std::find_if(lru.begin(), lru.end(), [this](CameraKey key) {
                const StoreEntry &entry = *entries[key];
                return entry.resident && entry.resident->pins == 0;
            });
            if (victim == lru.end())
                break;
            CameraKey key = *victim;
            StoreEntry &entry = *entries[key];
            assert(stats.residentHostBytes >= entry.resident->hostBytes);
            assert(stats.residentDeviceBytes >= entry.resident->deviceBytes);
            stats.residentHostBytes -= entry.resident->hostBytes;
            stats.residentDeviceBytes -= entry.resident->deviceBytes;
            entry.resident.reset();
            lru.erase(victim);
            ++stats.evictions;
        }
        assertLruLocked();
    }
    void release(CameraKey key, const std::shared_ptr<CameraPayloadGeneration> &generation,
                 std::uint64_t transientDeviceBytes = 0) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        assert(stats.residentDeviceBytes >= transientDeviceBytes);
        stats.residentDeviceBytes -= transientDeviceBytes;
        assert(key < entries.size());
        StoreEntry &entry = *entries[key];
        assert(entry.resident && entry.resident == generation && generation->pins > 0);
        if (!entry.resident || entry.resident != generation || generation->pins == 0)
            return;
        --generation->pins;
        if (!generation->pins)
            settleLocked();
        else
            assertLruLocked();
    }
};

struct MetadataCandidate {
    std::shared_ptr<PreparedCamera> prepared;
    std::optional<UndistortParams> undistort;
};

struct GenerationSnapshot {
    explicit GenerationSnapshot(const CameraPayloadGeneration &generation)
        : imagePyramids(generation.imagePyramids), maskPyramids(generation.maskPyramids),
          edgePyramids(generation.edgePyramids), deviceTensors(generation.deviceTensors),
          hostBytes(generation.hostBytes), deviceBytes(generation.deviceBytes), oversize(generation.oversize) {}
    void restore(CameraPayloadGeneration &generation) {
        generation.imagePyramids = std::move(imagePyramids);
        generation.maskPyramids = std::move(maskPyramids);
        generation.edgePyramids = std::move(edgePyramids);
        generation.deviceTensors = std::move(deviceTensors);
        generation.hostBytes = hostBytes;
        generation.deviceBytes = deviceBytes;
        generation.oversize = oversize;
    }
    std::unordered_map<int, torch::Tensor> imagePyramids, maskPyramids, edgePyramids;
    std::unordered_map<std::string, torch::Tensor> deviceTensors;
    std::uint64_t hostBytes, deviceBytes;
    bool oversize;
};

static bool requestMayMutate(const CameraStoreState &state, const CameraPayloadGeneration &generation,
                             const FrameRequest &request) {
    if (request.downscale > 1 && generation.imagePyramids.count(request.downscale) == 0)
        return true;
    if (request.needMask && request.downscale > 1 && generation.maskPyramids.count(request.downscale) == 0)
        return true;
    if (request.needEdgeMap && generation.edgePyramids.count(request.downscale) == 0)
        return true;
    if (!state.gpuCache || request.device == torch::kCPU)
        return false;
    for (char kind : {'i', 'm', 'e'}) {
        if ((kind == 'm' && (!request.needMask || !generation.mask.defined())) ||
            (kind == 'e' && !request.needEdgeMap))
            continue;
        if (generation.deviceTensors.count(deviceKey(request.device, kind, request.downscale)) == 0)
            return true;
    }
    return false;
}

static MetadataCandidate prepareMetadata(const Camera &descriptor, float initialDownscale, int actualWidth,
                                         int actualHeight) {
    if (descriptor.height <= 0 || descriptor.width <= 0 || actualWidth <= 0 || actualHeight <= 0)
        throw std::runtime_error("Invalid dimensions for camera " + std::to_string(descriptor.key));
    auto prepared = std::make_shared<PreparedCamera>();
    prepared->fx = descriptor.fx;
    prepared->fy = descriptor.fy;
    prepared->cx = descriptor.cx;
    prepared->cy = descriptor.cy;
    prepared->camToWorld = descriptor.camToWorld;
    prepared->filePath = descriptor.filePath;

    float actualScale = static_cast<float>(actualHeight) / static_cast<float>(descriptor.height);
    prepared->fx *= actualScale;
    prepared->fy *= actualScale;
    prepared->cx *= actualScale;
    prepared->cy *= actualScale;
    float scale = 1.0f / initialDownscale;
    prepared->fx *= scale;
    prepared->fy *= scale;
    prepared->cx *= scale;
    prepared->cy *= scale;
    prepared->width = (std::max)(1, cvRound(static_cast<float>(actualWidth) * scale));
    prepared->height = (std::max)(1, cvRound(static_cast<float>(actualHeight) * scale));

    MetadataCandidate result{prepared, std::nullopt};
    if (distorted(descriptor)) {
        result.undistort = computeUndistortParams(
            prepared->fx, prepared->fy, prepared->cx, prepared->cy, prepared->width, prepared->height, descriptor.k1,
            descriptor.k2, descriptor.k3, descriptor.k4, descriptor.k5, descriptor.k6, descriptor.p1, descriptor.p2);
        prepared->fx = result.undistort->dstFx;
        prepared->fy = result.undistort->dstFy;
        prepared->cx = result.undistort->dstCx;
        prepared->cy = result.undistort->dstCy;
        prepared->width = result.undistort->dstW;
        prepared->height = result.undistort->dstH;
    }
    return result;
}

static std::shared_ptr<const PreparedSourceMetadata> prepareSourceMetadata(const Camera &descriptor,
                                                                           float initialDownscale) {
#ifdef CAMERA_IMAGE_STORE_TESTING
    notifyCameraImageStoreTestHook(descriptor.key, CameraImageStoreTestEvent::SourceMetadataProbe);
#endif
    const cv::Size dimensions = imageDimensions(descriptor.filePath);
    MetadataCandidate candidate =
        prepareMetadata(descriptor, initialDownscale, dimensions.width, dimensions.height);
    auto result = std::make_shared<PreparedSourceMetadata>();
    result->prepared = std::move(candidate.prepared);
    result->undistort = std::move(candidate.undistort);
    result->sourceWidth = dimensions.width;
    result->sourceHeight = dimensions.height;
    return result;
}

template <typename Function>
static void parallelFor(std::size_t itemCount, std::size_t workerCount, Function function) {
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr firstFailure;
    std::mutex failureMutex;
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (std::size_t worker = 0; worker < workerCount; ++worker) {
        workers.emplace_back([&] {
            while (!failed.load()) {
                const std::size_t index = next.fetch_add(1);
                if (index >= itemCount)
                    return;
                try {
                    function(index);
                } catch (...) {
                    failed = true;
                    std::lock_guard<std::mutex> lock(failureMutex);
                    if (!firstFailure)
                        firstFailure = std::current_exception();
                }
            }
        });
    }
    for (auto &worker : workers)
        worker.join();
    if (firstFailure)
        std::rethrow_exception(firstFailure);
}

static std::shared_ptr<CameraPayloadGeneration> decodeCandidate(
    const Camera &descriptor, float initialDownscale,
    std::shared_ptr<const PreparedSourceMetadata> metadata, HostImageStorage imageStorage) {
#ifdef CAMERA_IMAGE_STORE_TESTING
    notifyCameraImageStoreTestHook(descriptor.key, CameraImageStoreTestEvent::SourceDecode);
#endif
    if (!metadata)
        metadata = prepareSourceMetadata(descriptor, initialDownscale);
    cv::Mat image = imreadRGB(descriptor.filePath);
    if (image.cols != metadata->sourceWidth || image.rows != metadata->sourceHeight)
        throw std::runtime_error("Image dimensions changed while loading camera " +
                                 std::to_string(descriptor.key));
    auto prepared = metadata->prepared;
    cv::Mat mask;
    if (!descriptor.maskPath.empty()) {
        mask = cv::imread(descriptor.maskPath, cv::IMREAD_GRAYSCALE);
        if (mask.empty())
            throw std::runtime_error("Cannot read mask " + descriptor.maskPath + " for camera " +
                                     std::to_string(descriptor.key));
    }
    if (initialDownscale > 1.0f) {
        float scale = 1.0f / initialDownscale;
        cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    if (!mask.empty()) {
        cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
        if (mask.rows != image.rows || mask.cols != image.cols)
            cv::resize(mask, mask, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_LINEAR);
    }
    if (metadata->undistort) {
        cv::Mat mapx, mapy, remapped;
        buildUndistortMaps(*metadata->undistort, mapx, mapy);
        cv::remap(image, remapped, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        image = remapped;
        if (!mask.empty()) {
            cv::remap(mask, remapped, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            mask = remapped;
        }
    }
    if (prepared->height != image.rows || prepared->width != image.cols)
        throw std::runtime_error("Prepared dimensions do not match decoded image for camera " +
                                 std::to_string(descriptor.key));
    auto generation = std::make_shared<CameraPayloadGeneration>();
    generation->prepared = prepared;
    generation->sourceMetadata = std::move(metadata);
    generation->image = imageStorage == HostImageStorage::UInt8 ? packedImageTensor(image)
                                                                 : imageToTensor(image).contiguous();
    if (!mask.empty()) {
        generation->mask =
            torch::from_blob(mask.data, {mask.rows, mask.cols}, torch::kU8).to(torch::kFloat32).div(255.0f).clone();
        generation->mask = (generation->mask >= 0.5f).to(torch::kFloat32);
    }
    generation->hostBytes = tensorBytes(generation->image) + tensorBytes(generation->mask);
    return generation;
}

static torch::Tensor &hostTensor(CameraPayloadGeneration &generation, char kind, int downscale) {
    if (kind == 'i' && downscale <= 1)
        return generation.image;
    if (kind == 'm' && downscale <= 1)
        return generation.mask;
    auto &map = kind == 'i'   ? generation.imagePyramids
                : kind == 'm' ? generation.maskPyramids
                              : generation.edgePyramids;
    auto found = map.find(downscale);
    if (found != map.end())
        return found->second;
    torch::Tensor result;
    if (kind == 'i') {
        cv::Mat image = tensorToImage(generation.image);
        cv::resize(image, image, cv::Size(image.cols / downscale, image.rows / downscale), 0, 0, cv::INTER_AREA);
        result = generation.image.scalar_type() == torch::kUInt8 ? packedImageTensor(image)
                                                                  : imageToTensor(image).contiguous();
    } else if (kind == 'm') {
        if (!generation.mask.defined()) {
            map.emplace(downscale, torch::Tensor());
            return map.at(downscale);
        }
        auto value = torch::nn::functional::interpolate(
            generation.mask.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{generation.mask.size(0) / downscale, generation.mask.size(1) / downscale})
                .mode(torch::kBilinear)
                .align_corners(false));
        result = (value.squeeze(0).squeeze(0) >= 0.5f).to(torch::kFloat32);
    } else {
        torch::Tensor &source = hostTensor(generation, 'i', downscale);
        cv::Mat image = tensorToImage(source), gray, edges;
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
        cv::Canny(gray, edges, 50, 150);
        result =
            torch::from_blob(edges.data, {edges.rows, edges.cols}, torch::kU8).to(torch::kFloat32).div(255.0f).clone();
    }
    const std::uint64_t bytes = tensorBytes(result);
    if (bytes > std::numeric_limits<std::uint64_t>::max() - generation.hostBytes)
        throw std::overflow_error("Host cache byte count overflow");
    auto inserted = map.emplace(downscale, std::move(result));
    generation.hostBytes += bytes;
    return inserted.first->second;
}

static torch::Tensor requestedTensor(CameraStoreState &state, CameraPayloadGeneration &generation, char kind,
                                     int downscale, const torch::Device &device, std::uint64_t oldDeviceBytes,
                                     std::uint64_t &transientBytes) {
    torch::Tensor &host = hostTensor(generation, kind, downscale);
    if (!host.defined())
        return host;
    const bool packedImage = kind == 'i' && host.scalar_type() == torch::kUInt8;
    auto materialize = [&] {
        if (packedImage)
            return host.to(device, torch::kFloat32).div_(255.0f);
        return host.to(device);
    };
    if (device == torch::kCPU)
        return packedImage ? materialize() : host;
    std::string key = deviceKey(device, kind, downscale);
    auto found = generation.deviceTensors.find(key);
    if (found != generation.deviceTensors.end())
        return found->second;
    const std::uint64_t bytes = packedImage ? floatImageBytes(host) : tensorBytes(host);
    if (!state.deviceBudget)
        state.deviceBudget = automaticDeviceCacheBudget();
    const std::uint64_t pendingCachedBytes = generation.deviceBytes - oldDeviceBytes;
    const bool fitsCache = bytes <= state.deviceBudget && state.stats.residentDeviceBytes <= state.deviceBudget - bytes &&
                           pendingCachedBytes <= state.deviceBudget - bytes - state.stats.residentDeviceBytes;
    if (!state.gpuCache || !fitsCache) {
        if (bytes > std::numeric_limits<std::uint64_t>::max() - transientBytes)
            throw std::overflow_error("Transient device byte count overflow");
        transientBytes += bytes;
        return materialize();
    }
    torch::Tensor value = materialize();
    if (bytes > std::numeric_limits<std::uint64_t>::max() - generation.deviceBytes)
        throw std::overflow_error("Device cache byte count overflow");
    auto inserted = generation.deviceTensors.emplace(key, std::move(value));
    generation.deviceBytes += bytes;
    return inserted.first->second;
}

CameraImageStore::CameraImageStore(std::vector<Camera> descriptors, std::uint64_t hostBudgetBytes,
                                   bool gpuCacheEnabled, float initialDownscale, HostImageStorage imageStorage)
    : state_(std::make_shared<CameraStoreState>()) {
    if (!hostBudgetBytes)
        throw std::invalid_argument("Host cache budget must be positive");
    if (initialDownscale < 1.0f)
        throw std::invalid_argument("Initial downscale must be at least 1");
    state_->budget = hostBudgetBytes;
    state_->gpuCache = gpuCacheEnabled;
    state_->initialDownscale = initialDownscale;
    state_->imageStorage = imageStorage;
    state_->entries.reserve(descriptors.size());
    state_->lru.reserve(descriptors.size());
    for (size_t i = 0; i < descriptors.size(); ++i) {
        if (descriptors[i].key != i)
            throw std::invalid_argument("Camera keys must be contiguous and stable");
        auto entry = std::make_unique<StoreEntry>();
        entry->descriptor = std::move(descriptors[i]);
        state_->entries.push_back(std::move(entry));
    }
}

CameraFrameLease CameraImageStore::acquire(CameraKey key, const FrameRequest &request) {
    return acquireImpl(key, request, true);
}

CameraFrameLease CameraImageStore::acquireImpl(CameraKey key, const FrameRequest &request, bool materializeImage) {
    if (request.downscale < 1)
        throw std::invalid_argument("Frame downscale must be positive");
    if (key >= state_->entries.size())
        throw std::out_of_range("Invalid camera key " + std::to_string(key));
    std::shared_ptr<CameraPayloadGeneration> generation;
    bool newlyPublished = false;
#ifdef CAMERA_IMAGE_STORE_TESTING
    bool startedLoading = false;
#endif
    bool reloading = false;
    {
        std::unique_lock<std::mutex> lock(state_->mutex);
        StoreEntry &entry = *state_->entries[key];
        while (entry.loading) {
            ++state_->stats.waits;
#ifdef CAMERA_IMAGE_STORE_TESTING
            notifyCameraImageStoreTestHook(key, CameraImageStoreTestEvent::Wait);
#endif
            entry.changed.wait(lock);
        }
        reloading = entry.hasPublishedGeneration;
        if (entry.resident) {
            generation = entry.resident;
            ++state_->stats.hits;
            ++generation->pins;
        } else {
            entry.loading = true;
#ifdef CAMERA_IMAGE_STORE_TESTING
            startedLoading = true;
#endif
            ++state_->stats.misses;
        }
    }
#ifdef CAMERA_IMAGE_STORE_TESTING
    if (startedLoading)
        notifyCameraImageStoreTestHook(key, CameraImageStoreTestEvent::AcquireLoading);
#endif
    if (!generation) {
        std::shared_ptr<CameraPayloadGeneration> candidate;
        try {
            std::shared_ptr<const PreparedSourceMetadata> sourceMetadata;
            {
                std::lock_guard<std::mutex> lock(state_->mutex);
                sourceMetadata = state_->entries[key]->sourceMetadata;
            }
            candidate = decodeCandidate(state_->entries[key]->descriptor, state_->initialDownscale, sourceMetadata,
                                        state_->imageStorage);
        } catch (...) {
            std::lock_guard<std::mutex> lock(state_->mutex);
            StoreEntry &entry = *state_->entries[key];
            entry.loading = false;
            ++state_->stats.decodeFailures;
            entry.changed.notify_all();
            throw;
        }
        std::lock_guard<std::mutex> lock(state_->mutex);
        candidate->pins = 1;
        generation = std::move(candidate);
        newlyPublished = true;
    }
    torch::Tensor image, mask, edge;
    std::uint64_t transientDeviceBytes = 0;
    std::uint64_t oldHostBytes = 0, oldDeviceBytes = 0;
    std::optional<GenerationSnapshot> snapshot;
    try {
        std::lock_guard<std::mutex> lock(state_->mutex);
        oldHostBytes = generation->hostBytes;
        oldDeviceBytes = generation->deviceBytes;
        if (!newlyPublished && requestMayMutate(*state_, *generation, request))
            snapshot.emplace(*generation);
        const PreparedCamera &camera = *generation->prepared;
        if (camera.width / request.downscale < 1 || camera.height / request.downscale < 1)
            throw std::invalid_argument("Frame downscale exceeds camera dimensions");
        if (materializeImage)
            image = requestedTensor(*state_, *generation, 'i', request.downscale, request.device, oldDeviceBytes,
                                    transientDeviceBytes);
        if (request.needMask)
            mask = requestedTensor(*state_, *generation, 'm', request.downscale, request.device, oldDeviceBytes,
                                   transientDeviceBytes);
        if (request.needEdgeMap)
            edge = requestedTensor(*state_, *generation, 'e', request.downscale, request.device, oldDeviceBytes,
                                   transientDeviceBytes);
        if (newlyPublished) {
            StoreEntry &entry = *state_->entries[key];
            entry.hasPublishedGeneration = true;
            entry.prepared = generation->prepared;
            entry.sourceMetadata = generation->sourceMetadata;
            entry.resident = generation;
            state_->promoteLocked(key);
            entry.loading = false;
            entry.changed.notify_all();
            if (reloading)
                ++state_->stats.reloads;
            state_->stats.residentHostBytes += generation->hostBytes;
            state_->stats.residentDeviceBytes += generation->deviceBytes + transientDeviceBytes;
        } else {
            state_->stats.residentHostBytes += generation->hostBytes - oldHostBytes;
            state_->stats.residentDeviceBytes += generation->deviceBytes - oldDeviceBytes + transientDeviceBytes;
            state_->promoteLocked(key);
        }
        if (!generation->oversize && generation->hostBytes > state_->budget) {
            ++state_->stats.oversizeAdmissions;
            generation->oversize = true;
            std::fprintf(stderr, "Camera %u host payload %llu bytes exceeds cache budget %llu bytes\n", key,
                         static_cast<unsigned long long>(generation->hostBytes),
                         static_cast<unsigned long long>(state_->budget));
        }
        state_->stats.peakHostBytes = std::max(state_->stats.peakHostBytes, state_->stats.residentHostBytes);
        state_->stats.peakDeviceBytes = std::max(state_->stats.peakDeviceBytes, state_->stats.residentDeviceBytes);
        state_->settleLocked();
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            if (snapshot)
                snapshot->restore(*generation);
            if (newlyPublished) {
                StoreEntry &entry = *state_->entries[key];
                entry.loading = false;
                entry.changed.notify_all();
                generation->pins = 0;
            }
        }
        if (!newlyPublished)
            state_->release(key, generation);
        throw;
    }
    return CameraFrameLease(state_, generation, key, std::move(image), std::move(mask), std::move(edge),
                            transientDeviceBytes);
}

bool CameraImageStore::primeResidentSetIfFits(const std::vector<CameraKey> &keys, std::size_t maxWorkers) {
    if (keys.empty())
        return true;
    for (CameraKey key : keys) {
        if (key >= state_->entries.size())
            throw std::out_of_range("Invalid camera key " + std::to_string(key));
    }
    const std::size_t processorCount = availableProcessorCount();
    const std::size_t requestedWorkers = maxWorkers ? (std::min)(maxWorkers, processorCount) : processorCount;
    parallelFor(keys.size(), (std::min)(requestedWorkers, keys.size()),
                [&](std::size_t index) { (void)metadata(keys[index]); });

    std::uint64_t estimatedBytes = 0;
    std::uint64_t largestDecodeScratch = 0;
    for (CameraKey key : keys) {
        const PreparedCamera &camera = metadata(key);
        const StoreEntry &entry = *state_->entries[key];
        const std::uint64_t pixels =
            static_cast<std::uint64_t>(camera.width) * static_cast<std::uint64_t>(camera.height);
        const std::uint64_t imageBytesPerPixel =
            state_->imageStorage == HostImageStorage::UInt8 ? 3 : 3 * sizeof(float);
        const std::uint64_t bytesPerPixel =
            imageBytesPerPixel + (entry.descriptor.maskPath.empty() ? 0 : sizeof(float));
        if (pixels > std::numeric_limits<std::uint64_t>::max() / bytesPerPixel ||
            estimatedBytes > std::numeric_limits<std::uint64_t>::max() - pixels * bytesPerPixel)
            throw std::overflow_error("Resident-set prime byte estimate overflow");
        estimatedBytes += pixels * bytesPerPixel;

        const auto sourcePixels = static_cast<std::uint64_t>(entry.sourceMetadata->sourceWidth) *
                                  static_cast<std::uint64_t>(entry.sourceMetadata->sourceHeight);
        const std::uint64_t maskChannels = entry.descriptor.maskPath.empty() ? 0 : 1;
        const std::uint64_t decodedChannels = 3 + maskChannels;
        if (sourcePixels > std::numeric_limits<std::uint64_t>::max() / decodedChannels ||
            pixels > std::numeric_limits<std::uint64_t>::max() / decodedChannels ||
            sourcePixels * decodedChannels >
                std::numeric_limits<std::uint64_t>::max() - pixels * decodedChannels)
            throw std::overflow_error("Resident-set prime decode estimate overflow");
        std::uint64_t decodeScratch = (sourcePixels + pixels) * decodedChannels;
        if (entry.sourceMetadata->undistort) {
            // remap keeps both float coordinate maps and a second output image
            // (plus a second output mask when present) live during decoding.
            const std::uint64_t remapBytesPerPixel = 2 * sizeof(float) + decodedChannels;
            if (pixels > std::numeric_limits<std::uint64_t>::max() / remapBytesPerPixel ||
                decodeScratch > std::numeric_limits<std::uint64_t>::max() - pixels * remapBytesPerPixel)
                throw std::overflow_error("Resident-set prime undistort estimate overflow");
            decodeScratch += pixels * remapBytesPerPixel;
        }
        largestDecodeScratch = (std::max)(largestDecodeScratch, decodeScratch);
    }
    if (estimatedBytes > state_->budget)
        return false;

    // Keep simultaneous source/output decode buffers to a bounded fraction of
    // effective RAM; the resident tensors themselves are covered by budget.
    const std::uint64_t transientAllowance = systemMemoryLimitBytes() / 8;
    const std::size_t memoryWorkers = largestDecodeScratch
                                          ? static_cast<std::size_t>((std::max)(std::uint64_t{1},
                                                                               transientAllowance /
                                                                                   largestDecodeScratch))
                                          : requestedWorkers;
    const std::size_t workerCount = (std::min)({requestedWorkers, keys.size(), memoryWorkers});
    parallelFor(keys.size(), workerCount, [&](std::size_t index) { acquireImpl(keys[index], {}, false) = {}; });
    return true;
}

const PreparedCamera &CameraImageStore::metadata(CameraKey key) {
    if (key >= state_->entries.size())
        throw std::out_of_range("Invalid camera key " + std::to_string(key));
    {
        std::unique_lock<std::mutex> lock(state_->mutex);
        StoreEntry &entry = *state_->entries[key];
        while (entry.loading) {
            ++state_->stats.waits;
#ifdef CAMERA_IMAGE_STORE_TESTING
            notifyCameraImageStoreTestHook(key, CameraImageStoreTestEvent::Wait);
#endif
            entry.changed.wait(lock);
        }
        if (entry.prepared)
            return *entry.prepared;
        entry.loading = true;
    }
#ifdef CAMERA_IMAGE_STORE_TESTING
    notifyCameraImageStoreTestHook(key, CameraImageStoreTestEvent::MetadataLoading);
#endif
    std::shared_ptr<const PreparedSourceMetadata> candidate;
    try {
        const Camera &descriptor = state_->entries[key]->descriptor;
        candidate = prepareSourceMetadata(descriptor, state_->initialDownscale);
    } catch (...) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        StoreEntry &entry = *state_->entries[key];
        entry.loading = false;
        ++state_->stats.decodeFailures;
        entry.changed.notify_all();
        throw;
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    StoreEntry &entry = *state_->entries[key];
    entry.sourceMetadata = std::move(candidate);
    entry.prepared = entry.sourceMetadata->prepared;
    entry.loading = false;
    entry.changed.notify_all();
    return *entry.prepared;
}
CacheStats CameraImageStore::stats() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->stats;
}
std::size_t CameraImageStore::size() const { return state_->entries.size(); }

CameraFrameLease::CameraFrameLease(std::shared_ptr<CameraStoreState> state,
                                   std::shared_ptr<CameraPayloadGeneration> generation, CameraKey key,
                                   torch::Tensor image, torch::Tensor mask, torch::Tensor edge,
                                   std::uint64_t transientDeviceBytes)
    : state_(std::move(state)), generation_(std::move(generation)), key_(key), image_(std::move(image)),
      mask_(std::move(mask)), edge_(std::move(edge)), transientDeviceBytes_(transientDeviceBytes) {}
CameraFrameLease::CameraFrameLease(CameraFrameLease &&other) noexcept { *this = std::move(other); }
CameraFrameLease &CameraFrameLease::operator=(CameraFrameLease &&other) noexcept {
    if (this != &other) {
        release();
        state_ = std::move(other.state_);
        generation_ = std::move(other.generation_);
        key_ = other.key_;
        image_ = std::move(other.image_);
        mask_ = std::move(other.mask_);
        edge_ = std::move(other.edge_);
        transientDeviceBytes_ = other.transientDeviceBytes_;
        other.transientDeviceBytes_ = 0;
    }
    return *this;
}
CameraFrameLease::~CameraFrameLease() noexcept { release(); }
void CameraFrameLease::release() noexcept {
    image_ = torch::Tensor();
    mask_ = torch::Tensor();
    edge_ = torch::Tensor();
    if (state_ && generation_)
        state_->release(key_, generation_, transientDeviceBytes_);
    transientDeviceBytes_ = 0;
    generation_.reset();
    state_.reset();
}
const PreparedCamera &CameraFrameLease::camera() const {
    if (!generation_)
        throw std::logic_error("Empty camera lease");
    return *generation_->prepared;
}
const torch::Tensor &CameraFrameLease::image() const { return image_; }
const torch::Tensor &CameraFrameLease::mask() const { return mask_; }
const torch::Tensor &CameraFrameLease::edgeMap() const { return edge_; }

std::uint64_t resolveHostCacheBudget(std::uint64_t requestedBytes) {
    if (requestedBytes)
        return requestedBytes;
    const std::uint64_t ram = systemMemoryLimitBytes();
    return std::min<std::uint64_t>(8ULL << 30, std::max<std::uint64_t>(512ULL << 20, ram / 4));
}
