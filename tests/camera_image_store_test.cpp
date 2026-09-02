#include "../camera_image_store.hpp"
#include "../cv_utils.hpp"
#include "camera_image_store_test_adapter.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <type_traits>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;
static int processId() {
#ifdef _WIN32
    return _getpid();
#else
    return getpid();
#endif
}
static void waitUntil(const std::atomic<int> &value, int expected, const char *description) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (value.load() != expected) {
        if (std::chrono::steady_clock::now() >= deadline) {
            std::cerr << "timed out waiting for " << description << std::endl;
            std::_Exit(1);
        }
        std::this_thread::yield();
    }
}
static void joinWithin(std::vector<std::thread> &threads, const std::atomic<int> &completed, int expected) {
    waitUntil(completed, expected, "camera-store worker threads");
    for (auto &thread : threads)
        thread.join();
}
static std::atomic<int> hookLoaderBlocked{0}, hookWaiters{0};
static std::atomic<int> hookSourceMetadataProbes{0};
static std::atomic<bool> hookReleaseLoader{false};
static std::atomic<bool> hookSecondDecodeEntered{false}, hookParallelDecodeObserved{false};
static CameraImageStoreTestEvent hookBlockingEvent = CameraImageStoreTestEvent::AcquireLoading;
static void loadingHook(CameraKey key, CameraImageStoreTestEvent event) noexcept {
    if (key != 0)
        return;
    if (event == CameraImageStoreTestEvent::Wait) {
        ++hookWaiters;
        return;
    }
    if (event != hookBlockingEvent)
        return;
    ++hookLoaderBlocked;
    while (!hookReleaseLoader.load())
        std::this_thread::yield();
}
static void resetLoadingHook(CameraImageStoreTestEvent blockingEvent) {
    hookLoaderBlocked = 0;
    hookWaiters = 0;
    hookReleaseLoader = false;
    hookBlockingEvent = blockingEvent;
    setCameraImageStoreTestHook(loadingHook);
}
static void decodeCountingHook(CameraKey, CameraImageStoreTestEvent event) noexcept {
    if (event == CameraImageStoreTestEvent::SourceMetadataProbe)
        ++hookSourceMetadataProbes;
}
static void parallelDecodeHook(CameraKey key, CameraImageStoreTestEvent event) noexcept {
    if (event != CameraImageStoreTestEvent::SourceDecode)
        return;
    if (key == 0) {
        ++hookLoaderBlocked;
        while (!hookReleaseLoader.load())
            std::this_thread::yield();
    } else if (key == 1) {
        hookSecondDecodeEntered = true;
        if (!hookReleaseLoader.load())
            hookParallelDecodeObserved = true;
    }
}
namespace ns {
InputData inputDataFromNerfStudio(const std::string &) { return {}; }
} // namespace ns
namespace cm {
InputData inputDataFromColmap(const std::string &) { return {}; }
} // namespace cm
namespace osfm {
InputData inputDataFromOpenSfM(const std::string &) { return {}; }
} // namespace osfm
namespace omvg {
InputData inputDataFromOpenMVG(const std::string &) { return {}; }
} // namespace omvg
#define ASSERT_TRUE(value)                                                                                             \
    do {                                                                                                               \
        if (!(value))                                                                                                  \
            throw std::runtime_error(std::string("assertion failed: ") + #value + " at line " +                        \
                                     std::to_string(__LINE__));                                                        \
    } while (0)

static void ppm(const fs::path &path, int width, int height) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << width << ' ' << height << "\n255\n";
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            char rgb[3] = {static_cast<char>(x * 7), static_cast<char>(y * 11), static_cast<char>((x + y) * 3)};
            out.write(rgb, 3);
        }
}
static void pgm(const fs::path &path, int width, int height) {
    std::ofstream out(path, std::ios::binary);
    out << "P5\n" << width << ' ' << height << "\n255\n";
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            char value = x < width / 2 ? static_cast<char>(255) : 0;
            out.write(&value, 1);
        }
}
static void addExifOrientation(const fs::path &path, unsigned char orientation) {
    std::ifstream in(path, std::ios::binary);
    std::vector<char> jpeg((std::istreambuf_iterator<char>(in)), {});
    ASSERT_TRUE(jpeg.size() > 2 && static_cast<unsigned char>(jpeg[0]) == 0xff &&
                static_cast<unsigned char>(jpeg[1]) == 0xd8);
    const unsigned char app1[] = {0xff, 0xe1, 0x00, 0x22, 'E', 'x',         'i', 'f', 0,    0,    'M', 'M',
                                  0,    42,   0,    0,    0,   8,           0,   1,   0x01, 0x12, 0,   3,
                                  0,    0,    0,    1,    0,   orientation, 0,   0,   0,    0,    0,   0};
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(jpeg.data(), 2);
    out.write(reinterpret_cast<const char *>(app1), sizeof(app1));
    out.write(jpeg.data() + 2, jpeg.size() - 2);
}
static Camera camera(CameraKey key, const fs::path &image, int declaredWidth = 32, int declaredHeight = 24) {
    Camera value(declaredWidth, declaredHeight, 30, 30, declaredWidth / 2.0f, declaredHeight / 2.0f, 0, 0, 0,
                           0, 0, torch::eye(4), image.string());
    value.key = key;
    return value;
}

int main() {
    static_assert(std::is_nothrow_destructible_v<CameraFrameLease>);
    static_assert(std::is_nothrow_move_assignable_v<CameraFrameLease>);
    fs::path root = fs::temp_directory_path() / ("opensplat-camera-store-" + std::to_string(processId()));
    fs::create_directories(root);
    try {
        fs::path one = root / "one.ppm", two = root / "two.ppm", three = root / "three.ppm",
                 mask = root / "one.png";
        ppm(one, 32, 24);
        ppm(two, 32, 24);
        ppm(three, 32, 24);
        pgm(mask, 32, 24);
        {
            cv::Mat pixels(23, 31, CV_8UC3, cv::Scalar(7, 11, 13));
            fs::path jpeg = root / "dimensions.jpg", png = root / "dimensions.png";
            ASSERT_TRUE(cv::imwrite(jpeg.string(), pixels) && cv::imwrite(png.string(), pixels));
            ASSERT_TRUE(imageDimensions(jpeg.string()) == cv::Size(31, 23));
            ASSERT_TRUE(imageDimensions(png.string()) == cv::Size(31, 23));
            addExifOrientation(jpeg, 6);
            auto oriented = imageDimensions(jpeg.string());
            ASSERT_TRUE(oriented == cv::Size(23, 31));
            ASSERT_TRUE(cv::imread(jpeg.string()).size() == imageDimensions(jpeg.string()));
        }
        {
            auto mismatched = camera(0, one, 40, 30);
            mismatched.k1 = .01f;
            mismatched.k2 = -.005f;
            CameraImageStore store({mismatched}, 1 << 20, false);
            const PreparedCamera &metadata = store.metadata(0);
            ASSERT_TRUE(metadata.width > 0 && metadata.height > 0);
            auto metadataStats = store.stats();
            ASSERT_TRUE(metadataStats.misses == 0);
            ASSERT_TRUE(metadataStats.residentHostBytes == 0 && metadataStats.peakHostBytes == 0);
            auto lease = store.acquire(0, {});
            ASSERT_TRUE(lease.camera().width == metadata.width && lease.camera().height == metadata.height);
            ASSERT_TRUE(store.stats().misses == 1);
        }
        {
            auto a = camera(0, one, 40, 30);
            a.maskPath = mask.string();
            a.k1 = .01f;
            a.k2 = -.005f;
            CameraImageStore store({a}, 1 << 20, false);
            auto first = store.acquire(0, FrameRequest{1, torch::kCPU, true, true});
            ASSERT_TRUE(first.camera().width > 0 &&
                        first.image().sizes() == torch::IntArrayRef({first.camera().height, first.camera().width, 3}));
            ASSERT_TRUE(first.mask().defined() && first.edgeMap().defined());
            auto metadata = first.camera();
            auto tensor = first.image().clone();
            first = {};
            auto second = store.acquire(0, FrameRequest{1, torch::kCPU, true, true});
            ASSERT_TRUE(std::abs(second.camera().fx - metadata.fx) < 1e-6f);
            ASSERT_TRUE(torch::allclose(second.image(), tensor));
            ASSERT_TRUE(store.stats().hits == 1);
            second = {};
            auto pyramid = store.acquire(0, FrameRequest{2, torch::kCPU, true, true});
            ASSERT_TRUE(pyramid.image().size(0) == metadata.height / 2 && pyramid.mask().size(1) == metadata.width / 2);
        }
        {
            auto distorted = camera(0, one, 40, 30);
            distorted.maskPath = mask.string();
            distorted.k1 = .01f;
            distorted.k2 = -.005f;
            CameraImageStore store({distorted}, 1, false);
            auto first = store.acquire(0, FrameRequest{1, torch::kCPU, true, true});
            auto image = first.image().clone();
            auto derivedMask = first.mask().clone();
            auto edge = first.edgeMap().clone();
            first = {};
            ASSERT_TRUE(store.stats().evictions == 1 && store.stats().residentHostBytes == 0);
            auto reloaded = store.acquire(0, FrameRequest{1, torch::kCPU, true, true});
            ASSERT_TRUE(torch::allclose(reloaded.image(), image));
            ASSERT_TRUE(torch::equal(reloaded.mask(), derivedMask));
            ASSERT_TRUE(torch::equal(reloaded.edgeMap(), edge));
            ASSERT_TRUE(store.stats().reloads == 1);
        }
        {
            const std::uint64_t bytes = 32 * 24 * 3 * 4;
            CameraImageStore store({camera(0, one), camera(1, two)}, bytes, false);
            {
                auto first = store.acquire(0, {});
                ASSERT_TRUE(store.stats().residentHostBytes == bytes);
            }
            {
                auto second = store.acquire(1, {});
                ASSERT_TRUE(store.stats().residentHostBytes <= bytes);
            }
            {
                auto firstAgain = store.acquire(0, {});
                ASSERT_TRUE(firstAgain.image().defined());
            }
            auto stats = store.stats();
            ASSERT_TRUE(stats.evictions >= 2);
            ASSERT_TRUE(stats.reloads == 1);
            ASSERT_TRUE(stats.peakHostBytes <= bytes * 2);
        }
        {
            const std::uint64_t bytes = 32 * 24 * 3 * 4;
            CameraImageStore store({camera(0, one), camera(1, two), camera(2, three)}, bytes * 2, false);
            store.acquire(0, {}) = {};
            store.acquire(1, {}) = {};
            const CacheStats beforeFailure = store.stats();
            bool failed = false;
            try {
                auto ignored = store.acquire(0, FrameRequest{64, torch::kCPU, false, false});
            } catch (const std::invalid_argument &) {
                failed = true;
            }
            ASSERT_TRUE(failed);
            const CacheStats afterFailure = store.stats();
            ASSERT_TRUE(afterFailure.hits == beforeFailure.hits + 1);
            ASSERT_TRUE(afterFailure.misses == beforeFailure.misses);
            ASSERT_TRUE(afterFailure.reloads == beforeFailure.reloads);
            ASSERT_TRUE(afterFailure.evictions == beforeFailure.evictions);
            ASSERT_TRUE(afterFailure.residentHostBytes == beforeFailure.residentHostBytes);
            ASSERT_TRUE(afterFailure.residentDeviceBytes == beforeFailure.residentDeviceBytes);
            store.acquire(2, {}) = {};
            const CacheStats beforeB = store.stats();
            auto b = store.acquire(1, {});
            const CacheStats afterB = store.stats();
            ASSERT_TRUE(afterB.hits == beforeB.hits + 1 && afterB.misses == beforeB.misses);
            b = {};
            const CacheStats beforeA = store.stats();
            auto a = store.acquire(0, {});
            const CacheStats afterA = store.stats();
            ASSERT_TRUE(afterA.misses == beforeA.misses + 1 && afterA.reloads == beforeA.reloads + 1);
        }
        {
            const std::uint64_t bytes = 32 * 24 * 3 * 4;
            CameraImageStore store({camera(0, one)}, bytes - 1, false);
            auto lease = store.acquire(0, {});
            ASSERT_TRUE(store.stats().oversizeAdmissions == 1);
            lease = {};
            ASSERT_TRUE(store.stats().residentHostBytes == 0);
        }
        {
            const std::uint64_t bytes = 32 * 24 * 3 * 4;
            CameraImageStore store({camera(0, one), camera(1, two)}, bytes, false);
            auto pinned = store.acquire(0, {});
            auto pressure = store.acquire(1, {});
            ASSERT_TRUE(store.stats().residentHostBytes == bytes * 2);
            pressure = {};
            ASSERT_TRUE(store.stats().residentHostBytes == bytes);
            ASSERT_TRUE(pinned.image().defined());
            const CacheStats beforePressureReload = store.stats();
            auto pressureAgain = store.acquire(1, {});
            ASSERT_TRUE(store.stats().reloads == beforePressureReload.reloads + 1);
            pressureAgain = {};
            const CacheStats beforePinnedHit = store.stats();
            auto pinnedAgain = store.acquire(0, {});
            ASSERT_TRUE(store.stats().hits == beforePinnedHit.hits + 1);
        }
        {
            const std::uint64_t bytes = 32 * 24 * 3 * 4;
            CameraImageStore store({camera(0, one), camera(1, two), camera(2, three)}, bytes * 2, false);
            store.acquire(0, {}) = {};
            store.acquire(1, {}) = {};
            auto pinned = store.acquire(0, {});
            store.acquire(2, {}) = {};
            const CacheStats beforeA = store.stats();
            auto a = store.acquire(0, {});
            ASSERT_TRUE(store.stats().hits == beforeA.hits + 1);
            a = {};
            pinned = {};
            const CacheStats beforeB = store.stats();
            auto b = store.acquire(1, {});
            ASSERT_TRUE(store.stats().reloads == beforeB.reloads + 1);
        }
        {
            const std::uint64_t bytes = 32 * 24 * 3 * 4;
            CameraImageStore store({camera(0, one), camera(1, two)}, bytes * 2, false);
            hookSourceMetadataProbes = 0;
            setCameraImageStoreTestHook(decodeCountingHook);
            ASSERT_TRUE(store.primeResidentSetIfFits({0, 1}, 2));
            setCameraImageStoreTestHook(nullptr);
            ASSERT_TRUE(hookSourceMetadataProbes == 2);
        }
        {
            const std::uint64_t bytes = 32 * 24 * 3 * 4;
            CameraImageStore store({camera(0, one), camera(1, two)}, bytes * 2, false);
            auto first = store.acquire(0, {});
            auto second = store.acquire(1, {});
            first = std::move(second);
            ASSERT_TRUE(first.camera().filePath == two.string());
            first = {};
            ASSERT_TRUE(store.stats().residentHostBytes == bytes * 2);
        }
        {
            CameraFrameLease lease;
            {
                CameraImageStore store({camera(0, one)}, 1 << 20, false);
                lease = store.acquire(0, {});
            }
            ASSERT_TRUE(lease.image().defined());
            lease = {};
        }
        {
            const std::uint64_t bytes = 32 * 24 * 3 * 4;
            CameraImageStore store({camera(0, one), camera(1, two), camera(2, three)}, bytes, false);
            for (int i = 0; i < 60; ++i) {
                store.acquire(static_cast<CameraKey>(i % 3), {}) = {};
                ASSERT_TRUE(store.stats().residentHostBytes <= bytes);
            }
        }
        {
            CameraImageStore store({camera(0, one)}, 1 << 20, false);
            std::atomic<int> ready{0};
            std::atomic<int> completed{0};
            std::vector<std::thread> threads;
            for (int i = 0; i < 4; ++i)
                threads.emplace_back([&] {
                    ++ready;
                    while (ready.load() != 4)
                        std::this_thread::yield();
                    auto lease = store.acquire(0, {});
                    ++completed;
                });
            joinWithin(threads, completed, 4);
            ASSERT_TRUE(store.stats().misses == 1 && store.stats().hits == 3);
        }
        {
            fs::path missingMask = root / "waiter-mask.pgm";
            auto descriptor = camera(0, one);
            descriptor.maskPath = missingMask.string();
            CameraImageStore store({descriptor}, 1 << 20, false);
            constexpr int waiterCount = 5;
            std::atomic<int> completed{0}, failures{0};
            std::vector<std::thread> threads;
            resetLoadingHook(CameraImageStoreTestEvent::AcquireLoading);
            threads.emplace_back([&] {
                try {
                    auto ignored = store.acquire(0, FrameRequest{1, torch::kCPU, true, false});
                } catch (const std::exception &) {
                    ++failures;
                }
                ++completed;
            });
            waitUntil(hookLoaderBlocked, 1, "the failing producer to enter Loading");
            for (int i = 0; i < waiterCount; ++i)
                threads.emplace_back([&] {
                    try {
                        auto ignored = store.acquire(0, FrameRequest{1, torch::kCPU, true, false});
                    } catch (const std::exception &) {
                        ++failures;
                    }
                    ++completed;
                });
            waitUntil(hookWaiters, waiterCount, "all same-key waiters");
            hookReleaseLoader = true;
            joinWithin(threads, completed, waiterCount + 1);
            setCameraImageStoreTestHook(nullptr);
            ASSERT_TRUE(failures == waiterCount + 1 && store.stats().waits >= waiterCount);
            pgm(missingMask, 32, 24);
            auto recovered = store.acquire(0, FrameRequest{1, torch::kCPU, true, false});
            ASSERT_TRUE(recovered.mask().defined() && store.stats().reloads == 0);
        }
        {
            CameraImageStore store({camera(0, one)}, 1 << 20, false);
            std::atomic<int> completed{0}, failures{0};
            int metadataWidth = 0;
            resetLoadingHook(CameraImageStoreTestEvent::MetadataLoading);
            std::thread metadataThread([&] {
                try {
                    metadataWidth = store.metadata(0).width;
                } catch (...) {
                    ++failures;
                }
                ++completed;
            });
            waitUntil(hookLoaderBlocked, 1, "metadata to enter Loading");
            std::thread acquireThread([&] {
                try {
                    auto lease = store.acquire(0, {});
                } catch (...) {
                    ++failures;
                }
                ++completed;
            });
            waitUntil(hookWaiters, 1, "acquire to wait for metadata");
            hookReleaseLoader = true;
            std::vector<std::thread> threads;
            threads.push_back(std::move(metadataThread));
            threads.push_back(std::move(acquireThread));
            joinWithin(threads, completed, 2);
            setCameraImageStoreTestHook(nullptr);
            ASSERT_TRUE(failures == 0 && metadataWidth == 32);
            ASSERT_TRUE(store.stats().misses == 1 && store.stats().waits >= 1);
        }
        {
            const std::uint64_t bytes = 32 * 24 * 3 * 4;
            CameraImageStore store({camera(0, one), camera(1, two)}, bytes, false);
            std::atomic<int> ready{0}, completed{0}, failures{0};
            std::thread first([&] {
                ++ready;
                while (ready.load() != 2)
                    std::this_thread::yield();
                try {
                    auto lease = store.acquire(0, {});
                    ++completed;
                    while (completed.load() != 2)
                        std::this_thread::yield();
                } catch (...) {
                    ++failures;
                    ++completed;
                }
            });
            std::thread second([&] {
                ++ready;
                while (ready.load() != 2)
                    std::this_thread::yield();
                try {
                    auto lease = store.acquire(1, {});
                    ++completed;
                    while (completed.load() != 2)
                        std::this_thread::yield();
                } catch (...) {
                    ++failures;
                    ++completed;
                }
            });
            std::vector<std::thread> threads;
            threads.push_back(std::move(first));
            threads.push_back(std::move(second));
            joinWithin(threads, completed, 2);
            ASSERT_TRUE(failures == 0 && store.stats().misses == 2);
            ASSERT_TRUE(store.stats().residentHostBytes == bytes);
        }
        {
            const std::uint64_t bytes = 32 * 24 * 3 * 4;
            CameraImageStore store({camera(0, one), camera(1, two)}, bytes * 2, false);
            hookLoaderBlocked = 0;
            hookReleaseLoader = false;
            hookSecondDecodeEntered = false;
            hookParallelDecodeObserved = false;
            setCameraImageStoreTestHook(parallelDecodeHook);
            std::atomic<bool> preloadResult{false}, preloadFailed{false};
            std::thread preloader([&] {
                try {
                    preloadResult = store.primeResidentSetIfFits({0, 1}, 2);
                } catch (...) {
                    preloadFailed = true;
                }
            });
            waitUntil(hookLoaderBlocked, 1, "the first source decode");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
            while (!hookSecondDecodeEntered.load() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
            hookReleaseLoader = true;
            preloader.join();
            setCameraImageStoreTestHook(nullptr);
            ASSERT_TRUE(!preloadFailed.load() && preloadResult.load() && hookParallelDecodeObserved.load());
            ASSERT_TRUE(store.stats().misses == 2 && store.stats().residentHostBytes == bytes * 2);
        }
        {
            const std::uint64_t bytes = 32 * 24 * 3 * 4;
            CameraImageStore store({camera(0, one), camera(1, two)}, bytes, false);
            ASSERT_TRUE(!store.primeResidentSetIfFits({0, 1}, 2));
            ASSERT_TRUE(store.stats().misses == 0 && store.stats().residentHostBytes == 0);
        }
        {
            const std::uint64_t packedBytes = 32 * 24 * 3;
            CameraImageStore store({camera(0, one), camera(1, two)}, packedBytes * 2, false, 1.0f,
                                   HostImageStorage::UInt8);
            ASSERT_TRUE(store.primeResidentSetIfFits({0, 1}, 2));
            ASSERT_TRUE(store.stats().residentHostBytes == packedBytes * 2);
            auto lease = store.acquire(0, {});
            ASSERT_TRUE(lease.image().scalar_type() == torch::kFloat32);
            ASSERT_TRUE(torch::equal(lease.image(), imageToTensor(imreadRGB(one.string()))));
            auto packedPyramid = store.acquire(0, FrameRequest{2, torch::kCPU, false, false});
            CameraImageStore reference({camera(0, one)}, 32 * 24 * 3 * 4, false);
            auto referencePyramid = reference.acquire(0, FrameRequest{2, torch::kCPU, false, false});
            ASSERT_TRUE(torch::equal(packedPyramid.image(), referencePyramid.image()));
        }
        {
            fs::path missing = root / "later.ppm";
            CameraImageStore store({camera(0, missing)}, 1 << 20, false);
            bool failed = false;
            try {
                auto ignored = store.acquire(0, {});
            } catch (const std::exception &) {
                failed = true;
            }
            ASSERT_TRUE(failed);
            ppm(missing, 32, 24);
            auto recovered = store.acquire(0, {});
            ASSERT_TRUE(recovered.image().defined());
            ASSERT_TRUE(store.stats().decodeFailures == 1);
        }
        {
            fs::path missingMask = root / "later-mask.pgm";
            auto descriptor = camera(0, one);
            descriptor.maskPath = missingMask.string();
            CameraImageStore store({descriptor}, 1 << 20, false);
            bool failed = false;
            try {
                auto ignored = store.acquire(0, FrameRequest{1, torch::kCPU, true, false});
            } catch (const std::exception &) {
                failed = true;
            }
            ASSERT_TRUE(failed);
            pgm(missingMask, 32, 24);
            auto recovered = store.acquire(0, FrameRequest{1, torch::kCPU, true, false});
            ASSERT_TRUE(recovered.mask().defined());
            ASSERT_TRUE(store.stats().decodeFailures == 1);
        }
        {
            fs::path missing = root / "metadata-later.ppm";
            CameraImageStore store({camera(0, missing)}, 1 << 20, false);
            bool failed = false;
            try {
                (void)store.metadata(0);
            } catch (const std::exception &) {
                failed = true;
            }
            ASSERT_TRUE(failed);
            ppm(missing, 32, 24);
            const PreparedCamera &recovered = store.metadata(0);
            ASSERT_TRUE(recovered.width == 32 && recovered.height == 24);
            auto stats = store.stats();
            ASSERT_TRUE(stats.decodeFailures == 1 && stats.misses == 0 && stats.peakHostBytes == 0);
        }
        {
            InputData input;
            input.cameras = {camera(0, one), camera(1, two)};
            input.assignCameraKeys();
            CameraSplit split = input.splitCameras(true, "two.ppm");
            ASSERT_TRUE(split.trainKeys.size() == 1 && split.trainKeys[0] == 0);
            ASSERT_TRUE(split.validationKey && *split.validationKey == 1);
        }
        if (torch::hasMPS()) {
            const std::uint64_t hostImageBytes = 32 * 24 * 3;
            const std::uint64_t deviceImageBytes = 32 * 24 * 3 * 4;
            CameraImageStore store({camera(0, one), camera(1, two)}, hostImageBytes * 2, true, 1.0f,
                                   HostImageStorage::UInt8);
            {
                auto first = store.acquire(0, FrameRequest{1, torch::kMPS, false, false});
                ASSERT_TRUE(first.image().device().is_mps());
                ASSERT_TRUE(torch::allclose(first.image().cpu(), imageToTensor(imreadRGB(one.string()))));
            }
            const CacheStats beforeFailure = store.stats();
            bool failed = false;
            try {
                auto ignored = store.acquire(1, FrameRequest{1, torch::kCUDA, false, false});
            } catch (const std::exception &) {
                failed = true;
            }
            const CacheStats afterFailure = store.stats();
            ASSERT_TRUE(failed);
            ASSERT_TRUE(afterFailure.residentDeviceBytes == beforeFailure.residentDeviceBytes);
            ASSERT_TRUE(afterFailure.evictions == beforeFailure.evictions);

            CameraImageStore rollbackStore({camera(0, one)}, hostImageBytes * 2, true, 1.0f,
                                           HostImageStorage::UInt8);
            rollbackStore.acquire(0, {}) = {};
            const CacheStats beforeRollback = rollbackStore.stats();
            failed = false;
            try {
                auto ignored = rollbackStore.acquire(0, FrameRequest{2, torch::kCUDA, false, false});
            } catch (const std::exception &) {
                failed = true;
            }
            const CacheStats afterRollback = rollbackStore.stats();
            ASSERT_TRUE(failed && afterRollback.residentHostBytes == beforeRollback.residentHostBytes);
            ASSERT_TRUE(afterRollback.residentDeviceBytes == beforeRollback.residentDeviceBytes);
            auto rolledBack = rollbackStore.acquire(0, FrameRequest{2, torch::kCPU, false, false});
            ASSERT_TRUE(rolledBack.image().defined() && rollbackStore.stats().reloads == 0);

            CameraImageStore transientStore({camera(0, one)}, hostImageBytes * 2, false, 1.0f,
                                            HostImageStorage::UInt8);
            auto transientOne = transientStore.acquire(0, FrameRequest{1, torch::kMPS, false, false});
            auto transientTwo = transientStore.acquire(0, FrameRequest{1, torch::kMPS, false, false});
            ASSERT_TRUE(transientStore.stats().residentDeviceBytes == deviceImageBytes * 2);
            transientOne = {};
            ASSERT_TRUE(transientStore.stats().residentDeviceBytes == deviceImageBytes);
            transientTwo = {};
            ASSERT_TRUE(transientStore.stats().residentDeviceBytes == 0);

            CameraImageStore evictionStore({camera(0, one), camera(1, two)}, hostImageBytes, true, 1.0f,
                                           HostImageStorage::UInt8);
            {
                auto first = evictionStore.acquire(0, FrameRequest{1, torch::kMPS, false, false});
                ASSERT_TRUE(first.image().device().is_mps());
            }
            const CacheStats beforeEviction = evictionStore.stats();
            {
                auto second = evictionStore.acquire(1, FrameRequest{1, torch::kMPS, false, false});
                ASSERT_TRUE(second.image().device().is_mps());
            }
            const CacheStats afterEviction = evictionStore.stats();
            ASSERT_TRUE(afterEviction.residentDeviceBytes == deviceImageBytes);
            ASSERT_TRUE(afterEviction.evictions == beforeEviction.evictions + 1);
        }
        {
            bool rejectedBudget = false;
            try {
                CameraImageStore invalid({camera(0, one)}, 0, false);
            } catch (const std::invalid_argument &) {
                rejectedBudget = true;
            }
            ASSERT_TRUE(rejectedBudget);

            CameraImageStore store({camera(0, one)}, 1 << 20, false);
            bool rejectedKey = false;
            bool rejectedDownscale = false;
            try {
                auto ignored = store.acquire(1, {});
            } catch (const std::out_of_range &) {
                rejectedKey = true;
            }
            try {
                auto ignored = store.acquire(0, FrameRequest{64, torch::kCPU, false, false});
            } catch (const std::invalid_argument &) {
                rejectedDownscale = true;
            }
            ASSERT_TRUE(rejectedKey && rejectedDownscale && store.stats().residentHostBytes == 0);
            auto recovered = store.acquire(0, {});
            ASSERT_TRUE(recovered.image().defined());
            recovered = {};
            bool rejectedZeroDownscale = false;
            try {
                auto ignored = store.acquire(0, FrameRequest{0, torch::kCPU, false, false});
            } catch (const std::invalid_argument &) {
                rejectedZeroDownscale = true;
            }
            ASSERT_TRUE(rejectedZeroDownscale);
        }
        fs::remove_all(root);
        std::cout << "camera_image_store_test passed" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        fs::remove_all(root);
        return 1;
    }
}
