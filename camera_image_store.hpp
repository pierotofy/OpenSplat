#ifndef CAMERA_IMAGE_STORE_H
#define CAMERA_IMAGE_STORE_H

#include "input_data.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <torch/torch.h>

struct PreparedCamera {
    int width = 0;
    int height = 0;
    float fx = 0;
    float fy = 0;
    float cx = 0;
    float cy = 0;
    torch::Tensor camToWorld;
    std::string filePath;
};

struct FrameRequest {
    int downscale = 1;
    torch::Device device = torch::kCPU;
    bool needMask = false;
    bool needEdgeMap = false;
};

enum class HostImageStorage { Float32, UInt8 };

struct CacheStats {
    std::uint64_t residentHostBytes = 0;
    std::uint64_t peakHostBytes = 0;
    std::uint64_t residentDeviceBytes = 0;
    std::uint64_t peakDeviceBytes = 0;
    // Hits and misses count lookup outcomes even when later request preparation
    // fails. Reloads and evictions count only committed state changes.
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t reloads = 0;
    std::uint64_t waits = 0;
    std::uint64_t decodeFailures = 0;
    std::uint64_t oversizeAdmissions = 0;
};

struct CameraStoreState;
struct CameraPayloadGeneration;

class CameraFrameLease {
  public:
    CameraFrameLease() = default;
    CameraFrameLease(CameraFrameLease &&other) noexcept;
    CameraFrameLease &operator=(CameraFrameLease &&other) noexcept;
    CameraFrameLease(const CameraFrameLease &) = delete;
    CameraFrameLease &operator=(const CameraFrameLease &) = delete;
    ~CameraFrameLease() noexcept;
    // Borrowed handles are valid for the lease lifetime. Retaining a copied
    // torch::Tensor handle makes that memory external to cache accounting.
    const PreparedCamera &camera() const;
    const torch::Tensor &image() const;
    const torch::Tensor &mask() const;
    const torch::Tensor &edgeMap() const;

  private:
    friend class CameraImageStore;
    CameraFrameLease(std::shared_ptr<CameraStoreState>, std::shared_ptr<CameraPayloadGeneration>, CameraKey,
                     torch::Tensor, torch::Tensor, torch::Tensor, std::uint64_t);
    void release() noexcept;
    std::shared_ptr<CameraStoreState> state_;
    std::shared_ptr<CameraPayloadGeneration> generation_;
    CameraKey key_ = 0;
    torch::Tensor image_, mask_, edge_;
    std::uint64_t transientDeviceBytes_ = 0;
};

class CameraImageStore {
  public:
    CameraImageStore(std::vector<Camera> descriptors, std::uint64_t hostBudgetBytes,
                     bool gpuCacheEnabled = true, float initialDownscale = 1.0f,
                     HostImageStorage imageStorage = HostImageStorage::Float32);
    CameraFrameLease acquire(CameraKey key, const FrameRequest &request);
    // Decode the complete key set in parallel only when its base host payload
    // fits the configured budget. Worker count adapts to available processors,
    // effective system memory, and source image size; maxWorkers is an optional cap.
    bool primeResidentSetIfFits(const std::vector<CameraKey> &keys, std::size_t maxWorkers = 0);
    // The returned reference remains valid only while this store exists.
    const PreparedCamera &metadata(CameraKey key);
    CacheStats stats() const;
    std::size_t size() const;
    CameraImageStore(const CameraImageStore &) = delete;
    CameraImageStore &operator=(const CameraImageStore &) = delete;
    CameraImageStore(CameraImageStore &&) noexcept = default;
    CameraImageStore &operator=(CameraImageStore &&) noexcept = default;

  private:
    CameraFrameLease acquireImpl(CameraKey key, const FrameRequest &request, bool materializeImage);
    std::shared_ptr<CameraStoreState> state_;
};

std::uint64_t resolveHostCacheBudget(std::uint64_t requestedBytes);
#endif
