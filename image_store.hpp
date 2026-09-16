#ifndef IMAGE_STORE_H
#define IMAGE_STORE_H

#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "input_data.hpp"

using Blob = std::shared_ptr<const std::vector<uint8_t>>;

// Thread-safe cache of compressed image bytes with a RAM cap. Least recently
// used entries are saved to a per-process directory under baseDir and reloaded on demand
class BlobCache{
public:
    BlobCache(uint64_t capBytes, const std::filesystem::path &baseDir);
    ~BlobCache();

    void put(const std::string &key, std::vector<uint8_t> &&bytes);
    Blob get(const std::string &key);
    bool has(const std::string &key) const;
    uint64_t residentBytes() const { return resident; }
    uint64_t capacityBytes() const { return cap; }

private:
    struct Entry{
        Blob bytes; // null when saved
        uint64_t size = 0;
        bool saved = false;
        std::list<std::string>::iterator lru;
    };
    void touchLocked(Entry &e, const std::string &key);
    void evictLocked(const std::string &keep);
    std::filesystem::path savePath(const std::string &key) const;

    mutable std::mutex m;
    std::unordered_map<std::string, Entry> map;
    std::list<std::string> lru; // front = most recently used
    uint64_t resident = 0;
    uint64_t cap;
    bool lowRamGuardTripped = false;
    std::filesystem::path saveDir;
};

// Preprocesses every camera image once (resize, undistort, downscale levels)
// into compressed blobs and updates the camera intrinsics and dimensions
struct ImageStore{
    ImageStore(uint64_t capBytes, const std::filesystem::path &baseDir);

    void prepare(std::vector<Camera> &cameras, float downscaleFactor, int numDownscales);
    void prepareCamera(Camera &cam, float downscaleFactor);

    static std::string imageKey(int imageId, int level);
    static std::string maskKey(int imageId, int level);

    BlobCache cache;
    int numLevels = 1;
};

#endif
