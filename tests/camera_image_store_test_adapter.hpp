#ifndef CAMERA_IMAGE_STORE_TEST_ADAPTER_H
#define CAMERA_IMAGE_STORE_TEST_ADAPTER_H

#include "../input_data.hpp"

enum class CameraImageStoreTestEvent {
    AcquireLoading,
    MetadataLoading,
    Wait,
    SourceDecode,
    SourceMetadataProbe
};
using CameraImageStoreTestHook = void (*)(CameraKey, CameraImageStoreTestEvent) noexcept;

void setCameraImageStoreTestHook(CameraImageStoreTestHook hook) noexcept;

#endif
