cmake_minimum_required(VERSION 3.21)
project(opensplat)

set(OPENSPLAT_BUILD_SIMPLE_TRAINER OFF CACHE BOOL "Build simple trainer applications")
set(GPU_RUNTIME "CUDA" CACHE STRING "HIP or CUDA")
set(OPENCV_DIR "OPENCV_DIR-NOTFOUND" CACHE PATH "Path to the OPENCV installation directory")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if(GPU_RUNTIME STREQUAL "CUDA")
    set(CMAKE_CUDA_ARCHITECTURES 70 75)
    find_package(CUDAToolkit)
    if (CUDAToolkit-NOTFOUND)
        message(WARNING "CUDA toolkit not found, building with CPU support only")
        set(GPU_RUNTIME "CPU")
    endif()
elseif(GPU_RUNTIME STREQUAL "HIP")
    set(USE_HIP ON CACHE BOOL "Use HIP for GPU acceleration")

    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    find_package(HIP REQUIRED)

    file(GLOB_RECURSE GSPLAT_GPU_SRC LIST_DIRECTORIES False vendor/gsplat/*.cu)
    set_source_files_properties(${GSPLAT_GPU_SRC} PROPERTIES LANGUAGE HIP)

    if(WIN32)
        set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
    else()
        set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
endif()

set(CMAKE_CXX_STANDARD 17)
if((GPU_RUNTIME STREQUAL "CUDA") OR (GPU_RUNTIME STREQUAL "HIP"))
    enable_language(${GPU_RUNTIME})
    set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
    set(${GPU_RUNTIME}_STANDARD 17)
endif()

if (NOT WIN32 AND NOT APPLE)
    set(STDPPFS_LIBRARY stdc++fs)
endif()

find_package(Torch REQUIRED)
find_package(OpenCV HINTS "${OPENCV_DIR}" REQUIRED)

if (NOT WIN32 AND NOT APPLE)
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)

set(GSPLAT_LIBS gsplat_cpu)
if((GPU_RUNTIME STREQUAL "CUDA") OR (GPU_RUNTIME STREQUAL "HIP"))
    add_library(gsplat vendor/gsplat/forward.cu vendor/gsplat/backward.cu vendor/gsplat/bindings.cu vendor/gsplat/helpers.cuh)
    list(APPEND GSPLAT_LIBS gsplat)
    if(GPU_RUNTIME STREQUAL "CUDA")
        set(GPU_LIBRARIES "cuda")
        target_link_libraries(gsplat PUBLIC cuda)
        set_target_properties(gsplat PROPERTIES CUDA_ARCHITECTURES "70;75")
    else(GPU_RUNTIME STREQUAL "HIP")
        set(GPU_INCLUDE_DIRS "${ROCM_ROOT}/include")
        target_compile_definitions(gsplat PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
    endif()
    target_include_directories(gsplat PRIVATE
        ${PROJECT_SOURCE_DIR}/vendor/glm        
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${TORCH_INCLUDE_DIRS}
    )
    set_target_properties(gsplat PROPERTIES LINKER_LANGUAGE CXX)
endif()

add_library(gsplat_cpu vendor/gsplat-cpu/gsplat_cpu.cpp)
target_include_directories(gsplat_cpu PRIVATE ${TORCH_INCLUDE_DIRS})

add_executable(opensplat opensplat.cpp point_io.cpp nerfstudio.cpp model.cpp kdtree_tensor.cpp spherical_harmonics.cpp cv_utils.cpp utils.cpp project_gaussians.cpp rasterize_gaussians.cpp ssim.cpp optim_scheduler.cpp colmap.cpp input_data.cpp tensor_math.cpp)
set_property(TARGET opensplat PROPERTY CXX_STANDARD 17)
target_include_directories(opensplat PRIVATE ${PROJECT_SOURCE_DIR}/vendor/glm ${GPU_INCLUDE_DIRS})
target_link_libraries(opensplat PUBLIC ${STDPPFS_LIBRARY} ${GPU_LIBRARIES} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
if(GPU_RUNTIME STREQUAL "HIP")
    target_compile_definitions(opensplat PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
elseif(GPU_RUNTIME STREQUAL "CUDA")
    target_compile_definitions(opensplat PRIVATE USE_CUDA)
endif()

if(OPENSPLAT_BUILD_SIMPLE_TRAINER)
    add_executable(simple_trainer simple_trainer.cpp project_gaussians.cpp rasterize_gaussians.cpp cv_utils.cpp)
    target_include_directories(simple_trainer PRIVATE ${PROJECT_SOURCE_DIR}/vendor/glm ${GPU_INCLUDE_DIRS})
    target_link_libraries(simple_trainer PUBLIC ${GPU_LIBRARIES} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
    set_property(TARGET simple_trainer PROPERTY CXX_STANDARD 17)
    if(GPU_RUNTIME STREQUAL "HIP")
        target_compile_definitions(simple_trainer PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
    elseif(GPU_RUNTIME STREQUAL "CUDA")
        target_compile_definitions(simple_trainer PRIVATE USE_CUDA)
    endif()
endif()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    file(GLOB OPENCV_DLL "${OPENCV_DIR}/x64/vc16/bin/opencv_world490.dll")
    set(DLLS_TO_COPY ${TORCH_DLLS} ${OPENCV_DLL})
    add_custom_command(TARGET opensplat
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DLLS_TO_COPY}
        $<TARGET_FILE_DIR:opensplat>)
endif (MSVC)
