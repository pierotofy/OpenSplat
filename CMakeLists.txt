cmake_minimum_required(VERSION 3.18)
project(opensplat)

set(OPENSPLAT_BUILD_SIMPLE_TRAINER OFF CACHE BOOL "Build simple trainer applications")
set(OPENCV_DIR "OPENCV_DIR-NOTFOUND" CACHE "OPENCV_DIR" "Path to the OPENCV installation directory")

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(CMAKE_CUDA_ARCHITECTURES 70 75)
set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD 17)

if (NOT WIN32 AND NOT APPLE)
    set(STDPPFS_LIBRARY stdc++fs)
endif()

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV HINTS "${OPENCV_DIR}" REQUIRED)

if (NOT WIN32 AND NOT APPLE)
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)

add_library(gsplat vendor/gsplat/forward.cu vendor/gsplat/backward.cu vendor/gsplat/bindings.cu vendor/gsplat/helpers.cuh)
target_link_libraries(gsplat PUBLIC cuda)
target_include_directories(gsplat PRIVATE
    ${PROJECT_SOURCE_DIR}/vendor/glm        
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${TORCH_INCLUDE_DIRS}
)
set_target_properties(gsplat PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(gsplat PROPERTIES CUDA_ARCHITECTURES "70;75")

add_executable(opensplat opensplat.cpp point_io.cpp nerfstudio.cpp model.cpp kdtree_tensor.cpp spherical_harmonics.cpp cv_utils.cpp utils.cpp project_gaussians.cpp rasterize_gaussians.cpp ssim.cpp)
target_include_directories(opensplat PRIVATE ${PROJECT_SOURCE_DIR}/vendor/glm)
target_link_libraries(opensplat PUBLIC ${STDPPFS_LIBRARY} cuda gsplat ${TORCH_LIBRARIES} ${OpenCV_LIBS})

if(OPENSPLAT_BUILD_SIMPLE_TRAINER)
    add_executable(simple_trainer simple_trainer.cpp project_gaussians.cpp rasterize_gaussians.cpp)
    target_include_directories(simple_trainer PRIVATE ${PROJECT_SOURCE_DIR}/vendor/glm)
    target_link_libraries(simple_trainer PUBLIC cuda gsplat ${TORCH_LIBRARIES} ${OpenCV_LIBS})
endif()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET example-app
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:example-app>)
# endif (MSVC)