cmake_minimum_required(VERSION 3.18)  # You can adjust the minimum required version
project(opensplat)

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
"Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
FORCE)
endif()

set(CMAKE_CUDA_ARCHITECTURES 70 75)  # Ti 2080 uses 75. V100 uses 70. RTX 4090 uses 89.
set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD 17)
set(PYTHON_EXECUTABLE /usr/bin/python3)
set(OPENCV_DIR "OPENCV_DIR-NOTFOUND" CACHE "OPENCV_DIR" "Path to the OPENCV installation directory")

if (NOT WIN32 AND NOT APPLE)
    set(STDPPFS_LIBRARY stdc++fs)
endif()

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
find_package(Torch REQUIRED)
find_package(OpenCV HINTS "${OPENCV_DIR}" REQUIRED)
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

# TODO: move project_gaussians.cpp, rasterize.cpp to another target?
add_executable(simple_trainer simple_trainer.cpp project_gaussians.cpp rasterize_gaussians.cpp)
target_include_directories(simple_trainer PRIVATE ${PROJECT_SOURCE_DIR}/vendor/glm)
target_link_libraries(simple_trainer PUBLIC cuda gsplat ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(opensplat opensplat.cpp point_io.cpp nerfstudio.cpp model.cpp kdtree_tensor.cpp spherical_harmonics.cpp cv_utils.cpp)
target_include_directories(opensplat PRIVATE ${PROJECT_SOURCE_DIR}/vendor/glm ${PROJECT_SOURCE_DIR}/vendor/json)
target_link_libraries(opensplat PUBLIC ${STDPPFS_LIBRARY} cuda gsplat ${TORCH_LIBRARIES} ${OpenCV_LIBS})


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