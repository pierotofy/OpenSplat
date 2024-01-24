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

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
find_package(Torch REQUIRED)

add_subdirectory(vendor/mve)

add_library(opensplat forward.cu backward.cu helpers.cuh)
target_link_libraries(opensplat PUBLIC cuda)
target_include_directories(opensplat PRIVATE
    ${PROJECT_SOURCE_DIR}/vendor/glm        
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
set_target_properties(opensplat PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(opensplat PROPERTIES CUDA_ARCHITECTURES "70;75")

# # To add an executable that uses the opensplat library,
add_executable(simple_trainer simple_trainer.cpp)
target_include_directories(simple_trainer PRIVATE ${PROJECT_SOURCE_DIR}/vendor/mve ${PROJECT_SOURCE_DIR}/vendor/glm)
target_link_libraries(simple_trainer PUBLIC cuda opensplat mve ${TORCH_LIBRARIES})

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