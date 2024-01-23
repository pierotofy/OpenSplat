project(mve)

find_package(PNG REQUIRED)

add_definitions(-DMVE_NO_JPEG_SUPPORT)
add_definitions(-DMVE_NO_TIFF_SUPPORT)

file(GLOB MVE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/mve/*.cc)
file(GLOB UTIL_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/util/*.cc)

add_library(mve ${MVE_SRC_FILES} ${UTIL_SRC_FILES})
target_include_directories(mve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PNG_INCLUDE_DIR} ${JPEG_INCLUDE_DIR} ${TIFF_INCLUDE_DIR})
target_link_libraries(mve PUBLIC png)