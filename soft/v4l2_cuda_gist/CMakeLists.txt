cmake_minimum_required(VERSION 3.10)
project(v4l_cuda_gist)

set(CMAKE_CXX_STANDARD 14)

#include_directories( /usr/src/linux-headers-4.9.140-tegra-ubuntu18.04_aarch64/nvidia/include/ )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS "OpenCV version: ${OpenCV_VERSION}")
MESSAGE(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV libs: ${OpenCV_LIBS}")
MESSAGE(STATUS "OpenCV libs dir: ${OpenCV_LIB_DIR}")


find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
MESSAGE(STATUS "CUDA version: ${CUDA_VERSION_STRING}")
MESSAGE(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
MESSAGE(STATUS "CUDA rt library: ${CUDA_CUDART_LIBRARY}")
MESSAGE(STATUS "CUDA nvcc: ${CUDA_NVCC_EXECUTABLE}")


IF (USE_CLANG_AS_HOST_COMPILER)
    SET(CUDA_HOST_COMPILER "/usr/bin/clang")
ENDIF ()


SET(CUDA_NVCC_FLAGS "-arch=sm_30")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

MESSAGE(STATUS "Using c compiler at path: ${CMAKE_C_COMPILER}")
MESSAGE(STATUS "Using cxx compiler at path: ${CMAKE_CXX_COMPILER}")

add_executable(${PROJECT_NAME}
        main.cpp
        Device.cpp
        Device.h
        LOG.h MemAlloc.h ProcessImage.cpp ProcessImage.h MemAlloc.cpp)

SET(CUDA_LIB_NAME "Part2CUDA")
SET(CUDA_SOURCES "histogram.cu")
CUDA_ADD_LIBRARY(${CUDA_LIB_NAME} ${CUDA_SOURCES} STATIC)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
        ${CUDA_LIB_NAME}
        ${OpenCV_LIBS}
        ${CUDA_CUDART_LIBRARY})
