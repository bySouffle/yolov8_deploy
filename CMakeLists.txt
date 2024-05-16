cmake_minimum_required(VERSION 3.16)
project(yolov8_deploy)

set(PROJECT_NAME yolov8_deploy)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# -------------- OpenCV  ------------------#
#   If custom install OpenCV, define OpenCV_DIR
#set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV 4.7 REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- OpenVINO  ------------------#
find_package(OpenVINO REQUIRED)


# -------------- Compile CUDA for FP16 inference if needed  ------------------#
option(USE_CUDA "Enable CUDA support" OFF)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA OFF)
endif ()
# -------------- ONNXRUNTIME  ------------------#
# Set ONNXRUNTIME_VERSION
set(ONNXRUNTIME_VERSION 1.17.3)
MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}")

if (WIN32)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
    endif ()
elseif (UNIX)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
    endif ()
elseif (APPLE)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}")
    # Apple X64 binary
    # set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-x64-${ONNXRUNTIME_VERSION}")
    # Apple Universal binary
    # set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}")
endif ()
message(STATUS "ONNXRUNTIME_ROOT: ${ONNXRUNTIME_ROOT}")

include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/include)
include_directories(${PROJECT_NAME} modules)
file(GLOB_RECURSE Model
        modules/*.h
        modules/*.cc
        modules/*.cpp
)

add_executable(${PROJECT_NAME} main.cpp ${Model})
target_link_libraries(${PROJECT_NAME} openvino::runtime)

if (WIN32)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (UNIX)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (APPLE)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
endif ()

# For windows system, copy onnxruntime.dll to the same folder of the executable file
if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif ()

