cmake_minimum_required(VERSION 3.15)
project(Test VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 将cmake文件夹添加到模块路径
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# 包含外部依赖配置
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/opencv.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/open3d.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/jsoncpp.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/eigen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tbb.cmake)

# 添加子目录
add_subdirectory(src)

# 创建可执行文件
add_executable(${PROJECT_NAME}_main main.cpp)


# PDAL
# find_package(PDAL REQUIRED)

# 链接库
target_link_libraries(${PROJECT_NAME}_main
    PRIVATE
        src_lib
        TBB::tbb
        # ${PDAL_LIBRARIES}
        Open3D::Open3D
        ${OpenCV_LIBS}
        jsoncpp_lib
)

# 包含头文件目录
target_include_directories(${PROJECT_NAME}_main
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

