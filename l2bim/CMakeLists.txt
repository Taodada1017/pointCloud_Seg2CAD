cmake_minimum_required(VERSION 3.15)
project(LiDAR2BIM VERSION 0.1.0 LANGUAGES CXX)
option(USE_BACKWARD "Enable backward-cpp (Windows may hit override issues)" OFF)

message("start make LiDAR2BIM")

if (MSVC)
  add_compile_options(/utf-8)
endif()


set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_STANDARD 17)

set(ALL_TARGET_LIBRARIES "")    
# /*include(cmake/boost.cmake)*/
include(cmake/eigen.cmake)
include(cmake/yaml.cmake)
include(cmake/glog.cmake)
include(cmake/pcl.cmake)
include(cmake/openmp.cmake)
include(cmake/global_definition.cmake)
include(cmake/opencv.cmake)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
)
include_directories(
        ${PROJECT_SOURCE_DIR}/Thirdparty/point-sam/include
)
include_directories(
        ${PROJECT_SOURCE_DIR}/Thirdparty/nanoflann/include
)
file(GLOB_RECURSE ALL_SRCS "src/*.c*" "include/*.h*")
if (NOT MSVC)
    list(APPEND ALL_TARGET_LIBRARIES stdc++fs)
endif()
list(APPEND ALL_TARGET_LIBRARIES tbb)

add_library(${PROJECT_NAME} STATIC ${ALL_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

add_subdirectory(Thirdparty/point-sam)
# ---- tweak compile options for point-sam AFTER it's added ----
# 有些版本导出的是别名 POINT_SAM::point_sam，这里做兼容处理
set(_PS_TARGET "")
if (TARGET point_sam)
  set(_PS_TARGET point_sam)
elseif (TARGET POINT_SAM::point_sam)
  get_target_property(_aliased POINT_SAM::point_sam ALIASED_TARGET)
  if (_aliased)
    set(_PS_TARGET ${_aliased})
  endif()
endif()

if (_PS_TARGET)
  if (MSVC)
    target_compile_options(${_PS_TARGET} PRIVATE /O2 /utf-8)
    target_compile_definitions(${_PS_TARGET} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  else()
    target_compile_options(${_PS_TARGET} PRIVATE -O3 -march=native)
  endif()
else()
  message(WARNING "point-sam target not found; skipped compile-options tweak.")
endif()

if (USE_BACKWARD)
  # Windows 下使用 PDB 符号后端
  set(BACKWARD_ENABLE ON CACHE BOOL "" FORCE)
  set(BACKWARD_HAS_BFD OFF CACHE BOOL "" FORCE)
  set(BACKWARD_HAS_DWARF OFF CACHE BOOL "" FORCE)
  set(BACKWARD_HAS_PDB_SYMBOLS ON CACHE BOOL "" FORCE)
  add_subdirectory(Thirdparty/backward-cpp)
endif()

add_subdirectory(Thirdparty/nanoflann)

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${ALL_TARGET_LIBRARIES}
        POINT_SAM::point_sam
)
message(STATUS "ALL_TARGET_LIBRARIES: ${ALL_TARGET_LIBRARIES}")
add_library(LiDAR2BIM::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# set bin output path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(reg_bm examples/cpp/reg_bm.cpp)
if (USE_BACKWARD)
  add_backward(reg_bm)
endif()
target_link_libraries(reg_bm PRIVATE ${PROJECT_NAME})

add_executable(demo_reg examples/cpp/demo_reg.cpp)
if (USE_BACKWARD)
  add_backward(demo_reg)
endif()
target_link_libraries(demo_reg PRIVATE ${PROJECT_NAME})


find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Projection example
add_executable(pcd_projection examples/cpp/pcd_projection.cpp)
if (USE_BACKWARD)
  add_backward(pcd_projection)
endif()
# Only compile minimal files needed for projection to avoid building whole lib
target_sources(pcd_projection PRIVATE
  src/frontend/geometry/point_cloud.cpp
  src/utils/raster_utils.cpp
)
target_link_libraries(pcd_projection PRIVATE ${PCL_LIBRARIES} opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_ximgproc)

# Projection + line detection example
add_executable(pcd2line_test examples/cpp/pcd2line_test.cpp)
if (USE_BACKWARD)
  add_backward(pcd2line_test)
endif()
target_include_directories(pcd2line_test PRIVATE
  ${PROJECT_SOURCE_DIR}/include/frontend/geometry
)
target_sources(pcd2line_test PRIVATE
  src/frontend/geometry/point_cloud.cpp
  src/frontend/geometry/lineseg.cpp
  src/frontend/geometry/corner.cpp
  src/frontend/geometry/ED.cpp
  src/frontend/geometry/EDColor.cpp
  src/frontend/geometry/NFA.cpp
  src/frontend/geometry/EDLines.cpp
  src/utils/raster_utils.cpp
)
target_link_libraries(pcd2line_test PRIVATE ${PCL_LIBRARIES} opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)

if (EXISTS ${PROJECT_SOURCE_DIR}/examples/cpp/tiled_pcd2line.cpp)
  add_executable(tiled_pcd2line examples/cpp/tiled_pcd2line.cpp)
  if (USE_BACKWARD)
    add_backward(tiled_pcd2line)
  endif()
  target_include_directories(tiled_pcd2line PRIVATE
    ${PROJECT_SOURCE_DIR}/include/frontend/geometry
  )
  target_sources(tiled_pcd2line PRIVATE
    src/frontend/geometry/point_cloud.cpp
    src/frontend/geometry/lineseg.cpp
    src/frontend/geometry/corner.cpp
    src/frontend/geometry/ED.cpp
    src/frontend/geometry/EDColor.cpp
    src/frontend/geometry/NFA.cpp
    src/frontend/geometry/EDLines.cpp
    src/utils/raster_utils.cpp
  )
  target_link_libraries(tiled_pcd2line PRIVATE ${PCL_LIBRARIES} opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)
endif()

if (EXISTS ${PROJECT_SOURCE_DIR}/examples/cpp/test_before.cpp)
  add_executable(test_before examples/cpp/test_before.cpp)
  if (USE_BACKWARD)
    add_backward(test_before)
  endif()
  target_include_directories(test_before PRIVATE
    ${PROJECT_SOURCE_DIR}/include/frontend/geometry
  )
  target_sources(test_before PRIVATE
    src/frontend/geometry/point_cloud.cpp
    src/frontend/geometry/lineseg.cpp
    src/frontend/geometry/corner.cpp
    src/frontend/geometry/ED.cpp
    src/frontend/geometry/EDColor.cpp
    src/frontend/geometry/NFA.cpp
    src/frontend/geometry/EDLines.cpp
    src/utils/raster_utils.cpp
  )
  target_link_libraries(test_before PRIVATE ${PCL_LIBRARIES} opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)
endif()

# Image post-process tool (binarize/close/connected-components filtering)
add_executable(img_plus examples/cpp/img_plus.cpp)
if (USE_BACKWARD)
  add_backward(img_plus)
endif()
target_link_libraries(img_plus PRIVATE opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)

# walls_2d_fill.pcd breakthrough experiment tool
add_executable(projection_2d examples/cpp/projection_2d.cpp)
if (USE_BACKWARD)
  add_backward(projection_2d)
endif()
target_link_libraries(projection_2d PRIVATE ${PCL_LIBRARIES} opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)
