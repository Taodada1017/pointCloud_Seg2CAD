set(FMFUSION_SRC
        cluster/PoseGraph.cpp
        mapping/VoxelHashMap.cpp
        mapping/SubVolume.cpp
        mapping/Detection.cpp
        mapping/Instance.cpp
        sgloop/Graph.cpp
        tools/Utility.cpp
        tools/IO.cpp
        mapping/SemanticMapping.cpp
)

# 创建库
add_library(src_lib SHARED ${FMFUSION_SRC})

# 设置库的属性
set_target_properties(src_lib PROPERTIES
    OUTPUT_NAME "src"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# 包含头文件目录
target_include_directories(src_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(src_lib
    PRIVATE
        Open3D::Open3D
        ${OpenCV_LIBS}
        jsoncpp_lib
)


# 设置头文件搜索路径
target_include_directories(src_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mapping
    ${CMAKE_CURRENT_SOURCE_DIR}/tool
    ${CMAKE_CURRENT_SOURCE_DIR}/cluster
    ${CMAKE_CURRENT_SOURCE_DIR}/sgloop
)