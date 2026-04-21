#ifndef VOXELHASHMAP_H
#define VOXELHASHMAP_H

#include <unordered_map>
#include <vector>
#include <memory>
#include <Eigen/Eigen>
#include <open3d/Open3D.h>
#include <open3d/geometry/PointCloud.h>
#include <iostream>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for.h>

#include "Common.h"


typedef std::shared_ptr<open3d::geometry::PointCloud> O3d_Cloud_Ptr;

// 体素坐标结构
struct VoxelCoord {
    int x, y, z;

    VoxelCoord(int x_ = 0, int y_ = 0, int z_ = 0) : x(x_), y(y_), z(z_) {}

    // 比较运算符，用于哈希
    bool operator==(const VoxelCoord& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// 体素哈希函数
struct VoxelHash {
    std::size_t operator()(const VoxelCoord& coord) const {
        // 使用简单的混合哈希
        return ((coord.x * 73856093) ^
            (coord.y * 19349663) ^
            (coord.z * 83492791)) & 0x7FFFFFFF;
    }

    // 静态hash方法 - TBB要求的
    static size_t hash(const VoxelCoord& coord) {
        // 计算哈希值
        return ((coord.x * 73856093) ^
            (coord.y * 19349663) ^
            (coord.z * 83492791)) & 0x7FFFFFFF;
    }

    // 静态equal方法 - TBB要求的
    static bool equal(const VoxelCoord& a, const VoxelCoord& b) {
        return a == b;  // 或者你的比较逻辑
    }

};



// 体素数据结构
struct VoxelData {
    std::vector<size_t> point_indices;  // 体素内的点的索引
    Eigen::Vector3d centroid;           // 体素的中心点（质心）
    int point_count;                    // 体素内的点数量
    bool is_visible;                    // 当前帧是否可见（用于动态目标检测）

    VoxelData() : point_count(0), is_visible(false) {}

    // 添加点到体素
    void add_point(size_t idx, const Eigen::Vector3d& point);
};

// 定义智能指针类型
typedef std::shared_ptr<VoxelData> VoxelDataPtr;

// 体素哈希映射类
class VoxelHashMap {
private:
    std::unordered_map<VoxelCoord, VoxelDataPtr, VoxelHash> voxel_map_;
    //tbb::concurrent_hash_map < VoxelCoord, VoxelDataPtr, VoxelHash> voxel_map_;
    O3d_Cloud_Ptr slam_point_cloud_; // SLAM点云
    double voxel_size_;  // 体素大小，单位：米

    // 世界坐标到体素坐标的转换
    VoxelCoord world_to_voxel(const Eigen::Vector3d& point) const {
        return VoxelCoord(
            static_cast<int>(std::floor(point.x() / voxel_size_)),
            static_cast<int>(std::floor(point.y() / voxel_size_)),
            static_cast<int>(std::floor(point.z() / voxel_size_))
        );
    }
    // 体素坐标到世界坐标的转换（返回体素中心）
    Eigen::Vector3d voxel_to_world(const VoxelCoord& coord) const {
        return Eigen::Vector3d(
            (coord.x + 0.5) * voxel_size_,
            (coord.y + 0.5) * voxel_size_,
            (coord.z + 0.5) * voxel_size_
        );
    }

public:
    VoxelHashMap(double voxel_size = 0.2);

    // 设置体素大小
    void set_voxel_size(double size) {
        voxel_size_ = size;
    }

    // 获取体素大小
    double get_voxel_size() const {
        return voxel_size_;
    }

    // 从点云构建体素哈希映射
    void build_from_pointcloud(const open3d::geometry::PointCloud& cloud);

    // 从点云构建体素哈希映射 - tbb
    void build_from_pointcloud_tbb(const open3d::geometry::PointCloud& cloud);

    // 获取指定坐标的体素
    VoxelDataPtr get_voxel(const VoxelCoord& coord);

    // 获取某点所在位置的体素
    VoxelDataPtr get_voxel_at_point(const Eigen::Vector3d& point);

    // 统计体素内信息
    void count_voxel_points(std::vector<Eigen::Vector3d>& points);

    // 获取obb框内的体素
    std::vector<VoxelDataPtr>get_voxels_in_obb(const open3d::geometry::OrientedBoundingBox& obb);


    std::vector<VoxelDataPtr> get_voxels_in_cameraFrustum(
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& pose,            // json
    double near_plane = 0.1,                // 可视化锥体近平面距离
    double far_plane = 5.0                 // 可视化锥体远平面距离
);

    std::vector<size_t> get_points_in_cameraFrustum(
        const open3d::camera::PinholeCameraIntrinsic& intrinsic,
        const Eigen::Matrix4d& camera_to_world,            // json
        double near_plane = 0.1,                // 可视化锥体近平面距离
        double far_plane = 5.0                 // 可视化锥体远平面距离
    );

    // 使用DDA寻找视野锥形体内的点
    std::vector<size_t>get_points_in_cameraFrustum_DDA(
            const open3d::camera::PinholeCameraIntrinsic& intrinsic,
            const Eigen::Matrix4d& camera_to_world,            // json
            double far_plane =5.0                // 可视化锥体远平面距离
        );

    std::vector<size_t>get_points_in_cameraFrustum_DDA_TBB_optimized(
        const open3d::camera::PinholeCameraIntrinsic& intrinsic,
        const Eigen::Matrix4d& camera_to_world,
        double far_plane = 5.0
    );


    // 获取轴对齐包围盒（AABB）内的所有体素
    std::vector<std::pair<VoxelCoord, VoxelDataPtr>>
        get_voxels_in_aabb(const Eigen::Vector3d& min_bound,
            const Eigen::Vector3d& max_bound);

    // 获取视锥体内的体素（简单AABB版本）
    std::vector<std::pair<VoxelCoord, VoxelDataPtr>>
        get_voxels_in_frustum_simple(const Eigen::Matrix4d& view_proj_matrix);

    // 获取所有体素
    std::vector<std::pair<VoxelCoord, VoxelDataPtr>> get_all_voxels();

    // 获取体素数量
    size_t size() const {
        return voxel_map_.size();
    }

    // 清除可见性标记（用于新一帧开始）
    void clear_visibility_flags();

    // 估算内存使用量（粗略）
    size_t estimate_memory_usage() const;

    // 下采样点云（可选颜色）
    std::shared_ptr<open3d::geometry::PointCloud>
        downsample_pointcloud(const open3d::geometry::PointCloud& cloud) const;
};

#endif