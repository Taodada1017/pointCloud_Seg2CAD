#include "VoxelHashMap.h"
#include <tbb/tbb.h>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_set.h>

// VoxelData 成员函数实现
void VoxelData::add_point(size_t idx, const Eigen::Vector3d& point) {
    point_indices.push_back(idx);

    // 增量更新质心，避免保存所有点
    /*centroid = centroid * (point_count / (point_count + 1.0)) +
        point / (point_count + 1.0);*/
    point_count++;
}

// VoxelHashMap 成员函数实现
VoxelHashMap::VoxelHashMap(double voxel_size) : voxel_size_(voxel_size) {}

// 构建体素网格
void VoxelHashMap::build_from_pointcloud(const open3d::geometry::PointCloud& cloud) {
    std::cout << "开始构建体素网格，体素正方体边长为：" << voxel_size_ << " 米" << std::endl;
    std::cout << "输入点云点数: " << cloud.points_.size() << std::endl;

    // 清空现有地图
    voxel_map_.clear();

    // 遍历点云
    for (size_t i = 0; i < cloud.points_.size(); ++i) {
        const auto& point = cloud.points_[i];

        // 计算体素坐标
        VoxelCoord coord = world_to_voxel(point);

        // 查找体素
        auto it = voxel_map_.find(coord);
        if (it != voxel_map_.end()) {
            // 体素已存在，添加点
            it->second->add_point(i, point);
        }
        else {
            // 创建新体素（使用智能指针）
            VoxelDataPtr voxel = std::make_shared<VoxelData>();
            voxel->add_point(i, point);
            voxel_map_[coord] = voxel;
        }
    }

    std::cout << "体素哈希映射构建完成" << std::endl;
    std::cout << "体素总数: " << voxel_map_.size() << std::endl;

    // 统计信息
    int max_points = 0;
    int min_points = INT_MAX;
    double avg_points = 0.0;
    int empty_voxels = 0;

    for (const auto& pair : voxel_map_) {
        int count = pair.second->point_count;
        if (count > max_points) max_points = count;
        if (count < min_points) min_points = count;
        avg_points += count;
        if (count == 0) empty_voxels++;
    }
    avg_points /= voxel_map_.size();

    std::cout << "体素内点数量统计:" << std::endl;
    std::cout << "  - 最多: " << max_points << std::endl;
    std::cout << "  - 最少: " << min_points << std::endl;
    std::cout << "  - 平均: " << avg_points << std::endl;
    std::cout << "  - 空体素: " << empty_voxels << std::endl;
}


// 构建体素网格 - tbb优化  (报错，没法用)
void VoxelHashMap::build_from_pointcloud_tbb(const open3d::geometry::PointCloud& cloud) {
    std::cout << "开始构建体素网格，体素正方体边长为：" << voxel_size_ << " 米" << std::endl;
    std::cout << "输入点云点数: " << cloud.points_.size() << std::endl;

    // 清空现有地图
    voxel_map_.clear();

    tbb::concurrent_hash_map<VoxelCoord, VoxelDataPtr, VoxelHash> voxel_map_2;


    tbb::parallel_for(tbb::blocked_range<size_t>(0, cloud.points_.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                const auto& point = cloud.points_[i];
                VoxelCoord coord = world_to_voxel(point);

                // 使用访问器保证线程安全
                typename decltype(voxel_map_2)::accessor accessor;
                if (voxel_map_2.insert(accessor, coord)) {
                    // 新插入，创建voxel
                    accessor->second = std::make_shared<VoxelData>();
                }
                // 已存在或新创建的voxel
                accessor->second->add_point(i, point);
            }
        }
    );

    // 并发遍历所有元素
    tbb::parallel_for_each(voxel_map_2.begin(), voxel_map_2.end(),
        [&](const std::pair<VoxelCoord, VoxelDataPtr>& item) {
            voxel_map_[item.first] = item.second;
        }
    );


    std::cout << "体素哈希映射构建完成" << std::endl;
    std::cout << "体素总数: " << voxel_map_.size() << std::endl;

    // 统计信息
    int max_points = 0;
    int min_points = INT_MAX;
    double avg_points = 0.0;
    int empty_voxels = 0;

    for (const auto& pair : voxel_map_) {
        int count = pair.second->point_count;
        if (count > max_points) max_points = count;
        if (count < min_points) min_points = count;
        avg_points += count;
        if (count == 0) empty_voxels++;
    }
    avg_points /= voxel_map_.size();

    std::cout << "体素内点数量统计:" << std::endl;
    std::cout << "  - 最多: " << max_points << std::endl;
    std::cout << "  - 最少: " << min_points << std::endl;
    std::cout << "  - 平均: " << avg_points << std::endl;
    std::cout << "  - 空体素: " << empty_voxels << std::endl;
}

VoxelDataPtr VoxelHashMap::get_voxel(const VoxelCoord& coord) {
    auto it = voxel_map_.find(coord);
    if (it != voxel_map_.end()) {
        return it->second;
    }
    return nullptr;  // 返回空智能指针
}

VoxelDataPtr VoxelHashMap::get_voxel_at_point(const Eigen::Vector3d& point) {
    VoxelCoord coord = world_to_voxel(point);
    return get_voxel(coord);
}

std::vector<std::pair<VoxelCoord, VoxelDataPtr>>
VoxelHashMap::get_voxels_in_aabb(const Eigen::Vector3d& min_bound,
    const Eigen::Vector3d& max_bound) {
    std::vector<std::pair<VoxelCoord, VoxelDataPtr>> result;

    // 计算体素坐标范围
    VoxelCoord min_coord = world_to_voxel(min_bound);
    VoxelCoord max_coord = world_to_voxel(max_bound);

    // 遍历范围内的体素
    for (int x = min_coord.x; x <= max_coord.x; ++x) {
        for (int y = min_coord.y; y <= max_coord.y; ++y) {
            for (int z = min_coord.z; z <= max_coord.z; ++z) {
                VoxelCoord coord(x, y, z);
                auto it = voxel_map_.find(coord);
                if (it != voxel_map_.end()) {
                    result.emplace_back(coord, it->second);
                }
            }
        }
    }

    return result;
}

// 寻找视野锥形体内的体素
std::vector<VoxelDataPtr>
VoxelHashMap::get_voxels_in_cameraFrustum(
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& pose,            // json
    double near_plane,                // 可视化锥体近平面距离
    double far_plane                // 可视化锥体远平面距离
) {

    std::vector<VoxelDataPtr> result;

    int image_width = intrinsic.width_;
    int image_height = intrinsic.height_;

    // 获取内参
    Eigen::Matrix3d K = intrinsic.intrinsic_matrix_;
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);

    Eigen::Matrix4d camera_to_world = pose;
    Eigen::Matrix3d R = camera_to_world.block<3, 3>(0, 0);
    Eigen::Vector3d t = camera_to_world.block<3, 1>(0, 3);

    //int z_size = (far_plane - near_plane) / voxel_size;
        double step = voxel_size_ / 2;
    // 避免重复添加
    std::unordered_map<VoxelCoord, bool, VoxelHash> is_added;

    int count = 0;

    for (double z_c = near_plane; z_c < far_plane; z_c += step) {
        double plane_width = (z_c / fx) * image_width;
        double plane_height = (z_c / fy) * image_height;

        int x_n = plane_width / voxel_size_;
        int y_n = plane_height / voxel_size_;

        double x_corner = -plane_width / 2;
        double y_corner = -plane_height / 2;
        
        double x_max = x_corner + (x_n + 1) * voxel_size_;
        double y_max = y_corner + (y_n + 1) * voxel_size_;

        

        for (double x_c=x_corner; x_c < x_max; x_c += step) {
            for (double y_c = y_corner; y_c < y_max; y_c += step) {

                count++;

                Eigen::Vector3d point_c(x_c, y_c, z_c);
                // 转换到世界坐标
                Eigen::Vector3d point_w = R * point_c + t;
                // 转为体素坐标
                VoxelCoord coord = world_to_voxel(point_w);

                if (is_added[coord])continue;

                // 寻找体素
                auto it = voxel_map_.find(coord);
                if (it != voxel_map_.end()) {
                        result.push_back(it->second);
                        is_added[coord] = true;
                }

            }

        }

    }
    std::cout << "循环次数：" << count << std::endl;
    std::cout << "视锥体内体素数量：" << result.size() << std::endl;

    return result;

}


// 寻找视野锥形体内的点
std::vector<size_t>
VoxelHashMap::get_points_in_cameraFrustum(
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,            // json
    double near_plane,                // 可视化锥体近平面距离
    double far_plane                // 可视化锥体远平面距离
) {

    std::cout << "==== 开始遍历寻找视锥体内点云 =====" << std::endl;


    std::vector<size_t> result;
    std::vector<VoxelDataPtr> voxel_vec;

    int image_width = intrinsic.width_;
    int image_height = intrinsic.height_;

    // 获取内参
    Eigen::Matrix3d K = intrinsic.intrinsic_matrix_;
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);

    //Eigen::Matrix4d camera_to_world = pose;
    Eigen::Matrix3d R = camera_to_world.block<3, 3>(0, 0);
    Eigen::Vector3d t = camera_to_world.block<3, 1>(0, 3);

    //int z_size = (far_plane - near_plane) / voxel_size;
    double step = voxel_size_ / 2;
    // 避免重复添加
    std::unordered_map<VoxelCoord, bool, VoxelHash> is_added;

    int count = 0;

    for (double z_c = near_plane; z_c < far_plane; z_c += step) {
        double plane_width = (z_c / fx) * image_width;
        double plane_height = (z_c / fy) * image_height;

        int x_n = plane_width / voxel_size_;
        int y_n = plane_height / voxel_size_;

        double x_corner = -plane_width / 2;
        double y_corner = -plane_height / 2;

        double x_max = x_corner + (x_n + 1) * voxel_size_;
        double y_max = y_corner + (y_n + 1) * voxel_size_;

        for (double x_c = x_corner; x_c < x_max; x_c += step) {
            for (double y_c = y_corner; y_c < y_max; y_c += step) {

                count++;

                Eigen::Vector3d point_c(x_c, y_c, z_c);
                // 转换到世界坐标
                Eigen::Vector3d point_w = R * point_c + t;
                // 转为体素坐标
                VoxelCoord coord = world_to_voxel(point_w);

                /*if (count < 20) {
                    if (count < 20) {
                        std::cout << "x y z:" << coord.x << " " << coord.y << " " << coord.z << std::endl;
                    }
                }*/

                if (is_added[coord])continue;

                // 寻找体素
                auto it = voxel_map_.find(coord);
                if (it != voxel_map_.end()) {
                    voxel_vec.push_back(it->second);
                    is_added[coord] = true;
                }

            }
        }

    }
    std::cout << "循环次数：" << count << std::endl;
    std::cout << "视锥体内体素数量：" << voxel_vec.size() << std::endl;

    // 获取所以体素里面的点云下标
    for (auto& voxel : voxel_vec) {
        for (size_t& index : voxel->point_indices) {
            result.push_back(index);
        }
    }

    std::cout << "找到视锥体内点云数量：" << result.size() << std::endl;

    return result;

}



// 统计凸包面上的点所在体素的信息
void VoxelHashMap::count_voxel_points(std::vector<Eigen::Vector3d>& points) {

    std::unordered_map<VoxelCoord, bool, VoxelHash> is_added;

    int voxel_count = 0;
    size_t points_count = 0;

    size_t min_points = 100000;
    size_t max_points = 0;

    for (auto point : points) {
        VoxelCoord coord = world_to_voxel(point);
        if (is_added.find(coord) != is_added.end())continue;
        if (voxel_map_.find(coord) == voxel_map_.end())continue;

        voxel_count++;
        size_t voxel_points = voxel_map_[coord]->point_count;
        points_count += voxel_points;
        if (voxel_points < min_points) {
            min_points = voxel_points;
        }

        if (voxel_points > max_points) {
            max_points = voxel_points;
        }

        is_added[coord] = true;
    }

    std::cout<<"体素内最大点数量：" << max_points << std::endl;
    std::cout<<"体素内最小点数量：" << min_points << std::endl;
    std::cout<<"体素内平均点数量：" << points_count / voxel_count << std::endl;




}


bool IsPointInObb(const Eigen::Vector3d &point,
    const open3d::geometry::OrientedBoundingBox& obb) {

    // 获取OBB参数
    Eigen::Vector3d center = obb.center_;
    Eigen::Matrix3d R = obb.R_;  // 局部到世界的旋转
    Eigen::Vector3d extent = obb.extent_;  // 半轴长度

    Eigen::Vector3d dx = R * Eigen::Vector3d(1, 0, 0);
    Eigen::Vector3d dy = R * Eigen::Vector3d(0, 1, 0);
    Eigen::Vector3d dz = R * Eigen::Vector3d(0, 0, 1);
    Eigen::Vector3d d = point - center;
    if (std::abs(d.dot(dx)) <= extent(0) / 2 &&
        std::abs(d.dot(dy)) <= extent(1) / 2 &&
        std::abs(d.dot(dz)) <= extent(2) / 2) {
        return true;
    }
    else return false;
}


// 获取obb框内的所有体素
std::vector<VoxelDataPtr>
VoxelHashMap::get_voxels_in_obb(const open3d::geometry::OrientedBoundingBox& obb) {


    std::vector<VoxelDataPtr> ans_voxels;

    // 1. 获取OBB的AABB
    auto aabb = obb.GetAxisAlignedBoundingBox();

    // 2. 计算AABB的体素范围
    VoxelCoord min_voxel = world_to_voxel(aabb.min_bound_);
    VoxelCoord max_voxel = world_to_voxel(aabb.max_bound_);

    // 体素的8个角点的相对偏移
    std::vector<Eigen::Vector3d> corner_offsets = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}, {1.0, 1.0, 1.0}
    };

    // 3. 遍历AABB内的所有体素
    for (int x = min_voxel.x; x <= max_voxel.x; ++x) {
        for (int y = min_voxel.y; y <= max_voxel.y; ++y) {
            for (int z = min_voxel.z; z <= max_voxel.z; ++z) {

                VoxelCoord coord(x, y, z );
                
                // 检查体素的8个角点
                bool is_inside = false;
                for (const auto& offset : corner_offsets) {
                    Eigen::Vector3d corner_point(
                        (x + offset.x()) * voxel_size_,
                        (y + offset.y()) * voxel_size_,
                        (z + offset.z()) * voxel_size_
                    );

                    if (IsPointInObb(corner_point, obb)) {
                        is_inside = true;
                        break;
                    }
                }

                if (is_inside) {
                    auto it = voxel_map_.find(coord);
                    if(it != voxel_map_.end()) ans_voxels.emplace_back(it->second);
                }

            }
        }
    }

    //std::cout << "找到obb框内体素数量：" << ans_voxels.size() << std::endl;

    return ans_voxels;

}


//
//std::vector<size_t> get_points_indices_in_obb(const open3d::geometry::OrientedBoundingBox& obb) {
//
//    std::vector<VoxelDataPtr> vo
//}




// 使用DDA寻找视野锥形体内的点
std::vector<size_t>
VoxelHashMap::get_points_in_cameraFrustum_DDA(
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,            // json
    double far_plane                // 可视化锥体远平面距离
) {

    std::cout<<"============ 开始使用DDA算法寻找视锥体内点云 ================" << std::endl;

    std::vector<size_t> result;
    std::vector<VoxelDataPtr> voxel_vec;

    int image_width = intrinsic.width_;
    int image_height = intrinsic.height_;

    // 获取内参
    Eigen::Matrix3d K = intrinsic.intrinsic_matrix_;
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);

    //Eigen::Matrix4d camera_to_world = pose;
    Eigen::Matrix3d R = camera_to_world.block<3, 3>(0, 0);
    Eigen::Vector3d t = camera_to_world.block<3, 1>(0, 3);

    if (voxel_size_ <= 0) {
        std::cout << "voxel_size_数据错误" << std::endl;
        
    }

    // 所有光线的起点
    double x0 = t.x() / voxel_size_;
    double y0 = t.y() / voxel_size_;
    double z0 = t.z() / voxel_size_;

    
    double plane_width = far_plane  * (image_width/ fx);
    double plane_height = far_plane * (image_height / fy);

    int x_n = plane_width / voxel_size_;
    int y_n = plane_height / voxel_size_;

    double x_corner = -plane_width / 2;
    double y_corner = -plane_height / 2;

    double x_max = x_corner + (x_n + 1) * voxel_size_;
    double y_max = y_corner + (y_n + 1) * voxel_size_;


    double step = voxel_size_ ;
    // 避免重复添加
    std::unordered_map<VoxelCoord, bool, VoxelHash> is_added;

    //std::cout << "参数处理完成" << std::endl;

    // 添加起点
    VoxelCoord coord = world_to_voxel(t);
    auto it = voxel_map_.find(coord);
    if (it != voxel_map_.end()) {
        voxel_vec.push_back(it->second);
        is_added[coord] = true;
    }

    //std::cout << "起点添加判断成功" << std::endl;

    int count = 0;

    // 遍历远平面的所有点，作为各个光线的终点
    for (double x_c = x_corner; x_c < x_max; x_c += step) {
        for (double y_c = y_corner; y_c < y_max; y_c += step) {

 
            //Eigen::Vector3d point_c(x_c, y_c, far_plane);
            Eigen::Vector3d point_c(x_c, -y_c, -far_plane);       // y\z轴取反
            // 转换到世界坐标 Eigen::Vector3d point_w = R * point_c + t;
            // 因为起点(相机位置)就是t，所以两者相减之后的d_xy就是起点到终点的位移
            Eigen::Vector3d d_xyz = R * point_c;  
            
            double dx = d_xyz.x() / voxel_size_;
            double dy = d_xyz.y() / voxel_size_;
            double dz = d_xyz.z() / voxel_size_;


            // 计算步数（取变化较大的方向）
            double steps = std::max({ fabs(dx), fabs(dy), fabs(dz) });

            if (steps == 0.0)continue;

            // 计算每一步的增量
            double xIncrement = dx / steps;
            double yIncrement = dy / steps;
            double zIncrement = dz / steps;


            // 从起点开始，逐步遍历到终点
            double x = x0;
            double y = y0;
            double z = z0;

            // 保证steps是体素坐标系
            for (int i = 0; i < static_cast<int>(std::round(steps)); ++i) {

                count++;

                x += xIncrement;
                y += yIncrement;
                z += zIncrement;

                // 寻找临近体素坐标
                VoxelCoord coord2 (std::floor(x), std::floor(y), std::floor(z));
                if (is_added[coord2])continue;
                // 寻找体素 
                it = voxel_map_.find(coord2);
                if (it != voxel_map_.end()) {
                    voxel_vec.push_back(it->second);
                    is_added[coord2] = true;
                    // 做一部分遮挡判断
                    if (it->second->point_count > 1500) {
                        break;
                    }
                }
            }
        }
    }

    //std::cout << "循环次数：" << count << std::endl;
    std::cout << "视锥体内体素数量：" << voxel_vec.size() << std::endl;

    // 获取所以体素里面的点云下标
    for (auto& voxel : voxel_vec) {
        for (size_t& index : voxel->point_indices) {
            result.push_back(index);
        }
    }

    std::cout << "找到视锥体内点云数量：" << result.size() << std::endl;

    return result;


}




// 在类中添加线程本地存储结构
struct ThreadLocalData {
    
    std::unordered_set<VoxelCoord, VoxelHash> local_visited;

    void clear() {
        local_visited.clear();
    }
};


std::vector<size_t>
VoxelHashMap::get_points_in_cameraFrustum_DDA_TBB_optimized(
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,
    double far_plane
) {
    std::cout << "============ 开始使用优化版TBB并行DDA算法 ================" << std::endl;

    std::vector<size_t> result;

    int image_width = intrinsic.width_;
    int image_height = intrinsic.height_;
    Eigen::Matrix3d K = intrinsic.intrinsic_matrix_;
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);

    Eigen::Matrix3d R = camera_to_world.block<3, 3>(0, 0);
    Eigen::Vector3d t = camera_to_world.block<3, 1>(0, 3);

    // 所有光线的起点（体素坐标系）
    double x0 = t.x() / voxel_size_;
    double y0 = t.y() / voxel_size_;
    double z0 = t.z() / voxel_size_;

    double plane_width = far_plane * (image_width / fx);
    double plane_height = far_plane * (image_height / fy);

    int x_n = static_cast<int>(plane_width / voxel_size_);
    int y_n = static_cast<int>(plane_height / voxel_size_);

    double x_corner = -plane_width / 2;
    double y_corner = -plane_height / 2;

    double x_max = x_corner + (x_n + 1) * voxel_size_;
    double y_max = y_corner + (y_n + 1) * voxel_size_;

    double step = voxel_size_;

    tbb::concurrent_vector<VoxelDataPtr> voxel_vec;
    tbb::concurrent_unordered_set<VoxelCoord, VoxelHash> visited_voxels;
    tbb::combinable<ThreadLocalData> thread_local_data([]() { return ThreadLocalData(); });

    tbb::atomic<size_t> total_count = 0;

    // 添加起点体素
    VoxelCoord start_coord = world_to_voxel(t);
    auto it = voxel_map_.find(start_coord);
    if (it != voxel_map_.end()) {
        voxel_vec.push_back(it->second);
        visited_voxels.insert(start_coord);
    }

    int x_steps = x_n + 1;
    int y_steps = y_n + 1;

    // 使用tbb::parallel_for_each进行更细粒度的并行
    tbb::parallel_for(0, x_steps, [&](int i) {
        auto& local_data = thread_local_data.local();

        double x_c = x_corner + i * step;

        for (int j = 0; j < y_steps; ++j) {
            double y_c = y_corner + j * step;

            Eigen::Vector3d point_c(x_c, y_c, far_plane);
            Eigen::Vector3d d_xyz = R * point_c;

            double dx = d_xyz.x() / voxel_size_;
            double dy = d_xyz.y() / voxel_size_;
            double dz = d_xyz.z() / voxel_size_;

            double steps = std::max({ std::fabs(dx), std::fabs(dy), std::fabs(dz) });
            if (steps == 0.0) continue;

            double xIncrement = dx / steps;
            double yIncrement = dy / steps;
            double zIncrement = dz / steps;

            double x = x0;
            double y = y0;
            double z = z0;

            int step_count = static_cast<int>(std::round(steps));
            total_count += step_count;

            for (int k = 0; k < step_count; ++k) {
                x += xIncrement;
                y += yIncrement;
                z += zIncrement;

                VoxelCoord coord(static_cast<int>(std::floor(x)),
                    static_cast<int>(std::floor(y)),
                    static_cast<int>(std::floor(z)));

                if (local_data.local_visited.find(coord) != local_data.local_visited.end()) {
                    continue;
                }

                if (visited_voxels.find(coord) != visited_voxels.end()) {
                    continue;
                }

                if (voxel_map_.find(coord) != voxel_map_.end()) {
                    local_data.local_visited.insert(coord);
                }
            }
        }
        });

    // 合并所有线程本地数据
    thread_local_data.combine_each([&](ThreadLocalData& data) {
        for (auto& coord : data.local_visited) {
            auto it = voxel_map_.find(coord);
            voxel_vec.push_back(it->second);
        }
        });

    std::cout << "循环次数：" << total_count << std::endl;
    std::cout << "视锥体内体素数量：" << voxel_vec.size() << std::endl;

    // 收集所有体素中的点云索引
    for (auto& voxel : voxel_vec) {
        result.insert(result.end(), voxel->point_indices.begin(), voxel->point_indices.end());
    }

    std::cout << "找到视锥体内点云数量：" << result.size() << std::endl;
    // ... 后续收集结果的代码与上面相同 ...

    return result;
}




std::vector<std::pair<VoxelCoord, VoxelDataPtr>>
VoxelHashMap::get_voxels_in_frustum_simple(const Eigen::Matrix4d& view_proj_matrix) {
    // 注意：这是极简的临时代码，实际应用中应该实现真实的视锥体裁剪
    // 此方法目前仅返回所有体素，用于功能占位

    std::vector<std::pair<VoxelCoord, VoxelDataPtr>> result;
    result.reserve(voxel_map_.size());

    for (auto& pair : voxel_map_) {
        result.emplace_back(pair.first, pair.second);
    }

    return result;
}

std::vector<std::pair<VoxelCoord, VoxelDataPtr>> VoxelHashMap::get_all_voxels() {
    std::vector<std::pair<VoxelCoord, VoxelDataPtr>> result;
    result.reserve(voxel_map_.size());

    for (auto& pair : voxel_map_) {
        result.emplace_back(pair.first, pair.second);
    }

    return result;
}

void VoxelHashMap::clear_visibility_flags() {
    for (auto& pair : voxel_map_) {
        pair.second->is_visible = false;
    }
}

size_t VoxelHashMap::estimate_memory_usage() const {
    size_t total = 0;

    // 哈希表桶内存
    total += voxel_map_.bucket_count() * sizeof(void*);

    // 键值对内存
    for (const auto& pair : voxel_map_) {
        total += sizeof(VoxelCoord);  // 键
        total += sizeof(VoxelDataPtr); // 智能指针本身
        total += sizeof(VoxelData);    // VoxelData对象
        total += pair.second->point_indices.capacity() * sizeof(size_t);
    }

    return total;
}

std::shared_ptr<open3d::geometry::PointCloud>
VoxelHashMap::downsample_pointcloud(const open3d::geometry::PointCloud& cloud) const {
    auto downsampled = std::make_shared<open3d::geometry::PointCloud>();

    for (const auto& pair : voxel_map_) {
        const auto& voxel = pair.second;
        if (voxel->point_count > 0) {
            // 使用体素质心作为下采样后的点
            downsampled->points_.push_back(voxel->centroid);

            // 可选：平均颜色
            if (!cloud.colors_.empty() && voxel->point_indices.size() > 0) {
                Eigen::Vector3d avg_color = Eigen::Vector3d::Zero();
                for (auto idx : voxel->point_indices) {
                    if (idx < cloud.colors_.size()) {
                        avg_color += cloud.colors_[idx];
                    }
                }
                avg_color /= voxel->point_indices.size();
                downsampled->colors_.push_back(avg_color);
            }
        }
    }

    return downsampled;
}