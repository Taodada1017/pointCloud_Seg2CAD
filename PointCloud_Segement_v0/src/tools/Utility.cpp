#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"
#include "Utility.h"
#include <open3d/geometry/Qhull.h>
#include <chrono>
#include <omp.h>
#include "TicToc.h"
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>


namespace fmfusion
{

namespace utility
{

// 基础版本：获取整数坐标颜色（不使用四舍五入）
Eigen::Vector3d GetPixelColorDirect(const open3d::geometry::Image& image, int x, int y) {
    // 边界检查
    x = std::max(0, std::min(x, image.width_ - 1));
    y = std::max(0, std::min(y, image.height_ - 1));

    Eigen::Vector3d color(0.5, 0.5, 0.5);

    if (image.bytes_per_channel_ == 1) {
        const uint8_t* data = image.PointerAt<uint8_t>(x, y, 0);
        if (image.num_of_channels_ >= 3) {
            color(0) = data[0] / 255.0;
            color(1) = data[1] / 255.0;
            color(2) = data[2] / 255.0;
        } else if (image.num_of_channels_ == 1) {
            double gray = data[0] / 255.0;
            color(0) = color(1) = color(2) = gray;
        }
    } else if (image.bytes_per_channel_ == 4) {
        const float* data = image.PointerAt<float>(x, y, 0);
        if (image.num_of_channels_ >= 3) {
            color(0) = data[0];
            color(1) = data[1];
            color(2) = data[2];
        } else if (image.num_of_channels_ == 1) {
            double gray = data[0];
            color(0) = color(1) = color(2) = gray;
        }
    }
    return color;
}
// 双线性插值版本
Eigen::Vector3d GetPixelColorBilinear(const open3d::geometry::Image& image, double u, double v) {
    int x0 = static_cast<int>(u);
    int y0 = static_cast<int>(v);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    double dx = u - x0;
    double dy = v - y0;

    // 边界检查
    x0 = std::max(0, std::min(x0, image.width_ - 1));
    y0 = std::max(0, std::min(y0, image.height_ - 1));
    x1 = std::max(0, std::min(x1, image.width_ - 1));
    y1 = std::max(0, std::min(y1, image.height_ - 1));

    Eigen::Vector3d c00 = GetPixelColorDirect(image, x0, y0);
    Eigen::Vector3d c10 = GetPixelColorDirect(image, x1, y0);
    Eigen::Vector3d c01 = GetPixelColorDirect(image, x0, y1);
    Eigen::Vector3d c11 = GetPixelColorDirect(image, x1, y1);
    // 双线性插值
    Eigen::Vector3d color = (1-dx)*(1-dy)*c00 + dx*(1-dy)*c10 +
                           (1-dx)*dy*c01 + dx*dy*c11;
    return color;
}
//球面翻转
std::vector<Eigen::Vector3d> spherical_flip(const std::vector<Eigen::Vector3d>& points) {
        if (points.empty()) return points;

        //size_t N = points.size();

        // 计算最大模长
        double max_norm = 0;
        for (const auto& p : points) {
            double norm = p.norm();
            if (norm > max_norm) max_norm = norm;
        }
        double radius = max_norm * 100.0;

        std::vector<Eigen::Vector3d> flipped_points;
        for (const auto& p : points) {
            double norm = p.norm();
            if (norm < 1e-6) continue;  // 避免除零
            Eigen::Vector3d unit_vec = p / norm;
            Eigen::Vector3d flipped = p + 2.0 * (radius - norm) * unit_vec;
            flipped_points.push_back(flipped);
        }
        return flipped_points;

    }


//球面翻转 - tbb优化
std::vector<Eigen::Vector3d> spherical_flip_tbb(const std::vector<Eigen::Vector3d>& points) {
    if (points.empty()) return points;

    size_t N = points.size();

    // 计算最大模长
    double max_norm = 0;
    for (const auto& p : points) {
        double norm = p.norm();
        if (norm > max_norm) max_norm = norm;
    }
    double radius = max_norm * 100.0;

    std::vector<Eigen::Vector3d> flipped_points(N);

    // 使用TBB并行化循环
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
                const Eigen::Vector3d& p = points[i];
                double norm = p.norm();
                if (norm < 1e-6) {
                    flipped_points[i] = Eigen::Vector3d::Zero();
                    continue;
                }
                Eigen::Vector3d unit_vec = p / norm;
                Eigen::Vector3d flipped = p + 2.0 * (radius - norm) * unit_vec;
                flipped_points[i] = flipped;
            }
        });

    return flipped_points;
}


// 计算是否在OBB范围内
bool isPointInOBBManual(const Eigen::Vector3d& point,
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



// 筛选相机视野下的全部可见点
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<size_t>, std::vector<Eigen::Vector2i>>
PointsInCameraView(
    open3d::geometry::PointCloud& slam_cloud,
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,
    double max_depth)
{
    auto cloud_in_view = std::make_shared<open3d::geometry::PointCloud>();  //相机视野下的全部可见点云
    std::vector<size_t> in_view_indices;    // 记录点云在原slam点云的下标
    std::vector<Eigen::Vector2i> final_projected_uvs;     // 记录点云在当前图像的像素点
    

    int image_width = intrinsic.width_;
    int image_height = intrinsic.height_;


    // 获取内参矩阵参数
    Eigen::Matrix3d K = intrinsic.intrinsic_matrix_;
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);

    // 世界坐标系到相机坐标系的变换
    Eigen::Matrix4d world_to_camera = camera_to_world.inverse();
    Eigen::Matrix3d R = world_to_camera.block<3, 3>(0, 0);
    Eigen::Vector3d t = world_to_camera.block<3, 1>(0, 3);

    size_t N = slam_cloud.points_.size();

    //  初步筛选相机前方的点
    std::vector<Eigen::Vector3d> camera_points;     // 初步筛选的点云
    std::vector<size_t> front_indices;              // 筛选点云对应的原始点云
    std::vector<Eigen::Vector2i> projected_uvs;     // 筛选点云对应的二维像素点

    camera_points.reserve(N / 5);  // 估计25%的点在视野内
    front_indices.reserve(N / 5);
    projected_uvs.reserve(N / 5);

    auto start = std::chrono::high_resolution_clock::now();
    // 采用并行运算，因此注意输出的下标数组并非严格递增

    // 分离条件判断，减少分支预测失败
#pragma omp parallel
    {
        std::vector<Eigen::Vector3d> local_camera_points;
        std::vector<size_t> local_front_indices;
        std::vector<Eigen::Vector2i> local_projected_uvs;

        // 预分配内存，减少重新分配
        local_camera_points.reserve(N / (5 * omp_get_num_threads()));
        local_front_indices.reserve(N / (5 * omp_get_num_threads()));
        local_projected_uvs.reserve(N / (5 * omp_get_num_threads()));

#pragma omp for nowait



        for (size_t i = 0; i < N; ++i) {
            const Eigen::Vector3d& world_point = slam_cloud.points_[i];
            Eigen::Vector3d camera_point_0 = R * world_point + t;
            Eigen::Vector3d camera_point(camera_point_0.x(), -camera_point_0.y(), -camera_point_0.z());

            // 先进行快速的深度判断
            double depth = camera_point(2);
            if (depth <= 0.1 || depth >= max_depth) continue;

            // 然后进行投影计算
            double u_d = camera_point(0) * fx / depth + cx;
            double v_d = camera_point(1) * fy / depth + cy;

            // 快速整数转换
            int u = int(u_d + 0.5f);
            int v = int(v_d + 0.5f);

            // 边界检查（使用整数比较更快）
            if (u >= 0 && u < image_width && v >= 0 && v < image_height) {
                // 如果需要更精确，可以再加浮点检查
                if (u_d >= 0 && u_d < image_width && v_d >= 0 && v_d < image_height) {
                    local_camera_points.push_back(camera_point);
                    local_front_indices.push_back(i);
                    local_projected_uvs.push_back(Eigen::Vector2i(u, v));
                }
            }
        }

        // 合并线程局部结果
#pragma omp critical
        {
            camera_points.insert(camera_points.end(),
                local_camera_points.begin(), local_camera_points.end());
            front_indices.insert(front_indices.end(),
                local_front_indices.begin(), local_front_indices.end());
            projected_uvs.insert(projected_uvs.end(),
                local_projected_uvs.begin(), local_projected_uvs.end());
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    // 计算耗时
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "遍历点云初筛耗时: " << duration.count() << " 微秒" << std::endl;

    if (front_indices.empty()) {
        return std::make_tuple(cloud_in_view, in_view_indices,final_projected_uvs);
    }

    //std::cout<<"3.初步筛选成功"<<std::endl;
    std::cout << "点云初筛耗后一共有点数量: " << front_indices.size() << std::endl;
    // 4.球面翻转
    

    // 5.添加原点，计算凸包，返回可见点
    start = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::Vector3d> flipped_points = spherical_flip(camera_points);
    /*end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "球面翻转耗时 " << duration.count() << " 微秒" << std::endl;*/

    std::vector<Eigen::Vector3d> hull_points = flipped_points;
    hull_points.insert(hull_points.begin(), Eigen::Vector3d(0, 0, 0));  // 添加相机原点

    //start = std::chrono::high_resolution_clock::now();
    auto result = open3d::geometry::Qhull::ComputeConvexHull(hull_points, true);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "凸包计算耗时 " << duration.count() << " 微秒" << std::endl;

    auto& vertex_indices = get<1>(result);  // 原始点云中的索引
    //std::cout<<"4.凸包计算成功"<< std::endl;
    // 6.计算与原始数据的的对应关系
    for (size_t i = 0; i < vertex_indices.size(); ++i) {
        // 相机点云下标。因为球面翻转后在begin添加了一个相机原点，因此要回到最初下标需要减一
        int camera_idx = vertex_indices[i]-1;         
        if(camera_idx>=0){
            int original_idx = front_indices[camera_idx];   //原点云下标
            if(original_idx < slam_cloud.points_.size()){   // 这个判断其实有点冗余，但安全
                in_view_indices.push_back(original_idx);    // 添加slam点云下标
                // 添加相机视野下的点云
                cloud_in_view->points_.push_back(slam_cloud.points_[original_idx]);
                cloud_in_view->colors_.push_back(slam_cloud.colors_[original_idx]);
                Eigen::Vector2i uv =  projected_uvs[camera_idx];
                final_projected_uvs.push_back(uv);          // 添加点云对应二维下标
                // 染色，或者做点其他你想映射的事情
//                Eigen::Vector3d pixel_color = GetPixelColorBilinear(rgb, uv[0], uv[1]);
//                slam_cloud.colors_[original_idx] = pixel_color;
            }
        }
    }
    std::cout << "已经获取视野内全部可见点，数量：" << in_view_indices.size() << std::endl;
    return std::make_tuple(cloud_in_view, in_view_indices, final_projected_uvs);
}



// 筛选相机视野下的全部可见点 - 体素网格优化
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<size_t>, std::vector<Eigen::Vector2i>>
PointsInCameraView_voxel(
    open3d::geometry::PointCloud& slam_cloud,
    VoxelHashMap& voxel_hash_map,
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,
    //const open3d::geometry::Image& rgb,
    double max_depth)
{
    TicToc timer;

    auto cloud_in_view = std::make_shared<open3d::geometry::PointCloud>();  //相机视野下的全部可见点云
    std::vector<size_t> in_view_indices;    // 记录点云在原slam点云的下标
    std::vector<Eigen::Vector2i> final_projected_uvs;     // 记录点云在当前图像的像素点

    int image_width = intrinsic.width_;
    int image_height = intrinsic.height_;

    // 获取内参矩阵参数
    Eigen::Matrix3d K = intrinsic.intrinsic_matrix_;
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);


    timer.tic();
    std::vector<size_t> points_indices_in_camera = voxel_hash_map.get_points_in_cameraFrustum_DDA(intrinsic, camera_to_world);
    
    std::cout << "获取视锥内体素初筛耗时:" << timer.toc() << " 毫秒" << std::endl;
    //std::cout << "体素内点云数量:" << points_indices_in_camera.size() << std::endl;


    // 世界坐标系到相机坐标系的变换
    Eigen::Matrix4d world_to_camera = camera_to_world.inverse();
    Eigen::Matrix3d R = world_to_camera.block<3, 3>(0, 0);
    Eigen::Vector3d t = world_to_camera.block<3, 1>(0, 3);

    size_t N = slam_cloud.points_.size();

    //  初步筛选相机前方的点
    std::vector<Eigen::Vector3d> camera_points;     // 初步筛选的点云
    std::vector<size_t> front_indices;              // 筛选点云对应的原始点云
    std::vector<Eigen::Vector2i> projected_uvs;     // 筛选点云对应的二维像素点

    camera_points.reserve(N / 5);  // 估计25%的点在视野内
    front_indices.reserve(N / 5);
    projected_uvs.reserve(N / 5);

    timer.tic();
    // 采用并行运算，因此注意输出的下标数组并非严格递增

    // 分离条件判断，减少分支预测失败

#pragma omp for nowait
        for (size_t &index : points_indices_in_camera) {
            const Eigen::Vector3d& world_point = slam_cloud.points_[index];
            Eigen::Vector3d camera_point_0 = R * world_point + t;

            // y,z轴取反
            Eigen::Vector3d camera_point(camera_point_0.x(), -camera_point_0.y(), -camera_point_0.z());

            // 先进行快速的深度判断
            double depth = camera_point(2);
            if (depth <= 0.1 || depth >= max_depth) continue;

            // 然后进行投影计算
            double u_d = camera_point(0) * fx / depth + cx;
            double v_d = camera_point(1) * fy / depth + cy;

            // 快速整数转换
            int u = int(u_d + 0.5f);
            int v = int(v_d + 0.5f);

            // 边界检查（使用整数比较更快）
            if (u >= 0 && u < image_width && v >= 0 && v < image_height) {
                // 如果需要更精确，可以再加浮点检查
                if (u_d >= 0 && u_d < image_width && v_d >= 0 && v_d < image_height) {
                    camera_points.push_back(camera_point);
                    front_indices.push_back(index);
                    projected_uvs.push_back(Eigen::Vector2i(u, v));
                }
            }
        }


    
    std::cout << "遍历点云初筛耗时: " << timer.toc() << " 毫秒" << std::endl;

    if (front_indices.empty()) {
        return std::make_tuple(cloud_in_view, in_view_indices, final_projected_uvs);
    }

    //std::cout<<"3.初步筛选成功"<<std::endl;
    std::cout << "点云初筛耗后一共有点数量: " << front_indices.size() << std::endl;
    // 4.球面翻转


    // 5.添加原点，计算凸包，返回可见点
    timer.tic();
    std::vector<Eigen::Vector3d> flipped_points = spherical_flip(camera_points);
    /*end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "球面翻转耗时 " << duration.count() << " 微秒" << std::endl;*/

    std::vector<Eigen::Vector3d> hull_points = flipped_points;
    hull_points.insert(hull_points.begin(), Eigen::Vector3d(0, 0, 0));  // 添加相机原点

    //start = std::chrono::high_resolution_clock::now();
    auto result = open3d::geometry::Qhull::ComputeConvexHull(hull_points, true);
    
    std::cout << "球面翻转+凸包计算耗时 " << timer.toc() << " 毫秒" << std::endl;

    auto& vertex_indices = get<1>(result);  // 原始点云中的索引
    //std::cout<<"4.凸包计算成功"<< std::endl;
    // 6.计算与原始数据的的对应关系
    for (size_t i = 0; i < vertex_indices.size(); ++i) {
        // 相机点云下标。因为球面翻转后在begin添加了一个相机原点，因此要回到最初下标需要减一
        int camera_idx = vertex_indices[i] - 1;
        if (camera_idx >= 0) {
            int original_idx = front_indices[camera_idx];   //原点云下标
            if (original_idx < slam_cloud.points_.size()) {   // 这个判断其实有点冗余，但安全
                in_view_indices.push_back(original_idx);    // 添加slam点云下标
                // 添加相机视野下的点云
                cloud_in_view->points_.push_back(slam_cloud.points_[original_idx]);
                cloud_in_view->colors_.push_back(slam_cloud.colors_[original_idx]);
                Eigen::Vector2i uv = projected_uvs[camera_idx];
                final_projected_uvs.push_back(uv);          // 添加点云对应二维下标
                // 染色，或者做点其他你想映射的事情
//                Eigen::Vector3d pixel_color = GetPixelColorBilinear(rgb, uv[0], uv[1]);
//                slam_cloud.colors_[original_idx] = pixel_color;
            }
        }
    }
    std::cout << "已经获取视野内全部可见点，数量：" << in_view_indices.size() << std::endl;
    return std::make_tuple(cloud_in_view, in_view_indices, final_projected_uvs);
}



// 筛选相机视野下的全部可见点 - 体素网格优化 - tbb并发优化
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<size_t>, std::vector<Eigen::Vector2i>>
PointsInCameraView_voxel_tbb(
    open3d::geometry::PointCloud& slam_cloud,
    VoxelHashMap& voxel_hash_map,
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,
    //const open3d::geometry::Image& rgb,
    double max_depth)
{
    TicToc timer;

    auto cloud_in_view = std::make_shared<open3d::geometry::PointCloud>();  //相机视野下的全部可见点云
    std::vector<size_t> in_view_indices;    // 记录点云在原slam点云的下标
    std::vector<Eigen::Vector2i> final_projected_uvs;     // 记录点云在当前图像的像素点

    int image_width = intrinsic.width_;
    int image_height = intrinsic.height_;

    // 获取内参矩阵参数
    Eigen::Matrix3d K = intrinsic.intrinsic_matrix_;
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);


    timer.tic();
    std::vector<size_t> points_indices_in_camera = voxel_hash_map.get_points_in_cameraFrustum_DDA(intrinsic, camera_to_world);

    std::cout << "获取视锥内体素初筛耗时:" << timer.toc() << " 毫秒" << std::endl;
    
    // 世界坐标系到相机坐标系的变换
    Eigen::Matrix4d world_to_camera = camera_to_world.inverse();
    Eigen::Matrix3d R = world_to_camera.block<3, 3>(0, 0);
    Eigen::Vector3d t = world_to_camera.block<3, 1>(0, 3);

    size_t N = slam_cloud.points_.size();

    //  初步筛选相机前方的点
    std::vector<Eigen::Vector3d> camera_points;     // 初步筛选的点云
    std::vector<size_t> front_indices;              // 筛选点云对应的原始点云
    std::vector<Eigen::Vector2i> projected_uvs;     // 筛选点云对应的二维像素点

    // 定义一个结构体来打包三个相关数据
    struct ProjectionResult {
        Eigen::Vector3d camera_point;
        size_t index;
        Eigen::Vector2i uv;

        ProjectionResult(const Eigen::Vector3d& cp, size_t idx, const Eigen::Vector2i& uv_coord)
            : camera_point(cp), index(idx), uv(uv_coord) {}
    };

    tbb::concurrent_vector<ProjectionResult> results;
    timer.tic();
    
    tbb::parallel_for(tbb::blocked_range<size_t>(0, points_indices_in_camera.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
                size_t index = points_indices_in_camera[i];
                const Eigen::Vector3d& world_point = slam_cloud.points_[index];
                Eigen::Vector3d camera_point_0 = R * world_point + t;

                // y,z轴取反
                Eigen::Vector3d camera_point(camera_point_0.x(), -camera_point_0.y(), -camera_point_0.z());

                // 先进行快速的深度判断
                double depth = camera_point(2);
                if (depth <= 0.1 || depth >= max_depth) continue;

                // 然后进行投影计算
                double u_d = camera_point(0) * fx / depth + cx;
                double v_d = camera_point(1) * fy / depth + cy;

                // 快速整数转换
                int u = int(u_d + 0.5f);
                int v = int(v_d + 0.5f);

                // 边界检查（使用整数比较更快）
                if (u >= 1 && u < image_width-1 && v >= 1 && v < image_height-1) {
                    // 如果需要更精确，可以再加浮点检查
                    //if (u_d >= 0 && u_d < image_width && v_d >= 0 && v_d < image_height) {
                        results.emplace_back(camera_point, index, Eigen::Vector2i(u, v));
                    //}
                }
            }
        });


    std::cout << "遍历点云初筛耗时: " << timer.toc() << " 毫秒" << std::endl;

    if (results.empty()) {
        return std::make_tuple(cloud_in_view, in_view_indices, final_projected_uvs);
    }

    timer.tic();
    // 最后将数据分配到三个vector中
    camera_points.reserve(results.size());
    front_indices.reserve(results.size());
    projected_uvs.reserve(results.size());

    for (const auto& result : results) {
        camera_points.emplace_back(result.camera_point);
        front_indices.emplace_back(result.index);
        projected_uvs.emplace_back(result.uv);
    }
    std::cout << "加载遍历结果耗时: " << timer.toc() << " 毫秒" << std::endl;

    if (front_indices.empty()) {
        return std::make_tuple(cloud_in_view, in_view_indices, final_projected_uvs);
    }

    //std::cout<<"3.初步筛选成功"<<std::endl;
    std::cout << "点云初筛耗后一共有点数量: " << front_indices.size() << std::endl;
    // 4.球面翻转


    // 5.添加原点，计算凸包，返回可见点
    /*timer.tic();
    std::vector<Eigen::Vector3d> flipped_points = spherical_flip(camera_points);
    std::cout << "初始球面翻转 " << timer.toc() << " 毫秒" << std::endl;*/

    timer.tic();
    std::vector<Eigen::Vector3d> flipped_points = spherical_flip_tbb(camera_points);
    std::cout << "tbb球面翻转 " << timer.toc() << " 毫秒" << std::endl;

    timer.tic();
    std::vector<Eigen::Vector3d> hull_points = flipped_points;
    //hull_points.insert(hull_points.begin(), Eigen::Vector3d(0, 0, 0));  // 添加相机原点
    hull_points.emplace_back(Eigen::Vector3d(0, 0, 0));  // 添加相机原点
    std::cout << "添加相机原点 " << timer.toc() << " 毫秒" << std::endl;

    timer.tic();
    auto result = open3d::geometry::Qhull::ComputeConvexHull(hull_points, true);
    std::cout << "凸包计算 " << timer.toc() << " 毫秒" << std::endl;

    auto& vertex_indices = get<1>(result);  // 原始点云中的索引
    //std::cout<<"4.凸包计算成功"<< std::endl;
    // 6.计算与原始数据的的对应关系
    size_t n_res = vertex_indices.size() - 1;
    cloud_in_view->points_.reserve(n_res);
    cloud_in_view->colors_.reserve(n_res);
    in_view_indices.reserve(n_res);
    final_projected_uvs.reserve(n_res);

    //for (size_t i = 0; i < vertex_indices.size(); ++i) {
    //    // 相机点云下标。因为球面翻转后在begin添加了一个相机原点，因此要回到最初下标需要减一
    //    int camera_idx = vertex_indices[i] - 1;
    //    if (camera_idx >= 0) {
    //        int original_idx = front_indices[camera_idx];   //原点云下标
    //        if (original_idx < slam_cloud.points_.size()) {   // 这个判断其实有点冗余，但安全
    //            in_view_indices.emplace_back(original_idx);    // 添加slam点云下标
    //            // 添加相机视野下的点云
    //            cloud_in_view->points_.emplace_back(slam_cloud.points_[original_idx]);
    //            cloud_in_view->colors_.emplace_back(slam_cloud.colors_[original_idx]);
    //            Eigen::Vector2i uv = projected_uvs[camera_idx];
    //            final_projected_uvs.emplace_back(uv);          // 添加点云对应二维下标
    //        }
    //    }
    //}
    for (size_t i = 0; i < vertex_indices.size(); ++i) {
        // 相机点云下标。因为球面翻转后在begin添加了一个相机原点，因此要回到最初下标需要减一
        int camera_idx = vertex_indices[i];
        if (camera_idx >= front_indices.size())continue;
        int original_idx = front_indices[camera_idx];   //原点云下标
               
        in_view_indices.emplace_back(original_idx);    // 添加slam点云下标
                    // 添加相机视野下的点云
        cloud_in_view->points_.emplace_back(slam_cloud.points_[original_idx]);
        cloud_in_view->colors_.emplace_back(slam_cloud.colors_[original_idx]);
        Eigen::Vector2i uv = projected_uvs[camera_idx];
        final_projected_uvs.emplace_back(uv);          // 添加点云对应二维下标
        
    }

    //voxel_hash_map.count_voxel_points(cloud_in_view->points_);
    std::cout << "已经获取视野内全部可见点，数量：" << in_view_indices.size() << std::endl;
    return std::make_tuple(cloud_in_view, in_view_indices, final_projected_uvs);
}



// 加上深度容错率-筛选相机视野下的全部可见点 - 体素网格优化
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<size_t>, std::vector<Eigen::Vector2i>>
PointsInCameraView_voxel_zbuffer(
    open3d::geometry::PointCloud& slam_cloud,
    VoxelHashMap& voxel_hash_map,
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,
    double depth_threshold,  // 深度一致性阈值
    double max_depth)
{
    auto cloud_in_view = std::make_shared<open3d::geometry::PointCloud>();  //相机视野下的全部可见点云
    std::vector<size_t> in_view_indices;    // 记录点云在原slam点云的下标
    std::vector<Eigen::Vector2i> final_projected_uvs;     // 记录点云在当前图像的像素点


    int image_width = intrinsic.width_;
    int image_height = intrinsic.height_;


    // 获取内参矩阵参数
    Eigen::Matrix3d K = intrinsic.intrinsic_matrix_;
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);


    TicToc timer;
    timer.tic();

    std::vector<size_t> points_indices_in_camera = voxel_hash_map.get_points_in_cameraFrustum_DDA(intrinsic, camera_to_world);
    
    std::cout << "获取视锥内体素初筛耗时:" << timer.toc() << " 毫秒" << std::endl;
    //std::cout << "体素内点云数量:" << points_indices_in_camera.size() << std::endl;


    // 世界坐标系到相机坐标系的变换
    Eigen::Matrix4d world_to_camera = camera_to_world.inverse();
    Eigen::Matrix3d R = world_to_camera.block<3, 3>(0, 0);
    Eigen::Vector3d t = world_to_camera.block<3, 1>(0, 3);

    size_t N = slam_cloud.points_.size();

    //  初步筛选相机前方的点
    std::vector<Eigen::Vector3d> camera_points;     // 初步筛选的点云
    std::vector<size_t> front_indices;              // 筛选点云对应的原始点云
    std::vector<Eigen::Vector2i> projected_uvs;     // 筛选点云对应的二维像素点

    camera_points.reserve(N / 5);  // 估计25%的点在视野内
    front_indices.reserve(N / 5);
    projected_uvs.reserve(N / 5);

    timer.tic();
    // 采用并行运算，因此注意输出的下标数组并非严格递增

    // 分离条件判断，减少分支预测失败

#pragma omp for nowait
    for (size_t& index : points_indices_in_camera) {
        const Eigen::Vector3d& world_point = slam_cloud.points_[index];
        Eigen::Vector3d camera_point_0 = R * world_point + t;

        // y,z轴取反
        Eigen::Vector3d camera_point(camera_point_0.x(), -camera_point_0.y(), -camera_point_0.z());

        // 先进行快速的深度判断
        double depth = camera_point(2);
        if (depth <= 0.1 || depth >= max_depth) continue;

        // 然后进行投影计算
        double u_d = camera_point(0) * fx / depth + cx;
        double v_d = camera_point(1) * fy / depth + cy;

        // 快速整数转换
        int u = int(u_d + 0.5f);
        int v = int(v_d + 0.5f);

        // 边界检查（使用整数比较更快）
        if (u >= 1 && u < image_width-1 && v >= 1 && v < image_height-1) {
            // 如果需要更精确，可以再加浮点检查
            //if (u_d >= 0 && u_d < image_width && v_d >= 0 && v_d < image_height) {
                camera_points.push_back(camera_point);
                front_indices.push_back(index);
                projected_uvs.push_back(Eigen::Vector2i(u, v));
           // }
        }
    }


    std::cout << "遍历点云初筛耗时: " << timer.toc() << " 毫秒" << std::endl;

    if (front_indices.empty()) {
        return std::make_tuple(cloud_in_view, in_view_indices, final_projected_uvs);
    }

    //std::cout<<"3.初步筛选成功"<<std::endl;
    std::cout << "点云初筛耗后一共有点数量: " << front_indices.size() << std::endl;
    // 4.球面翻转


    // 5.添加原点，计算凸包，返回可见点
    timer.tic();
    std::vector<Eigen::Vector3d> flipped_points = spherical_flip(camera_points);
   
    std::vector<Eigen::Vector3d> hull_points = flipped_points;
    hull_points.insert(hull_points.begin(), Eigen::Vector3d(0, 0, 0));  // 添加相机原点
 
    auto result = open3d::geometry::Qhull::ComputeConvexHull(hull_points, true);
    
    std::cout << "球面翻转+凸包计算耗时 " << timer.toc() << " 毫秒" << std::endl;

    auto& vertex_indices = get<1>(result);  // 原始点云中的索引
    //std::cout<<"4.凸包计算成功"<< std::endl;
    // 
    std::vector<std::vector<double>> min_depths(image_width, std::vector<double>(image_height, max_depth));

    timer.tic();
    // 找到每个像素点做映射点云的最近点
    for (size_t i = 0; i < vertex_indices.size(); ++i) {
        int camera_idx = vertex_indices[i] - 1;
        if (camera_idx < 0)continue;
        double depth = camera_points[camera_idx](2);
        Eigen::Vector2i uv = projected_uvs[camera_idx];

        int u = uv[0];
        int v = uv[1];
        if (depth < min_depths[u][v]) {
            min_depths[u][v] = depth;
        }
    }
    std::cout << "寻找每个像素最近点耗时： " << timer.toc() << " 毫秒" << std::endl;
    timer.tic();
    // 再次遍历视锥内的点做深度判断
    for (size_t i = 0; i < front_indices.size(); i++) {
        Eigen::Vector2i uv = projected_uvs[i];
        int& u = uv[0];
        int& v = uv[1];
        if (min_depths[u][v] == max_depth)continue;
        if (std::abs(camera_points[i](2) - min_depths[u][v]) > depth_threshold) {
            continue;
        }

        size_t original_idx = front_indices[i];
        in_view_indices.push_back(original_idx);    // 添加slam点云下标
        cloud_in_view->points_.push_back(slam_cloud.points_[original_idx]);// 添加相机视野下的点云
        cloud_in_view->colors_.push_back(slam_cloud.colors_[original_idx]);
        final_projected_uvs.push_back(uv);          // 添加点云对应二维下标
   
    }
    std::cout << "深度容错检测耗时：： " << timer.toc() << " 毫秒" << std::endl;

    std::cout << "已经获取视野内全部可见点，数量：" << in_view_indices.size() << std::endl;
    return std::make_tuple(cloud_in_view, in_view_indices, final_projected_uvs);
}

// tbb并发优化-加上深度容错率-筛选相机视野下的全部可见点 - 体素网格优化
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<size_t>, std::vector<Eigen::Vector2i>>
PointsInCameraView_voxel_zbuffer_tbb(
    open3d::geometry::PointCloud& slam_cloud,
    VoxelHashMap& voxel_hash_map,
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,
    double depth_threshold,  // 深度一致性阈值
    double max_depth)
{
    auto cloud_in_view = std::make_shared<open3d::geometry::PointCloud>();  //相机视野下的全部可见点云
    std::vector<size_t> in_view_indices;    // 记录点云在原slam点云的下标
    std::vector<Eigen::Vector2i> final_projected_uvs;     // 记录点云在当前图像的像素点


    int image_width = intrinsic.width_;
    int image_height = intrinsic.height_;


    // 获取内参矩阵参数
    Eigen::Matrix3d K = intrinsic.intrinsic_matrix_;
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);


    TicToc timer;
    timer.tic();

    std::vector<size_t> points_indices_in_camera = voxel_hash_map.get_points_in_cameraFrustum_DDA(intrinsic, camera_to_world);

    std::cout << "获取视锥内体素初筛耗时:" << timer.toc() << " 毫秒" << std::endl;
    //std::cout << "体素内点云数量:" << points_indices_in_camera.size() << std::endl;


    // 世界坐标系到相机坐标系的变换
    Eigen::Matrix4d world_to_camera = camera_to_world.inverse();
    Eigen::Matrix3d R = world_to_camera.block<3, 3>(0, 0);
    Eigen::Vector3d t = world_to_camera.block<3, 1>(0, 3);

    size_t N = slam_cloud.points_.size();

    //  初步筛选相机前方的点
    std::vector<Eigen::Vector3d> camera_points;     // 初步筛选的点云
    std::vector<size_t> front_indices;              // 筛选点云对应的原始点云
    std::vector<Eigen::Vector2i> projected_uvs;     // 筛选点云对应的二维像素点

    //camera_points.reserve(N / 5);  // 估计25%的点在视野内
    //front_indices.reserve(N / 5);
    //projected_uvs.reserve(N / 5);


    // 定义一个结构体来打包三个相关数据
    struct ProjectionResult {
        Eigen::Vector3d camera_point;
        size_t index;
        Eigen::Vector2i uv;

        ProjectionResult(const Eigen::Vector3d& cp, size_t idx, const Eigen::Vector2i& uv_coord)
            : camera_point(cp), index(idx), uv(uv_coord) {}
    };

    tbb::concurrent_vector<ProjectionResult> results;

    timer.tic();
    // 采用并行运算，因此注意输出的下标数组并非严格递增

    // 分离条件判断，减少分支预测失败

   
    tbb::parallel_for(tbb::blocked_range<size_t>(0, points_indices_in_camera.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
                size_t index = points_indices_in_camera[i];
                const Eigen::Vector3d& world_point = slam_cloud.points_[index];
                Eigen::Vector3d camera_point_0 = R * world_point + t;

                // y,z轴取反
                Eigen::Vector3d camera_point(camera_point_0.x(), -camera_point_0.y(), -camera_point_0.z());

                // 先进行快速的深度判断
                double depth = camera_point(2);
                if (depth <= 0.1 || depth >= max_depth) continue;

                // 然后进行投影计算
                double u_d = camera_point(0) * fx / depth + cx;
                double v_d = camera_point(1) * fy / depth + cy;

                // 快速整数转换
                int u = int(u_d + 0.5f);
                int v = int(v_d + 0.5f);

                // 边界检查（使用整数比较更快）
                if (u >= 1 && u < image_width-1 && v >= 1 && v < image_height-1) {
                    // 如果需要更精确，可以再加浮点检查
                    //if (u_d >= 0 && u_d < image_width && v_d >= 0 && v_d < image_height) {
                        results.emplace_back(camera_point, index, Eigen::Vector2i(u, v));
                    //}
                }
            }
        });


    std::cout << "遍历点云初筛耗时: " << timer.toc() << " 毫秒" << std::endl;
   
    if (results.empty()) {
        return std::make_tuple(cloud_in_view, in_view_indices, final_projected_uvs);
    }

    timer.tic();
    // 最后将数据分配到三个vector中
    camera_points.reserve(results.size());
    front_indices.reserve(results.size());
    projected_uvs.reserve(results.size());

    for (const auto& result : results) {
        camera_points.emplace_back(result.camera_point);
        front_indices.emplace_back(result.index);
        projected_uvs.emplace_back(result.uv);
    }
    std::cout << "加载遍历结果耗时: " << timer.toc() << " 毫秒" << std::endl;


    //std::cout<<"3.初步筛选成功"<<std::endl;
    std::cout << "点云初筛耗后一共有点数量: " << results.size() << std::endl;
    // 4.球面翻转


    // 5.添加原点，计算凸包，返回可见点
    timer.tic();
    //std::vector<Eigen::Vector3d> flipped_points = spherical_flip(camera_points);
    std::vector<Eigen::Vector3d> flipped_points = spherical_flip_tbb(camera_points);
    std::cout << "球面翻转: " << timer.toc() << " 毫秒" << std::endl;

    std::vector<Eigen::Vector3d> hull_points = flipped_points;
    hull_points.insert(hull_points.begin(), Eigen::Vector3d(0, 0, 0));  // 添加相机原点

    timer.tic();
    auto result = open3d::geometry::Qhull::ComputeConvexHull(hull_points, true);
    std::cout << "凸包计算耗时 " << timer.toc() << " 毫秒" << std::endl;

    auto& vertex_indices = get<1>(result);  // 原始点云中的索引
    //std::cout<<"4.凸包计算成功"<< std::endl;
    // 
    std::vector<std::vector<double>> min_depths(image_width, std::vector<double>(image_height, max_depth));

    timer.tic();
    // 找到每个像素点做映射点云的最近点
    for (size_t i = 0; i < vertex_indices.size(); ++i) {

        int camera_idx = vertex_indices[i] - 1;
        if (camera_idx < 0)continue;
        double depth = camera_points[camera_idx](2);

        Eigen::Vector2i uv = projected_uvs[camera_idx];
        int u = uv[0];
        int v = uv[1];
        if (depth < min_depths[u][v]) {
            min_depths[u][v] = depth;
        }
    }
    std::cout << "寻找每个像素最近点耗时： " << timer.toc() << " 毫秒" << std::endl;

    timer.tic();
    // 再次遍历视锥内的点做深度判断
    for (size_t i = 0; i < front_indices.size(); i++) {

        Eigen::Vector2i uv = projected_uvs[i];
        int& u = uv[0];
        int& v = uv[1];

        if (min_depths[u][v] == max_depth)continue;
        if (std::abs(camera_points[i](2) - min_depths[u][v]) > depth_threshold) {
            continue;
        }
        size_t original_idx = front_indices[i];

        in_view_indices.push_back(original_idx);    // 添加slam点云下标
        cloud_in_view->points_.push_back(slam_cloud.points_[original_idx]);// 添加相机视野下的点云
        cloud_in_view->colors_.push_back(slam_cloud.colors_[original_idx]);
        final_projected_uvs.push_back(uv);          // 添加点云对应二维下标

    }
    std::cout << "深度容错检测耗时：： " << timer.toc() << " 毫秒" << std::endl;

    std::cout << "已经获取视野内全部可见点，数量：" << in_view_indices.size() << std::endl;
    return std::make_tuple(cloud_in_view, in_view_indices, final_projected_uvs);
}




// 根据每个二维检测物体的像素点再筛选该物体的点云
std::set<size_t> GetInstanceIndices(std::vector<size_t>& in_view_indices,
                                  std::vector<Eigen::Vector2i>& projected_uvs,
                                  const cv::Mat& mask
                                  //std::string name
//                                  open3d::geometry::PointCloud& slam_cloud,
//                                  const open3d::geometry::Image& rgb
)
{
    std::set<size_t> result;
    //visualize_mask_image(mask , name);
    assert (in_view_indices.size() == projected_uvs.size() ), "in_view_indices and projected_uvs have different size";

    for(size_t i=0;i<projected_uvs.size();i++){
        Eigen::Vector2i uv =  projected_uvs[i];
        int u = uv[0];
        int v = uv[1];
        if(mask.at<uint8_t>(v,u)>0){
            result.insert(in_view_indices[i]);
            // 染色查看正确性
//            Eigen::Vector3d pixel_color = GetPixelColorBilinear(rgb, uv[0], uv[1]);
//            slam_cloud.colors_[in_view_indices[i]] = pixel_color;
        }
    }

    return result;
}

// 根据每个二维检测物体的像素点再筛选该物体的点云 - tbb并发优化
std::set<size_t> GetInstanceIndices_tbb(std::vector<size_t>& in_view_indices,
    std::vector<Eigen::Vector2i>& projected_uvs,
    const cv::Mat& mask
)
{
    assert(in_view_indices.size() == projected_uvs.size() &&
        "in_view_indices and projected_uvs have different size");

    // 使用线程安全的concurrent_vector
    tbb::concurrent_vector<size_t> concurrent_result;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, projected_uvs.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                const Eigen::Vector2i& uv = projected_uvs[i];
                const int& u = uv[0];
                const int& v = uv[1];
                //if (u >= 0 && v >= 0 && u < mask.cols && v < mask.rows) {
                    if (mask.at<uint8_t>(v, u) > 0) {
                        concurrent_result.emplace_back(in_view_indices[i]);
                    }
                //}
            }
        }
    );

    // 转换为set去重排序
    return std::set<size_t>(concurrent_result.begin(), concurrent_result.end());
}


// 暂时无法用
// 根据每个二维检测物体的像素点再筛选该物体的点云 - 尝试一次性获取所有实例的点云，但由于在之前读取图像时会筛掉一些idx，所有图像中最大掩码值和 n_det 不一定相同
std::vector<std::set<size_t>> GetInstanceIndices_all(std::vector<size_t>& in_view_indices,
    std::vector<Eigen::Vector2i>& projected_uvs,
    const cv::Mat& detection_map,              // 全图掩码
    const int& n_k
)
{
    std::vector<std::set<size_t>> result(n_k);

    assert(in_view_indices.size() == projected_uvs.size()), "in_view_indices and projected_uvs have different size";

    for (size_t i = 0; i < projected_uvs.size(); i++) {
        Eigen::Vector2i uv = projected_uvs[i];
        int u = uv[0];
        int v = uv[1];
        int idx = detection_map.at<uint8_t>(v, u);
        if (!idx)continue;
        if (idx > n_k) {
            std::cout << "掩码值超出" << std::endl;
            continue;
        }
        result[idx-1].insert(in_view_indices[i]);
    }
    return result;
}







// 根据已有的 相机可见点、对应像素点、该instance在相机中的可见点
std::shared_ptr<cv::Mat> prjection_cloud_to_mask(const std::vector<size_t> &in_view_indices,
                                                 const std::set<size_t> &observed_instance_points,
                                                 const std::vector<Eigen::Vector2i> &projected_uvs,
                                                 const open3d::camera::PinholeCameraIntrinsic &intrinsic
)
{
    auto mask = std::make_shared<cv::Mat>(cv::Mat::zeros(intrinsic.height_, intrinsic.width_, CV_8UC1));

    // 转为有序的set
    std::set<size_t> in_view_indices_set(in_view_indices.begin(), in_view_indices.end());

    

    assert(in_view_indices.size() == projected_uvs.size()), "in_view_indices and projected_uvs 有不同size";

    for (size_t i = 0; i < in_view_indices.size(); i++) {
        if (observed_instance_points.find(in_view_indices[i]) != observed_instance_points.end()) {
            Eigen::Vector2i uv = projected_uvs[i];
            if(uv[1] < mask->rows && uv[0] < mask->cols)
                mask->at<uint8_t>(uv[1], uv[0]) = 1;
        }
    }


    auto depth_out = std::make_shared<cv::Mat>(cv::Mat::zeros(intrinsic.height_, intrinsic.width_, CV_8UC1));
    int dilation_size = 5;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    cv::dilate(*mask, *depth_out, element);

    return depth_out;

    //return mask;

}






//////////////////////////////////////////////////////////////////////////

// 将掩码可视化
void visualize_mask_image(const cv::Mat &mask , std::string first_name){

    int width = mask.cols;
    int height = mask.rows;
    // 可视化mask
    open3d::geometry::Image masked_vis;
    masked_vis.Prepare(width, height, 3, 1);
    uint8_t* data_ptr = masked_vis.data_.data();


    size_t mask_count = 0;

    // 双层循环遍历深度图像的每个像素
    for(int v=0; v<height;v++){
        for(int u=0; u<width;u++){
            // 检查当前像素在掩码中是否有效（掩码值>0）
            if(mask.at<uint8_t>(v,u)>0){

                mask_count++;
                int idx = (v * width + u) * 3;
                // 设置RGB颜色
                data_ptr[idx] = static_cast<uint8_t>(u % 256);      // R: 基于x位置
                data_ptr[idx + 1] = static_cast<uint8_t>(v % 256);  // G: 基于y位置
                data_ptr[idx + 2] = 100;                          // B: 固定值
            }
        }
    }

    string name = first_name + ".png";
    std::cout << "保存了图像: " << name << std::endl;
    std::cout << "该mask共有像素点: " << mask_count << std::endl;
    open3d::io::WriteImage(name, masked_vis);

}

std::vector<std::string> split_str(const std::string s, const std::string delim) 
{
    std::vector<std::string> list;
    auto start = 0U;
    auto end = s.find(delim);
    while (true) {
        list.push_back(s.substr(start, end - start));
        if (end == std::string::npos)
            break;
        start = end + delim.length();
        end = s.find(delim, start);
    }
    return list;
}

bool int_to_bool(int flag)
{
    if(flag>0) return true;
    else return false;
}

// 读取config
fmfusion::Config *create_scene_graph_config(const std::string &config_file, bool verbose)
{
    fmfusion::Config *config = new fmfusion::Config();
    // cancer
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        open3d::utility::LogWarning("Failed to open config file: {}", config_file);
        return nullptr;
    }
    else{
        std::string dataset_name = fs["dataset"];
        if(dataset_name.find("fusion_portable")!=string::npos)
            config->dataset = fmfusion::Config::DATASET_TYPE::FUSION_PORTABLE;
        else if(dataset_name.find("realsense")!=string::npos)
            config->dataset = fmfusion::Config::DATASET_TYPE::REALSENSE;
        else if(dataset_name.find("scannet")!=string::npos)
            config->dataset = fmfusion::Config::DATASET_TYPE::SCANNET;
        else if(dataset_name.find("matterport")!=string::npos)
            config->dataset = fmfusion::Config::DATASET_TYPE::MATTERPORT;
        else if (dataset_name.find("rio")!=string::npos)
            config->dataset = fmfusion::Config::DATASET_TYPE::RIO;
        else
            open3d::utility::LogWarning("Unknown dataset type: {}", dataset_name);
        
        int img_width = fs["image_width"];
        int img_height = fs["image_height"];
        double fx = fs["camera_fx"];
        double fy = fs["camera_fy"];
        double cx = fs["camera_cx"];
        double cy = fs["camera_cy"];

        // Mapping
        auto mapping_fs = fs["Mapping"];
        config->instance_cfg.voxel_length = mapping_fs["voxel_length"];
        config->instance_cfg.sdf_trunc = mapping_fs["sdf_trunc"];
        config->instance_cfg.min_voxel_weight = mapping_fs["min_voxel_weight"];
        config->instance_cfg.intrinsic.SetIntrinsics(img_width,img_height,fx,fy,cx,cy);

        config->mapping_cfg.depth_scale = mapping_fs["depth_scale"];
        config->mapping_cfg.depth_max = mapping_fs["depth_max"];
        config->mapping_cfg.min_active_points = mapping_fs["min_active_points"];

        config->mapping_cfg.min_det_masks = mapping_fs["min_det_masks"];
        config->mapping_cfg.max_box_area_ratio = mapping_fs["max_box_area_ratio"];
        config->mapping_cfg.query_depth_vx_size = mapping_fs["query_depth_vx_size"];
        config->mapping_cfg.dilation_size = mapping_fs["dilate_kernal"];
        config->mapping_cfg.min_iou = mapping_fs["min_iou"];
        config->mapping_cfg.search_radius = mapping_fs["search_radius"];

        config->mapping_cfg.shape_min_points = mapping_fs["shape_min_points"];
        mapping_fs["bayesian_semantic_likelihood"] >> config->mapping_cfg.bayesian_semantic_likelihood;
        if(config->mapping_cfg.bayesian_semantic_likelihood.size()>0){
            config->mapping_cfg.bayesian_semantic = true;
        }

        config->mapping_cfg.merge_iou = mapping_fs["merge_iou"];
        config->mapping_cfg.merge_inflation = mapping_fs["merge_inflation"];
        config->mapping_cfg.realtime_merge_floor = int_to_bool(mapping_fs["realtime_merge_floor"]);
        config->mapping_cfg.update_period = mapping_fs["update_period"];
        config->mapping_cfg.recent_window_size = mapping_fs["recent_window_size"];

        mapping_fs["save_da_dir"] >> config->mapping_cfg.save_da_dir;

        // Graph config
        auto graph_config_fs = fs["Graph"];
        config->graph.edge_radius_ratio = graph_config_fs["edge_radius_ratio"];
        config->graph.voxel_size = graph_config_fs["voxel_size"];
        config->graph.involve_floor_edge = int_to_bool(graph_config_fs["involve_floor_edge"]);
        graph_config_fs["ignore_labels"]>>config->graph.ignore_labels;

        //
        auto shape_fs = fs["ShapeEncoder"];
        config->shape_encoder.init_voxel_size = shape_fs["init_voxel_size"];
        config->shape_encoder.init_radius = shape_fs["init_radius"];
        config->shape_encoder.K_shape_samples = shape_fs["K_shape_samples"];
        config->shape_encoder.K_match_samples = shape_fs["K_match_samples"];
        shape_fs["padding"] >> config->shape_encoder.padding;

        //
        auto sgnet_config_fs = fs["SGNet"];
        config->sgnet.triplet_number = sgnet_config_fs["triplet_number"];
        config->sgnet.warm_up_iter = sgnet_config_fs["warm_up"];
        config->sgnet.instance_match_threshold = sgnet_config_fs["instance_match_threshold"];

        //
        auto lcd_fs = fs["LoopDetector"];
        config->loop_detector.fuse_shape = int_to_bool(lcd_fs["fuse_shape"]);
        config->loop_detector.lcd_nodes = lcd_fs["lcd_nodes"];
        config->loop_detector.recall_nodes = lcd_fs["recall_nodes"];

        //
        auto reg_fs = fs["Registration"];
        float noise_bound = reg_fs["noise_bound"];
        config->reg.noise_bound_vec = {noise_bound};

        // Close and print
        fs.release();
        if (verbose){
            auto message = config_to_message(*config);
            cout << message << endl;
        }        
        return config;
    }
}

std::string config_to_message(const fmfusion::Config &config)
{
    std::stringstream message;
    switch (config.dataset)
    {
    case fmfusion::Config::DATASET_TYPE::FUSION_PORTABLE:
        message << "dataset: fusion_portable\n";
        break;
    case fmfusion::Config::DATASET_TYPE::REALSENSE:
        message << "dataset: realsense\n";
        break;
    case fmfusion::Config::DATASET_TYPE::SCANNET:
        message << "dataset: scannet\n";
        break;
    default:
        break;
    }
    message <<"Instance: \n"<<config.instance_cfg.print_msg();
    message <<"Mapping: \n"<<config.mapping_cfg.print_msg();
    message <<"Graph: \n"<<config.graph.print_msg();
    message <<"SGNet: \n"<<config.sgnet.print_msg();
    message <<"Shape encoder: \n" <<config.shape_encoder.print_msg();
    message <<"Loop detector: \n" <<config.loop_detector.print_msg();
    message <<"Registration: \n"<<config.reg.print_msg();

    return message.str();
}

//cancer
//���ص�ǰ֡�������������浽detections��detectionsֻ�ǵ�֡�ļ����

bool LoadPredictions(const std::string &folder_path, const std::string &frame_name,
                    const MappingConfig &mapping_cfg, const int &img_width, const int &img_height,
                    std::vector<DetectionPtr> &detections)
{
    const int MAX_BOX_AREA = mapping_cfg.max_box_area_ratio * (img_width * img_height);

    auto detection_fs = std::make_shared<fmfusion::DetectionFile>(mapping_cfg.min_det_masks, MAX_BOX_AREA);
    std::string json_file_dir = folder_path + "/" + frame_name + "_label.json";
    std::string instance_file_dir = folder_path + "/" + frame_name + "_mask.png";

    if(open3d::io::ReadIJsonConvertible(json_file_dir, *detection_fs)){
        bool read_mask = detection_fs->updateInstanceMap(instance_file_dir);
        if(read_mask){            
            detections = detection_fs->detections;
            // cout<<"Load "<<detections.size()<<" detections correct"<<endl;
            return true;
        }
        else return false;
    }
    else return false;
}


// Render-��Ⱦ
std::shared_ptr<cv::Mat> RenderDetections(const std::shared_ptr<cv::Mat> &rgb_img,
                                        const std::vector<DetectionPtr> &detections, 
                                        const std::unordered_map<InstanceId,CvMatPtr> &instances_mask,
                                        const Eigen::VectorXi &matches, 
                                        const std::unordered_map<InstanceId,Eigen::Vector3d> &instance_colors)
{
    auto detection_img = std::make_shared<cv::Mat>(rgb_img->clone());
    auto detection_mask = std::make_shared<cv::Mat>(cv::Mat::zeros(rgb_img->rows, rgb_img->cols, CV_8UC3));
    auto instance_img = std::make_shared<cv::Mat>(cv::Mat::zeros(rgb_img->rows, rgb_img->cols, CV_8UC3));
    if(detections.size()<1) return detection_img;

    int k=0;
    for(auto detection:detections){
        cv::Scalar box_color;
        if(matches(k)>0) box_color = cv::Scalar(0,255,0);
        else if(matches(k)<0) box_color = cv::Scalar(0,0,255);  // invalid
        else box_color = cv::Scalar(255,0,0); // create new

        cv::rectangle(*detection_img, 
                    cv::Point(detection->bbox_.u0,detection->bbox_.v0),
                    cv::Point(detection->bbox_.u1,detection->bbox_.v1), 
                    box_color, 
                    1);
        std::string label_score_str = detection->extract_label_string();
        cv::putText(*detection_img, 
                    label_score_str, 
                    cv::Point(detection->bbox_.u0+1,detection->bbox_.v0+10), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    cv::Scalar(255,255,255), 
                    1);
        k++;

        // detection-wise color
        cv::Scalar det_color = cv::Scalar(rand()%255,rand()%255,rand()%255);
        detection_mask->setTo(det_color, detection->instances_idxs_);
    }

    for (auto &instance:instances_mask){
        cv::Scalar inst_color_cv;
        if(instance_colors.empty())
            inst_color_cv = cv::Scalar(rand()%255,rand()%255,rand()%255);
        else{
            const Eigen::Vector3d inst_color_vec = 255 * instance_colors.at(instance.first);
            inst_color_cv = cv::Scalar(inst_color_vec[0],inst_color_vec[1],inst_color_vec[2]);
        }

        instance_img->setTo(inst_color_cv, *instance.second);
    }

    // concatenate images
    cv::addWeighted(*detection_img, 1.0, *detection_mask, 0.5, 0.0, *detection_img);
    auto out_img = std::make_shared<cv::Mat>(cv::Mat::zeros(rgb_img->rows, rgb_img->cols*2, CV_8UC3));
    cv::hconcat(*detection_img, *instance_img, *out_img);

    // draw matches
    int K = matches.size();
    cv::Scalar line_color = cv::Scalar(255,255,0);
    for(int k_=0;k_<K;k_++){
        if(matches(k_)>0){
            cv::Point detection_centroid = detections[k_]->get_box_center();
            auto matched_instance = instances_mask.find(matches(k_)); // [H,W], CV_8UC1
            cv::Mat instance_uvs;
            cv::findNonZero(*matched_instance->second, instance_uvs);
            cv::Point instance_centroid = cv::Point(cv::mean(instance_uvs)[0],cv::mean(instance_uvs)[1]);
            cv::line(*out_img, detection_centroid, cv::Point(instance_centroid.x+rgb_img->cols,instance_centroid.y), line_color, 1);
        }
    }


    return out_img;
}


std::shared_ptr<cv::Mat> PrjectionCloudToDepth(const open3d::geometry::PointCloud& cloud, 
    const Eigen::Matrix4d &pose_inverse,const open3d::camera::PinholeCameraIntrinsic& intrinsic, int dilation_size)
{
    auto depth = std::make_shared<cv::Mat>(cv::Mat::zeros(intrinsic.height_, intrinsic.width_, CV_8UC1));
    if(!cloud.HasPoints()) return depth;
    open3d::geometry::PointCloud points_camera(cloud.points_);
    points_camera.Transform(pose_inverse);
    // std::cout << "points_camera.points_.size(): " << points_camera.points_.size() << std::endl;

    int count = 0;
    for (const Eigen::Vector3d &point: points_camera.points_){
        Eigen::Vector3d point_normalized = point / point[2];
        Eigen::Vector3d uv_homograph = intrinsic.intrinsic_matrix_ * point_normalized;
        int u_ = round(uv_homograph[0]);
        int v_ = round(uv_homograph[1]);
        if(u_ >= 0 && u_ < intrinsic.width_ && v_ >= 0 && v_ < intrinsic.height_){
            depth->at<uint8_t>(v_,u_) = round(point[2] * 1000.0);
            count ++;
        }
    }
    // std::cout << "projected points: " << count << std::endl;
    // Expand depth by dilation
    auto depth_out = std::make_shared<cv::Mat>(cv::Mat::zeros(intrinsic.height_, intrinsic.width_, CV_8UC1));
    // int dilation_size = 5;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    cv::dilate(*depth, *depth_out, element);

    return depth_out;
}






// cancer
// ����mask.pngɸѡ��ǰinstance����Ч���mat
bool create_masked_rgbd(const open3d::geometry::Image &rgb, 
                        const open3d::geometry::Image &float_depth, 
                        const cv::Mat &mask,
                        const int &min_points,
                        std::shared_ptr<open3d::geometry::RGBDImage> &masked_rgbd)
{
    open3d::geometry::Image masked_depth;
    assert (float_depth.width_ == mask.cols && float_depth.height_ == mask.rows), "depth and mask have different size";
    assert (float_depth.num_of_channels_==1), "depth has more than one channel";
    assert (float_depth.bytes_per_channel_==4), "depth is not in float";
    masked_rgbd->color_ = rgb;

    float CLIP_RATIO = 0.1;
    masked_depth.Prepare(float_depth.width_, float_depth.height_, 1, 4);
    std::vector<float> valid_depth_array;

    for(int v=0; v<float_depth.height_;v++){
        for(int u=0; u<float_depth.width_;u++){
            if(mask.at<uint8_t>(v,u)>0){
                *masked_depth.PointerAt<float>(u,v) = *float_depth.PointerAt<float>(u,v);
                if(*masked_depth.PointerAt<float>(u,v)>0.2) {
                    valid_depth_array.push_back(*masked_depth.PointerAt<float>(u,v));
                }
            }
        }
    }
    if(valid_depth_array.size()<min_points) return false;
    else{
        // sort depth array
        std::sort(valid_depth_array.begin(), valid_depth_array.end());
        // double min_depth_clip = valid_depth_array[std::floor(valid_depth_array.size()*CLIP_RATIO)];
        double max_depth_clip = valid_depth_array[std::ceil(valid_depth_array.size()*(1-CLIP_RATIO))];
        // std::cout<<"["<<min_depth_clip<<","<<max_depth_clip<<"]"<<std::endl;
        masked_depth.ClipIntensity(0.0,max_depth_clip);

        masked_rgbd->depth_ = masked_depth;
        return true;
        // return valid_depth_array.size();
    }
}

O3d_Image_Ptr extract_masked_o3d_image(const O3d_Image &depth, const O3d_Image &mask)
{
    auto masked_depth = std::make_shared<open3d::geometry::Image>();
    masked_depth->Prepare(depth.width_, depth.height_, 1, 4);
    const unsigned char ZERO_MASK = 0;
    for(int v=0; v<depth.height_;v++){
        for(int u=0; u<depth.width_;u++){
            if(mask.PointerAt<unsigned char>(u,v)> &ZERO_MASK){
                *masked_depth->PointerAt<float>(u,v) = *depth.PointerAt<float>(u,v);
            }
        }
    }
    return masked_depth;
}

void random_sample(const std::vector<int> &indices, const int &sample_size, std::vector<int> &sampled_indices)
{
    std::vector<int> shuffled_indices = indices;
    std::random_shuffle(shuffled_indices.begin(), shuffled_indices.end());
    sampled_indices = std::vector<int>(shuffled_indices.begin(), shuffled_indices.begin()+sample_size);
}

bool write_config(const std::string &output_dir, const fmfusion::Config &config)
{
    std::ofstream file(output_dir);

    if (file.is_open()){
        file << "Mapping: " << std::endl;
        file << config.mapping_cfg.print_msg();
        file << "Graph: " << std::endl;
        file << config.graph.print_msg();
        file << "SGNet: " << std::endl;
        file << config.sgnet.print_msg();
        file << "Shape encoder: " << std::endl;
        file << config.shape_encoder.print_msg();
        file << "Loop detector: " << std::endl;
        file << config.loop_detector.print_msg();
        file << "Registration: " << std::endl;
        file << config.reg.print_msg();

        file.close();
        return true;
    }
    else return false;

}

}
}
