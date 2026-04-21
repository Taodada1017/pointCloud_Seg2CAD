#ifndef FMFUSION_UTILITY_H
#define FMFUSION_UTILITY_H
#include <fstream>
#include <string>
#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"
#include "Common.h"
#include "mapping/Instance.h"
#include "mapping/VoxelHashMap.h"
namespace fmfusion
{

namespace utility
{

// 基础版本：获取整数坐标颜色（不使用四舍五入）
Eigen::Vector3d GetPixelColorDirect(const open3d::geometry::Image& image, int x, int y);
// 双线性插值版本
Eigen::Vector3d GetPixelColorBilinear(const open3d::geometry::Image& image, double u, double v);

//球面翻转
std::vector<Eigen::Vector3d> spherical_flip(const std::vector<Eigen::Vector3d>& points);

//球面翻转 - tbb并发
std::vector<Eigen::Vector3d> spherical_flip_tbb(const std::vector<Eigen::Vector3d>& points);


// 计算点是否在obb框内
bool isPointInOBBManual(const Eigen::Vector3d& point,
    const open3d::geometry::OrientedBoundingBox& obb);


// 筛选相机视野下的可见点
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<size_t>, std::vector<Eigen::Vector2i>>
PointsInCameraView(
    open3d::geometry::PointCloud& slam_cloud,
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,
    //const open3d::geometry::Image& rgb,
    double max_depth = 10.0);


// 筛选相机视野下的全部可见点 - 体素网格优化
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<size_t>, std::vector<Eigen::Vector2i>>
PointsInCameraView_voxel(
    open3d::geometry::PointCloud& slam_cloud,
    VoxelHashMap& voxel_hash_map,
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,
    //const open3d::geometry::Image& rgb,
    double max_depth = 5.0);


// 筛选相机视野下的全部可见点 - 体素网格优化 - tbb并发
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<size_t>, std::vector<Eigen::Vector2i>>
PointsInCameraView_voxel_tbb(
    open3d::geometry::PointCloud& slam_cloud,
    VoxelHashMap& voxel_hash_map,
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,
    //const open3d::geometry::Image& rgb,
    double max_depth = 5.0);


// 加上深度容错率-筛选相机视野下的全部可见点 - 体素网格优化
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<size_t>, std::vector<Eigen::Vector2i>>
PointsInCameraView_voxel_zbuffer(
    open3d::geometry::PointCloud& slam_cloud,
    VoxelHashMap& voxel_hash_map,
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,
    double depth_threshold = 0.2,  // 深度一致性阈值
    //const open3d::geometry::Image& rgb,
    double max_depth = 5.0);

// tbb并发优化-加上深度容错率-筛选相机视野下的全部可见点 - 体素网格优化
std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::vector<size_t>, std::vector<Eigen::Vector2i>>
PointsInCameraView_voxel_zbuffer_tbb(
    open3d::geometry::PointCloud& slam_cloud,
    VoxelHashMap& voxel_hash_map,
    const open3d::camera::PinholeCameraIntrinsic& intrinsic,
    const Eigen::Matrix4d& camera_to_world,
    double depth_threshold = 0.2,  // 深度一致性阈值
    //const open3d::geometry::Image& rgb,
    double max_depth = 5.0);


// 将掩码图可视化
void visualize_mask_image(const cv::Mat &mask , std::string first_name);

// 根据每个二维检测物体的像素点再筛选该物体的点云
std::set<size_t> GetInstanceIndices(std::vector<size_t>& in_view_indices,
                                  std::vector<Eigen::Vector2i>& projected_uvs,
                                  const cv::Mat& mask
                                  //std::string name
//                                  open3d::geometry::PointCloud& slam_cloud,
//                                  const open3d::geometry::Image& rgb
);

// 根据每个二维检测物体的像素点再筛选该物体的点云 - tbb并发优化
std::set<size_t> GetInstanceIndices_tbb(std::vector<size_t>& in_view_indices,
    std::vector<Eigen::Vector2i>& projected_uvs,
    const cv::Mat& mask
);

// 暂时无法用
// 根据每个二维检测物体的像素点再筛选该物体的点云 
std::vector<std::set<size_t>> GetInstanceIndices_all(std::vector<size_t>& in_view_indices,
    std::vector<Eigen::Vector2i>& projected_uvs,
    const cv::Mat& detection_map,              // 全图掩码
    const int &n_k
);



// 根据已有的 相机可见点、对应像素点、该instance在相机中的可见点

std::shared_ptr<cv::Mat> prjection_cloud_to_mask(const std::vector<size_t> &in_view_indices,
    const std::set<size_t> &observed_instance_points,
    const std::vector<Eigen::Vector2i> &projected_uvs,
    const open3d::camera::PinholeCameraIntrinsic &intrinsic);


//////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> 
    split_str(const std::string s, const std::string delim);

fmfusion::Config *create_scene_graph_config(const std::string &config_file, bool verbose);

std::string config_to_message(const fmfusion::Config &config);

template <typename T>
inline std::vector<T> update_masked_vec(const std::vector<T> &pairs, const std::vector<bool> &prune_masks)
{
    assert(pairs.size() == prune_masks.size() && "Size mismatch");
    std::vector<T> pruned_pairs;
    for(int i=0;i<prune_masks.size();i++){
        if(prune_masks[i]) pruned_pairs.push_back(pairs[i]);
    }
    return pruned_pairs;
}

// 加载检测二维分割结果
bool LoadPredictions(const std::string &folder_path, const std::string &frame_name, 
                    const MappingConfig &mapping_cfg, const int &img_width, const int &img_height,
                    std::vector<DetectionPtr> &detections);

std::shared_ptr<cv::Mat> RenderDetections(const std::shared_ptr<cv::Mat> &rgb_img,
    const std::vector<fmfusion::DetectionPtr> &detections, const std::unordered_map<InstanceId,CvMatPtr> &instances_mask,
    const Eigen::VectorXi &matches, const std::unordered_map<InstanceId,Eigen::Vector3d> &instance_colors);

std::shared_ptr<cv::Mat> PrjectionCloudToDepth(const open3d::geometry::PointCloud& cloud, 
    const Eigen::Matrix4d &pose_inverse,const open3d::camera::PinholeCameraIntrinsic& intrinsic, int dilation_size);

bool create_masked_rgbd(
    const open3d::geometry::Image &rgb, const open3d::geometry::Image &float_depth, const cv::Mat &mask,
    const int &min_points,
    std::shared_ptr<open3d::geometry::RGBDImage> &masked_rgbd);

bool write_config(const std::string &output_dir, const fmfusion::Config &config);
 
}

O3d_Image_Ptr extract_masked_o3d_image(const O3d_Image &depth, const O3d_Image &mask);

void random_sample(const std::vector<int> &indices, const int &sample_size, std::vector<int> &sampled_indices);


}

#endif //FMFUSION_UTILITY_H
