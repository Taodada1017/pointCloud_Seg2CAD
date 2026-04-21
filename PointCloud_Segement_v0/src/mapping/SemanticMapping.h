#ifndef FMFUSION_SEMANTICMAPPING_H
#define FMFUSION_SEMANTICMAPPING_H

#include <unordered_map>
#include <fstream>

#include "Common.h"
#include "tools/Color.h"
#include "tools/Utility.h"
#include "tools/TicToc.h"
#include "Instance.h"
#include "SemanticDict.h"
#include "BayesianLabel.h"
#include "VoxelHashMap.h"



namespace fmfusion {

class SemanticMapping {
public:
    SemanticMapping(const MappingConfig &mapping_cfg, const InstanceConfig &instance_cfg, O3d_Cloud_Ptr &slam_cloud);

    ~SemanticMapping() {};

public:
//    void integrate(const int &frame_id,
//                    const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose,
//                    std::vector<DetectionPtr> &detections);

    void build_voxel_hashmap(double voxel_size = 2.4);

    // 重载自己的融合函数
    void integrate(const int &frame_id,
                    //const std::shared_ptr<open3d::geometry::Image> &rgb_image, 
                    const Eigen::Matrix4d &pose,
                    //const cv::Mat& detection_map,
                    std::vector<DetectionPtr> &detections);
    // 将instance的颜色赋予给slam
    void ColorSlamWithInstances();

    int merge_overlap_instances(std::vector<InstanceId> instance_list = std::vector<InstanceId>());

    int merge_floor_2(bool verbose = false);
    int merge_ceiling(bool verbose = false);

    int merge_floor(bool verbose=false);

    int merge_overlap_structural_instances(bool merge_all = true);

    int merge_other_instances(std::vector<InstancePtr> &instances);
    
    void extract_point_cloud(const std::vector<InstanceId> instance_list = std::vector<InstanceId>());

    /// \brief  Extract and update bounding box for each instance.
    void extract_bounding_boxes();

    int update_instances(const int &cur_frame_id, const std::vector<InstanceId> &instance_list);

    /// @brief  clean all and update all.
    void refresh_all_semantic_dict();

    std::shared_ptr<open3d::geometry::PointCloud> export_global_pcd(bool filter = false, float vx_size = -1.0);

    std::vector<Eigen::Vector3d> export_instance_centroids(int earliest_frame_id = -1,
                                                        bool verbose=false) const;

    std::vector<std::string> export_instance_annotations(int earliest_frame_id = -1
                                                        ) const;

    bool query_instance_info(const std::vector<InstanceId> &names,
                                std::vector<Eigen::Vector3f> &centroids,
                                std::vector<std::string> &labels);

    void remove_invalid_instances();

    /// \brief  Get geometries for each instance.
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>>
    get_geometries(bool point_cloud = true, bool bbox = false);

    bool is_empty() { return instance_map.empty(); }

    InstancePtr get_instance(const InstanceId &name) { return instance_map[name]; }

    void Transform(const Eigen::Matrix4d &pose);

    /// @brief
    /// @param path output sequence folder
    /// @return
    bool Save(const std::string &path);

    bool load(const std::string &path);


    // 保存与读取
    
    bool save_all_instances_2(const std::string& root_path);
    bool load_all_instances_2(const std::string& root_path);
    void save_cloud(const std::string& output_path);

    // 在SLAM中去除实例
    O3d_Cloud_Ptr remove_instances_from_slam(const std::vector<InstanceId>& instance_ids);
    O3d_Cloud_Ptr remove_all_instances();

    // 使用obb框来填充实例
    void fill_instances();
    void update_instances_cloud();

    std::vector<ObbPtr> get_instances_obb();
    std::vector<AABBPtr> get_instances_aabb();


    /// \brief  Export instances to the vector.
    ///         Filter instances that are too small or not been observed for a long time.
    void export_instances(std::vector<InstanceId> &names, std::vector<InstancePtr> &instances,
                            int earliest_frame_id = 0);

protected:
    /// \brief  match vector in [K,1], K is the number of detections;
    /// If detection k is associated, match[k] = matched_instance_id
    int
    data_association(const std::vector<DetectionPtr> &detections, const std::vector<InstanceId> &active_instances,
                        Eigen::VectorXi &matches,
                        std::vector<std::pair<InstanceId, InstanceId>> &ambiguous_pairs);
    int
    data_association_2(const std::vector<set<size_t>>& detections_points,    // 二维检测物体的像素点对应的点云
        const std::vector<InstanceId>& active_instances,
        Eigen::VectorXi& matches,
        std::vector<std::pair<InstanceId, InstanceId>>& ambiguous_pairs);


    // 七、创建新实例。已修改
    InstanceId create_new_instance(const std::vector<LabelScore>& labels, const unsigned int &frame_id,
                            const std::set<size_t> &cloud_indices_);

    std::vector<InstanceId> search_active_instances(const O3d_Cloud_Ptr& in_view_cloud,        // 相机视野下的可见点云
                                                                     const std::vector<size_t>& in_view_indices, // 可见点云对应的原始下标
                                                                     const std::vector<Eigen::Vector2i>& projected_uvs,
                                                                     const double search_radius = 5.0);

    void update_active_instances(const std::vector<InstanceId> &active_instances);

    void update_recent_instances(const int &frame_id,
                                    const std::vector<InstanceId> &active_instances,
                                    const std::vector<InstanceId> &new_instances);

    bool IsSemanticSimilar(const std::unordered_map<std::string, float> &measured_labels_a,
                            const std::unordered_map<std::string, float> &measured_labels_b);

    /// \brief  Compute the 2D IoU (horizontal plane) between two oriented bounding boxes.
    double Compute2DIoU(const open3d::geometry::OrientedBoundingBox &box_a,
                        const open3d::geometry::OrientedBoundingBox &box_b);

    /// \brief  Compute the 3D IoU between two point clouds.
    /// \param cloud_a, point cloud of the larger instance
    /// \param cloud_b, point cloud of the smaller instance
    double Compute3DIoU(const O3d_Cloud_Ptr &cloud_a, const O3d_Cloud_Ptr &cloud_b, double inflation = 1.0);

    int merge_ambiguous_instances(const std::vector<std::pair<InstanceId, InstanceId>> &ambiguous_pairs);

    // Recent observed instances
    std::unordered_set<InstanceId> recent_instances;

private:
    // Config config_;
    MappingConfig mapping_config;
    InstanceConfig instance_config;
    std::unordered_map<InstanceId, InstancePtr> instance_map;
    std::vector<int> point_map;         // 记录每个点对应的实例id,用于反向查找
    std::unordered_map<std::string, std::vector<InstanceId>> label_instance_map;
    SemanticDictServer semantic_dict_server;
    BayesianLabel *bayesian_label;
    O3d_Cloud_Ptr slam_point_cloud; // SLAM点云

    InstanceId latest_created_instance_id;
    int last_cleanup_frame_id;
    int last_update_frame_id;

    // 将点云体素储存
    VoxelHashMap* voxel_hashmap; //体素网格

    std::unordered_map<InstanceId, bool> instanceID_map_;


};


} // namespace fmfusion


#endif //FMFUSION_SEMANTICMAPPING_H
