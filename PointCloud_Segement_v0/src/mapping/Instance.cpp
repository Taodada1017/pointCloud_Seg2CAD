#include "Instance.h"


namespace fmfusion {



    Eigen::Vector3d IdToColorHash(uint32_t id) {
        // 使用哈希混合颜色通道，使相邻ID颜色差异大
        /*std::hash<uint32_t> hasher;
        uint32_t hash = hasher(id);*/

        // 混合算法，使颜色分布更均匀
        uint8_t r = (id >> 16) & 0xFF;
        uint8_t g = (id >> 8) & 0xFF;
        uint8_t b = id & 0xFF;

        // 避免太暗或太亮的颜色，提高可视性
        /*r = std::clamp(r, static_cast<uint8_t>(30), static_cast<uint8_t>(225));
        g = std::clamp(g, static_cast<uint8_t>(30), static_cast<uint8_t>(225));
        b = std::clamp(b, static_cast<uint8_t>(30), static_cast<uint8_t>(225));*/

        return Eigen::Vector3d(r / 255.0, g / 255.0, b / 255.0);
    }


    Instance::Instance(const InstanceId id, const unsigned int frame_id, const InstanceConfig &config,
                       const O3d_Cloud_Ptr &slam_cloud) :
            id_(id), frame_id_(frame_id), update_frame_id(frame_id),
            config_(config), bayesian_label(false),
            slam_point_cloud_(slam_cloud)
    {

        point_cloud = std::make_shared<open3d::geometry::PointCloud>();
        merged_cloud = std::make_shared<open3d::geometry::PointCloud>();
        min_box = std::make_shared<open3d::geometry::OrientedBoundingBox>();
        predicted_label = std::make_pair("unknown", 0.0);

        color_ = IdToColorHash(id);     // 将id编码进颜色通道

        std::srand(std::time(nullptr));
//        color_ = Eigen::Vector3d((double) rand() / RAND_MAX,
//                                (double) rand() / RAND_MAX,
//                                (double) rand() / RAND_MAX);
        centroid = Eigen::Vector3d(0.0, 0.0, 0.0);
        normal = Eigen::Vector3d(0.0, 0.0, 0.0);
        observation_count = 1;

    }


    void Instance::init_bayesian_fusion(const std::vector<std::string> &label_set)
    {
        semantic_labels = std::vector<std::string>(label_set);
        probability_vector = Eigen::VectorXf::Zero(label_set.size());
        bayesian_label = true;
    }

    // 融合其他instance的下标数组
    void Instance::integrate(const int &frame_id, const std::set<size_t> &other_cloud_indices, bool if_update_cloud){

        //std::cout << "实例 " << id_ << " 开始融合下标数组，已有数组大小：" << cloud_indices_.size() << endl;
        //std::cout << "要融合的数组大小为" << other_cloud_indices.size() << endl;

        frame_id_ = frame_id;
        // 合并set
        cloud_indices_.insert(other_cloud_indices.begin(), other_cloud_indices.end());

        //std::cout << "融合之后数组大小为：" << cloud_indices_.size() << endl;

        if(if_update_cloud){
            extract_write_point_cloud();
        }

    }

    // 使用可见点云对应的原始下标，找到当前视野下该instance的下标
    std::set<size_t> Instance::query_observed_points(const std::vector<size_t>& in_view_indices) {
        // 转为有序的set,复杂度为 nlogn
        std::set<size_t> set1(in_view_indices.begin(), in_view_indices.end());
        
        std::set<size_t> intersection;

        // 寻找两个set的交集。时间复杂度：O(set1.size() + set2.size())
        std::set_intersection(set1.begin(), set1.end(),
                              cloud_indices_.begin(), cloud_indices_.end(),
                              std::inserter(intersection, intersection.begin()));

        return intersection;
        
    }
    
        //    void Instance::integrate(const int &frame_id,
        //                             const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image,
        //                             const Eigen::Matrix4d &pose) {
        //        volume_->Integrate(*rgbd_image, config_.intrinsic, pose);
        //        frame_id_ = frame_id;
        //    }
        // 
            // 原函数，使用统计方法去除离群点
            //void Instance::filter_pointcloud_statistic() {
            //    if (point_cloud) {
            //        size_t old_points_number = point_cloud->points_.size();
            //        O3d_Cloud_Ptr output_cloud;
            //        std::tie(output_cloud, std::ignore) = point_cloud->RemoveStatisticalOutliers(20, 2.0);
            //        point_cloud = output_cloud;
            //        // std::cout<<"!!!!Filter point cloud from "<<old_points_number<<" to "<<point_cloud->points_.size()<<std::endl;
            //    }
            //}
    

    void Instance::filter_pointcloud_statistic() {
        if (point_cloud) {
            size_t old_points_number = point_cloud->points_.size();

            size_t old_indices_size = cloud_indices_.size();

            O3d_Cloud_Ptr output_cloud;
            std::vector<size_t> old_cloud_indices;
            std::tie(output_cloud, old_cloud_indices) = point_cloud->RemoveStatisticalOutliers(30, 1.5);
            point_cloud = output_cloud;

            std::set<size_t> new_cloud_indices_;
            std::vector<size_t> old_indices(cloud_indices_.begin(), cloud_indices_.end());  //将原下标数组转为vector

            for (size_t index : old_cloud_indices) {
                new_cloud_indices_.insert(old_indices[index]);
            }

            cloud_indices_ = new_cloud_indices_; //TODO风险

            size_t new_indices_size = cloud_indices_.size();


            //std::cout<<"!!!!Filter point cloud from "<<old_points_number<<" to "<<point_cloud->points_.size()<<std::endl;

            //std::cout << "!!!!Filter point cloud from " << old_indices_size << " to " << new_indices_size << std::endl;

        }
    }

    // 已修改。根据下标数组提取点云
    void Instance::extract_write_point_cloud() {

        point_cloud->Clear();

        point_cloud->points_.reserve(cloud_indices_.size());

        for(auto& index : cloud_indices_){
            if(index < slam_point_cloud_->points_.size())
                point_cloud->points_.emplace_back(slam_point_cloud_->points_[index]);
        }
        if (point_cloud->HasPoints()) {
//            point_cloud->VoxelDownSample(config_.voxel_length);
//            point_cloud->PaintUniformColor(color_);
            centroid = point_cloud->GetCenter();
        }
        else{
            std::cerr<<"Instance "<<id_<<" has no point cloud.\n";
        }

    }

    void Instance::merge_with_2(const std::set<size_t>& other_cloud_indices,
        const std::unordered_map<std::string, float>& label_measurements,
        const int& observations_) {
        
        cloud_indices_.insert(other_cloud_indices.begin(), other_cloud_indices.end());

        // update labels
        for (const auto label_score : label_measurements) {
            if (measured_labels.find(label_score.first) == measured_labels.end()) {
                measured_labels[label_score.first] = label_score.second;
            }
            else {
                measured_labels[label_score.first] += label_score.second;
            }
            // update prediction
            if (!bayesian_label
                && measured_labels[label_score.first] > predicted_label.second) {
                predicted_label = std::make_pair(label_score.first, measured_labels[label_score.first]);
            }
        }
        observation_count += observations_;

        //
        CreateMinimalBoundingBox();

    }

    bool Instance::filter_pointcloud_by_cluster() {
        if (point_cloud == nullptr) {
            return false;
        } else {
            auto labels = point_cloud->ClusterDBSCAN(config_.cluster_eps, config_.cluster_min_points, true);
            std::unordered_map<int, int> cluster_counts; // cluster_id:count
            const size_t old_points_number = point_cloud->points_.size();

            // create cluster
            for (auto label: labels) {
                if (cluster_counts.find(label) == cluster_counts.end()) {
                    cluster_counts[label] = 1;
                } else {
                    cluster_counts[label] += 1;
                }
            }

            // find invalid cluster
            std::unordered_set<int> invalid_cluster({-1});

            // filter
            size_t k = 0;
            for (size_t i = 0; i < old_points_number; i++) {
                if (invalid_cluster.find(labels[i]) != invalid_cluster.end()) { //invalid
                    continue;
                } else { //valid
                    point_cloud->points_[k] = point_cloud->points_[i];
                    if (point_cloud->HasNormals()) point_cloud->normals_[k] = point_cloud->normals_[i];
                    if (point_cloud->HasCovariances()) point_cloud->covariances_[k] = point_cloud->covariances_[i];
                    if (point_cloud->HasColors()) point_cloud->colors_[k] = point_cloud->colors_[i];
                    k++;
                }
            }
            point_cloud->points_.resize(k);
            point_cloud->PaintUniformColor(color_);
            o3d_utility::LogInfo("Filter point cloud from {} to {}.", old_points_number, k);
            return true;
        }
    }

    void Instance::CreateMinimalBoundingBox() {
        // if(point_cloud->points_.size()<10){
        if (get_cloud_size() < 10) {
            return;
        }
        auto complete_cloud = get_complete_cloud();

        using namespace open3d::geometry;
        std::shared_ptr<TriangleMesh> mesh;
        std::tie(mesh, std::ignore) = complete_cloud->ComputeConvexHull(false);
        double min_vol = -1;
        min_box->Clear();
        PointCloud hull_pcd;
        for (auto &tri: mesh->triangles_) {
            hull_pcd.points_ = mesh->vertices_;
            Eigen::Vector3d a = mesh->vertices_[tri(0)];
            Eigen::Vector3d b = mesh->vertices_[tri(1)];
            Eigen::Vector3d c = mesh->vertices_[tri(2)];
            Eigen::Vector3d u = b - a;
            Eigen::Vector3d v = c - a;
            Eigen::Vector3d w = u.cross(v);
            v = w.cross(u);
            u = u / u.norm();
            v = v / v.norm();
            w = w / w.norm();
            Eigen::Matrix3d m_rot;
            // m_rot << u[0], v[0], w[0], u[1], v[1], w[1], u[2], v[2], w[2];
            // set roll and pitch to zero
            m_rot << u[0], v[0], 0, u[1], v[1], 0, 0, 0, 1;
            hull_pcd.Rotate(m_rot.inverse(), a);

            const auto aabox = hull_pcd.GetAxisAlignedBoundingBox();
            double volume = aabox.Volume();
            if (min_vol == -1. || volume < min_vol) {
                min_vol = volume;
                *min_box = aabox.GetOrientedBoundingBox();
                min_box->Rotate(m_rot, a);
            }
        }
        min_box->color_ = color_;

        //
        if (predicted_label.first == "floor") {
            min_box->center_[2] = min_box->center_[2] - min_box->extent_[2] / 2 + 0.05;
            min_box->extent_[2] = 0.02;
        }
    }

    void Instance::merge_with(const O3d_Cloud_Ptr &other_cloud,
                              const std::unordered_map<std::string, float> &label_measurements,
                              const int &observations_) {
        // merge point cloud
        *merged_cloud += *other_cloud;
        merged_cloud->VoxelDownSample(config_.voxel_length);
        merged_cloud->PaintUniformColor(color_);

        // update labels
        for (const auto label_score: label_measurements) {
            if (measured_labels.find(label_score.first) == measured_labels.end()) {
                measured_labels[label_score.first] = label_score.second;
            } else {
                measured_labels[label_score.first] += label_score.second;
            }
            // update prediction
            if (!bayesian_label
                &&measured_labels[label_score.first] > predicted_label.second) {
                predicted_label = std::make_pair(label_score.first, measured_labels[label_score.first]);
            }
        }
        observation_count += observations_;

        //
        CreateMinimalBoundingBox();

    }

    bool Instance::update_semantic_probability(const Eigen::VectorXf &probability_vector_)
    {
        if(probability_vector_.size() != probability_vector.size()){ 
            std::cerr<<"Probability vector size mismatch.\n";
            return false;
        }
        else {
            probability_vector += probability_vector_;
            extract_bayesian_prediciton();
            return true;
        }
    }

    // 更新语义标签（已经修改）
    void Instance::update_label(const std::vector<LabelScore> &labels) {
        for (const auto &label_score: labels) {
            if (measured_labels.find(label_score.first) == measured_labels.end()) {
                measured_labels[label_score.first] = label_score.second;
            } else {
                measured_labels[label_score.first] += label_score.second;
            }

            // Update prediction
            if (!bayesian_label 
                && measured_labels[label_score.first] > predicted_label.second) {
                predicted_label = std::make_pair(label_score.first, measured_labels[label_score.first]);
            }
        }
        observation_count++;
    }

    O3d_Cloud_Ptr Instance::get_point_cloud() const {
        return point_cloud;
    }

    void Instance::extract_bayesian_prediciton()
    {
        if(bayesian_label){
            int max_idx;
            probability_vector.maxCoeff(&max_idx);
            Eigen::VectorXf probability_normalized = probability_vector.normalized();
            predicted_label = std::make_pair(semantic_labels[max_idx], 
                                            probability_normalized[max_idx]);
        }
        else std::cerr<<"Instance "<<id_
                    <<" is not in Bayesian fusion mode. \n";
    }


    bool Instance::update_point_cloud(int cur_frame_id, int min_frame_gap) {
        if (cur_frame_id - update_frame_id < min_frame_gap) {
            return false;
        } else {
            extract_write_point_cloud();
            filter_pointcloud_statistic();
            CreateMinimalBoundingBox();
            update_frame_id = cur_frame_id;
            return true;
        }

    }

    void Instance::load_previous_labels(const std::string &labels_str) {
        std::stringstream ss(labels_str);
        std::string label_score;

        while (std::getline(ss, label_score, ',')) {
            std::stringstream ss2(label_score); // label_name(score)
            std::string label;
            float score;
            std::getline(ss2, label, '(');
            ss2 >> score;
            measured_labels[label] = score;
            if (score > predicted_label.second) {
                predicted_label = std::make_pair(label, score);
            }
            // cout<<label<<":"<<score<<",";
        }
        // cout<<"\n";
    }

    size_t Instance::get_cloud_size() const {
        return cloud_indices_.size();
    }

    O3d_Cloud_Ptr Instance::get_complete_cloud() const {
        if (merged_cloud->HasPoints()) {
            auto complete_cloud = std::make_shared<O3d_Cloud>();
            *complete_cloud += *point_cloud;
            *complete_cloud += *merged_cloud;
            complete_cloud->VoxelDownSample(config_.voxel_length);
            return complete_cloud;
        } else {
            return point_cloud;
        }
    }


}