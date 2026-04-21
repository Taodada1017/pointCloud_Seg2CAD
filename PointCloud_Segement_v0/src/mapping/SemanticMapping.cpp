#include "SemanticMapping.h"
#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <ctime>
#include <random>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for_each.h>
namespace fmfusion
{

SemanticMapping::SemanticMapping(const MappingConfig &mapping_cfg, const InstanceConfig &instance_cfg, O3d_Cloud_Ptr &slam_cloud)
    : mapping_config(mapping_cfg), instance_config(instance_cfg), slam_point_cloud(slam_cloud), semantic_dict_server()
{
    open3d::utility::LogInfo("Initialize SceneGraph server");
    latest_created_instance_id = 0;
    last_cleanup_frame_id = 0;
    last_update_frame_id = 0;

    if(slam_point_cloud){
        cout << "SemanticMapping成功读取slam点云，点数: " << slam_point_cloud->points_.size() << endl;
        std::cout << "slam_cloud对象地址: " << slam_point_cloud.get() << std::endl;
    }

    point_map.assign(slam_point_cloud->points_.size(), -1); // 初始化点云映射，-1表示未匹配

    // 分离体素网格构建
    //// 初始化体素网格
    //voxel_hashmap = new VoxelHashMap(0.24);
    //voxel_hashmap->build_from_pointcloud(*slam_cloud);

    if(mapping_config.bayesian_semantic){
        bayesian_label = new BayesianLabel(mapping_config.bayesian_semantic_likelihood, true);
    }
    else bayesian_label = nullptr;

}

void SemanticMapping::build_voxel_hashmap(double voxel_size) {
    // 初始化体素网格
    voxel_hashmap = new VoxelHashMap(voxel_size);
    voxel_hashmap->build_from_pointcloud(*slam_point_cloud);
}


//////////////////////////////////////////////////////////////////////////////////

// 一、重载自己的融合函数
void SemanticMapping::integrate(const int &frame_id,
               //const std::shared_ptr<open3d::geometry::Image> &rgb_image, 
               const Eigen::Matrix4d &pose,
               //const cv::Mat& detection_map,
               std::vector<DetectionPtr> &detections)
{
    TicToc timer;

    open3d::utility::LogInfo("## Integrate SceneGraph");

    int n_det = detections.size();  //检测数量
    std::vector<InstanceId> new_instances;  // 存储新创建的实例

    if(n_det < 1){
        std::cout<<"检测的数量太少"<<endl;
        return;
    }

    open3d::camera::PinholeCameraIntrinsic camera_intrinsic = instance_config.intrinsic;

    // 1.找相机视野内的点云
    if (!voxel_hashmap) {
        std::cout << "未初始化voxel_hashmap" << std::endl;
        return;
    }
    auto result = utility::PointsInCameraView_voxel_tbb(*slam_point_cloud, *voxel_hashmap, camera_intrinsic, pose);
    //auto result = utility::PointsInCameraView(*slam_point_cloud, camera_intrinsic, pose);

    O3d_Cloud_Ptr in_view_cloud = std::get<0>(result);       // 可见点云
    std::vector<size_t> in_view_indices = std::get<1>(result);    // 可见点云对应的原始下标
    std::vector<Eigen::Vector2i> projected_uvs = std::get<2>(result);      // 可见点云对应的图像二维坐标

    // 2.搜索相机视野下的实例
    auto active_instances = search_active_instances(in_view_cloud, in_view_indices, projected_uvs);
    o3d_utility::LogInfo("{:d} active instances are found.", active_instances.size());// 记录找到的活跃实例数量


    // 3.获取每个detections二维像素点对应的点云

    std::vector<set<size_t>> detections_points;

    timer.tic();
    for (int k_ = 0; k_ < n_det; k_++) {
        // 取名，用于可视化mask，检查筛选的正确性
        //int idx = frame_id * 100 + k_;
        //std::string name = to_string(idx);

        // 二次筛选，找到每个instance的点云下标
        std::set<size_t> instance_indices = utility::GetInstanceIndices_tbb(in_view_indices, projected_uvs, detections[k_]->instances_idxs_);

        //TODO 有可能存在序号对不上的问题，如果语义标签错误查此
        detections_points.emplace_back(instance_indices);
    }
    //detections_points = utility::GetInstanceIndices_tbb(in_view_indices, projected_uvs, detection_map, n_det);
    std::cout << "二次筛选耗时：" << timer.toc() << "ms" << std::endl;
    


    // 4.数据关联
    Eigen::VectorXi matches;// 用于存储检测与实例的匹配结果
    std::vector<std::pair<InstanceId, InstanceId>> ambiguous_pairs;// 存储模糊匹配对（多个实例可能与同一个检测匹配）
    int count_m = data_association_2(detections_points,
        active_instances,
        matches,
        ambiguous_pairs); // (n_det,)



    for(int k_=0; k_<n_det; k_++){
        std::set<size_t> instance_indices = detections_points[k_];
        if (instance_indices.size() < 100) {
            o3d_utility::LogInfo("实例点云数量太少");
            continue;
        }
        if (matches(k_) > 0) {

            //std::cout << "检测物体 " << k_ << "匹配的instance为 " << matches(k_) << std::endl;
            auto matched_instance = instance_map[matches(k_)];
            if (!matched_instance) {
                std::cout << "无法找到对应实例" << std::endl;
                continue;
            }
            //matched_instance->update_point_cloud
            matched_instance->integrate(frame_id, instance_indices);
            matched_instance->update_label(detections[k_]->labels_);// 用检测的标签信息更新实例的语义标签
            if (bayesian_label) {
                Eigen::VectorXf probability_vector;
                matched_instance->init_bayesian_fusion(bayesian_label->get_label_vec());
                bayesian_label->update_measurements(detections[k_]->labels_,
                    probability_vector);
                matched_instance->update_semantic_probability(probability_vector);
            }
            matched_instance->update_point_cloud(frame_id, mapping_config.update_period); // 更新实例的点云表示（定期更新，避免每帧都更新）

        }
        else {

            InstanceId added_idx = create_new_instance(detections[k_]->labels_, frame_id, instance_indices);
            new_instances.emplace_back(added_idx);

        }
         
    }
    
}


// 二、搜索活跃实例。根据已有的
std::vector<InstanceId> 
SemanticMapping::search_active_instances(const O3d_Cloud_Ptr &in_view_cloud,        // 当前帧相机视野下的可见点云
                                        const std::vector<size_t> &in_view_indices, // 可见点云对应的原始下标
                                        const std::vector<Eigen::Vector2i> &projected_uvs,           
                                        const double search_radius)
{
    std::vector<InstanceId> active_instances;
    const size_t MIN_UNITS= 1;
    if (instance_map.empty()) return active_instances;
    open3d::utility::Timer timer;
    // float query_time_ms = 0.0;
    // float projection_time_ms = 0.0;
    Eigen::Vector3d depth_cloud_center = in_view_cloud->GetCenter();

    // 先根据距离初筛一遍
    std::vector<InstanceId> target_instances;
    for(auto &instance_j:instance_map){
        double dist = (depth_cloud_center   - instance_j.second->centroid).norm();
        if(dist<search_radius) target_instances.emplace_back(instance_j.first);
    }

    //std::cout << "距离范围内的实例数量：" << target_instances.size() << std::endl;

#pragma omp parallel for default(none) shared(target_instances, active_instances)
    for(const auto &idx:target_instances){
        InstancePtr instance_j = instance_map[idx];
        // 找到观测到的该instance的点
        std::set<size_t> observed_instance_points = instance_j->query_observed_points(in_view_indices);
        //std::cout << "获取了该实例的可见点" << std::endl;
        if(observed_instance_points.size()>mapping_config.min_active_points){
#pragma omp critical
            
                // 更新活跃实例的可见mask
                //std::cout << "找到活跃实例：" << idx << std::endl;
                // 
                // 更新instance的二维可见点
                //std::shared_ptr<cv::Mat> observed_image_mask_ = utility::prjection_cloud_to_mask(in_view_indices, observed_instance_points,
                    //projected_uvs, instance_config.intrinsic);
                //std::cout << "投影到二维完成" << std::endl;
                // 
                //instance_j->observed_image_mask = observed_image_mask_;  // 保证一次性赋予
                //std::cout << "赋予实例二维投影结果"<< std::endl;
                // 
                // 疑似有竞争问题
                instance_j->observed_points_indices = observed_instance_points;  // 更新实例在当前视野下的可见点
                //std::cout << "更新实例在当前视野下的可见点" << std::endl;
            //std::string name = to_string(idx);
            //utility::visualize_mask_image(*(instance_j->observed_image_mask), name);
            active_instances.emplace_back(idx);
        }
    }
    
    return active_instances;
}

// 三、更新活跃instance的 observed_image_mask
void SemanticMapping::update_active_instances(const std::vector<InstanceId> &active_instances)
{
    for(InstanceId j_:active_instances){
        auto instance_j = instance_map[j_];
        instance_j->observed_image_mask.reset();
        // instance_j->fast_update_centroid();
    }
}

// 四、更新
void SemanticMapping::update_recent_instances(const int &frame_id,
                                            const std::vector<InstanceId> &active_instances,
                                            const std::vector<InstanceId> &new_instances)
{
    std::vector<InstanceId> invalid_instances; // remove from recent instances
    
    //
    for(InstanceId j_:active_instances) recent_instances.emplace(j_);
    for(InstanceId j_:new_instances) recent_instances.emplace(j_);
    
    //!debug: some instances should not be cleared
    for(auto idx:recent_instances){
        auto inst = instance_map.find(idx);
        if(inst==instance_map.end()) continue;

        if((frame_id-inst->second->frame_id_) > mapping_config.recent_window_size){
            if(!inst->second->point_cloud->HasPoints()){ // remove instances has not been re-observed for long time
                instance_map.erase(inst);
            }

        }
    }

    // Clean up
    for(auto idx:invalid_instances){
        recent_instances.erase(idx);
    }



}

// 五_1、原数据关联 二维iou
int SemanticMapping::data_association(const std::vector<DetectionPtr> &detections, 
                                    const std::vector<InstanceId> &active_instances,
                                    Eigen::VectorXi &matches,
                                    std::vector<std::pair<InstanceId,InstanceId>> &ambiguous_pairs)
{
    int K = detections.size();
    int M = active_instances.size();

    // Eigen::VectorXi 
    matches = Eigen::VectorXi::Zero(K);
    if (M<1) return 0;

    Eigen::MatrixXd iou = Eigen::MatrixXd::Zero(K,M);
    Eigen::MatrixXi assignment = Eigen::MatrixXi::Zero(K,M);
    Eigen::MatrixXi assignment_colwise = Eigen::MatrixXi::Zero(K,M);
    Eigen::MatrixXi assignment_rowise = Eigen::MatrixXi::Zero(K,M);

    for (int k_=0;k_<K;k_++){
        const auto &zk = detections[k_];
        double zk_area = double(cv::countNonZero(zk->instances_idxs_));
        for (int m_=0;m_<M;m_++){
            auto instance_m = instance_map[active_instances[m_]];
            cv::Mat overlap = instance_m->observed_image_mask->mul(zk->instances_idxs_);
            double overlap_area = double(cv::countNonZero(overlap));
            double instance_area = double(cv::countNonZero(*instance_m->observed_image_mask));
            iou(k_,m_) = overlap_area/(zk_area+instance_area-overlap_area);
        }
    } 

    // Find the maximum match for each colume
    for (int m_=0;m_<M;m_++){
        int max_row;
        double max_iou = iou.col(m_).maxCoeff(&max_row);
        if (max_iou>mapping_config.min_iou)assignment_colwise(max_row,m_) = 1;
    }

    // Find the maximum match for each row
    // std::vector<std::pair<InstanceId,InstanceId>> ambiguous_pairs;
    for (int k_=0;k_<K;k_++){
        int max_col;
        double max_iou = iou.row(k_).maxCoeff(&max_col);
        if (max_iou>mapping_config.min_iou)assignment_rowise(k_,max_col) = 1;

        // Find ambiguous pairs and search their 3D overlap later
        Eigen::ArrayXd row_correlated = iou.row(k_).array(); //- mapping_config.min_iou * Eigen::ArrayXd::Ones(K);
        int row_correlated_num = (row_correlated>0).count();
        if(row_correlated_num>1){
            row_correlated[max_col] = 0.0;
            int second_max_col;
            row_correlated.maxCoeff(&second_max_col);
            ambiguous_pairs.emplace_back(std::make_pair(active_instances[max_col],active_instances[second_max_col]));
        }
    }

    assignment = assignment_colwise + assignment_rowise;

    // export matches
    int count= 0;
    std::vector<InstanceId> matched_instances;
    std::vector<InstanceId> unmatched_instances;    
    for (int k_=0;k_<K;k_++){
        for (int m_=0;m_<M;m_++){
            if (assignment(k_,m_)==2){
                matches(k_) = active_instances[m_];
                matched_instances.emplace_back(active_instances[m_]);
                count ++;
                break;
            }
        }
    }

    //
    for(int m_=0;m_<M;m_++){
        if(assignment.col(m_).sum()<2) unmatched_instances.emplace_back(active_instances[m_]);
    }

    // std::cout<<iou<<std::endl;   
    // std::cout<<assignment<<std::endl;
    o3d_utility::LogInfo("{} associations out of {} detections and {} active instances.",
                        count,K,M);

    return count;
}

// 五_2、修改数据关联，
int SemanticMapping::data_association_2(const std::vector<set<size_t>>& detections_points,    // 二维检测物体的像素点对应的点云
    const std::vector<InstanceId>& active_instances,
    Eigen::VectorXi& matches,
    std::vector<std::pair<InstanceId, InstanceId>>& ambiguous_pairs)
{
    int K = detections_points.size();
    int M = active_instances.size();

    // Eigen::VectorXi 
    matches = Eigen::VectorXi::Zero(K);
    if (M < 1) return 0;

    Eigen::MatrixXd iou = Eigen::MatrixXd::Zero(K, M);
    Eigen::MatrixXi assignment = Eigen::MatrixXi::Zero(K, M);
    Eigen::MatrixXi assignment_colwise = Eigen::MatrixXi::Zero(K, M);
    Eigen::MatrixXi assignment_rowise = Eigen::MatrixXi::Zero(K, M);

    for (int k_ = 0; k_ < K; k_++) {
        // 获取实例一的点云数组
        const std::set<size_t> &set1 = detections_points[k_];
        for (int m_ = 0; m_ < M; m_++) {
            // 获取实例二的点云数组
            auto instance_m = instance_map[active_instances[m_]];
            std::set<size_t> &set2 = instance_m->observed_points_indices;
            //  求交集
            std::set<int> intersection;
            std::set_intersection(
                set1.begin(), set1.end(),
                set2.begin(), set2.end(),
                std::inserter(intersection, intersection.begin())
            );
            // 计算iou
            double overlap_area = double(intersection.size());
            double union_area = double(set1.size() + set2.size() - intersection.size());
            iou(k_, m_) = overlap_area / union_area;
        }
    }
    // 计算每个实例相匹配的最大检测物体的iou最大
    for (int m_ = 0; m_ < M; m_++) {
        int max_row;
        double max_iou = iou.col(m_).maxCoeff(&max_row);
        if (max_iou > mapping_config.min_iou)assignment_colwise(max_row, m_) = 1;
    }

    // Find the maximum match for each row
    // std::vector<std::pair<InstanceId,InstanceId>> ambiguous_pairs;
    for (int k_ = 0; k_ < K; k_++) {
        int max_col;
        double max_iou = iou.row(k_).maxCoeff(&max_col);
        if (max_iou > mapping_config.min_iou)assignment_rowise(k_, max_col) = 1;

        // Find ambiguous pairs and search their 3D overlap later
        Eigen::ArrayXd row_correlated = iou.row(k_).array(); //- mapping_config.min_iou * Eigen::ArrayXd::Ones(K);
        int row_correlated_num = (row_correlated > 0).count();
        if (row_correlated_num > 1) {
            row_correlated[max_col] = 0.0;
            int second_max_col;
            row_correlated.maxCoeff(&second_max_col);
            ambiguous_pairs.emplace_back(std::make_pair(active_instances[max_col], active_instances[second_max_col]));
        }
    }

    assignment = assignment_colwise + assignment_rowise;

    // export matches
    int count = 0;
    std::vector<InstanceId> matched_instances;
    std::vector<InstanceId> unmatched_instances;
    for (int k_ = 0; k_ < K; k_++) {
        for (int m_ = 0; m_ < M; m_++) {
            if (assignment(k_, m_) == 2) {
                matches(k_) = active_instances[m_];
                matched_instances.emplace_back(active_instances[m_]);
                count++;
                break;
            }
        }
    }

    //
    for (int m_ = 0; m_ < M; m_++) {
        if (assignment.col(m_).sum() < 2) unmatched_instances.emplace_back(active_instances[m_]);
    }

    // std::cout<<iou<<std::endl;   
    // std::cout<<assignment<<std::endl;
    o3d_utility::LogInfo("{} associations out of {} detections and {} active instances.",
        count, K, M);

    return count;
}


// 六、
void SemanticMapping::refresh_all_semantic_dict()
{
    semantic_dict_server.clear();
    for(auto instance:instance_map){
        auto instance_labels = instance.second->get_predicted_class();
        semantic_dict_server.update_instance(instance_labels.first,instance.first);
    }
    auto query_result = semantic_dict_server.query_instances("floor");
    std::cout<<"一共有地板数量：" << query_result.size() << std::endl;

    query_result = semantic_dict_server.query_instances("ceiling");
    std::cout << "一共有天花板数量：" << query_result.size() << std::endl;

}

// 七、创建新实例
InstanceId SemanticMapping::create_new_instance(const std::vector<LabelScore>& labels, const unsigned int &frame_id,
                                         const std::set<size_t> &cloud_indices_)
{

    //// 初始化随机数种子
    //std::srand(static_cast<uint32_t>(std::time(nullptr)));
    //// 生成24位以内的随机正整数
    //uint32_t random_num = std::rand() % 16777215 + 1;

    // 创建随机数引擎
    std::random_device rd;  // 用于获取真随机数种子
    std::mt19937 gen(rd()); // 使用Mersenne Twister引擎

    // 定义分布范围 [1, 16777215] 或 [0, 16777215]
    std::uniform_int_distribution<uint32_t> dist(1, 16777215); // 包含1和16777215

    // 生成随机数
    uint32_t random_number = dist(gen);
    while (instanceID_map_.find(random_number)!= instanceID_map_.end()) {
        random_number = dist(gen);
    }
    instanceID_map_[random_number] = true;

    auto instance = std::make_shared<Instance>(//latest_created_instance_id+1,
                                                random_number,
                                                frame_id,instance_config,
                                               slam_point_cloud);
    instance->integrate(frame_id, cloud_indices_, true);
    instance->update_label(labels);
    //instance->color_ = InstanceColorBar20[instance->get_id()%InstanceColorBar20.size()];
    if(bayesian_label){
        Eigen::VectorXf probability_vector;
        instance->init_bayesian_fusion(bayesian_label->get_label_vec());
        bayesian_label->update_measurements(labels,
                                            probability_vector);
        instance->update_semantic_probability(probability_vector);
    }

    instance_map.emplace(instance->get_id(),instance);
    latest_created_instance_id = instance->get_id();
    std::cout<<"创建新instance，id: "<< latest_created_instance_id << std::endl;
    return latest_created_instance_id;

}

// 将instance的颜色赋予给slam
void SemanticMapping::ColorSlamWithInstances(){

    // 从大到小着色，防止吞并

    //std::vector<InstanceId> target_instances;
    //std::unordered_map<InstanceId, size_t> instance_size;

    //for (const auto& [id, inst_ptr] : instance_map) {
    //    if (!inst_ptr) {
    //        std::cerr << "[Save Warning] 实例 " << id << " 的指针为空，跳过。" << std::endl;
    //        continue;
    //    }
    //    if (inst_ptr->get_cloud_indices().size() < 100) {
    //        continue;
    //    }
    //    target_instances.emplace_back(id);
    //    instance_size[id] = inst_ptr->get_cloud_indices().size();
    //}

    //// 将instance按照点云大小排序
    //std::sort(target_instances.begin(), target_instances.end(),
    //    [&instance_size](const InstanceId& a, const InstanceId& b) {
    //        return instance_size[a] > instance_size[b];
    //    });


    for(auto &instance_j:instance_map){
        std::set<size_t> &indices = instance_j.second->cloud_indices_;
        Eigen::Vector3d &color = instance_j.second->color_;

        tbb::parallel_for_each(indices.begin(), indices.end(),
            [&](const size_t &index) {
                    slam_point_cloud->colors_[index] = color;
            });
        //for(auto &index:indices){
        //    slam_point_cloud->colors_[index] = color;
        //}
    }
}


// 二进制储存下标数组
bool SemanticMapping::save_all_instances_2(const std::string& root_path) {
    // 1. 构建文件完整路径
    std::string points_file_path = root_path + "/points.txt";
    std::string info_file_path = root_path + "/info.txt";

    // 2. 以二进制模式打开 points.txt 文件用于写入
    std::ofstream points_file(points_file_path, std::ios::out | std::ios::binary);
    if (!points_file.is_open()) {
        std::cerr << "[Save Error] 无法创建/写入(二进制) points.txt 文件: " << points_file_path << std::endl;
        return false;
    }

    // 3. 以文本模式打开 info.txt 文件
    std::ofstream info_file(info_file_path);
    if (!info_file.is_open()) {
        std::cerr << "[Save Error] 无法创建/写入 info.txt 文件: " << info_file_path << std::endl;
        points_file.close();
        return false;
    }

    // 4. 遍历全局 instance_map 并写入数据
    for (const auto& [id, inst_ptr] : instance_map) {
        if (!inst_ptr) {
            std::cerr << "[Save Warning] 实例 " << id << " 的指针为空，跳过。" << std::endl;
            continue;
        }

        // 4.1 写入 points.txt (二进制格式)
        const auto& indices = inst_ptr->get_cloud_indices();
        size_t indices_count = indices.size();

        // 写入实例ID (uint32_t)
        uint32_t id_to_write = static_cast<uint32_t>(id);
        points_file.write(reinterpret_cast<const char*>(&id_to_write), sizeof(id_to_write));

        // 写入下标数量 (size_t)
        points_file.write(reinterpret_cast<const char*>(&indices_count), sizeof(indices_count));

        // 写入下标数组 (size_t 数组)
        // 先将 std::set<size_t> 转为 std::vector<size_t> 以便连续内存访问
        std::vector<size_t> indices_vec(indices.begin(), indices.end());
        if (!indices_vec.empty()) {
            points_file.write(reinterpret_cast<const char*>(indices_vec.data()),
                indices_count * sizeof(size_t));
        }

        // 3.2 写入 info.txt
        // 获取 measured_labels 并格式化为字符串 "label:score,label:score,..."
        std::stringstream labels_stream;
        auto measured_labels = inst_ptr->get_measured_labels();
        bool is_first_label = true;
        for (const auto& [label, score] : measured_labels) {
            if (!is_first_label) {
                labels_stream << ",";
            }
            labels_stream << label << ":" << score;
            is_first_label = false;
        }
        info_file << id << " " << inst_ptr->frame_id_ << " " << labels_stream.str() << std::endl;
    }

    // 4. 关闭文件并返回
    points_file.close();
    info_file.close();
    std::cout << "[Save Success] 实例数据已保存至目录: " << root_path << std::endl;
    return true;
}

// 读取二进制格式
bool SemanticMapping::load_all_instances_2(const std::string& root_path) {
    // 1. 构建文件完整路径
    std::string points_file_path = root_path + "/points.txt";
    std::string info_file_path = root_path + "/info.txt";

    // 2. 以二进制模式打开 points.txt 文件
    std::ifstream points_file(points_file_path, std::ios::in | std::ios::binary);
    if (!points_file.is_open()) {
        std::cerr << "[Load Error] 无法以二进制模式打开 points.txt 文件: " << points_file_path << std::endl;
        return false;
    }

    // 3. 以文本模式打开 info.txt 文件
    std::ifstream info_file(info_file_path);
    if (!info_file.is_open()) {
        std::cerr << "[Load Error] 无法打开 info.txt 文件: " << info_file_path << std::endl;
        points_file.close();
        return false;
    }

    // 3. 清空当前全局 instance_map
    instance_map.clear();

    // 4. 首先，解析 info.txt 文件，创建 Instance 对象并设置基础信息
    std::string line;
    std::unordered_map<InstanceId, InstancePtr> temp_loaded_map;

    while (std::getline(info_file, line)) {
        std::istringstream iss(line);
        InstanceId id;
        unsigned int frame_id;
        std::string labels_str;

        // 读取每行的基本格式：id frame_id label1:score1,label2:score2,...
        if (!(iss >> id >> frame_id >> labels_str)) {
            std::cerr << "[Load Error] info.txt 行格式错误: " << line << std::endl;
            continue;
        }

        // 4.1 创建新的 Instance 对象
        // 注意：Instance 构造函数需要 (id, frame_id, config, slam_cloud)
        auto new_instance = std::make_shared<fmfusion::Instance>(id, frame_id, instance_config, slam_point_cloud);

        // 4.2 解析并设置 measured_labels
        std::unordered_map<std::string, float> measured_labels;
        std::istringstream label_stream(labels_str);
        std::string label_pair;

        double max_score = 0.0;
        LabelScore predicted_label;

        while (std::getline(label_stream, label_pair, ',')) {
            size_t colon_pos = label_pair.find(':');
            if (colon_pos != std::string::npos) {
                std::string label = label_pair.substr(0, colon_pos);
                float score = std::stof(label_pair.substr(colon_pos + 1));
                measured_labels[label] = score;

                if (score > max_score) {
                    predicted_label = std::make_pair(label, score);
                    max_score = score;
                }
            }
        }
        new_instance->set_measured_labels(measured_labels);
        new_instance->set_predicted_label(predicted_label);
        //new_instance->color_ = InstanceColorBar40[id % InstanceColorBar40.size()];

        // 4.3 将临时实例存入映射表
        temp_loaded_map[id] = new_instance;
    }
    info_file.close();

    // 6. 解析 points.txt 文件（二进制格式）
    while (points_file) {
        // 6.1 读取实例ID
        uint32_t id_bin;
        points_file.read(reinterpret_cast<char*>(&id_bin), sizeof(id_bin));
        if (points_file.eof()) {
            break; // 正常读到文件末尾
        }
        if (!points_file) {
            std::cerr << "[Load Error] 读取 points.txt 时无法读取实例ID或已到达文件末尾。" << std::endl;
            break;
        }
        InstanceId id = static_cast<InstanceId>(id_bin);

        // 6.2 读取下标数量
        size_t indices_count = 0;
        points_file.read(reinterpret_cast<char*>(&indices_count), sizeof(indices_count));
        if (!points_file || indices_count > 100000000) { // 简单合理性检查：下标数量不超过1亿
            std::cerr << "[Load Error] points.txt 中实例 " << id << " 的下标数量读取失败或值异常: " << indices_count << std::endl;
            break;
        }

        // 6.3 读取下标数组
        std::vector<size_t> indices_vec(indices_count);
        if (indices_count > 0) {
            points_file.read(reinterpret_cast<char*>(indices_vec.data()), indices_count * sizeof(size_t));
            if (!points_file) {
                std::cerr << "[Load Error] points.txt 中实例 " << id << " 的下标数组数据不完整。" << std::endl;
                break;
            }
        }

        // 6.4 查找对应的 Instance 对象并设置下标
        auto it = temp_loaded_map.find(id);
        if (it == temp_loaded_map.end()) {
            std::cerr << "[Load Warning] points.txt 中实例 ID " << id << " 在 info.txt 中未找到，跳过其点云数据。" << std::endl;
            // 注意：即使info.txt中没有对应记录，我们仍需跳过这个实例的二进制数据块，
            // 但上面的 read 已经消费了数据，所以这里直接继续循环即可。
            continue;
        }

        // 将 vector 转换为 set 并设置到实例中
        std::set<size_t> indices(indices_vec.begin(), indices_vec.end());
        it->second->set_cloud_indices(indices);

        // 提取每个实例的点云，获得中心点（可选）
        it->second->extract_write_point_cloud();
    }
    points_file.close();

    // 6. 将成功加载的实例交换到全局 instance_map
    instance_map.swap(temp_loaded_map);

    std::cout << "[Load Success] 从目录加载了 " << instance_map.size() << " 个实例: " << root_path << std::endl;
    return true;
}

// 储存实例点云
void SemanticMapping::save_cloud(const std::string& output_path) {
    //std::vector<InstanceId> target_instances;
    
    for (const auto& [id, inst_ptr] : instance_map) {
        if (!inst_ptr) {
            std::cerr << "[Save Warning] 实例 " << id << " 的指针为空，跳过。" << std::endl;
            continue;
        }
        open3d::io::WritePointCloud(output_path +"/" + std::to_string(id) + ".ply", *(inst_ptr->point_cloud));
    }
}


// 重写地板合并
int SemanticMapping::merge_floor_2(bool verbose)
{
    std::vector<InstanceId> target_instances = semantic_dict_server.query_instances("floor");
    std::vector<InstanceId> carpet_instances = semantic_dict_server.query_instances("carpet");
    for (auto& carpet_id : carpet_instances) target_instances.emplace_back(carpet_id);
    if (target_instances.size() < 2) return 0;
    if (verbose) std::cout << "Merging " << target_instances.size() << " floor instances\n";

    InstancePtr root_floor; //instance_map[target_instances[0]];
    Eigen::Vector3d root_center;
    int root_points = 0;
    for (int i = 0; i < target_instances.size(); i++) { // Find the largest floor instance
        if (instance_map.find(target_instances[i]) == instance_map.end()) continue;
        auto instance = instance_map[target_instances[i]];

        if (instance->get_cloud_indices().size() > root_points) {
            root_floor = instance;
            root_center = instance->centroid;
            root_points = instance->get_cloud_indices().size();
        }
    }
    if (root_points < 500) return 0;
    if (verbose) std::cout << "Root floor has " << root_points << " points\n";

    // Iterate and merge
    int count = 0;
    int debug = 0;
    for (int i = 0; i < target_instances.size(); i++) {
        if (target_instances[i] == root_floor->get_id()) continue;
        if (instance_map.find(target_instances[i]) == instance_map.end()) continue;
        auto instance = instance_map[target_instances[i]];
        if (instance->get_cloud_indices().size() < 500)  continue;
        double dist_z = (root_center - instance->centroid)[2];
        if (dist_z < 1.0 && dist_z > -1.0) { //merge
            root_floor->merge_with_2(
                instance->get_cloud_indices(),
                instance->get_measured_labels(),
                instance->get_observation_count());
            if (bayesian_label) {
                Eigen::VectorXf probability_vector;
                bayesian_label->update_measurements(instance->get_measured_labels(),
                    probability_vector);
                root_floor->update_semantic_probability(probability_vector);
            }

            instance_map.erase(instance->get_id());
            count++;
        }

        debug++;
    }
    if (verbose) std::cout << debug << " floor instances are checked\n";

    std::cout << "Merged " << count << " floor instances\n";
    return count;
}

// （添加）合并天花板
int SemanticMapping::merge_ceiling(bool verbose)
{
    std::vector<InstanceId> target_instances = semantic_dict_server.query_instances("ceiling");
    
    if (target_instances.size() < 2) return 0;
    if (verbose) std::cout << "Merging " << target_instances.size() << " ceiling instances\n";

    InstancePtr root_floor; //instance_map[target_instances[0]];
    Eigen::Vector3d root_center;
    int root_points = 0;
    for (int i = 0; i < target_instances.size(); i++) { // Find the largest floor instance
        if (instance_map.find(target_instances[i]) == instance_map.end()) continue;
        auto instance = instance_map[target_instances[i]];

        if (instance->get_cloud_indices().size() > root_points) {
            root_floor = instance;
            root_center = instance->centroid;
            root_points = instance->get_cloud_indices().size();
        }
    }
    if (root_points < 500) return 0;
    if (verbose) std::cout << "Root floor has " << root_points << " points\n";

    // Iterate and merge
    int count = 0;
    int debug = 0;
    for (int i = 0; i < target_instances.size(); i++) {
        if (target_instances[i] == root_floor->get_id()) continue;
        if (instance_map.find(target_instances[i]) == instance_map.end()) continue;
        auto instance = instance_map[target_instances[i]];
        if (instance->get_cloud_indices().size() < 500)  continue;
        double dist_z = (root_center - instance->centroid)[2];
        if (dist_z < 1.0 && dist_z > -1.0) { //merge
            root_floor->merge_with_2(
                instance->get_cloud_indices(),
                instance->get_measured_labels(),
                instance->get_observation_count());
            if (bayesian_label) {
                Eigen::VectorXf probability_vector;
                bayesian_label->update_measurements(instance->get_measured_labels(),
                    probability_vector);
                root_floor->update_semantic_probability(probability_vector);
            }

            instance_map.erase(instance->get_id());
            count++;
        }

        debug++;
    }
    if (verbose) std::cout << debug << " ceiling instances are checked\n";

    std::cout << "Merged " << count << " ceiling instances\n";
    return count;
}

// 更新各个实例的点云
void SemanticMapping::update_instances_cloud() {
    
    tbb::parallel_for_each(
        instance_map.begin(),
        instance_map.end(),
        [](auto& kv) {
            InstancePtr instance = kv.second;
            if (instance->cloud_indices_.size() < 20) {
                return;
            }

            LabelScore predicted_label = instance->get_predicted_class();
            if (predicted_label.first == "floor" || predicted_label.first == "ceiling") {
                return;
            }

            instance->extract_write_point_cloud();
        }
    );

    std::cout << "成功更新所有实例的点云" << std::endl;
}



// 使用obb框来填充实例
void SemanticMapping::fill_instances(){

    TicToc timer;

    timer.tic();

    std::vector<InstanceId> target_instances;
    std::unordered_map<InstanceId, size_t> instance_size;

    for (const auto& [id, inst_ptr] : instance_map) {
        if (!inst_ptr) {
            std::cerr << "[Save Warning] 实例 " << id << " 的指针为空，跳过。" << std::endl;
            continue;
        }
        LabelScore predicted_label = inst_ptr->get_predicted_class();
        if (predicted_label.first == "floor" || predicted_label.first == "ceiling" || predicted_label.first == "carpet") {
            continue;
        }
        target_instances.emplace_back(id);
        instance_size[id] = inst_ptr->get_cloud_indices().size();
    }

    // 将instance按照点云大小排序
    std::sort(target_instances.begin(), target_instances.end(),
        [&instance_size](const InstanceId& a, const InstanceId& b) {
            return instance_size[a] < instance_size[b];
        });

    for (const auto& id : target_instances) {
        InstancePtr instance = instance_map[id];
        if (instance->cloud_indices_.size() < 20) {
            continue;
        }
        //std::cout << "当前填充的实例点云大小为：" << instance_size[id] << std::endl;
        instance->extract_write_point_cloud();
        instance->min_box = std::make_shared<open3d::geometry::OrientedBoundingBox>(
            instance->point_cloud->GetMinimalOrientedBoundingBox() );

        assert(voxel_hashmap), "未构建体素网格";
        std::vector<VoxelDataPtr> voxel_list = voxel_hashmap->get_voxels_in_obb(*(instance->min_box));
        //std::cout<<"当前实例的体素数量为：" << voxel_list.size() << std::endl;

        if (voxel_list.size() == 0) {
            continue;
        }
        tbb::concurrent_vector<size_t> concurrent_vec;
        tbb::parallel_for_each(
            voxel_list.begin(),
            voxel_list.end(),
            [&](VoxelDataPtr voxel) {
                for (size_t& index : voxel->point_indices) {
                    if (point_map[index]>0)continue;
                    auto& point = slam_point_cloud->points_[index];
                    if (utility::isPointInOBBManual(point, *(instance->min_box))) {
                        point_map[index] = id;
                        concurrent_vec.emplace_back(index);
                    }
                        
                }
            }
        );

        instance->cloud_indices_.insert(concurrent_vec.begin(), concurrent_vec.end());

    }

    // 遍历获取每个实例的obb框
    //tbb::parallel_for_each(
    //    instance_map.begin(),
    //    instance_map.end(),
    //    [&](auto& kv) {  
    //        InstancePtr instance = kv.second;
    //        if (instance->cloud_indices_.size() < 20) {
    //            return;
    //        }

    //        LabelScore predicted_label = instance->get_predicted_class();
    //        if (predicted_label.first == "floor" || predicted_label.first == "ceiling" || predicted_label.first == "carpet") {
    //            return;
    //        }

    //        instance->extract_write_point_cloud();
    //        instance->min_box = std::make_shared<open3d::geometry::OrientedBoundingBox>(
    //            instance->point_cloud->GetMinimalOrientedBoundingBox() );


    //        assert(voxel_hashmap), "未构建体素网格";
    //        
    //        std::vector<VoxelDataPtr> voxel_list = voxel_hashmap->get_voxels_in_obb(*(instance->min_box));
    //        for (VoxelDataPtr &voxel : voxel_list) {
    //            for (size_t& index : voxel->point_indices) {

    //                auto& point = slam_point_cloud->points_[index];
    //                if (utility::isPointInOBBManual(point, *(instance->min_box)))
    //                    instance->cloud_indices_.insert(index);
    //            }
    //        }


    //    }
    //);


    std::cout << "填充实例完成,耗时:" << timer.toc() << std::endl;
}


// 获取每个实例的obb框
std::vector<ObbPtr> SemanticMapping::get_instances_obb() {

    std::vector<ObbPtr> obb_list;

    // 遍历获取每个实例的obb框
    tbb::parallel_for_each(
        instance_map.begin(),
        instance_map.end(),
        [](auto& kv) {
            InstancePtr instance = kv.second;
            if (instance->cloud_indices_.size() < 20) {
                return;
            }

            LabelScore predicted_label = instance->get_predicted_class();
            if (predicted_label.first == "floor" || predicted_label.first == "ceiling") {
                return;
            }

            instance->extract_write_point_cloud();
            instance->min_box = std::make_shared<open3d::geometry::OrientedBoundingBox>(
                instance->point_cloud->GetMinimalOrientedBoundingBox());
            instance->min_box->color_ = Eigen::Vector3d(0, 1, 0);
        }
    );

    std::cout << "成功并发计算得到所有obb框" << std::endl;

    for (const auto& [id, inst_ptr] : instance_map) {

        if (!inst_ptr) {
            std::cerr << "[Save Warning] 实例 " << id << " 的指针为空，跳过。" << std::endl;
            continue;
        }
        if (inst_ptr->cloud_indices_.size() < 100) {
            continue;
        }

        LabelScore predicted_label = inst_ptr->get_predicted_class();
        if (predicted_label.first == "floor" || predicted_label.first == "ceiling") {
            continue;
        }

        obb_list.emplace_back(inst_ptr->min_box);
    }

    std::cout << "成功获取所有实例的obb框" << std::endl;
    return obb_list;
}

// 获取每个实例的aabb框
std::vector<AABBPtr> SemanticMapping::get_instances_aabb() {

    std::vector<AABBPtr> aabb_list;

    // 遍历获取每个实例的obb框
    tbb::parallel_for_each(
        instance_map.begin(),
        instance_map.end(),
        [](auto& kv) {
            InstancePtr instance = kv.second;
            if (instance->cloud_indices_.size() < 20) {
                return;
            }

            LabelScore predicted_label = instance->get_predicted_class();
            if (predicted_label.first == "floor" || predicted_label.first == "ceiling") {
                return;
            }

            instance->extract_write_point_cloud();
            instance->aabb_box = std::make_shared<open3d::geometry::AxisAlignedBoundingBox>(
                instance->point_cloud->GetAxisAlignedBoundingBox());
            instance->aabb_box->color_ = Eigen::Vector3d(0, 1, 0);
        }
    );

    std::cout << "成功并发计算得到所有aabb框" << std::endl;

    for (const auto& [id, inst_ptr] : instance_map) {

        if (!inst_ptr) {
            std::cerr << "[Save Warning] 实例 " << id << " 的指针为空，跳过。" << std::endl;
            continue;
        }
        if (inst_ptr->cloud_indices_.size() < 100) {
            continue;
        }

        LabelScore predicted_label = inst_ptr->get_predicted_class();
        if (predicted_label.first == "floor" || predicted_label.first == "ceiling") {
            continue;
        }

        aabb_list.emplace_back(inst_ptr->aabb_box);
    }

    std::cout << "成功获取所有实例的aabb框" << std::endl;
    return aabb_list;
}
// 剔除实例
O3d_Cloud_Ptr SemanticMapping::remove_all_instances() {

    // 遍历全局 instance_map
    std::set<size_t> all_instances_indices;

    for (const auto& [id, inst_ptr] : instance_map) {
        if (!inst_ptr) {
            std::cerr << "[Save Warning] 实例 " << id << " 的指针为空，跳过。" << std::endl;
            continue;
        }

        LabelScore predicted_label = inst_ptr->get_predicted_class();
        if (predicted_label.first == "floor" || predicted_label.first == "ceiling" || predicted_label.first == "carpet") {
            continue;
        }

        const auto& indices = inst_ptr->get_cloud_indices();
        for (size_t idx : indices) {
            all_instances_indices.insert(idx);
        }

    }

    if (all_instances_indices.empty()) {
        std::cout << "无实例可以移除" << std::endl;
        return slam_point_cloud;
    }

    // 转为vector
    vector<size_t> instances_indices(all_instances_indices.begin(), all_instances_indices.end());

    size_t N = slam_point_cloud->points_.size();
    size_t M = instances_indices.size();

    if (N < M) {
        std::cout << "实例点云数量错误" << std::endl;
        return slam_point_cloud;
    }

    auto removed_cloud = std::make_shared<open3d::geometry::PointCloud>();
    removed_cloud->points_.reserve(N - M);
    removed_cloud->colors_.reserve(N - M);

    int i = 0, j = 0;

    while (i < N && j < M) {

        if (i < instances_indices[j]) {
            removed_cloud->points_.emplace_back(slam_point_cloud->points_[i]);
            removed_cloud->colors_.emplace_back(slam_point_cloud->colors_[i]);
            i++;
        }
        else if (i == instances_indices[j]) {
            i++;
            j++;
        }
        else {
            j++;
        }
    }

    while (i < N) {
        removed_cloud->points_.emplace_back(slam_point_cloud->points_[i]);
        removed_cloud->colors_.emplace_back(slam_point_cloud->colors_[i]);
        i++;
    }

    return removed_cloud;

}


// 
bool SemanticMapping::IsSemanticSimilar (const std::unordered_map<std::string,float> &measured_labels_a,
    const std::unordered_map<std::string,float> &measured_labels_b)
{
    if(measured_labels_a.size()<1 || measured_labels_b.size()<1) return false;

    for (const auto &label_score_a:measured_labels_a){
        for (const auto &label_score_b:measured_labels_b){
            if(label_score_a.first==label_score_b.first)return true;
        }
    }
    return false;
}

double SemanticMapping::Compute2DIoU(
    const open3d::geometry::OrientedBoundingBox &box_a, const open3d::geometry::OrientedBoundingBox &box_b)
{
    auto box_a_aligned = box_a.GetAxisAlignedBoundingBox();
    auto box_b_aligned = box_b.GetAxisAlignedBoundingBox();

    // extract corners
    Eigen::Vector3d a0 = box_a_aligned.GetMinBound();
    Eigen::Vector3d a1 = box_a_aligned.GetMaxBound();
    Eigen::Vector3d b0 = box_b_aligned.GetMinBound();
    Eigen::Vector3d b1 = box_b_aligned.GetMaxBound();

    // find overlapped rectangle
    double x0 = std::max(a0(0),b0(0));
    double y0 = std::max(a0(1),b0(1));
    double x1 = std::min(a1(0),b1(0));
    double y1 = std::min(a1(1),b1(1));

    if(x0>x1 || y0>y1) return 0.0;

    // iou
    double intersection_area = ((x1-x0)*(y1-y0));
    double area_a = (a1(0)-a0(0))*(a1(1)-a0(1));
    double area_b = (b1(0)-b0(0))*(b1(1)-b0(1));
    double iou = intersection_area/(area_a+area_b-intersection_area+0.000001);

    // std::cout<<"floor iou: "<<iou<<std::endl;
    // std::cout<<area_a<<","<<area_b<<","<<intersection_area<<std::endl;

    return iou;
}

// todo: try compute the real IoU
double SemanticMapping::Compute3DIoU (const O3d_Cloud_Ptr &cloud_a, const O3d_Cloud_Ptr &cloud_b, double inflation)
{
    auto vxgrid_a = open3d::geometry::VoxelGrid::CreateFromPointCloud(*cloud_a, inflation * instance_config.voxel_length);
    std::vector<bool> overlap = vxgrid_a->CheckIfIncluded(cloud_b->points_);
    double iou = double(std::count(overlap.begin(), overlap.end(), true)) / double(overlap.size()+0.000001);
    return iou;
}
// 
int SemanticMapping::merge_overlap_instances(std::vector<InstanceId> instance_list)
{
    double SEARCH_DISTANCE = 3.0; // in meters
    std::vector<InstanceId> target_instances;
    if(instance_list.empty()){
        // 默认尝试融合场景中所有实例
        for(const auto &instance_j:instance_map) target_instances.emplace_back(instance_j.first);
    }
    else{
        target_instances = instance_list;
    }
    if(target_instances.size()<3) return 0;
    int old_instance_number = target_instances.size();
    // std::cout<<"Trying to merge "<< target_instances.size()<<" instances\n";

    // 更新实例点云
    for (int i = 0; i < target_instances.size(); i++) {
        auto instance_i = instance_map[target_instances[i]];
        instance_i->extract_write_point_cloud();
    }

    // Find overlap instances
    open3d::utility::Timer timer;
    timer.Start();
    std::unordered_set<InstanceId> remove_instances;
    for(int i=0;i<target_instances.size();i++){
        auto instance_i = instance_map[target_instances[i]];
        if (!instance_i->point_cloud)
            o3d_utility::LogWarning("Instance {:d} has no point cloud",instance_i->get_id());
        // std::cout<<"instance "<<target_instances[i]<<": ";
        // std::string label_i = instance_i->get_predicted_class().first;
        if (instance_i->point_cloud->points_.size()<30) continue;
        for(int j=i+1;j<target_instances.size();j++){
            if(remove_instances.find(target_instances[j])!=remove_instances.end()) 
                continue;
            auto instance_j = instance_map[target_instances[j]];
            if (!instance_j->point_cloud)
                o3d_utility::LogWarning("Instance {:d} has no point cloud",instance_j->get_id());
            // std::cout<<target_instances[j] <<":"<<instance_j->get_predicted_class().first<<", "
            if (instance_j->point_cloud->points_.size()<30) continue;

            double dist = (instance_i->centroid-instance_j->centroid).norm();
            if (!IsSemanticSimilar(instance_i->get_measured_labels(), instance_j->get_measured_labels()) ||
                dist > SEARCH_DISTANCE) {
                //std::cout << "实例 " << instance_i->get_id() << " 和实例 " << instance_j->get_id() << " 语义不相似" << std::endl;
                continue;
            }

            // Compute Spatial IoU
            InstancePtr large_instance, small_instance;
            if(instance_i->point_cloud->points_.size()>instance_j->point_cloud->points_.size()){
                large_instance = instance_i;
                small_instance = instance_j;
            }
            else{
                large_instance = instance_j;
                small_instance = instance_i;
            }
            double iou = Compute3DIoU(large_instance->point_cloud,
                                    small_instance->point_cloud,
                                    mapping_config.merge_inflation);

            // Merge
            if(iou>mapping_config.merge_iou){
                large_instance->merge_with_2(
                    small_instance->get_cloud_indices(),
                    small_instance->get_measured_labels(),
                    small_instance->get_observation_count());
                if(bayesian_label){
                    Eigen::VectorXf probability_vector;
                    bayesian_label->update_measurements(small_instance->get_measured_labels(),
                                                        probability_vector);
                    large_instance->update_semantic_probability(probability_vector);
                }

                remove_instances.insert(small_instance->get_id());
                // std::cout<<small_instance->id_<<" merged into "<<large_instance->id_<<std::endl;
                if(small_instance->get_id()==instance_i->get_id()) break;
            }   
        }
    }

    // Remove merged instances
    for(auto &instance_id:remove_instances){
        instance_map.erase(instance_id);
    }
    timer.Stop();

    std::cout<<"Merged "<<remove_instances.size()<<"/"<<old_instance_number<<" instances by 3D IoU."
                <<"It takes "<< std::fixed<<std::setprecision(1)<<timer.GetDurationInMillisecond()<<" ms.\n";

    return remove_instances.size();
}

int SemanticMapping::merge_floor(bool verbose)
{
    std::vector<InstanceId> target_instances = semantic_dict_server.query_instances("floor");
    std::vector<InstanceId> carpet_instances = semantic_dict_server.query_instances("carpet");
    for(auto &carpet_id:carpet_instances) target_instances.emplace_back(carpet_id);
    if(target_instances.size()<2) return 0;
    if (verbose) std::cout<<"Merging "<< target_instances.size()<<" floor instances\n";

    InstancePtr root_floor; //instance_map[target_instances[0]];
    Eigen::Vector3d root_center;
    int root_points = 0;
    for(int i=0;i<target_instances.size();i++){ // Find the largest floor instance
        if(instance_map.find(target_instances[i])==instance_map.end()) continue;
        auto instance = instance_map[target_instances[i]];

        if(instance->point_cloud->points_.size()>root_points){
            root_floor = instance;
            root_center = instance->centroid;
            root_points = instance->point_cloud->points_.size();
        }
    }
    if(root_points<1000) return 0;
    if(verbose) std::cout<<"Root floor has "<<root_points<<" points\n";

    // Iterate and merge
    int count = 0;
    int debug = 0;
    for(int i=0;i<target_instances.size();i++){
        if(target_instances[i]==root_floor->get_id()) continue;
        if(instance_map.find(target_instances[i])==instance_map.end()) continue;
        auto instance = instance_map[target_instances[i]];
        if(instance->get_complete_cloud()->points_.size()<500 ||
            instance->get_observation_count()<mapping_config.min_observation)  continue;
        double dist_z = (root_center-instance->centroid)[2];
        if(dist_z<1.0){ //merge
            root_floor->merge_with(
                instance->get_complete_cloud(),
                instance->get_measured_labels(),
                instance->get_observation_count());
            if(bayesian_label){
                Eigen::VectorXf probability_vector;
                bayesian_label->update_measurements(instance->get_measured_labels(),
                                                    probability_vector);
                root_floor->update_semantic_probability(probability_vector);
            }

            instance_map.erase(instance->get_id());
            count++;
        }

        debug++;
    }
    if(verbose) std::cout<<debug<<" floor instances are checked\n";

    std::cout<<"Merged " <<count<<" floor instances\n";
    return count;
}

int SemanticMapping::merge_overlap_structural_instances(bool merge_all)
{
    assert(false); // Abandoned
    std::vector<InstanceId> target_instances;
    for(auto &instance_j:instance_map){
        if(instance_j.second->get_predicted_class().first=="floor")
            target_instances.emplace_back(instance_j.first);
    }
    if(target_instances.size()<2) return 0;

    if(merge_all){
        InstancePtr largest_floor;
        size_t larget_floor_size=0;

        for(auto idx:target_instances){
            auto instance = instance_map[idx];
            if(instance->point_cloud->points_.size()>larget_floor_size){
                largest_floor = instance;
                larget_floor_size = instance->point_cloud->points_.size();
            }
        }

        for(auto idx:target_instances){
            if(idx==largest_floor->get_id()) continue;
            auto instance = instance_map[idx];
            largest_floor->merge_with(
                instance->point_cloud,
                instance->get_measured_labels(),
                instance->get_observation_count());
            instance_map.erase(idx);
        }

        o3d_utility::LogInfo("Merged {:d} floor instances in one floor.",target_instances.size());
        return target_instances.size()-1;
    }

    //todo:remove
    int old_instance_number = target_instances.size();
    std::unordered_set<InstanceId> remove_instances;
    for(int i=0;i<target_instances.size();i++){
        auto instance_i = instance_map[target_instances[i]];
        std::string label_i = instance_i->get_predicted_class().first;
        for(int j=i+1;j<target_instances.size();j++){
            auto instance_j = instance_map[target_instances[j]];

            // Compute 2D IoU
            InstancePtr large_instance, small_instance;
            if(instance_i->point_cloud->points_.size()>instance_j->point_cloud->points_.size()){
                large_instance = instance_i;
                small_instance = instance_j;
            }
            else{
                large_instance = instance_j;
                small_instance = instance_i;
            }
            
            double iou = Compute2DIoU(*large_instance->min_box, *small_instance->min_box);
            
            // Merge
            if(iou>0.03){
                large_instance->merge_with(
                    small_instance->point_cloud,small_instance->get_measured_labels(),small_instance->get_observation_count());
                remove_instances.insert(small_instance->get_id());
                // std::cout<<small_instance->id_<<" merged into "<<large_instance->id_<<std::endl;
                if(small_instance->get_id()==instance_i->get_id()) break;
            }   
        }
    }

    // remove merged instances
    for(auto &instance_id:remove_instances){
        instance_map.erase(instance_id);
    }


}

int SemanticMapping::merge_ambiguous_instances(const std::vector<std::pair<InstanceId,InstanceId>> &ambiguous_pairs)
{
    int count = 0;
    for(const auto &pair:ambiguous_pairs){
        auto instance_i = instance_map[pair.first];
        auto instance_j = instance_map[pair.second];
        if(instance_i->point_cloud && instance_j->point_cloud){        
            continue;
            double iou = Compute3DIoU(instance_i->point_cloud,instance_j->point_cloud);
        }
        else{
            O3d_Cloud_Ptr cloud_ptr;

            if(instance_i->point_cloud){
                cloud_ptr = instance_i->point_cloud;
            }
            else if(instance_j->point_cloud){
                cloud_ptr = instance_j->point_cloud;
            }
            else continue;

        }

    }
    o3d_utility::LogInfo("Merged {:d} ambiguous instances by 3D IoU.",count);
    return count;
}

void SemanticMapping::extract_bounding_boxes()
{
    open3d::utility::Timer timer;
    timer.Start();
    int count = 0;
    std::cout<<"Extract bounding boxes for "<<instance_map.size()<<" instances\n";

    for (const auto &instance: instance_map){
        instance.second->filter_pointcloud_statistic();

        if(instance.second->get_cloud_size()>mapping_config.shape_min_points){
            instance.second->CreateMinimalBoundingBox();
            count++;
            // if(instance.second->min_box->IsEmpty()) count++;
        }
    }
    timer.Stop();
    o3d_utility::LogInfo("Extract {:d} valid bounding box in {:f} ms",count,timer.GetDurationInMillisecond());
}

std::shared_ptr<open3d::geometry::PointCloud> SemanticMapping::export_global_pcd(bool filter, float vx_size)
{
    auto global_pcd = std::make_shared<open3d::geometry::PointCloud>();
    for(const auto &inst:instance_map){
        if(filter && inst.second->get_cloud_size()<mapping_config.shape_min_points) continue;
        *global_pcd += *inst.second->get_complete_cloud();

    }
    if(vx_size>0.0) global_pcd = global_pcd->VoxelDownSample(vx_size);

    return global_pcd;
}

std::vector<Eigen::Vector3d> 
SemanticMapping::export_instance_centroids(int earliest_frame_id,
                                            bool verbose)const
{
    std::vector<Eigen::Vector3d> centroids;
    std::stringstream msg;
    for(const auto &inst:instance_map){
        int tmp_idx = inst.second->frame_id_;
        int pts_size = inst.second->get_cloud_size();
        if(pts_size>mapping_config.shape_min_points &&
            tmp_idx>=earliest_frame_id){
            centroids.emplace_back(inst.second->centroid);
        }
        msg<<inst.second->frame_id_
            <<"("<< pts_size<<")"
            <<",";
    }
    if(verbose) std::cout<<msg.str()<<"\n";
    o3d_utility::LogInfo("{:d} instance centroids are exported.",centroids.size());
    return centroids;
}

std::vector<std::string> SemanticMapping::export_instance_annotations(int earliest_frame_id)const
{
    std::vector<std::string> annotations;
    for(const auto &inst:instance_map){
        int tmp_idx = inst.second->frame_id_;
        if(inst.second->get_cloud_size()>mapping_config.shape_min_points &&
            tmp_idx>=earliest_frame_id)
            annotations.emplace_back(inst.second->get_predicted_class().first);
    }
    return annotations;
}

std::vector<std::shared_ptr<const open3d::geometry::Geometry>> SemanticMapping::get_geometries(bool point_cloud, bool bbox)
{
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> viz_geometries;
    for (const auto &instance: instance_map){
        if(instance.second->get_cloud_size()<mapping_config.shape_min_points) continue;
        viz_geometries.emplace_back(instance.second->point_cloud);
        if(bbox&&!instance.second->min_box->IsEmpty()){ 
            viz_geometries.emplace_back(instance.second->min_box);
        }
    }
    return viz_geometries;
}

void SemanticMapping::Transform(const Eigen::Matrix4d &pose)
{
    for (const auto &instance: instance_map){
        instance.second->point_cloud->Transform(pose);
        instance.second->centroid = instance.second->point_cloud->GetCenter();
    }
}

void SemanticMapping::extract_point_cloud(const std::vector<InstanceId> instance_list)
{
    std::vector<InstanceId> target_instances;
    if (instance_list.empty()){
        for(auto &instance:instance_map){
            target_instances.emplace_back(instance.first);
        }
    }
    else target_instances = instance_list;
    
    o3d_utility::LogInfo("Extract point cloud for {:d} instances",target_instances.size());
    for(const InstanceId idx:target_instances){
        if(instance_map.find(idx)==instance_map.end()) continue; //
        instance_map[idx]->extract_write_point_cloud();
    }
    o3d_utility::LogInfo("Extracted point cloud.");
}

bool SemanticMapping::Save(const std::string &path)
{
    using namespace o3d_utility::filesystem;
    if(!DirectoryExists(path)) MakeDirectory(path);

    open3d::geometry::PointCloud global_instances_pcd;

    typedef std::pair<InstanceId,std::string> InstanceInfo;
    std::vector<InstanceInfo> instance_info;
    std::vector<std::string> instance_box_info; // id:x,y,z;qw,qx,qy,qz;sx,sy,sz

    for (const auto &instance: instance_map){
        if(!instance.second->point_cloud) continue;
        LabelScore semantic_class_score = instance.second->get_predicted_class();
        auto instance_cloud = instance.second->get_complete_cloud(); //instance.second->point_cloud;
        if(instance.second->get_cloud_size()<mapping_config.shape_min_points) continue;

        global_instances_pcd += *instance_cloud;
        stringstream ss; // instance info string
        ss<<std::setw(4)<<std::setfill('0')<<instance.second->get_id();
        open3d::io::WritePointCloud(path+"/"+ss.str()+".ply",*instance_cloud);

        ss<<";"
            <<semantic_class_score.first<<"("<<std::fixed<<std::setprecision(2)<<semantic_class_score.second<<")"<<";"
            <<instance.second->get_observation_count()<<";"
            <<instance.second->get_measured_labels_string()<<";"
            <<instance_cloud->points_.size()<<";\n";
        instance_info.emplace_back(instance.second->get_id(),ss.str());

        if(!instance.second->min_box->IsEmpty()){
            stringstream box_ss;
            box_ss<<std::setw(4)<<std::setfill('0')<<instance.second->get_id()<<";";
            auto box = instance.second->min_box;
            box_ss<<box->center_(0)<<","<<box->center_(1)<<","<<box->center_(2)<<";"
                <<box->R_.coeff(0,0)<<","<<box->R_.coeff(0,1)<<","<<box->R_.coeff(0,2)<<","<<box->R_.coeff(1,0)<<","<<box->R_.coeff(1,1)<<","<<box->R_.coeff(1,2)<<","<<box->R_.coeff(2,0)<<","<<box->R_.coeff(2,1)<<","<<box->R_.coeff(2,2)<<";"
                <<box->extent_(0)<<","<<box->extent_(1)<<","<<box->extent_(2)<<";\n";
            instance_box_info.emplace_back(box_ss.str());
        }

        // Eigen::Vector3d pt_centroid = instance_cloud->GetCenter();
        // Eigen::Vector3d vl_centroid = instance.second->centroid;
        // std::cout<<instance.second->id_<<":"<<pt_centroid.transpose()<<";  "<<vl_centroid.transpose()<<"\n";

        o3d_utility::LogInfo("Instance {:s} has {:d} points",semantic_class_score.first, instance_cloud->points_.size());
    }

    // Sort instance info and write it to text 
    std::sort(instance_info.begin(),instance_info.end(),[](const InstanceInfo &a, const InstanceInfo &b){
        return a.first<b.first;
    });    
    std::ofstream ofs(path+"/instance_info.txt",std::ofstream::out);
    ofs<<"# instance_id;semantic_class(aggregate_score);observation_count;label_measurements;points_number\n";
    for (const auto &info:instance_info){
        ofs<<info.second;
    }
    ofs.close();

    // Sort box info and write it to text
    std::sort(instance_box_info.begin(),instance_box_info.end(),[](const std::string &a, const std::string &b){
        return std::stoi(a.substr(0,4))<std::stoi(b.substr(0,4));
    });
    std::ofstream ofs_box(path+"/instance_box.txt",std::ofstream::out);
    ofs_box<<"# instance_id;center_x,center_y,center_z;R00,R01,R02,R10,R11,R12,R20,R21,R22;extent_x,extent_y,extent_z\n";
    for (const auto &info:instance_box_info){
        ofs_box<<info;
    }
    ofs_box.close();

    // Save global instance map
    if(global_instances_pcd.points_.size()<1) return false;

    open3d::io::WritePointCloud(path+"/instance_map.ply",global_instances_pcd);
    o3d_utility::LogWarning("Save {} semantic instances to {:s}",instance_info.size(),path);

    return true;
}

// 加载instance
bool SemanticMapping::load(const std::string &path)
{
//    o3d_utility::LogInfo("Load SceneGraph from {:s}",path);
//    using namespace o3d_utility::filesystem;
//    if(!DirectoryExists(path)) return false;
//
//    // Load instance info
//    std::ifstream ifs(path+"/instance_info.txt",std::ifstream::in);
//    std::string line;
//    std::getline(ifs,line); // skip header
//    while(std::getline(ifs,line)){
//        std::stringstream ss(line);
//        std::string instance_id_str;
//        std::getline(ss,instance_id_str,';');
//        InstanceId instance_id = std::stoi(instance_id_str);
//        std::string label_score_str, observ_str, label_measurments_str, observation_count_str;
//        std::getline(ss,label_score_str,';');
//        std::getline(ss,observ_str,';');
//        std::getline(ss,label_measurments_str,';');
//
//        InstancePtr instance_toadd = std::make_shared<Instance>(instance_id,10,instance_config);
//        instance_toadd->load_previous_labels(label_measurments_str);
//        instance_toadd->load_obser_count(std::stoi(observ_str));
//        instance_toadd->point_cloud = open3d::io::CreatePointCloudFromFile(path+"/"+instance_id_str+".ply");
//        instance_toadd->centroid = instance_toadd->point_cloud->GetCenter();
//        instance_toadd->color_ = InstanceColorBar20[instance_id%InstanceColorBar20.size()];
//        if(bayesian_label){
//            Eigen::VectorXf probability_vector;
//            instance_toadd->init_bayesian_fusion(bayesian_label->get_label_vec());
//            bayesian_label->update_measurements(instance_toadd->get_measured_labels(),
//                                                probability_vector);
//            instance_toadd->update_semantic_probability(probability_vector);
//        }
//
//        instance_map.emplace(instance_id,instance_toadd);
//    }
//
//    o3d_utility::LogInfo("Load {:d} instances",instance_map.size());

    return true;
}

void SemanticMapping::export_instances(
    std::vector<InstanceId> &names, std::vector<InstancePtr> &instances, int earliest_frame_id)
{
    std::stringstream msg;
    msg<<"latest frames: ";
    for(auto &instance:instance_map){
        if(!instance.second->point_cloud) continue;
        msg<<instance.second->frame_id_<<",";
        if (instance.second->get_cloud_size() >mapping_config.shape_min_points&&
            instance.second->frame_id_>earliest_frame_id){
            names.emplace_back(instance.first);
            instances.emplace_back(instance.second);
        }
    }
    msg<<"\n";
    o3d_utility::LogInfo("{:s}",msg.str());
}

bool SemanticMapping::query_instance_info(const std::vector<InstanceId> &names,
                                        std::vector<Eigen::Vector3f> &centroids, 
                                        std::vector<std::string> &labels)
{
    for(auto &name:names){
        if(instance_map.find(name)==instance_map.end()) continue;
        auto instance = instance_map[name];
        centroids.emplace_back(instance->centroid.cast<float>());
        labels.emplace_back(instance->get_predicted_class().first);
    }

    if(centroids.size()<1) return false;
    else return true;
}


int SemanticMapping::merge_other_instances(std::vector<InstancePtr> &instances)
{
    int count = 0;
    for(auto &instance:instances){
        if(instance->point_cloud->points_.size()<mapping_config.shape_min_points) continue;
        // todo: initialize new instance instead
        instance->change_id(latest_created_instance_id+1);
        // instance->id_ = latest_created_instance_id+1;
        instance_map.emplace(instance->get_id(),instance);
        latest_created_instance_id = instance->get_id();
        count ++;
    }
    o3d_utility::LogInfo("Merge {:d} instances",count);
    return count;
}

} // namespace fmfusion
