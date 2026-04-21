
// 在包含pen3D头文件之前添加这些定义
#define FMT_HEADER_ONLY 0
#define FMT_USE_VARIADIC_TEMPLATES 1
#define FMT_USE_RVALUE_REFERENCES 1

//#include <pdal/PointTable.hpp>
//#include <pdal/PointView.hpp>
//#include <pdal/StageFactory.hpp>
//#include <pdal/io/LasReader.hpp>
//#include <pdal/io/LasHeader.hpp>
//#include <pdal/Options.hpp>

#include <Eigen/Eigen>
#include <open3d/Open3D.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <open3d/geometry/Qhull.h>
#include "json/json.h"
#include <chrono>
#include <omp.h>
#include "json/json.h"

#include "tools/Utility.h"
#include "tools/IO.h"
#include "tools/TicToc.h"
#include "mapping/SemanticMapping.h"
#include "mapping/VoxelHashMap.h"
#include "Common.h"


using namespace std;
using namespace open3d;
using namespace geometry;
using namespace camera;
//using namespace pdal;

using namespace fmfusion;

//// 读取las文件为open3d的点云
//std::shared_ptr<open3d::geometry::PointCloud> ReadLASWithPDAL(const std::string& filename) {
//    // 创建点云对象
//    auto cloud = std::make_shared<geometry::PointCloud>();
//
//    try {
//        // PDAL 读取 LAS
//        pdal::Option las_opt("filename", filename);
//        pdal::Options opts;
//        opts.add(las_opt);
//
//        pdal::LasReader reader;
//        reader.setOptions(opts);
//
//        pdal::PointTable table;
//        reader.prepare(table);
//        pdal::PointViewSet viewSet = reader.execute(table);
//
//        if (viewSet.empty()) {
//            //open3d::utility::LogWarning("Failed to read LAS file: {}", filename);
//            return cloud;
//        }
//
//        pdal::PointViewPtr view = *viewSet.begin();
//
//        // 获取点数量
//        size_t num_points = view->size();
//        if (num_points == 0) {
//            return cloud;
//        }
//
//        // 准备 Open3D 点云数据
//        cloud->points_.resize(num_points);
//        cloud->colors_.resize(num_points);
//
//        // 复制点数据
//        for (size_t i = 0; i < num_points; ++i) {
//            // 获取坐标
//            double x = view->getFieldAs<double>(pdal::Dimension::Id::X, i);
//            double y = view->getFieldAs<double>(pdal::Dimension::Id::Y, i);
//            double z = view->getFieldAs<double>(pdal::Dimension::Id::Z, i);
//
//            cloud->points_[i] = Eigen::Vector3d(x, y, z);
//
//            // 获取颜色（如果有）
//            if (view->hasDim(pdal::Dimension::Id::Red)) {
//                uint16_t r = view->getFieldAs<uint16_t>(pdal::Dimension::Id::Red, i);
//                uint16_t g = view->getFieldAs<uint16_t>(pdal::Dimension::Id::Green, i);
//                uint16_t b = view->getFieldAs<uint16_t>(pdal::Dimension::Id::Blue, i);
//
//                // 转换为 0-1 范围的浮点数
//                cloud->colors_[i] = Eigen::Vector3d(
//                    r / 65535.0,  // 16位转[0,1]
//                    g / 65535.0,
//                    b / 65535.0
//                );
//            }
//            else {
//                // 没有颜色时设为白色
//                cloud->colors_[i] = Eigen::Vector3d(1, 1, 1);
//            }
//        }
//
//        //open3d::utility::LogInfo("Successfully read {} points from LAS file.", num_points);
//
//    }
//    catch (const std::exception& e) {
//        //open3d::utility::LogError("Error reading LAS file: {}", e.what());
//    }
//
//    return cloud;
//}


//using namespace geometry;
//using namespace camera;

int main(int argc, char* argv[]){


    using namespace fmfusion;
    //using namespace std;
    using namespace open3d::utility::filesystem;
    using namespace open3d::io;


    TicToc timer;
    timer.tic();

    // 项目文件夹设置

    std::string config_file = "config.yaml";    // 配置文件
    std::string root_dir = "data/12-08";        // 输入数据根目录
    std::string prediction_folder = "prediction_no_augment";      // 图像分割结果   _mask.png和 _label.json所在文件夹
    std::string output_folder = root_dir + "/outputs/obb-merge02";  //输出文件目录
    //std::string association_name = "association_name.txt";
    //std::string trajectory_name = "trajectory.log";
    //std::string slam_name = "colorized.pcd";

    //::string output_file = "segement_1208_46.pcd";
    //std::string output_file = "removed.pcd";
    std::string slam_name = "colorized.pcd";       //输入文件夹目录下的点云名


    if (argc == 3) {
        root_dir = std::string(argv[1]);
        output_folder = std::string(argv[2]);
        std::cout << "数据输入路径: " << root_dir << std::endl;
        std::cout << "数据输出路径: " << output_folder << std::endl;
    }
    else {
        std::cout << "指向文件数指令错误，采用默认路径" << std::endl;
        std::cout << "数据输入路径: " << root_dir << std::endl;
        std::cout << "数据输出路径: " << output_folder << std::endl;
    }



    bool if_load = false;


    //统核心Inits
    fmfusion::Config *global_config;
    SemanticMapping *semantic_mapping;
    
    // 读取config
    {
    global_config = fmfusion::utility::create_scene_graph_config(root_dir + "/" + config_file, true);
    if(output_folder.size() > 0 && !DirectoryExists(output_folder))
            MakeDirectory(output_folder);
    std::ofstream out_file(output_folder + "/config.txt");
    out_file << fmfusion::utility::config_to_message(*global_config);
    out_file.close();
    }

     //读取slam文件
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    if (open3d::io::ReadPointCloud(root_dir + "/" + slam_name, *cloud)) {
        cout << "成功读取SLAM点云，点数: " << cloud->points_.size() << endl;
    } else {
        cerr << "无法读取SLAM文件" << endl;
        return 0;
    }
    //auto cloud = ReadLASWithPDAL(root_dir + "/" + slam_name);

    // 添加一个点，不然最后全是黑色，原因未知
    cloud->points_.push_back(Eigen::Vector3d(0, 0, 0));
    cloud->colors_.resize(cloud->points_.size(), Eigen::Vector3d(0.5, 0.5, 0.5));
    std::cout << "cloud对象地址: " << cloud.get() << std::endl;
    std::cout << "slam点云数量: " << cloud->points_.size() << std::endl;

    // 初始化映射模块
    semantic_mapping = new SemanticMapping(global_config->mapping_cfg, global_config->instance_cfg, cloud);
    std::cout << "初始加载耗时: " << timer.toc() << " 毫秒" << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    
    if (if_load) {
        semantic_mapping->load_all_instances_2(output_folder);
    }
    else {

        timer.tic();
        semantic_mapping->build_voxel_hashmap(0.26);
        std::cout << "构建体素网格耗时: " << timer.toc() << " 毫秒" << std::endl;

// 读取RGB数据和位姿
        std::vector<string> rgb_table;
        std::vector<Eigen::Matrix4d> pose_table;                    //位姿信息

        // 读取RGB文件路径并读取对应的位姿
        IO::construct_sorted_frame_table_3_2(root_dir, rgb_table, pose_table);

        if(rgb_table.empty()){
            std::cout << "No RGB frames found in " << root_dir.c_str() << std::endl;
            return 0;
        }
        else {
            std::cout << "find frames : " << rgb_table.size() << std::endl;
        }

    int seq_id = 0;

    for(int k=0;k< rgb_table.size();k++){
        string rgb_dir = rgb_table[k];
        string rgb_name = rgb_dir.substr(rgb_dir.find_last_of("/")+1);
        rgb_name = rgb_name.substr(0,rgb_name.find_last_of("."));
        std::cout << "================= 遍历到图像名字: " << rgb_name << " ================== " << endl;
        seq_id += 6;//字符串转整数

        // 读取RGB
        /*bool success = open3d::io::ReadImage(rgb_dir, *color);
        if (!success) {std::cerr << "错误: 无法读取图像文件 " << rgb_dir << std::endl;continue;}*/

        //加载检测结果
        bool loaded = false;
        std::vector<DetectionPtr> detections;//检测结果
        //加载预先计算好的检测结果
        timer.tic();
        loaded = fmfusion::utility::LoadPredictions(root_dir+'/'+prediction_folder, rgb_name,
                                                global_config->mapping_cfg, global_config->instance_cfg.intrinsic.width_, global_config->instance_cfg.intrinsic.height_,
                                                detections);
        //////////////
        ////读取图像分割全图掩码
        //std::string instance_file = root_dir + '/' + prediction_folder + "/" + rgb_name + "_mask.png";
        //cv::Mat detection_map = cv::imread(instance_file, -1);
        //if (detection_map.empty()) {
        //    std::cout << "错误：无法读取图像文件: " << instance_file << std::endl;
        //    loaded = false;
        //}
        /////////////////
        std::cout << "加载耗分割模型: " << timer.toc() << " 毫秒" << std::endl;

        if(!loaded) {
            std::cout << "加载二维分割结果失败"<< std::endl;
            continue;
        }
        timer.tic();
        semantic_mapping->integrate(seq_id, 
                                    pose_table[k], 
                                    //detection_map,
                                    detections);
   
        std::cout << "逻辑推理: " << timer.toc() << " 毫秒" << std::endl;

    }

    semantic_mapping->refresh_all_semantic_dict();

    //semantic_mapping->merge_floor_2(true);
    semantic_mapping->merge_ceiling(true);
    semantic_mapping->fill_instances();
    semantic_mapping->merge_overlap_instances();

    semantic_mapping->save_all_instances_2(output_folder);
    
    }
    

    timer.tic();
    semantic_mapping->ColorSlamWithInstances();
    auto removed_slam_cloud = semantic_mapping->remove_all_instances();
    std::cout << "可视化: " << timer.toc() << " 毫秒" << std::endl;
    

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start2);
    std::cout << "总时间: " << duration.count() << " 微秒" << std::endl;

    // 输出点云文件
    open3d::io::WritePointCloud(output_folder + "/segement.pcd", *cloud);
    open3d::io::WritePointCloud(output_folder + "/removed.pcd", *removed_slam_cloud);

    
    // 可视化结果
    open3d::visualization::Visualizer visualizer;
    visualizer.CreateVisualizerWindow("Camera View Coloring", 1200, 800);

    // 添加点云
    visualizer.AddGeometry(cloud);

    
    //std::vector<ObbPtr> obb_list = semantic_mapping->get_instances_obb();
    //for (auto& obbPtr : obb_list) {
    //    visualizer.AddGeometry(obbPtr);
    //}

    //std::vector<AABBPtr> aabb_list = semantic_mapping->get_instances_aabb();
    //for (auto& AABBPtr : aabb_list) {
    //    visualizer.AddGeometry(AABBPtr);
    //}


    // 添加坐标系（可选）
    auto coordinate_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame(1.0);
    visualizer.AddGeometry(coordinate_frame);

    // 设置视角以便更好地观察
    visualizer.GetViewControl().SetFront(Eigen::Vector3d(2.4395, -1.04235, 2.10451));
    visualizer.GetViewControl().SetLookat(Eigen::Vector3d(1.0, 0.0, 0.0));
    visualizer.GetViewControl().SetUp(Eigen::Vector3d(0.0, 0.0, 1.0));

    // 运行可视化
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    

    return 0;
}

