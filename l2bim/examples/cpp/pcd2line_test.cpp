/**
 * @file pcd2line_test.cpp
 * @brief 点云转线段工具，将3D点云数据投影到2D平面并检测线段
 * 
 * 该程序的主要功能：
 * 1. 加载3D点云数据（支持PCD和PLY格式）
 * 2. 使用RANSAC算法分割地面和墙壁点云
 * 3. 将墙壁点云投影到2D平面
 * 4. 生成点云的栅格图像表示
 * 5. 使用霍夫变换检测线段
 * 6. 合并和筛选相似线段
 * 7. 检测线段交点（角点）
 * 8. 保存处理后的2D点云、栅格图像、检测到的线段和角点
 * 
 * 该工具主要用于LiDAR2BIM-Registration项目中的前端几何处理，为后续的点云配准和BIM模型生成提供基础数据。
 */

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

#include "frontend/geometry/point_cloud.h"
#include "frontend/geometry/lineseg.h"
#include "frontend/geometry/EDLines.h"
#include "utils/raster_utils.h"

namespace fs = std::filesystem;

/**
 * @brief 加载点云文件（支持PCD和PLY格式）
 * 
 * @param path 点云文件路径
 * @return pcl::PointCloud<pcl::PointXYZ>::Ptr 加载的点云数据指针
 * @throws std::runtime_error 如果文件加载失败或文件格式不支持
 */
static pcl::PointCloud<pcl::PointXYZ>::Ptr loadCloud(const std::string& path) {
    // 创建点云指针
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    
    // 根据文件扩展名选择加载方式
    if (path.size() >= 4 && path.substr(path.size()-4) == ".pcd") {
        if (pcl::io::loadPCDFile(path, *cloud) != 0) 
            throw std::runtime_error("Failed to load PCD: " + path);
    } else if (path.size() >= 4 && path.substr(path.size()-4) == ".ply") {
        if (pcl::io::loadPLYFile(path, *cloud) != 0) 
            throw std::runtime_error("Failed to load PLY: " + path);
    } else {
        throw std::runtime_error("Unsupported file extension: " + path);
    }
    
    // 检查点云是否有足够的点
    if (cloud->points.size() < 3) 
        throw std::runtime_error("Too few points in cloud");
    
    return cloud;
}

/**
 * @brief 使用RANSAC算法分割地面和墙壁点云
 * 
 * @param src 输入的点云数据
 * @param ground 输出的地面点云
 * @param walls 输出的墙壁点云
 * @param dist_thresh 距离阈值，用于判断点是否属于平面
 * @param ransac_n RANSAC算法中随机采样的点数
 * @param max_iter RANSAC算法的最大迭代次数
 * @throws std::runtime_error 如果RANSAC分割失败
 */
static void splitGroundWalls(pcl::PointCloud<pcl::PointXYZ>::Ptr src,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr& ground,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr& walls,
                             double dist_thresh=0.3, int ransac_n=3, int max_iter=1000) {
    // 创建RANSAC分割对象
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);       // 优化平面系数
    // 约束平面法向接近 Z 轴，避免把大墙面误分割成“地面”
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setAxis(Eigen::Vector3f(0.f, 0.f, 1.f));
    seg.setEpsAngle(static_cast<float>(20.0 * M_PI / 180.0));
    seg.setMethodType(pcl::SAC_RANSAC);      // 使用RANSAC算法
    seg.setMaxIterations(max_iter);          // 设置最大迭代次数
    seg.setDistanceThreshold(dist_thresh);   // 设置距离阈值
    seg.setInputCloud(src);                  // 设置输入点云

    // 执行分割
    pcl::ModelCoefficients coefficients;     // 存储平面系数
    pcl::PointIndices inliers;               // 存储平面内点的索引
    seg.segment(inliers, coefficients);      // 执行分割
    
    // 检查分割是否成功
    if (inliers.indices.empty()) 
        throw std::runtime_error("RANSAC plane segmentation failed");

    // 创建点云提取对象
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(src);

    // 初始化输出点云
    ground.reset(new pcl::PointCloud<pcl::PointXYZ>);
    walls.reset(new pcl::PointCloud<pcl::PointXYZ>);

    // 提取地面点云（平面内的点）
    pcl::PointIndices::Ptr inliersPtr(new pcl::PointIndices(inliers));
    extract.setIndices(inliersPtr);
    extract.setNegative(false);  // 提取平面内的点
    extract.filter(*ground);

    // 提取墙壁点云（平面外的点）
    extract.setNegative(true);   // 提取平面外的点
    extract.filter(*walls);
}

/**
 * @brief 将点云的XY坐标平移到正坐标系
 * 
 * @param cloud 输入的点云数据，会被直接修改
 */
static void shiftToPositiveXY(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    // 计算点云的边界框
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    
    // 创建平移变换矩阵
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T(0,3) = -min_pt.x();  // X方向平移
    T(1,3) = -min_pt.y();  // Y方向平移
    
    // 应用变换
    pcl::transformPointCloud(*cloud, *cloud, T);
}

/**
 * @brief 将3D点云转换为2D点集（只保留XY坐标）
 * 
 * @param cloud 输入的3D点云
 * @return std::vector<Eigen::Vector2d> 输出的2D点集
 */
static std::vector<Eigen::Vector2d> to2DPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    std::vector<Eigen::Vector2d> pts;
    pts.reserve(cloud->points.size());  // 预分配内存
    
    // 转换每个点的XY坐标
    for (const auto& p : cloud->points) 
        pts.emplace_back(p.x, p.y);
    
    return pts;
}

/**
 * @brief 使用EDLines算法检测线段
 * 
 * @param img 输入的灰度图像
 * @param threshold 未使用，保持接口兼容
 * @param minLineLength 最小线段长度
 * @param maxLineGap 未使用，保持接口兼容
 * @return LineSegments 检测到的线段集合
 */
static LineSegments houghLineDetection(const cv::Mat& img, int threshold=60, int minLineLength=30, int maxLineGap=30) {
    LineSegments linesegs;
    
    // 确保输入图像是灰度图
    cv::Mat grayImg;
    if (img.channels() == 3) {
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
    } else {
        grayImg = img.clone();
    }
    
    // 创建EDLines对象并检测线段
    //edlines::EDLines edLines(grayImg, 1.0, minLineLength, 6.0, 1.3);
    edlines::EDLines edLines(grayImg, 1.0, -1, 6.0, 1.3);
    
    // 获取检测到的线段
    std::vector<edlines::LS> lines = edLines.getLines();
    
    // 转换为LineSegment对象
    for (const auto& line : lines) {
        Eigen::Vector2d p1(line.start.x, line.start.y);
        Eigen::Vector2d p2(line.end.x, line.end.y);
        linesegs.append(LineSegment(p1, p2));
    }
    
    return linesegs;
}

/**
 * @brief 合并相似线段
 * 
 * @param linesegs 输入的线段集合
 * @param angleThd 角度阈值（度）
 * @param ptThd 点距离阈值
 * @return LineSegments 合并后的线段集合
 */
static LineSegments mergeSimilarLines(const LineSegments& linesegs, double angleThd=10.0, double ptThd=10.0) {
    LineSegments merged;
    
    // 遍历所有线段
    for (const auto& line : linesegs) {
        bool isMerged = false;
        
        // 检查是否可以与已合并的线段合并
        for (auto& mergedLine : merged) {
            if (line.similar(mergedLine, angleThd, ptThd)) {
                mergedLine.update(line, true);  // 平均更新线段
                isMerged = true;
                break;
            }
        }
        
        // 如果没有合并，则添加到结果中
        if (!isMerged) {
            merged.append(line);
        }
    }
    
    return merged;
}

/**
 * @brief 过滤短线段
 * 
 * @param linesegs 输入的线段集合
 * @param minLength 线段的最小长度
 * @return LineSegments 过滤后的线段集合
 */
static LineSegments filterShortLines(const LineSegments& linesegs, double minLength=60.0) {
    LineSegments filtered;
    
    // 遍历所有线段
    for (const auto& line : linesegs) {
        if (line.getLength() >= minLength) {
            filtered.append(line);
        }
    }
    
    return filtered;
}

/**
 * @brief 保存线段到文件
 * 
 * @param linesegs 线段集合
 * @param filePath 文件路径
 */
static void saveLinesToFile(const LineSegments& linesegs, const std::string& filePath) {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    
    // 写入线段数量
    int count = 0;
    for (const auto& line : linesegs) count++;
    file << count << std::endl;
    
    // 写入每个线段
    for (const auto& line : linesegs) {
        Eigen::Vector2d p1 = line.getPointA();
        Eigen::Vector2d p2 = line.getPointB();
        file << p1.x() << " " << p1.y() << " " << p2.x() << " " << p2.y() << std::endl;
    }
    
    file.close();
}

/**
 * @brief 保存角点到文件
 * 
 * @param corners 角点集合
 * @param filePath 文件路径
 */
static void saveCornersToFile(const std::vector<CornerPtr>& corners, const std::string& filePath) {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    
    // 写入角点数量
    file << corners.size() << std::endl;
    
    // 写入每个角点
    for (const auto& corner : corners) {
        if (corner) {
            Eigen::Vector2d pos = corner->getPosition();
            file << pos.x() << " " << pos.y() << std::endl;
        }
    }
    
    file.close();
}

/**
 * @brief 程序主函数
 * 
 * 命令行参数：
 *   pcd2line_test <input.pcd/ply> <out_dir> [gridSize] [lowerZ] [upperZ] [voxel] [minLineLength]
 * 
 * 参数说明：
 *   input.pcd/ply: 输入的3D点云文件路径
 *   out_dir: 输出目录路径
 *   gridSize: 可选，栅格图像的网格大小，默认0.05
 *   lowerZ: 可选，高度带的下限，默认-1.5
 *   upperZ: 可选，高度带的上限，默认-1.0
 *   voxel: 可选，体素大小，默认0.01
 *   minLineLength: 可选，最小线段长度，默认30
 * 
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return int 程序执行结果，0表示成功，非0表示失败
 */
int main(int argc, char** argv) {
    // 记录程序开始时间
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    // 检查命令行参数
    if (argc < 3) {
        std::cerr << "Usage: pcd2line_test <input.pcd/ply> <out_dir> [gridSize] [lowerZ] [upperZ] [voxel] [minLineLength]\n";
        return 1;
    }
    
    // 解析命令行参数
    std::string in_path = argv[1];
    std::string out_dir = argv[2];
    //in_path="./data/input_other.pcd";
    //out_dir="./output/detectlines_once";  //固定参数 输出目录
    double gridSize = (argc >= 4) ? std::stod(argv[3]) : 0.05;  // 默认网格大小：0.05
    float lowerZ = (argc >= 5) ? static_cast<float>(std::stod(argv[4])) : -1.5f;  // 默认高度下限：-1.5
    float upperZ = (argc >= 6) ? static_cast<float>(std::stod(argv[5])) : -1.0f;  // 默认高度上限：-1.0
    float voxel = (argc >= 7) ? static_cast<float>(std::stod(argv[6])) : 0.01f;   // 默认体素大小：0.01
    int minLineLength = (argc >= 8) ? std::stoi(argv[7]) : 30;  // 默认最小线段长度：30

    // 创建输出目录
    fs::create_directories(out_dir);

    try {
        // 加载点云数据
        auto cloud = loadCloud(in_path);
        std::cout << "[pcd2line] loaded cloud: " << cloud->points.size() << " points\n";
        
        // 分割地面和墙壁
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground, walls;
        splitGroundWalls(cloud, ground, walls);
        std::cout << "[pcd2line] ground: " << ground->points.size() << " | walls: " << walls->points.size() << "\n";

        // 创建PCD2d对象，用于3D到2D的投影
        PCD2d pcd2d(walls, ground);
        pcd2d.SetHeightBand(lowerZ, upperZ);  // 设置高度带范围
        pcd2d.SetVoxelSize(voxel);             // 设置体素大小
        
        std::cout << "[pcd2line] height band: [" << lowerZ << ", " << upperZ << "], voxel: " << voxel << "\n";
        
        // 执行3D到2D的投影
        if (!pcd2d.GetPcd2d("default")) {
            std::cerr << "PCD2d.GetPcd2d() failed\n";
            return 1;
        }
        
        // 获取投影后的2D墙壁点云
        auto walls2d = pcd2d.getWalls2d();
        if (!walls2d || walls2d->points.empty()) {
            std::cerr << "walls2d empty\n";
            return 1;
        }
        std::cout << "[pcd2line] walls2d after flatten: " << walls2d->points.size() << " points\n";

        // 计算点云边界
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*walls2d, min_pt, max_pt);

        // 计算图像尺寸
        int img_width = std::ceil((max_pt.x() - min_pt.x()) / gridSize + 2);
        int img_height = std::ceil((max_pt.y() - min_pt.y()) / gridSize + 2);

        printf("[pcd2line] 原始点云数据 projected walls2d bounds: X[%.2f, %.2f], Y[%.2f, %.2f], size: %d x %d\n",
               min_pt.x(), max_pt.x(), min_pt.y(), max_pt.y(), img_width, img_height);
        
        // 将2D点云平移到正坐标系
        shiftToPositiveXY(walls2d);

        pcl::getMinMax3D(*walls2d, min_pt, max_pt);

        // 计算图像尺寸
        img_width = std::ceil((max_pt.x() - min_pt.x()) / gridSize + 2);
        img_height = std::ceil((max_pt.y() - min_pt.y()) / gridSize + 2);

        printf("[pcd2line] 移动到正坐标系后 projected walls2d bounds: X[%.2f, %.2f], Y[%.2f, %.2f], size: %d x %d\n",
               min_pt.x(), max_pt.x(), min_pt.y(), max_pt.y(), img_width, img_height);

        // 保存2D点云数据
        std::string out_pcd = (fs::path(out_dir) / "walls_2d.pcd").string();
        pcl::io::savePCDFileBinary(out_pcd, *walls2d);
        std::cout << "[pcd2line] saved 2D point cloud: " << out_pcd << "\n";

        // 将2D点云转换为Eigen向量集合
        auto pts2d = to2DPoints(walls2d);
        
        // 创建栅格化对象
        PointRaster raster(pts2d, gridSize);
        raster.rasterize(pts2d, 1.0, true, 0);  // 执行栅格化

        // 将栅格数据转换为图像
        cv::Mat imgF = raster.toImage(false, 0);
        double minVal = 0.0, maxVal = 0.0;
        cv::minMaxLoc(imgF, &minVal, &maxVal);
        
        // 归一化图像到0-255范围
        if (maxVal > 0.0) {
            imgF.convertTo(imgF, CV_8U, 255.0 / maxVal);
        } else {
            imgF.convertTo(imgF, CV_8U);
        }
        
        // 反色，使点变为白色，背景为黑色
        cv::Mat img = imgF;
        
        std::cout << "[pcd2line] raster image size: " << img.rows << " x " << img.cols
                  << ", maxVal: " << maxVal << "\n";
        
        // 保存栅格图像
        // 如果需要白色背景和黑色墙壁，可以取消下面一行的注释
        // cv::bitwise_not(img, img);
        std::string out_png = (fs::path(out_dir) / "projection.png").string();
        cv::imwrite(out_png, img);
        std::cout << "[pcd2line] saved projection image: " << out_png << "\n";

        // 线段检测：HoughLinesP 更适合输入边缘图/二值边界。
        // 投影图是“点密度/强度”灰度图，直接做霍夫会不稳定，因此先进行预处理再提边。
        
        // 1. 对比度增强：使用CLAHE解决光照不均匀问题，降低clipLimit避免过度增强
        // cv::Mat img_clahe;
        // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.0, cv::Size(8, 8));
        // clahe->apply(img, img_clahe);
        
        // // 2. 保边去噪：使用双边滤波，降低滤波强度以保留更多边缘细节
        // cv::Mat img_bilateral;
        // cv::bilateralFilter(img_clahe, img_bilateral, 5, 50, 50);
        
        // // 3. 二值化处理：调整阈值以适应点云密度
        // cv::Mat img_bin;
        // cv::threshold(img_bilateral, img_bin, 50, 255, cv::THRESH_BINARY);
        
        // // 4. 边缘检测：Canny算子提取边缘，降低阈值以检测更多边缘
        // cv::Mat edges;
        // cv::Canny(img_bilateral, edges, 30, 100, 3);

        // 检测线段
        LineSegments linesegs = houghLineDetection(img, 60, minLineLength, 30);
        int rawLines = 0;
        for (const auto& line : linesegs) rawLines++;
        std::cout << "[pcd2line] detected " << rawLines << " raw lines\n";

        // 合并相似线段
        // LineSegments mergedLines = mergeSimilarLines(linesegs, 10.0, 10.0);
        // int mergedCount = 0;
        // for (const auto& line : mergedLines) mergedCount++;
        // std::cout << "[pcd2line] merged to " << mergedCount << " lines\n";

        // // 过滤短线段（与命令行 minLineLength 保持一致）
        // LineSegments filteredLines = filterShortLines(mergedLines, static_cast<double>(minLineLength));
        // int filteredCount = 0;
        // for (const auto& line : filteredLines) filteredCount++;
        // std::cout << "[pcd2line] filtered to " << filteredCount << " lines\n";

        // 创建彩色图像版本用于绘制线段
        cv::Mat img_color;
        cv::cvtColor(img, img_color, cv::COLOR_GRAY2BGR);
        int lineType = cv::LINE_AA;
        // 在彩色图像上绘制检测到的线段（红色）
        for (const auto& line : linesegs) {
            Eigen::Vector2d p1 = line.getPointA();
            Eigen::Vector2d p2 = line.getPointB();
            cv::Point pt1(static_cast<int>(p1.x()), static_cast<int>(p1.y()));
            cv::Point pt2(static_cast<int>(p2.x()), static_cast<int>(p2.y()));
            cv::line(img_color, pt1, pt2, cv::Scalar(0, 0, 255), 1, lineType); // 红色，线宽1
        }

        // 保存带有线段的彩色图像
        std::string out_png_lines = (fs::path(out_dir) / "projection_with_lines.png").string();
        cv::imwrite(out_png_lines, img_color);
        std::cout << "[pcd2line] saved projection with lines: " << out_png_lines << "\n";

        // 检测角点
        std::vector<CornerPtr> corners = linesegs.intersections(0.1);
        std::cout << "[pcd2line] detected " << corners.size() << " corners\n";

        // 保存线段和角点 (保存角点的函数还没写)
        std::string out_raw_lines = (fs::path(out_dir) / "lines_raw.txt").string();
        std::string out_merged_lines = (fs::path(out_dir) / "lines_merged.txt").string();
        std::string out_corners = (fs::path(out_dir) / "corners.txt").string();
        
        // saveLinesToFile(linesegs, out_raw_lines);
        // saveLinesToFile(filteredLines, out_merged_lines);
        // saveCornersToFile(corners, out_corners);
        saveLinesToFile(linesegs, out_raw_lines);
        saveCornersToFile(corners, out_corners);
        
        std::cout << "[pcd2line] saved raw lines: " << out_raw_lines << "\n";
        //std::cout << "[pcd2line] saved merged lines: " << out_merged_lines << "\n";
        std::cout << "[pcd2line] saved corners: " << out_corners << "\n";

        // 计算并输出程序运行时间
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
        std::cout << "[timing] total: " << total_ms << " ms\n";

    } catch (const std::exception& e) {
        // 捕获并处理异常
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}