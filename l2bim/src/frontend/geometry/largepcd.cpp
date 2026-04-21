// 基础依赖：PCL + Eigen + OpenCV + C++标准库（添加OpenCV用于生成投影图像）
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm> // 新增：用于空间排序
#include <opencv2/opencv.hpp>

// 项目自定义头文件
#include <utils/raster_utils.h>

// PCL核心头文件
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// Eigen数学库
#include <Eigen/Core>

// -------------------------- 核心适配：复刻VoxelMapPlus的VOXEL_LOC结构 --------------------------
struct VOXEL_LOC {
    int64_t x, y, z;

    VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

    friend bool operator==(const VOXEL_LOC& a, const VOXEL_LOC& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
};

// 为VOXEL_LOC实现哈希器（用于unordered_map）
namespace std {
    template<> struct hash<VOXEL_LOC> {
        size_t operator()(const VOXEL_LOC& loc) const {
            size_t h1 = hash<int64_t>()(loc.x);
            size_t h2 = hash<int64_t>()(loc.y);
            size_t h3 = hash<int64_t>()(loc.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

// -------------------------- 类型定义 + 配置参数 --------------------------
// 适配无特征点云：仅保留x/y/z/intensity（删除normal/curvature）
using PointType = pcl::PointXYZI;          
using PointCloudPtr = pcl::PointCloud<PointType>::Ptr; // 点云智能指针

// 表示一个带边界信息的点云块
struct CloudBlock {
    PointCloudPtr cloud;                 // 点云数据
    double min_x, min_y, min_z;          // 边界框最小值
    double max_x, max_y, max_z;          // 边界框最大值
    CloudBlock() : min_x(0), min_y(0), min_z(0), max_x(0), max_y(0), max_z(0) {
        cloud = PointCloudPtr(new pcl::PointCloud<PointType>());
    }
};

// 全局配置（可直接修改参数）
struct SplitConfig {
    double voxel_size = 0.5;                // 基础体素尺寸（会自动调大减少小体素）
    std::string input_pcd = "/path/to/your/large_cloud.pcd"; // 输入千万级点云路径
    std::string output_dir = "./split_blocks/";              // 分块保存目录
    int min_points_per_block = 500000;      // 单块最小点数（50万）
    int max_points_per_block = 2000000;     // 单块最大点数（200万）
    double grid_size = 0.1;                 // 投影图像栅格大小（米）
};

// -------------------------- 工具函数 --------------------------
// 创建目录（跨平台兼容）
bool createDir(const std::string& dir_path) {
#ifdef _WIN32
    std::string cmd = "md " + dir_path;
#else
    std::string cmd = "mkdir -p " + dir_path;
#endif
    return system(cmd.c_str()) == 0;
}

// -------------------------- 核心函数：按空间坐标排序体素（解决重叠问题） --------------------------
std::vector<std::pair<VOXEL_LOC, PointCloudPtr>> sortVoxelsBySpatialPosition(
    std::unordered_map<VOXEL_LOC, PointCloudPtr>& voxel_clouds
) {
    // 转换为vector以便排序
    std::vector<std::pair<VOXEL_LOC, PointCloudPtr>> sorted_voxels(voxel_clouds.begin(), voxel_clouds.end());

    // 按x→y→z优先级排序（保证空间相邻的体素连续遍历）
    std::sort(sorted_voxels.begin(), sorted_voxels.end(),
        [](const std::pair<VOXEL_LOC, PointCloudPtr>& a, 
           const std::pair<VOXEL_LOC, PointCloudPtr>& b) {
            if (a.first.x != b.first.x) return a.first.x < b.first.x;
            if (a.first.y != b.first.y) return a.first.y < b.first.y;
            return a.first.z < b.first.z;
        }
    );

    return sorted_voxels;
}

// -------------------------- 核心函数：合并小体素块（保证点数范围+无空间重叠） --------------------------
std::vector<CloudBlock> mergeSmallVoxels(
    std::unordered_map<VOXEL_LOC, PointCloudPtr>& voxel_clouds,
    int min_points,
    int max_points,
    double voxel_size
) {
    std::vector<CloudBlock> final_blocks;
    CloudBlock temp_block;
    temp_block.cloud->reserve(max_points); // 预分配内存

    // 关键修复：先按空间坐标排序体素，避免重叠
    auto sorted_voxels = sortVoxelsBySpatialPosition(voxel_clouds);

    // 遍历排序后的体素，累积点数到目标范围
    for (auto& [voxel_loc, cloud] : sorted_voxels) {
        // 跳过空体素
        if (cloud->empty()) continue;

        // 计算当前体素的边界
        double v_min_x = voxel_loc.x * voxel_size;
        double v_min_y = voxel_loc.y * voxel_size;
        double v_min_z = voxel_loc.z * voxel_size;
        double v_max_x = (voxel_loc.x + 1) * voxel_size;
        double v_max_y = (voxel_loc.y + 1) * voxel_size;
        double v_max_z = (voxel_loc.z + 1) * voxel_size;

        // 若当前临时块+当前体素块超过最大值 → 保存临时块
        if (temp_block.cloud->size() + cloud->size() > max_points) {
            // 临时块点数达标则保存
            if (temp_block.cloud->size() >= min_points) {
                final_blocks.push_back(temp_block);
                temp_block = CloudBlock();
                temp_block.cloud->reserve(max_points);
                // 初始化新临时块的边界
                temp_block.min_x = v_min_x;
                temp_block.min_y = v_min_y;
                temp_block.min_z = v_min_z;
                temp_block.max_x = v_max_x;
                temp_block.max_y = v_max_y;
                temp_block.max_z = v_max_z;
            }
        } else if (temp_block.cloud->empty()) {
            // 初始化第一个体素的边界
            temp_block.min_x = v_min_x;
            temp_block.min_y = v_min_y;
            temp_block.min_z = v_min_z;
            temp_block.max_x = v_max_x;
            temp_block.max_y = v_max_y;
            temp_block.max_z = v_max_z;
        } else {
            // 更新临时块的边界
            temp_block.min_x = std::min(temp_block.min_x, v_min_x);
            temp_block.min_y = std::min(temp_block.min_y, v_min_y);
            temp_block.min_z = std::min(temp_block.min_z, v_min_z);
            temp_block.max_x = std::max(temp_block.max_x, v_max_x);
            temp_block.max_y = std::max(temp_block.max_y, v_max_y);
            temp_block.max_z = std::max(temp_block.max_z, v_max_z);
        }
        // 合并当前体素块到临时块（空间连续，无重叠）
        *temp_block.cloud += *cloud;
    }

    // 处理最后一个临时块
    if (temp_block.cloud->size() >= min_points) {
        final_blocks.push_back(temp_block);
    }

    std::cout << "\n[合并小体素完成] " << std::endl;
    std::cout << "原始体素块数: " << voxel_clouds.size() << std::endl;
    std::cout << "最终分块数: " << final_blocks.size() << std::endl;
    std::cout << "单块点数限制: " << min_points << " ~ " << max_points << std::endl;

    return final_blocks;
}

// -------------------------- 核心函数：体素分割+点数限制+无空间重叠 --------------------------
std::vector<CloudBlock> splitLargeCloudWithLimit(
    const PointCloudPtr& large_cloud,
    const SplitConfig& config
) {
    // 1. 初始化体素映射
    std::unordered_map<VOXEL_LOC, PointCloudPtr> voxel_clouds;
    double adjusted_voxel_size = config.voxel_size * 2; // 调大体素尺寸，减少小体素

    // 2. 按体素分割超大点云
    std::cout << "[开始体素分割] 基础体素尺寸: " << config.voxel_size 
              << " | 调整后体素尺寸: " << adjusted_voxel_size << std::endl;
    for (const auto& point : large_cloud->points) {
        // 跳过无效点
        if (std::isnan(point.x) || std::isinf(point.x) ||
            std::isnan(point.y) || std::isinf(point.y) ||
            std::isnan(point.z) || std::isinf(point.z)) {
            continue;
        }

        // 计算体素坐标
        int64_t vx = static_cast<int64_t>(floor(point.x / adjusted_voxel_size));
        int64_t vy = static_cast<int64_t>(floor(point.y / adjusted_voxel_size));
        int64_t vz = static_cast<int64_t>(floor(point.z / adjusted_voxel_size));
        VOXEL_LOC voxel_loc(vx, vy, vz);

        // 初始化体素点云
        if (voxel_clouds.find(voxel_loc) == voxel_clouds.end()) {
            voxel_clouds[voxel_loc] = PointCloudPtr(new pcl::PointCloud<PointType>());
            voxel_clouds[voxel_loc]->reserve(config.max_points_per_block);
        }

        // 添加点（保留原始特征：x/y/z/intensity）
        voxel_clouds[voxel_loc]->push_back(point);
    }
    std::cout << "[体素分割完成] 生成原始体素块数: " << voxel_clouds.size() << std::endl;

    // 3. 合并小体素块（空间排序+点数限制+无重叠）
    return mergeSmallVoxels(voxel_clouds, config.min_points_per_block, config.max_points_per_block, adjusted_voxel_size);
}

// -------------------------- 核心函数：生成点云投影图像 --------------------------
// 重载版本：使用全局原点确保所有分块图像坐标系对齐
void generateProjectionImage(const CloudBlock& block, const std::string& output_dir, int block_idx, 
                            const Eigen::Vector2d& global_origin, double gridSize = 0.1) {
    // 创建img子文件夹路径
    std::string img_dir = output_dir + "/img";
    
    // 创建img子文件夹
    createDir(img_dir);
    
    // 将3D点云转换为2D点向量（投影到XY平面）
    std::vector<Eigen::Vector2d> pts2d;
    for (const auto& point : block.cloud->points) {
        pts2d.emplace_back(point.x, point.y);
    }
    
    // 创建PointRaster对象
    PointRaster raster(gridSize);
    // 设置全局原点，确保所有分块使用相同坐标系
    raster.setOrigin(global_origin);
    // 初始化栅格
    raster.initRaster(pts2d);
    // 栅格化点云
    raster.rasterize(pts2d, 1.0, true, 0); // 栅格化点云，invert=1.0，override=true，width=0
    
    // 生成图像
    cv::Mat imgF = raster.toImage(false, 0);
    
    // 归一化图像到0-255范围
    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(imgF, &minVal, &maxVal);
    
    cv::Mat img;
    if (maxVal > 0.0) {
        imgF.convertTo(img, CV_8U, 255.0 / maxVal);
    } else {
        imgF.convertTo(img, CV_8U);
    }
    
    // 生成图像文件名，包含栅格大小信息
    std::string image_path = img_dir + "/block_" + std::to_string(block_idx) + "_projection.png";
    
    // 保存图像
    if (!cv::imwrite(image_path, img)) {
        std::cerr << "[警告] 保存投影图像失败: " << image_path << std::endl;
        return;
    }
    
    std::cout << "[保存投影图像] " << image_path << " | 分辨率: " << img.cols << "x" << img.rows << std::endl;
    std::cout << "          栅格大小: " << gridSize << "米" << std::endl;
    std::cout << "          全局原点: (" << global_origin.x() << ", " << global_origin.y() << ")" << std::endl;
}

// 兼容旧版本的函数包装
void generateProjectionImage(const CloudBlock& block, const std::string& output_dir, int block_idx) {
    // 计算当前分块的原点（兼容旧版本，不推荐使用）
    Eigen::Vector2d block_origin(block.min_x, block.min_y);
    generateProjectionImage(block, output_dir, block_idx, block_origin);
}

// -------------------------- 核心函数：保存分块点云 --------------------------
void saveBlocks(const std::vector<CloudBlock>& final_blocks, const SplitConfig& config) {
    // 创建输出目录
    if (!createDir(config.output_dir)) {
        std::cerr << "[错误] 无法创建输出目录: " << config.output_dir << std::endl;
        return;
    }
    
    // 计算全局边界（用于生成投影图像时的全局原点）
    double global_min_x = std::numeric_limits<double>::max();
    double global_min_y = std::numeric_limits<double>::max();
    for (const auto& block : final_blocks) {
        global_min_x = std::min(global_min_x, block.min_x);
        global_min_y = std::min(global_min_y, block.min_y);
    }
    Eigen::Vector2d global_origin(global_min_x, global_min_y);
    
    std::cout << "[全局边界] min_x=" << global_min_x << ", min_y=" << global_min_y << std::endl;
    std::cout << "[全局原点] (" << global_origin.x() << ", " << global_origin.y() << ")" << std::endl;

    // 遍历分块保存
    int block_idx = 0;
    std::ofstream block_info(config.output_dir + "/block_info.txt");
    block_info << "block_idx,min_x,min_y,min_z,max_x,max_y,max_z,points_count\n";
    
    for (const auto& block : final_blocks) {
        if (block.cloud->empty()) continue;

        // 生成文件名（包含点数，方便查看）
        std::string save_path = config.output_dir 
            + "/block_" + std::to_string(block_idx)
            + "_points_" + std::to_string(block.cloud->size()) + ".pcd";

        // 保存二进制PCD（高效+保留所有特征）
        if (pcl::io::savePCDFileBinary(save_path, *block.cloud) != 0) {
            std::cerr << "[警告] 保存分块 " << block_idx << " 失败" << std::endl;
            continue;
        }

        // 保存块的边界信息
        block_info << block_idx << ","
                   << block.min_x << "," << block.min_y << "," << block.min_z << ","
                   << block.max_x << "," << block.max_y << "," << block.max_z << ","
                   << block.cloud->size() << "\n";

        // 打印空间范围信息
        std::cout << "[保存分块] " << save_path << " | 点数: " << block.cloud->size() << std::endl;
        std::cout << "          空间范围: X[" << block.min_x << ", " << block.max_x 
                  << "], Y[" << block.min_y << ", " << block.max_y 
                  << "], Z[" << block.min_z << ", " << block.max_z << "]" << std::endl;
        
        // 生成并保存投影图像（使用全局原点和配置的栅格大小）
        generateProjectionImage(block, config.output_dir, block_idx, global_origin, config.grid_size);
        
        block_idx++;
    }
    block_info.close();

    // 生成文件列表（用于后续合并）
    std::ofstream file_list(config.output_dir + "/file_list.txt");
    for (int i = 0; i < final_blocks.size(); i++) {
        if (final_blocks[i].cloud->empty()) continue;
        file_list << config.output_dir << "/block_" << i 
                  << "_points_" << final_blocks[i].cloud->size() << ".pcd\n";
    }
    file_list.close();
}

// -------------------------- 核心函数：合并分块验证 --------------------------
PointCloudPtr mergeBlocks(const SplitConfig& config) {
    PointCloudPtr merged_cloud(new pcl::PointCloud<PointType>());
    merged_cloud->reserve(50000000); // 预分配千万级内存

    // 读取分块列表
    std::ifstream file_list(config.output_dir + "/file_list.txt");
    if (!file_list.is_open()) {
        std::cerr << "[错误] 无法读取分块列表: " << config.output_dir << "/file_list.txt" << std::endl;
        return merged_cloud;
    }

    // 合并所有分块
    std::string file_path;
    int block_count = 0;
    while (std::getline(file_list, file_path)) {
        PointCloudPtr block_cloud(new pcl::PointCloud<PointType>());
        if (pcl::io::loadPCDFile<PointType>(file_path, *block_cloud) != 0) {
            std::cerr << "[警告] 加载分块失败: " << file_path << std::endl;
            continue;
        }
        *merged_cloud += *block_cloud;
        block_count++;
    }
    file_list.close();

    // 保存合并后的点云
    std::string merged_path = config.output_dir + "/merged_cloud.pcd";
    pcl::io::savePCDFileBinary(merged_path, *merged_cloud);
    std::cout << "\n[合并完成] " << std::endl;
    std::cout << "合并分块数: " << block_count << std::endl;
    std::cout << "合并后总点数: " << merged_cloud->size() << std::endl;
    std::cout << "合并结果保存路径: " << merged_path << std::endl;

    return merged_cloud;
}

// -------------------------- 主函数（流程入口） --------------------------
int main(int argc, char** argv) {
    // 1. 配置参数（修改这里的路径和点数限制）
    SplitConfig config;
    config.voxel_size = 0.5;                // 基础体素尺寸
    config.input_pcd = "./data/input_other.pcd"; // 替换为你的千万级点云路径
    config.output_dir = "./output_large";   // 分块保存目录
    config.min_points_per_block = 500000;   // 单块最小50万点
    config.max_points_per_block = 2000000;  // 单块最大200万点
    config.grid_size = 0.1;                 // 投影图像栅格大小（米），与合并时保持一致
    
    // 2. 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "用法: " << argv[0] << " [选项]" << std::endl;
            std::cout << "选项:" << std::endl;
            std::cout << "  --input <path>      设置输入点云路径" << std::endl;
            std::cout << "  --output <path>     设置输出目录" << std::endl;
            std::cout << "  --voxel <size>      设置体素尺寸（默认: 0.5）" << std::endl;
            std::cout << "  --min_points <num>  设置单块最小点数（默认: 500000）" << std::endl;
            std::cout << "  --max_points <num>  设置单块最大点数（默认: 2000000）" << std::endl;
            std::cout << "  --grid <size>       设置投影图像栅格大小（米，默认: 0.1）" << std::endl;
            std::cout << "  --help, -h          显示帮助信息" << std::endl;
            return 0;
        } else if (arg == "--input" && i + 1 < argc) {
            config.input_pcd = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--voxel" && i + 1 < argc) {
            config.voxel_size = std::stod(argv[++i]);
        } else if (arg == "--min_points" && i + 1 < argc) {
            config.min_points_per_block = std::stoi(argv[++i]);
        } else if (arg == "--max_points" && i + 1 < argc) {
            config.max_points_per_block = std::stoi(argv[++i]);
        } else if (arg == "--grid" && i + 1 < argc) {
            config.grid_size = std::stod(argv[++i]);
        } else {
            std::cerr << "未知选项: " << arg << std::endl;
            std::cerr << "使用 --help 查看用法" << std::endl;
            return 1;
        }
    }
    
    // 打印配置信息
    std::cout << "[配置信息]" << std::endl;
    std::cout << "输入点云路径: " << config.input_pcd << std::endl;
    std::cout << "输出目录: " << config.output_dir << std::endl;
    std::cout << "体素尺寸: " << config.voxel_size << std::endl;
    std::cout << "单块点数范围: " << config.min_points_per_block << " ~ " << config.max_points_per_block << std::endl;
    std::cout << "投影栅格大小: " << config.grid_size << " 米" << std::endl;

    // 2. 加载超大点云（适配无特征的PointXYZI格式）
    std::cout << "[开始加载点云] 路径: " << config.input_pcd << std::endl;
    PointCloudPtr large_cloud(new pcl::PointCloud<PointType>());
    if (pcl::io::loadPCDFile<PointType>(config.input_pcd, *large_cloud) != 0) {
        std::cerr << "[错误] 加载点云失败！请检查路径是否正确: " << config.input_pcd << std::endl;
        return -1;
    }
    std::cout << "[加载完成] 输入点云总点数: " << large_cloud->size() << std::endl;

    // 3. 分割点云（体素分割+空间排序+点数限制+无重叠）
    std::vector<CloudBlock> final_blocks = splitLargeCloudWithLimit(large_cloud, config);

    // 4. 保存分块点云
    std::cout << "\n[开始保存分块]..." << std::endl;
    saveBlocks(final_blocks, config);

    // 5. 合并分块验证（可选，注释掉可跳过）
    mergeBlocks(config);

    return 0;
}