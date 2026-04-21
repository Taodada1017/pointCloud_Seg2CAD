// 合并投影图像的示例代码
// 假设每个点云块的投影图像已经生成，命名为block_0_projection.png, block_1_projection.png等
// 该代码演示如何根据block_info.txt中的边界信息将这些图像合并成完整的投影图像

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <algorithm>

// 表示一个带边界信息的图像块
struct ImageBlock {
    int block_idx;
    double min_x, min_y, min_z;
    double max_x, max_y, max_z;
    int points_count;
    cv::Mat image;
};

// 从block_info.txt中读取块信息
std::vector<ImageBlock> readBlockInfo(const std::string& block_info_path) {
    std::vector<ImageBlock> blocks;
    std::ifstream file(block_info_path);
    if (!file.is_open()) {
        std::cerr << "无法打开block_info.txt文件: " << block_info_path << std::endl;
        return blocks;
    }
    
    std::string line;
    // 跳过表头
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        ImageBlock block;
        sscanf(line.c_str(), "%d,%lf,%lf,%lf,%lf,%lf,%lf,%d", 
               &block.block_idx, 
               &block.min_x, &block.min_y, &block.min_z,
               &block.max_x, &block.max_y, &block.max_z,
               &block.points_count);
        blocks.push_back(block);
    }
    
    file.close();
    return blocks;
}

// 读取投影图像
cv::Mat readProjectionImage(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "无法读取投影图像: " << image_path << std::endl;
    }
    return image;
}

// 合并投影图像
cv::Mat mergeProjectionImages(const std::vector<ImageBlock>& blocks, 
                             const std::string& image_dir, 
                             const std::string& image_prefix = "block_",
                             const std::string& image_suffix = "_projection.png",
                             double grid_size = 0.1) { // 栅格大小，默认0.02米，直接决定图像分辨率
    if (blocks.empty()) {
        std::cerr << "没有可用的图像块" << std::endl;
        return cv::Mat();
    }
    
    // 计算全局边界
    double global_min_x = std::numeric_limits<double>::max();
    double global_min_y = std::numeric_limits<double>::max();
    double global_max_x = std::numeric_limits<double>::lowest();
    double global_max_y = std::numeric_limits<double>::lowest();
    
    for (const auto& block : blocks) {
        global_min_x = std::min(global_min_x, block.min_x);
        global_min_y = std::min(global_min_y, block.min_y);
        global_max_x = std::max(global_max_x, block.max_x);
        global_max_y = std::max(global_max_y, block.max_y);
    }

    std::cout << "全局边界: " << std::endl;
    std::cout << "X[" << global_min_x << ", " << global_max_x << "]" << std::endl;
    std::cout << "Y[" << global_min_y << ", " << global_max_y << "]" << std::endl;
    
    
    // 计算图像分辨率（直接由栅格大小决定，1个栅格对应1个像素）
    int width = static_cast<int>((global_max_x - global_min_x) / grid_size) + 1;
    int height = static_cast<int>((global_max_y - global_min_y) / grid_size) + 1;

    std::cout << "全局图像分辨率: " << width << "x" << height << std::endl;
    
    // 创建最终图像（黑色背景）
    cv::Mat merged_image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // 遍历所有图像块，将它们复制到最终图像的正确位置
    for (const auto& block : blocks) {
        // 读取投影图像（支持两种路径：直接在image_dir下或在image_dir/img/下）
        std::string image_path = image_dir + "/" + image_prefix + std::to_string(block.block_idx) + image_suffix;
        cv::Mat image = readProjectionImage(image_path);
        
        // 如果直接路径不存在，尝试从img子目录读取
        if (image.empty()) {
            std::string img_subdir_path = image_dir + "/img/" + image_prefix + std::to_string(block.block_idx) + image_suffix;
            image = readProjectionImage(img_subdir_path);
            if (image.empty()) {
                continue;
            }
        }
        
        // 1. 转置图像：解决grid[0]对应行，grid[1]对应列的问题
        cv::transpose(image, image);
        
        // 2. 垂直翻转：解决OpenCV图像y轴向下，点云y轴向上的问题
        cv::flip(image, image, 0);
        
        // 计算图像块的空间范围
        double block_width = block.max_x - block.min_x;
        double block_height = block.max_y - block.min_y;
        
        // 计算图像块应该有的大小（基于栅格大小，与PointRaster.initRaster保持一致）
        // 转置+翻转后，宽高恢复正常
        int expected_width = static_cast<int>(std::ceil(block_width / grid_size)) + 2;
        int expected_height = static_cast<int>(std::ceil(block_height / grid_size)) + 2;
        
        // 调整图像大小以匹配预期大小，使用最近邻插值（与生成图像保持一致）
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(expected_width, expected_height), 0, 0, cv::INTER_NEAREST);
        
        // 计算图像块在最终图像中的位置
        int start_x = static_cast<int>(std::round((block.min_x - global_min_x) / grid_size));
        int start_y = height - static_cast<int>(std::round((block.max_y - global_min_y) / grid_size));
        
        // 获取调整后的图像大小
        int img_width = resized_image.cols;
        int img_height = resized_image.rows;
        
        // 确保不会超出最终图像的边界
        int end_x = std::min(start_x + img_width, width);
        int end_y = std::min(start_y + img_height, height);
        int actual_width = end_x - start_x;
        int actual_height = end_y - start_y;
        
        // 调试输出
        std::cout << "  块信息: 空间范围 X[" << block.min_x << ", " << block.max_x 
                  << "], Y[" << block.min_y << ", " << block.max_y << "]" << std::endl;
        std::cout << "  计算位置: start_x=" << start_x << ", start_y=" << start_y 
                  << ", 期望大小=" << expected_width << "x" << expected_height 
                  << ", 实际大小=" << img_width << "x" << img_height << std::endl;
        
        if (actual_width > 0 && actual_height > 0) {
            cv::Mat roi = merged_image(cv::Rect(start_x, start_y, actual_width, actual_height));
            cv::Mat image_roi = resized_image(cv::Rect(0, 0, actual_width, actual_height));
            
            // 处理可能的重叠区域（这里使用简单的覆盖，如果有像素不为0则替换）
            for (int y = 0; y < actual_height; y++) {
                for (int x = 0; x < actual_width; x++) {
                    cv::Vec3b pixel = image_roi.at<cv::Vec3b>(y, x);
                    if (pixel != cv::Vec3b(0, 0, 0)) { // 只有非黑色像素才替换
                        roi.at<cv::Vec3b>(y, x) = pixel;
                    }
                }
            }
        }
        
        std::cout << "已合并块 " << block.block_idx << " 到位置 (" << start_x << ", " << start_y << ")" << std::endl;
    }
    
    // 边界平滑处理：对合并后的图像进行轻微的高斯模糊，减少分块边界差异
    cv::GaussianBlur(merged_image, merged_image, cv::Size(3, 3), 0);
    
    return merged_image;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "用法: " << argv[0] << " <block_info.txt路径> <投影图像目录> [输出图像路径] [栅格大小]" << std::endl;
        std::cerr << "示例: " << argv[0] << " block_info.txt img merged.png 0.1" << std::endl;
        return 1;
    }
    
    std::string block_info_path = argv[1];
    std::string image_dir = argv[2];
    std::string output_path = (argc >= 4) ? argv[3] : "merged_projection.png";
    double grid_size = (argc >= 5) ? std::stod(argv[4]) : 0.1; // 默认栅格大小为0.02米，直接决定图像分辨率
    
    // 读取块信息
    std::vector<ImageBlock> blocks = readBlockInfo(block_info_path);
    if (blocks.empty()) {
        std::cerr << "没有读取到块信息" << std::endl;
        return 1;
    }
    
    // 合并投影图像
    cv::Mat merged_image = mergeProjectionImages(blocks, image_dir, "block_", "_projection.png", grid_size);
    if (merged_image.empty()) {
        std::cerr << "合并图像失败" << std::endl;
        return 1;
    }
    
    // 保存合并后的图像
    if (!cv::imwrite(output_path, merged_image)) {
        std::cerr << "保存合并图像失败" << std::endl;
        return 1;
    }
    
    std::cout << "合并图像已保存到: " << output_path << std::endl;
    std::cout << "合并图像大小: " << merged_image.cols << " x " << merged_image.rows << std::endl;
    std::cout << "使用的栅格大小: " << grid_size << "米" << std::endl;
    
    return 0;
}
