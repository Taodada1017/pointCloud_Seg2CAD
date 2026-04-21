#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @file img_plus.cpp
 * @brief 图像优化工具，用于测试投影图的形态学闭运算和连通区域过滤
 * 
 * 该程序的主要功能：
 * 1. 读取输入图像
 * 2. 形态学闭运算（独立函数，修复空洞与断裂）
 * 3. 连通区域面积过滤（独立函数，面积阈值与形状约束）
 * 4. 保存处理后的图像
 */

/**
 * @brief 形态学闭运算处理函数
 * 
 * @param input_img 输入二值化图像
 * @param kernel_size 闭运算核大小
 * @return cv::Mat 闭运算处理后的图像
 */
cv::Mat morphologicalCloseImage(const cv::Mat& input_img, int kernel_size = 3) {
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::Mat close_img;
    cv::morphologyEx(input_img, close_img, cv::MORPH_CLOSE, close_kernel);
    std::cout << "[img_plus] 完成形态学闭运算" << std::endl;
    return close_img;
}

/**
 * @brief 连通区域过滤函数
 * 
 * @param input_img 输入二值化图像
 * @param min_area 最小连通区域面积
 * @param max_area 最大连通区域面积
 * @param aspect_ratio_thresh 长宽比阈值
 * @param adaptive 是否启用自适应参数
 * @return cv::Mat 连通区域过滤后的图像
 */
cv::Mat filterConnectedComponents(const cv::Mat& input_img, double min_area = 50.0, 
                                 double max_area = 10000.0, double aspect_ratio_thresh = 10.0,
                                 bool adaptive = false) {
    // 自适应参数计算
    if (adaptive) {
        // 计算图像的总像素面积
        double img_area = input_img.cols * input_img.rows;
        
        // 根据图像大小动态计算min_area
        // 设置为图像面积的0.0001倍，确保小图像和大图像都有合适的阈值
        min_area = img_area * 0.0001;
        
        // 确保min_area有一个合理的最小值（避免过小）
        if (min_area < 10.0) {
            min_area = 10.0;
        }
        
        // 动态计算max_area，设置为图像面积的0.1倍，或设为无穷大
        // 这里设为图像面积的0.1倍，确保大型建筑结构能被保留
        max_area = img_area * 0.1;
        
        // 或者直接设为一个非常大的值（接近无穷大）
        // max_area = std::numeric_limits<double>::max();
        
        std::cout << "[img_plus] 自适应参数: " << std::endl;
        std::cout << "  图像大小: " << input_img.cols << " x " << input_img.rows << std::endl;
        std::cout << "  计算的min_area: " << min_area << std::endl;
        std::cout << "  计算的max_area: " << max_area << std::endl;
    }
    
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(input_img, labels, stats, centroids, 8, CV_32S);
    
    cv::Mat filtered_img = cv::Mat::zeros(input_img.size(), CV_8U);
    
    for (int i = 1; i < num_labels; i++) { // 跳过背景（标签0）
        double area = stats.at<int>(i, cv::CC_STAT_AREA);
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        
        // 计算长宽比
        double aspect_ratio = static_cast<double>(std::max(width, height)) / std::min(width, height);
        
        // 应用面积阈值和形状约束
        if (area >= min_area && area <= max_area && aspect_ratio <= aspect_ratio_thresh) {
            // 保留符合条件的连通区域
            for (int y = top; y < top + height; y++) {
                for (int x = left; x < left + width; x++) {
                    if (labels.at<int>(y, x) == i) {
                        filtered_img.at<uchar>(y, x) = 255;
                    }
                }
            }
        }
    }
    std::cout << "[img_plus] 完成连通区域过滤" << std::endl;
    return filtered_img;
}

/**
 * @brief 图像优化处理函数（组合调用三个独立函数）
 * 
 * @param input_path 输入图像路径
 * @param output_path 输出图像路径
 * @param close_kernel_size 闭运算核大小
 * @param min_area 最小连通区域面积
 * @param max_area 最大连通区域面积
 * @param aspect_ratio_thresh 长宽比阈值
 */
void optimizeImage(const std::string& input_path, const std::string& output_path,
                   int close_kernel_size = 3,
                   double min_area = 50.0, double max_area = 10000.0,
                   double aspect_ratio_thresh = 10.0) {
    // 读取输入图像
    cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: 无法读取图像文件 " << input_path << std::endl;
        return;
    }
    
    std::cout << "[img_plus] 输入图像大小: " << img.rows << " x " << img.cols << std::endl;

    cv::Mat current_img = img;
    current_img = morphologicalCloseImage(current_img, close_kernel_size);
    current_img = filterConnectedComponents(current_img, min_area, max_area, aspect_ratio_thresh);

    cv::imwrite(output_path, current_img);
    std::cout << "[img_plus] 已保存处理后的图像: " << output_path << std::endl;
}

/**
 * @brief 程序主函数
 * 
 * 命令行参数：
 *   img_plus <input_image> <output_image> [options]
 * 
 * 选项说明：
 *   -c [kernel_size]: 执行形态学闭运算
 *     kernel_size: 闭运算核大小
 *   -f [min_area] [max_area] [aspect_ratio]: 执行连通区域过滤
 *     min_area: 最小连通区域面积
 *     max_area: 最大连通区域面积
 *     aspect_ratio: 长宽比阈值
 *   -a: 启用自适应连通区域过滤参数（自动根据图像大小计算min_area和max_area）
 * 
 * 示例：
 *   仅闭运算：img_plus input.png output.png -c 3
 *   仅连通区域过滤：img_plus input.png output.png -f 50 10000 10
 *   闭运算+连通区域过滤：img_plus input.png output.png -c 3 -f 50 10000 10
 *   闭运算+自适应过滤：img_plus input.png output.png -c 3 -f 0 0 10 -a
 * 
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return int 程序执行结果，0表示成功，非0表示失败
 */
int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc < 3) {
        std::cerr << "Usage: img_plus <input_image> <output_image> [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  -c [kernel_size]: 执行形态学闭运算\n";
        std::cerr << "  -f [min_area] [max_area] [aspect_ratio]: 执行连通区域过滤\n";
        std::cerr << "  -a: 启用自适应连通区域过滤参数\n";
        std::cerr << "      自动根据图像大小计算min_area和max_area\n";
        std::cerr << "\nExamples:\n";
        std::cerr << "  Only close operation: img_plus input.png output.png -c 3\n";
        std::cerr << "  Only connected components filtering: img_plus input.png output.png -f 50 10000 10\n";
        std::cerr << "  Adaptive filtering: img_plus input.png output.png -c 3 -f 0 0 10 -a\n";
        return 1;
    }
    
    // 解析命令行参数
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    
    // 读取输入图像
    cv::Mat input_img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (input_img.empty()) {
        std::cerr << "Error: 无法读取图像文件 " << input_path << std::endl;
        return 1;
    }
    
    std::cout << "[img_plus] 输入图像大小: " << input_img.rows << " x " << input_img.cols << std::endl;
    
    // 初始化处理参数
    bool do_close = false;
    int close_kernel_size = 3;
    
    bool do_filter = false;
    double min_area = 50.0;
    double max_area = 10000.0;
    double aspect_ratio_thresh = 10.0;
    bool adaptive_filter = false;
    
    // 解析选项
    int i = 3;
    while (i < argc) {
        std::string option = argv[i];
        if (option == "-c") {
            do_close = true;
            if (i + 1 < argc) close_kernel_size = std::stoi(argv[++i]);
        } else if (option == "-f") {
            do_filter = true;
            if (i + 1 < argc) min_area = std::stod(argv[++i]);
            if (i + 1 < argc) max_area = std::stod(argv[++i]);
            if (i + 1 < argc) aspect_ratio_thresh = std::stod(argv[++i]);
        } else if (option == "-a") {
            // 启用自适应连通区域过滤参数
            adaptive_filter = true;
        } else {
            std::cerr << "Error: 未知选项 " << option << std::endl;
            return 1;
        }
        i++;
    }
    
    // 输出处理参数
    std::cout << "[img_plus] 处理参数: " << std::endl;
    if (do_close) {
        std::cout << "  闭运算: 启用" << std::endl;
        std::cout << "    核大小: " << close_kernel_size << std::endl;
    }
    if (do_filter) {
        std::cout << "  连通区域过滤: 启用" << std::endl;
        std::cout << "    最小面积: " << min_area << std::endl;
        std::cout << "    最大面积: " << max_area << std::endl;
        std::cout << "    长宽比阈值: " << aspect_ratio_thresh << std::endl;
    }
    
    try {
        // 执行图像处理
        cv::Mat current_img = input_img.clone();

        // 闭运算处理
        if (do_close) {
            current_img = morphologicalCloseImage(current_img, close_kernel_size);
        }
        
        // 连通区域过滤
        if (do_filter) {
            current_img = filterConnectedComponents(current_img, min_area, max_area, aspect_ratio_thresh, adaptive_filter);
        }
        
        // 保存处理后的图像
        cv::imwrite(output_path, current_img);
        std::cout << "[img_plus] 已保存处理后的图像: " << output_path << std::endl;
        std::cout << "[img_plus] 图像处理完成！" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
