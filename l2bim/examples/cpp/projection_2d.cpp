#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace fs = std::filesystem;

namespace
{
    struct Options
    {
        std::string inputPath;
        std::string outDir;

        // If > 0, use fixed raster grid size (meter) instead of auto-infer.
        double gridSize = 0.02;

        double densityPercentile = 78.0; // Keep top-density pixels as wall candidates.
        int smoothKSize = 5;
        int closeIter = 2;
        int openIter = 1;
        int minArea = 120;
        double minAspect = 1.6;

        bool exportDebug = true;
    };

    struct RasterMeta
    {
        double minX = 0.0;
        double minY = 0.0;
        double grid = 0.02;
    };

    static pcl::PointCloud<pcl::PointXYZ>::Ptr loadCloud(const std::string &path)
    {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        if (pcl::io::loadPCDFile(path, *cloud) != 0)
        {
            throw std::runtime_error("Failed to load PCD: " + path);
        }
        if (cloud->empty())
        {
            throw std::runtime_error("Input cloud is empty: " + path);
        }
        return cloud;
    }

    static double robustAxisStep(std::vector<double> values)
    {
        if (values.size() < 2)
            return 0.0;
        std::sort(values.begin(), values.end());
        std::vector<double> diffs;
        diffs.reserve(values.size());
        for (size_t i = 1; i < values.size(); ++i)
        {
            const double d = values[i] - values[i - 1];
            if (d > 1e-6 && d < 0.5)
            {
                diffs.push_back(d);
            }
        }
        if (diffs.empty())
            return 0.0;
        const size_t mid = diffs.size() / 2;
        std::nth_element(diffs.begin(), diffs.begin() + mid, diffs.end());
        return diffs[mid];
    }

    static double inferGridSize(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        std::vector<double> xs;
        std::vector<double> ys;
        xs.reserve(cloud->size());
        ys.reserve(cloud->size());
        for (const auto &p : cloud->points)
        {
            xs.push_back(static_cast<double>(p.x));
            ys.push_back(static_cast<double>(p.y));
        }

        const double sx = robustAxisStep(xs);
        const double sy = robustAxisStep(ys);

        double g = 0.0;
        if (sx > 0.0 && sy > 0.0)
            g = std::min(sx, sy);
        else if (sx > 0.0)
            g = sx;
        else if (sy > 0.0)
            g = sy;

        if (g <= 0.0)
            g = 0.02;
        return std::clamp(g, 0.001, 0.2);
    }

    static cv::Mat rasterizeDensity(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, RasterMeta &meta, double fixedGrid)
    {
        double minX = static_cast<double>(cloud->points[0].x);
        double minY = static_cast<double>(cloud->points[0].y);
        double maxX = minX;
        double maxY = minY;
        for (const auto &p : cloud->points)
        {
            const double x = static_cast<double>(p.x);
            const double y = static_cast<double>(p.y);
            minX = std::min(minX, x);
            minY = std::min(minY, y);
            maxX = std::max(maxX, x);
            maxY = std::max(maxY, y);
        }

        double grid = fixedGrid;
        if (grid <= 0.0)
            grid = inferGridSize(cloud);
        grid = std::clamp(grid, 0.001, 0.2);
        const int rows = std::max(1, static_cast<int>(std::ceil((maxX - minX) / grid)) + 1);
        const int cols = std::max(1, static_cast<int>(std::ceil((maxY - minY) / grid)) + 1);

        cv::Mat density = cv::Mat::zeros(rows, cols, CV_32F);
        for (const auto &p : cloud->points)
        {
            const int r = static_cast<int>(std::floor((static_cast<double>(p.x) - minX) / grid));
            const int c = static_cast<int>(std::floor((static_cast<double>(p.y) - minY) / grid));
            if (r < 0 || r >= rows || c < 0 || c >= cols)
                continue;
            density.at<float>(r, c) += 1.0f;
        }

        meta.minX = minX;
        meta.minY = minY;
        meta.grid = grid;
        return density;
    }

    static double percentileNonZero(const cv::Mat &src, double percentile)
    {
        CV_Assert(src.type() == CV_32F);
        std::vector<float> vals;
        vals.reserve(static_cast<size_t>(src.rows) * static_cast<size_t>(src.cols) / 4);
        for (int r = 0; r < src.rows; ++r)
        {
            const float *ptr = src.ptr<float>(r);
            for (int c = 0; c < src.cols; ++c)
            {
                if (ptr[c] > 0.0f)
                    vals.push_back(ptr[c]);
            }
        }
        if (vals.empty())
            return 0.0;

        percentile = std::clamp(percentile, 0.0, 100.0);
        const size_t idx = static_cast<size_t>((percentile / 100.0) * static_cast<double>(vals.size() - 1));
        std::nth_element(vals.begin(), vals.begin() + idx, vals.end());
        return static_cast<double>(vals[idx]);
    }

    static cv::Mat normalizeToU8(const cv::Mat &src)
    {
        cv::Mat out;
        double minVal = 0.0, maxVal = 0.0;
        cv::minMaxLoc(src, &minVal, &maxVal);
        if (maxVal <= minVal)
        {
            src.convertTo(out, CV_8U);
            return out;
        }
        src.convertTo(out, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        return out;
    }

    static cv::Mat filterComponents(const cv::Mat &binary, int minArea, double minAspect)
    {
        CV_Assert(binary.type() == CV_8U);
        cv::Mat labels, stats, centroids;
        const int n = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

        cv::Mat out = cv::Mat::zeros(binary.size(), CV_8U);
        for (int i = 1; i < n; ++i)
        {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            if (area < minArea || w <= 0 || h <= 0)
                continue;

            const double aspect = static_cast<double>(std::max(w, h)) / static_cast<double>(std::min(w, h));
            const bool keep = (aspect >= minAspect) || (area >= 4 * minArea);
            if (!keep)
                continue;

            cv::Mat mask;
            cv::compare(labels, i, mask, cv::CMP_EQ);
            out.setTo(255, mask);
        }
        return out;
    }

    static cv::Mat morphSkeleton(const cv::Mat &binary)
    {
        CV_Assert(binary.type() == CV_8U);
        cv::Mat img;
        cv::threshold(binary, img, 0, 255, cv::THRESH_BINARY);

        cv::Mat skel = cv::Mat::zeros(img.size(), CV_8U);
        cv::Mat temp, eroded;
        const cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

        while (true)
        {
            cv::erode(img, eroded, element);
            cv::dilate(eroded, temp, element);
            cv::subtract(img, temp, temp);
            cv::bitwise_or(skel, temp, skel);
            eroded.copyTo(img);
            if (cv::countNonZero(img) == 0)
                break;
        }
        return skel;
    }

    static cv::Mat houghRefine(const cv::Mat &thin)
    {
        CV_Assert(thin.type() == CV_8U);
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(thin, lines, 1.0, CV_PI / 180.0, 30, 18.0, 6.0);

        cv::Mat out = cv::Mat::zeros(thin.size(), CV_8U);
        for (const auto &ln : lines)
        {
            cv::line(out, cv::Point(ln[0], ln[1]), cv::Point(ln[2], ln[3]), cv::Scalar(255), 1, cv::LINE_8);
        }
        return out;
    }

    static pcl::PointCloud<pcl::PointXYZ>::Ptr maskToCloud(const cv::Mat &mask, const RasterMeta &meta)
    {
        auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        if (mask.empty())
            return out;

        out->points.reserve(static_cast<size_t>(cv::countNonZero(mask)));
        for (int r = 0; r < mask.rows; ++r)
        {
            const uchar *ptr = mask.ptr<uchar>(r);
            for (int c = 0; c < mask.cols; ++c)
            {
                if (ptr[c] == 0)
                    continue;
                const float x = static_cast<float>(meta.minX + (static_cast<double>(r) + 0.5) * meta.grid);
                const float y = static_cast<float>(meta.minY + (static_cast<double>(c) + 0.5) * meta.grid);
                out->points.emplace_back(x, y, 0.0f);
            }
        }
        out->width = static_cast<uint32_t>(out->points.size());
        out->height = 1;
        out->is_dense = true;
        return out;
    }

    static void writeImage(const fs::path &path, const cv::Mat &img)
    {
        cv::imwrite(path.string(), img);
    }

    static int runProjection2D(const Options &opt)
    {
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();

        fs::create_directories(opt.outDir);

        auto cloud = loadCloud(opt.inputPath);
        std::cout << "[projection_2d] input points: " << cloud->size() << "\n";

        RasterMeta meta;
        const cv::Mat densityRaw = rasterizeDensity(cloud, meta, opt.gridSize);
        std::cout << "[projection_2d] raster: " << densityRaw.rows << " x " << densityRaw.cols
                  << ", grid=" << meta.grid << "\n";

        cv::Mat densitySmooth;
        const int k = std::max(3, opt.smoothKSize | 1);
        cv::GaussianBlur(densityRaw, densitySmooth, cv::Size(k, k), 1.2);

        const double thr = percentileNonZero(densitySmooth, opt.densityPercentile);
        cv::Mat denseMask;
        cv::threshold(densitySmooth, denseMask, thr, 255.0, cv::THRESH_BINARY);
        denseMask.convertTo(denseMask, CV_8U);

        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        if (opt.closeIter > 0)
            cv::morphologyEx(denseMask, denseMask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), opt.closeIter);
        if (opt.openIter > 0)
            cv::morphologyEx(denseMask, denseMask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), opt.openIter);

        denseMask = filterComponents(denseMask, std::max(1, opt.minArea), std::max(1.0, opt.minAspect));

        const cv::Mat thinMask = morphSkeleton(denseMask);
        const cv::Mat houghMask = houghRefine(thinMask);
        const cv::Mat &finalLineMask = (cv::countNonZero(houghMask) > 0) ? houghMask : thinMask;

        auto denseCloud = maskToCloud(denseMask, meta);
        auto thinCloud = maskToCloud(finalLineMask, meta);

        const fs::path outDir(opt.outDir);
        pcl::io::savePCDFileBinary((outDir / "walls_dense_only.pcd").string(), *denseCloud);
        pcl::io::savePCDFileBinary((outDir / "walls_line_only.pcd").string(), *thinCloud);
        pcl::io::savePCDFileBinary((outDir / "walls_2d_breakthrough.pcd").string(), *thinCloud);

        if (opt.exportDebug)
        {
            writeImage(outDir / "density_raw_u8.png", normalizeToU8(densityRaw));
            writeImage(outDir / "density_smooth_u8.png", normalizeToU8(densitySmooth));
            writeImage(outDir / "walls_dense_mask.png", denseMask);
            writeImage(outDir / "walls_thin_mask.png", thinMask);
            writeImage(outDir / "walls_hough_mask.png", houghMask);

            cv::Mat overlay;
            cv::cvtColor(normalizeToU8(densitySmooth), overlay, cv::COLOR_GRAY2BGR);
            overlay.setTo(cv::Scalar(0, 255, 255), denseMask);
            overlay.setTo(cv::Scalar(0, 0, 255), thinMask);
            overlay.setTo(cv::Scalar(0, 255, 0), finalLineMask);
            writeImage(outDir / "walls_breakthrough_overlay.png", overlay);
        }

        const auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
        std::cout << "[projection_2d] density percentile threshold=" << thr << " (p=" << opt.densityPercentile << ")\n";
        std::cout << "[projection_2d] dense mask points=" << denseCloud->size() << "\n";
        std::cout << "[projection_2d] line mask points=" << thinCloud->size() << "\n";
        std::cout << "[projection_2d] output dir: " << outDir.string() << "\n";
        std::cout << "[projection_2d] total: " << totalMs << " ms\n";
        return 0;
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: projection_2d <walls_2d_fill.pcd> <out_dir>"
                     " [densityPercentile=78] [smoothKSize=5] [closeIter=2] [openIter=1] [minArea=120] [minAspect=1.6] [gridSize=0.02]\n";
        return 1;
    }

    Options opt;
    opt.inputPath = argv[1];
    opt.outDir = argv[2];
    if (argc >= 4)
        opt.densityPercentile = std::stod(argv[3]);
    if (argc >= 5)
        opt.smoothKSize = std::stoi(argv[4]);
    if (argc >= 6)
        opt.closeIter = std::stoi(argv[5]);
    if (argc >= 7)
        opt.openIter = std::stoi(argv[6]);
    if (argc >= 8)
        opt.minArea = std::stoi(argv[7]);
    if (argc >= 9)
        opt.minAspect = std::stod(argv[8]);
    if (argc >= 10)
        opt.gridSize = std::stod(argv[9]);

    try
    {
        return runProjection2D(opt);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[projection_2d][error] " << e.what() << "\n";
        return 1;
    }
}
