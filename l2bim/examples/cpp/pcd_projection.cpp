#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#if defined(__has_include)
#if __has_include(<opencv2/ximgproc.hpp>)
#include <opencv2/ximgproc.hpp>
#define HAS_OPENCV_XIMGPROC 1
#else
#define HAS_OPENCV_XIMGPROC 0
#endif
#else
#define HAS_OPENCV_XIMGPROC 0
#endif
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

#include "frontend/geometry/point_cloud.h"
#include "utils/raster_utils.h"

namespace fs = std::filesystem;

namespace
{

    inline double kPi()
    {
        return std::acos(-1.0);
    }

    struct ProjectionOptions
    {
        std::string inputPath;
        std::string outDir;

        double gridSize = 0.05;
        double resolutionScale = 2.0;
        float lowerZ = -1.5f;
        float upperZ = -1.0f;
        float voxel = 0.01f;

        // Multi-slice voting for robust wall projection.
        int numSlices = 10;
        int minVotes = 6;

        // Ground segmentation
        double groundDistThresh = 0.3;
        int groundMaxIter = 1000;
        int groundRansacN = 3;

        bool shiftToPositiveXY = true;

        // Output a thin binary line map (edges) without changing the original projection.png.
        // This avoids large filled blobs while still providing topology-stable line inputs.
        bool saveLineBinary = true;

        // Occupancy threshold on projection (single stroke like projection.png)
        // pixels > occThreshold become 255
        int occThreshold = 5;

        // Denoise before line extraction
        int medianKSize = 3;

        // Optional morphology on the produced binary map
        int binDilate = 0; // thicken a bit
        int binClose = 1;  // connect tiny gaps (iterations)

        // Extra post-process (ported from img_plus)
        int closeKernelSize = 7; // kernel size for MORPH_CLOSE
        bool ccEnable = false;   // connected components filtering
        double ccMinArea = 50.0;
        double ccMaxArea = 10000.0;
        double ccAspectRatio = 10.0;
        bool ccAdaptive = false;

        // Export line-like 2D pcd from raster boundary and use it as walls_2d.pcd.
        // The original filled 2d cloud will still be saved as walls_2d_fill.pcd.
        bool exportLineLikePcd = true;

        // Skeletonization + line fitting
        bool useXimgprocThinning = true;
        int houghThreshold = 35;
        double houghMinLineLength = 25.0;
        double houghMaxLineGap = 6.0;
        double mergeAngleDeg = 5.0;
        double mergeDistPx = 2.0;
        double mergeGapPx = 12.0;

        // Raster sharpness control
        bool sharpRaster = false;
    };

}

static pcl::PointCloud<pcl::PointXYZ>::Ptr loadCloud(const std::string &path)
{
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".pcd")
    {
        if (pcl::io::loadPCDFile(path, *cloud) != 0)
            throw std::runtime_error("Failed to load PCD: " + path);
    }
    else if (path.size() >= 4 && path.substr(path.size() - 4) == ".ply")
    {
        if (pcl::io::loadPLYFile(path, *cloud) != 0)
            throw std::runtime_error("Failed to load PLY: " + path);
    }
    else
    {
        throw std::runtime_error("Unsupported file extension: " + path);
    }
    if (cloud->points.size() < 3)
        throw std::runtime_error("Too few points in cloud");
    return cloud;
}

static void splitGroundWalls(pcl::PointCloud<pcl::PointXYZ>::Ptr src,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr &ground,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr &walls,
                             Eigen::Vector3f &groundNormal,
                             double dist_thresh = 0.3, int ransac_n = 3, int max_iter = 1000)
{
    (void)ransac_n;
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    // Constrain the plane normal to be close to Z axis so we find the ground plane
    // instead of accidentally fitting a large wall plane.
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setAxis(Eigen::Vector3f(0.f, 0.f, 1.f));
    seg.setEpsAngle(static_cast<float>(20.0 * kPi() / 180.0));
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iter);
    seg.setDistanceThreshold(dist_thresh);
    seg.setInputCloud(src);

    pcl::ModelCoefficients coefficients;
    pcl::PointIndices inliers;
    seg.segment(inliers, coefficients);
    if (inliers.indices.empty())
        throw std::runtime_error("RANSAC plane segmentation failed");

    if (coefficients.values.size() >= 3)
    {
        groundNormal = Eigen::Vector3f(coefficients.values[0], coefficients.values[1], coefficients.values[2]);
        if (groundNormal.norm() > 1e-6f)
        {
            groundNormal.normalize();
        }
        else
        {
            groundNormal = Eigen::Vector3f(0.f, 0.f, 1.f);
        }
    }
    else
    {
        groundNormal = Eigen::Vector3f(0.f, 0.f, 1.f);
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(src);

    ground.reset(new pcl::PointCloud<pcl::PointXYZ>);
    walls.reset(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointIndices::Ptr inliersPtr(new pcl::PointIndices(inliers));
    extract.setIndices(inliersPtr);
    extract.setNegative(false);
    extract.filter(*ground);

    extract.setNegative(true);
    extract.filter(*walls);
}

static void rotateCloudInPlace(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const Eigen::Matrix3f &R)
{
    if (!cloud || cloud->points.empty())
        return;

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3, 3>(0, 0) = R;
    pcl::transformPointCloud(*cloud, *cloud, T);
}

static Eigen::Matrix3f estimateYawAlignmentFromWalls(const pcl::PointCloud<pcl::PointXYZ>::Ptr &walls)
{
    if (!walls || walls->points.size() < 10)
    {
        return Eigen::Matrix3f::Identity();
    }

    Eigen::Vector2d mean(0.0, 0.0);
    for (const auto &p : walls->points)
    {
        mean.x() += static_cast<double>(p.x);
        mean.y() += static_cast<double>(p.y);
    }
    mean /= static_cast<double>(walls->points.size());

    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (const auto &p : walls->points)
    {
        Eigen::Vector2d d(static_cast<double>(p.x) - mean.x(), static_cast<double>(p.y) - mean.y());
        cov += d * d.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
    if (es.info() != Eigen::Success)
    {
        return Eigen::Matrix3f::Identity();
    }

    Eigen::Vector2d principal = es.eigenvectors().col(1);
    const double theta = std::atan2(principal.y(), principal.x());
    const float c = static_cast<float>(std::cos(-theta));
    const float s = static_cast<float>(std::sin(-theta));

    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    R(0, 0) = c;
    R(0, 1) = -s;
    R(1, 0) = s;
    R(1, 1) = c;
    return R;
}

static void alignWallsToXYZ(pcl::PointCloud<pcl::PointXYZ>::Ptr walls,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr ground,
                            const Eigen::Vector3f &groundNormal)
{
    if (!walls || walls->points.empty())
        return;

    Eigen::Vector3f n = groundNormal;
    if (n.norm() < 1e-6f)
    {
        n = Eigen::Vector3f(0.f, 0.f, 1.f);
    }
    n.normalize();
    if (n.z() < 0.f)
    {
        n = -n;
    }

    // Step 1: align ground normal to +Z.
    const Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(n, Eigen::Vector3f::UnitZ());
    const Eigen::Matrix3f Rz = q.toRotationMatrix();
    rotateCloudInPlace(walls, Rz);
    rotateCloudInPlace(ground, Rz);

    // Step 2: align dominant horizontal wall direction to +X.
    const Eigen::Matrix3f Ryaw = estimateYawAlignmentFromWalls(walls);
    rotateCloudInPlace(walls, Ryaw);
    rotateCloudInPlace(ground, Ryaw);
}

static void shiftToPositiveXY(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T(0, 3) = -min_pt.x();
    T(1, 3) = -min_pt.y();
    pcl::transformPointCloud(*cloud, *cloud, T);
}

static std::vector<Eigen::Vector2d> to2DPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    std::vector<Eigen::Vector2d> pts;
    pts.reserve(cloud->points.size());
    for (const auto &p : cloud->points)
        pts.emplace_back(p.x, p.y);
    return pts;
}

static int chooseSupersampleFactor(const Eigen::Vector2i &size)
{
    // Limit memory/time: keep hi-res image under ~40M pixels.
    const long long basePixels = 1LL * static_cast<long long>(size[0]) * static_cast<long long>(size[1]);
    if (basePixels <= 0)
        return 1;
    const long long maxPixels = 40LL * 1000LL * 1000LL;
    const double raw = std::sqrt(static_cast<double>(maxPixels) / static_cast<double>(basePixels));
    int ss = static_cast<int>(std::floor(raw));
    ss = std::clamp(ss, 1, 4);
    return ss;
}

static cv::Mat rasterizePointsToCounts32F_AA(const std::vector<Eigen::Vector2d> &pts2d,
                                             const Eigen::Vector2d &origin,
                                             const Eigen::Vector2i &size,
                                             double gridSize,
                                             int supersample)
{
    CV_Assert(size[0] > 0 && size[1] > 0);
    CV_Assert(gridSize > 0.0);
    supersample = std::clamp(supersample, 1, 8);

    if (supersample == 1)
    {
        cv::Mat img = cv::Mat::zeros(size[0], size[1], CV_32F);

        for (const auto &p : pts2d)
        {
            const Eigen::Vector2d g = (p - origin) / gridSize;
            const double gx = g.x();
            const double gy = g.y();

            const int x0 = static_cast<int>(std::floor(gx));
            const int y0 = static_cast<int>(std::floor(gy));
            const float fx = static_cast<float>(gx - x0);
            const float fy = static_cast<float>(gy - y0);

            const int xs[2] = {x0, x0 + 1};
            const int ys[2] = {y0, y0 + 1};
            const float wx[2] = {1.0f - fx, fx};
            const float wy[2] = {1.0f - fy, fy};

            for (int xi = 0; xi < 2; ++xi)
            {
                const int x = xs[xi];
                if (x < 0 || x >= size[0])
                    continue;
                for (int yi = 0; yi < 2; ++yi)
                {
                    const int y = ys[yi];
                    if (y < 0 || y >= size[1])
                        continue;
                    const float w = wx[xi] * wy[yi];
                    img.at<float>(x, y) += w;
                }
            }
        }
        return img;
    }

    // SSAA: render into higher-res grid then downsample.
    const int hiRows = size[0] * supersample;
    const int hiCols = size[1] * supersample;
    cv::Mat hi = cv::Mat::zeros(hiRows, hiCols, CV_32F);
    const double hiGrid = gridSize / static_cast<double>(supersample);

    for (const auto &p : pts2d)
    {
        const Eigen::Vector2d g = (p - origin) / hiGrid;
        const double gx = g.x();
        const double gy = g.y();

        const int x0 = static_cast<int>(std::floor(gx));
        const int y0 = static_cast<int>(std::floor(gy));
        const float fx = static_cast<float>(gx - x0);
        const float fy = static_cast<float>(gy - y0);

        const int xs[2] = {x0, x0 + 1};
        const int ys[2] = {y0, y0 + 1};
        const float wx[2] = {1.0f - fx, fx};
        const float wy[2] = {1.0f - fy, fy};

        for (int xi = 0; xi < 2; ++xi)
        {
            const int x = xs[xi];
            if (x < 0 || x >= hiRows)
                continue;
            for (int yi = 0; yi < 2; ++yi)
            {
                const int y = ys[yi];
                if (y < 0 || y >= hiCols)
                    continue;
                const float w = wx[xi] * wy[yi];
                hi.at<float>(x, y) += w;
            }
        }
    }

    cv::Mat down;
    cv::resize(hi, down, cv::Size(size[1], size[0]), 0.0, 0.0, cv::INTER_AREA);
    return down;
}

static cv::Mat morphologicalCloseImage(const cv::Mat &input_img, int kernel_size, int iterations)
{
    CV_Assert(!input_img.empty());
    kernel_size = std::max(1, kernel_size);
    iterations = std::max(0, iterations);
    if (iterations == 0)
        return input_img;
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::Mat close_img;
    cv::morphologyEx(input_img, close_img, cv::MORPH_CLOSE, close_kernel, cv::Point(-1, -1), iterations);
    return close_img;
}

static cv::Mat filterConnectedComponents(const cv::Mat &input_img,
                                         double min_area,
                                         double max_area,
                                         double aspect_ratio_thresh,
                                         bool adaptive)
{
    CV_Assert(!input_img.empty());
    CV_Assert(input_img.type() == CV_8U);

    if (adaptive)
    {
        const double img_area = static_cast<double>(input_img.cols) * static_cast<double>(input_img.rows);
        min_area = img_area * 0.0001;
        if (min_area < 10.0)
            min_area = 10.0;
        max_area = img_area * 0.1;
    }

    cv::Mat labels, stats, centroids;
    const int num_labels = cv::connectedComponentsWithStats(input_img, labels, stats, centroids, 8, CV_32S);

    cv::Mat filtered_img = cv::Mat::zeros(input_img.size(), CV_8U);
    for (int i = 1; i < num_labels; i++)
    { // skip background label 0
        const double area = static_cast<double>(stats.at<int>(i, cv::CC_STAT_AREA));
        const int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(i, cv::CC_STAT_TOP);
        const int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        if (width <= 0 || height <= 0)
            continue;
        const double aspect_ratio = static_cast<double>(std::max(width, height)) / static_cast<double>(std::min(width, height));

        if (area < min_area || area > max_area || aspect_ratio > aspect_ratio_thresh)
        {
            continue;
        }

        const cv::Rect roi(left, top, width, height);
        cv::Mat labels_roi = labels(roi);
        cv::Mat out_roi = filtered_img(roi);
        cv::Mat mask;
        cv::compare(labels_roi, i, mask, cv::CMP_EQ);
        out_roi.setTo(255, mask);
    }
    return filtered_img;
}

static cv::Mat makeLineBinaryFromProjection(const cv::Mat &grayU8,
                                            int occThreshold,
                                            int medianKSize,
                                            int binDilate,
                                            int binClose,
                                            int closeKernelSize,
                                            bool ccEnable,
                                            double ccMinArea,
                                            double ccMaxArea,
                                            double ccAspectRatio,
                                            bool ccAdaptive)
{
    CV_Assert(!grayU8.empty());
    cv::Mat gray;
    if (grayU8.channels() == 3)
    {
        cv::cvtColor(grayU8, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = grayU8;
    }
    if (gray.type() != CV_8U)
    {
        gray.convertTo(gray, CV_8U);
    }

    // Occupancy-like binary mask (single stroke, not double edges)
    const cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    const int thr = std::clamp(occThreshold, 0, 255);
    cv::Mat smoothed;
    cv::GaussianBlur(gray, smoothed, cv::Size(5, 5), 1.0);

    cv::Mat bin;
    cv::threshold(smoothed, bin, std::max(thr, 5), 255, cv::THRESH_BINARY);

    if (medianKSize > 1)
    {
        if (medianKSize % 2 == 0)
            ++medianKSize;
        cv::medianBlur(bin, bin, std::max(3, medianKSize));
    }

    if (binClose > 0)
    {
        const cv::Mat closeKernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(std::max(1, closeKernelSize), std::max(1, closeKernelSize)));
        cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, closeKernel, cv::Point(-1, -1), binClose);
    }
    if (binDilate > 0)
    {
        cv::dilate(bin, bin, kernel3, cv::Point(-1, -1), binDilate);
    }

    if (ccEnable)
    {
        bin = filterConnectedComponents(bin, ccMinArea, ccMaxArea, ccAspectRatio, ccAdaptive);
    }
    return bin;
}

static cv::Mat skeletonizeBinary(const cv::Mat &bin, bool useXimgprocThinning)
{
    CV_Assert(!bin.empty());
    CV_Assert(bin.type() == CV_8U);

    cv::Mat clean;
    cv::threshold(bin, clean, 0, 255, cv::THRESH_BINARY);

#if HAS_OPENCV_XIMGPROC
    if (useXimgprocThinning)
    {
        cv::Mat thin;
        cv::ximgproc::thinning(clean, thin, cv::ximgproc::THINNING_ZHANGSUEN);
        return thin;
    }
#else
    (void)useXimgprocThinning;
#endif

    // Fallback: distance-transform ridge extraction
    cv::Mat dist;
    cv::distanceTransform(clean, dist, cv::DIST_L2, 3);

    cv::Mat distDilated;
    cv::dilate(dist, distDilated, cv::Mat());

    cv::Mat localMaxMask;
    cv::compare(dist, distDilated, localMaxMask, cv::CMP_GE);

    cv::Mat thickMask;
    cv::threshold(dist, thickMask, 0.7, 255, cv::THRESH_BINARY);
    thickMask.convertTo(thickMask, CV_8U);

    cv::Mat localMaxMaskU8;
    localMaxMask.convertTo(localMaxMaskU8, CV_8U);

    cv::Mat ridge;
    cv::bitwise_and(localMaxMaskU8, thickMask, ridge);
    return ridge;
}

static double wrapAngle180(double angleDeg)
{
    while (angleDeg < 0.0)
        angleDeg += 180.0;
    while (angleDeg >= 180.0)
        angleDeg -= 180.0;
    return angleDeg;
}

static double segmentAngleDeg(const cv::Vec4i &seg)
{
    const double dx = static_cast<double>(seg[2] - seg[0]);
    const double dy = static_cast<double>(seg[3] - seg[1]);
    return wrapAngle180(std::atan2(dy, dx) * 180.0 / kPi());
}

static double angleDifferenceDeg(double a, double b)
{
    const double d = std::fabs(a - b);
    return std::min(d, 180.0 - d);
}

static bool fitClusterLine(const std::vector<cv::Vec4i> &segs,
                           cv::Point2f &linePoint,
                           cv::Point2f &lineDir,
                           double &tMin,
                           double &tMax,
                           double &lineAngleDeg)
{
    if (segs.empty())
        return false;

    std::vector<cv::Point2f> pts;
    pts.reserve(segs.size() * 2);
    for (const auto &s : segs)
    {
        pts.emplace_back(static_cast<float>(s[0]), static_cast<float>(s[1]));
        pts.emplace_back(static_cast<float>(s[2]), static_cast<float>(s[3]));
    }

    if (pts.size() < 2)
        return false;

    cv::Vec4f line;
    cv::fitLine(pts, line, cv::DIST_L2, 0, 0.01, 0.01);
    lineDir = cv::Point2f(line[0], line[1]);
    linePoint = cv::Point2f(line[2], line[3]);

    const float norm = std::sqrt(lineDir.x * lineDir.x + lineDir.y * lineDir.y);
    if (norm < 1e-6f)
        return false;
    lineDir.x /= norm;
    lineDir.y /= norm;

    tMin = std::numeric_limits<double>::max();
    tMax = std::numeric_limits<double>::lowest();
    for (const auto &p : pts)
    {
        const cv::Point2f d = p - linePoint;
        const double t = static_cast<double>(d.x * lineDir.x + d.y * lineDir.y);
        tMin = std::min(tMin, t);
        tMax = std::max(tMax, t);
    }

    lineAngleDeg = wrapAngle180(std::atan2(lineDir.y, lineDir.x) * 180.0 / kPi());
    return true;
}

static cv::Mat redrawMergedHoughLines(const cv::Mat &thin,
                                      int houghThreshold,
                                      double houghMinLineLength,
                                      double houghMaxLineGap,
                                      double mergeAngleDeg,
                                      double mergeDistPx,
                                      double mergeGapPx)
{
    CV_Assert(!thin.empty());
    CV_Assert(thin.type() == CV_8U);

    std::vector<cv::Vec4i> rawSegs;
    cv::HoughLinesP(
        thin,
        rawSegs,
        1.0,
        CV_PI / 180.0,
        std::max(1, houghThreshold),
        std::max(1.0, houghMinLineLength),
        std::max(0.0, houghMaxLineGap));

    cv::Mat merged = cv::Mat::zeros(thin.size(), CV_8U);
    if (rawSegs.empty())
        return merged;

    std::vector<std::vector<cv::Vec4i>> clusters;
    for (const auto &seg : rawSegs)
    {
        const cv::Point2f p1(static_cast<float>(seg[0]), static_cast<float>(seg[1]));
        const cv::Point2f p2(static_cast<float>(seg[2]), static_cast<float>(seg[3]));
        const cv::Point2f mid = 0.5f * (p1 + p2);
        const double angle = segmentAngleDeg(seg);

        int bestCluster = -1;
        for (int i = 0; i < static_cast<int>(clusters.size()); ++i)
        {
            cv::Point2f linePoint, lineDir;
            double tMin = 0.0, tMax = 0.0, lineAngle = 0.0;
            if (!fitClusterLine(clusters[i], linePoint, lineDir, tMin, tMax, lineAngle))
                continue;

            if (angleDifferenceDeg(angle, lineAngle) > mergeAngleDeg)
                continue;

            const cv::Point2f d = mid - linePoint;
            const double perpDist = std::fabs(d.x * lineDir.y - d.y * lineDir.x);
            if (perpDist > mergeDistPx)
                continue;

            const double t1 = static_cast<double>((p1.x - linePoint.x) * lineDir.x + (p1.y - linePoint.y) * lineDir.y);
            const double t2 = static_cast<double>((p2.x - linePoint.x) * lineDir.x + (p2.y - linePoint.y) * lineDir.y);
            const double sMin = std::min(t1, t2);
            const double sMax = std::max(t1, t2);
            const bool nearRange = (sMax >= tMin - mergeGapPx) && (sMin <= tMax + mergeGapPx);
            if (!nearRange)
                continue;

            bestCluster = i;
            break;
        }

        if (bestCluster >= 0)
        {
            clusters[bestCluster].push_back(seg);
        }
        else
        {
            clusters.push_back({seg});
        }
    }

    for (const auto &cluster : clusters)
    {
        cv::Point2f linePoint, lineDir;
        double tMin = 0.0, tMax = 0.0, lineAngle = 0.0;
        if (!fitClusterLine(cluster, linePoint, lineDir, tMin, tMax, lineAngle))
            continue;

        const cv::Point2f pA = linePoint + lineDir * static_cast<float>(tMin);
        const cv::Point2f pB = linePoint + lineDir * static_cast<float>(tMax);

        cv::line(merged, pA, pB, cv::Scalar(255), 1, cv::LINE_8);
    }

    return merged;
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr maskToPointCloud2D(const cv::Mat &mask,
                                                              const Eigen::Vector2d &origin,
                                                              double gridSize)
{
    auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (mask.empty() || gridSize <= 0.0)
        return out;

    out->points.reserve(static_cast<size_t>(cv::countNonZero(mask)));
    for (int row = 0; row < mask.rows; ++row)
    {
        const uchar *ptr = mask.ptr<uchar>(row);
        for (int col = 0; col < mask.cols; ++col)
        {
            if (ptr[col] == 0)
                continue;
            const float x = static_cast<float>(origin.x() + (static_cast<double>(row) + 0.5) * gridSize);
            const float y = static_cast<float>(origin.y() + (static_cast<double>(col) + 0.5) * gridSize);
            out->points.emplace_back(x, y, 0.0f);
        }
    }
    out->width = static_cast<uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = true;
    return out;
}

static int runPcdProjection(const ProjectionOptions &opt)
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    if (opt.inputPath.empty())
        throw std::runtime_error("inputPath is empty");
    if (opt.outDir.empty())
        throw std::runtime_error("outDir is empty");

    fs::create_directories(opt.outDir);

    auto cloud = loadCloud(opt.inputPath);
    std::cout << "[projection] loaded cloud: " << cloud->points.size() << " points\n";

    pcl::PointCloud<pcl::PointXYZ>::Ptr ground, walls;
    Eigen::Vector3f groundNormal(0.f, 0.f, 1.f);
    splitGroundWalls(cloud, ground, walls, groundNormal, opt.groundDistThresh, opt.groundRansacN, opt.groundMaxIter);
    std::cout << "[projection] ground: " << ground->points.size() << " | walls: " << walls->points.size() << "\n";

    alignWallsToXYZ(walls, ground, groundNormal);
    std::cout << "[projection] walls aligned to XYZ axes\n";

    if (!walls || walls->points.empty())
    {
        std::cerr << "walls empty\n";
        return 1;
    }

    const int numSlices = std::max(1, opt.numSlices);
    const int minVotes = std::max(1, opt.minVotes);
    if (opt.upperZ <= opt.lowerZ)
    {
        std::cerr << "Invalid height band: upperZ must be greater than lowerZ\n";
        return 1;
    }

    double effectiveLowerZ = static_cast<double>(opt.lowerZ);
    double effectiveUpperZ = static_cast<double>(opt.upperZ);
    double wallsMinZ = static_cast<double>(walls->points[0].z);
    double wallsMaxZ = static_cast<double>(walls->points[0].z);
    int inBandCount = 0;
    for (const auto &p : walls->points)
    {
        const double z = static_cast<double>(p.z);
        wallsMinZ = std::min(wallsMinZ, z);
        wallsMaxZ = std::max(wallsMaxZ, z);
        if (z >= effectiveLowerZ && z <= effectiveUpperZ)
        {
            ++inBandCount;
        }
    }
    if (inBandCount == 0)
    {
        effectiveLowerZ = wallsMinZ;
        effectiveUpperZ = wallsMaxZ;
        std::cout << "[projection] warning: no points in configured Z band ["
                  << opt.lowerZ << ", " << opt.upperZ << "]"
                  << ", fallback to walls Z range [" << effectiveLowerZ << ", " << effectiveUpperZ << "]\n";
    }
    if (effectiveUpperZ <= effectiveLowerZ)
    {
        std::cerr << "Invalid effective Z band after fallback\n";
        return 1;
    }
    const double sliceHeight = (effectiveUpperZ - effectiveLowerZ) / static_cast<double>(numSlices);

    double shiftX = 0.0;
    double shiftY = 0.0;
    if (opt.shiftToPositiveXY)
    {
        float rawMinX = walls->points[0].x;
        float rawMinY = walls->points[0].y;
        for (const auto &p : walls->points)
        {
            rawMinX = std::min(rawMinX, p.x);
            rawMinY = std::min(rawMinY, p.y);
        }
        shiftX = -static_cast<double>(rawMinX);
        shiftY = -static_cast<double>(rawMinY);
    }

    // Build a flattened 2D cloud from all wall points (used as fill backup output).
    auto walls2d = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    walls2d->points.reserve(walls->points.size());
    for (const auto &p : walls->points)
    {
        walls2d->points.emplace_back(
            static_cast<float>(static_cast<double>(p.x) + shiftX),
            static_cast<float>(static_cast<double>(p.y) + shiftY),
            0.0f);
    }
    walls2d->width = static_cast<uint32_t>(walls2d->points.size());
    walls2d->height = 1;
    walls2d->is_dense = true;

    std::vector<Eigen::Vector2d> pts2d;
    pts2d.reserve(walls2d->points.size());
    for (const auto &p : walls2d->points)
    {
        pts2d.emplace_back(p.x, p.y);
    }
    if (pts2d.empty())
    {
        std::cerr << "walls2d empty\n";
        return 1;
    }

    std::cout << "[projection] height band: [" << effectiveLowerZ << ", " << effectiveUpperZ << "], voxel: " << opt.voxel
              << ", numSlices: " << numSlices << ", minVotes: " << minVotes << "\n";
    std::cout << "[projection] walls2d after flatten: " << walls2d->points.size() << " points\n";

    std::string out_pcd = (fs::path(opt.outDir) / "walls_2d.pcd").string();
    std::string out_pcd_fill = (fs::path(opt.outDir) / "walls_2d_fill.pcd").string();
    pcl::io::savePCDFileBinary(out_pcd_fill, *walls2d);

    std::cout << "[projection] converting to 2D points: " << pts2d.size() << "\n";

    // Check min/max of 2D points for sanity
    if (!pts2d.empty())
    {
        double minX = pts2d[0].x(), maxX = pts2d[0].x();
        double minY = pts2d[0].y(), maxY = pts2d[0].y();
        for (const auto &p : pts2d)
        {
            minX = std::min(minX, p.x());
            maxX = std::max(maxX, p.x());
            minY = std::min(minY, p.y());
            maxY = std::max(maxY, p.y());
        }
        std::cout << "[projection] 2D bounds: X [" << minX << ", " << maxX << "], Y [" << minY << ", " << maxY << "]\n";
    }

    const double resolutionScale = std::max(1.0, opt.resolutionScale);
    const double effectiveGridSize = opt.gridSize / resolutionScale;

    double minX = pts2d[0].x(), maxX = pts2d[0].x();
    double minY = pts2d[0].y(), maxY = pts2d[0].y();
    for (const auto &p : pts2d)
    {
        minX = std::min(minX, p.x());
        maxX = std::max(maxX, p.x());
        minY = std::min(minY, p.y());
        maxY = std::max(maxY, p.y());
    }
    const Eigen::Vector2d origin(minX, minY);
    const int rows = std::max(1, static_cast<int>(std::ceil((maxX - minX) / effectiveGridSize)) + 1);
    const int cols = std::max(1, static_cast<int>(std::ceil((maxY - minY) / effectiveGridSize)) + 1);
    const Eigen::Vector2i size(rows, cols);

    // Multi-slice voting accumulator in a unified raster coordinate system.
    cv::Mat voteAccumulator = cv::Mat::zeros(size[0], size[1], CV_32S);
    const int ss = opt.sharpRaster ? 1 : chooseSupersampleFactor(size);

    const fs::path slicesDir = fs::path(opt.outDir) / "slices";
    fs::create_directories(slicesDir);

    int nonEmptySlices = 0;
    for (int i = 0; i < numSlices; ++i)
    {
        const double zStart = effectiveLowerZ + static_cast<double>(i) * sliceHeight;
        const double zEnd = zStart + sliceHeight;
        const bool isLastSlice = (i == numSlices - 1);

        std::vector<Eigen::Vector2d> slicePts;
        slicePts.reserve(walls->points.size() / static_cast<size_t>(numSlices) + 1);
        for (const auto &p : walls->points)
        {
            const double z = static_cast<double>(p.z);
            const bool inSlice = isLastSlice ? (z >= zStart && z <= zEnd) : (z >= zStart && z < zEnd);
            if (!inSlice)
                continue;

            const double x = static_cast<double>(p.x) + shiftX;
            const double y = static_cast<double>(p.y) + shiftY;
            slicePts.emplace_back(x, y);
        }

        cv::Mat slicePreview = cv::Mat::zeros(size[0], size[1], CV_8U);
        if (!slicePts.empty())
        {
            ++nonEmptySlices;
            const cv::Mat sliceCounts = rasterizePointsToCounts32F_AA(
                slicePts,
                origin,
                size,
                effectiveGridSize,
                ss);

            cv::Mat sliceMask32F;
            cv::threshold(sliceCounts, sliceMask32F, 0.0, 1.0, cv::THRESH_BINARY);

            cv::Mat sliceBin32S;
            sliceMask32F.convertTo(sliceBin32S, CV_32S);
            cv::add(voteAccumulator, sliceBin32S, voteAccumulator);

            sliceMask32F.convertTo(slicePreview, CV_8U, 255.0);
        }

        const std::string sliceName = "slice_" + cv::format("%02d", i) + ".png";
        cv::imwrite((slicesDir / sliceName).string(), slicePreview);
    }

    cv::Mat img;
    cv::Mat voteAccumulatorF;
    voteAccumulator.convertTo(voteAccumulatorF, CV_32F);
    cv::threshold(voteAccumulatorF, img, static_cast<double>(minVotes) - 0.5, 255.0, cv::THRESH_BINARY);
    img.convertTo(img, CV_8U);

    double minVotesVal = 0.0, maxVotesVal = 0.0;
    cv::minMaxLoc(voteAccumulatorF, &minVotesVal, &maxVotesVal);
    cv::Mat voteVis;
    if (maxVotesVal > 0.0)
    {
        voteAccumulatorF.convertTo(voteVis, CV_8U, 255.0 / maxVotesVal);
    }
    else
    {
        voteAccumulatorF.convertTo(voteVis, CV_8U);
    }
    cv::imwrite((fs::path(opt.outDir) / "projection_vote_counts.png").string(), voteVis);

    std::cout << "[projection] raster image size: " << img.rows << " x " << img.cols
              << ", maxVotes: " << maxVotesVal
              << ", resolutionScale: " << resolutionScale
              << ", effectiveGrid: " << effectiveGridSize
              << ", sharpRaster: " << (opt.sharpRaster ? 1 : 0)
              << ", nonEmptySlices: " << nonEmptySlices << "\n";

    std::string out_png = (fs::path(opt.outDir) / "projection.png").string();
    cv::imwrite(out_png, img);

    // Extra output: thin binary line map + optional line-like walls_2d.pcd export.
    if (opt.saveLineBinary)
    {
        cv::Mat lineBin = makeLineBinaryFromProjection(img,
                                                       opt.occThreshold,
                                                       opt.medianKSize,
                                                       opt.binDilate,
                                                       opt.binClose,
                                                       opt.closeKernelSize,
                                                       opt.ccEnable,
                                                       opt.ccMinArea,
                                                       opt.ccMaxArea,
                                                       opt.ccAspectRatio,
                                                       opt.ccAdaptive);
        const std::string out_lines = (fs::path(opt.outDir) / "projection_lines_bin.png").string();
        cv::imwrite(out_lines, lineBin);

        cv::Mat lineThin = skeletonizeBinary(lineBin, opt.useXimgprocThinning);
        const std::string out_lines_thin = (fs::path(opt.outDir) / "projection_lines_thin.png").string();
        cv::imwrite(out_lines_thin, lineThin);

        cv::Mat lineRefined = redrawMergedHoughLines(
            lineThin,
            opt.houghThreshold,
            opt.houghMinLineLength,
            opt.houghMaxLineGap,
            opt.mergeAngleDeg,
            opt.mergeDistPx,
            opt.mergeGapPx);
        const std::string out_lines_refined = (fs::path(opt.outDir) / "projection_lines_refined.png").string();
        cv::imwrite(out_lines_refined, lineRefined);

        const cv::Mat &lineForExport = (cv::countNonZero(lineRefined) > 0) ? lineRefined : lineThin;

        bool linePcdWritten = false;
        if (opt.exportLineLikePcd)
        {
            auto walls2dLine = maskToPointCloud2D(lineForExport, origin, effectiveGridSize);
            if (walls2dLine && !walls2dLine->empty())
            {
                pcl::io::savePCDFileBinary(out_pcd, *walls2dLine);
                pcl::io::savePCDFileBinary((fs::path(opt.outDir) / "walls_2d_line.pcd").string(), *walls2dLine);
                linePcdWritten = true;
                std::cout << "[projection] exported line-like walls_2d.pcd points=" << walls2dLine->points.size() << "\n";
            }
        }
        if (!linePcdWritten)
        {
            pcl::io::savePCDFileBinary(out_pcd, *walls2d);
            std::cout << "[projection] line-like export unavailable, fallback walls_2d.pcd=filling cloud\n";
        }

        // Optional overlay for quick visual confirmation
        cv::Mat overlay;
        cv::cvtColor(img, overlay, cv::COLOR_GRAY2BGR);
        overlay.setTo(cv::Scalar(0, 255, 255), lineBin);
        overlay.setTo(cv::Scalar(0, 0, 255), lineThin);
        overlay.setTo(cv::Scalar(0, 255, 0), lineRefined);
        const std::string out_overlay = (fs::path(opt.outDir) / "projection_lines_overlay.png").string();
        cv::imwrite(out_overlay, overlay);
    }
    else
    {
        pcl::io::savePCDFileBinary(out_pcd, *walls2d);
    }

    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    std::cout << "[timing] total: " << total_ms << " ms\n";

    std::cout << "[projection] 2D points: " << walls2d->points.size() << "\n";
    std::cout << "[projection] saved PCD: " << out_pcd << "\n";
    std::cout << "[projection] saved fill PCD backup: " << out_pcd_fill << "\n";
    std::cout << "[projection] saved PNG: " << out_png << "\n";
    std::cout << "[projection] saved vote counts: " << (fs::path(opt.outDir) / "projection_vote_counts.png").string() << "\n";
    std::cout << "[projection] saved per-slice projections dir: " << (fs::path(opt.outDir) / "slices").string() << "\n";
    if (opt.saveLineBinary)
    {
        std::cout << "[projection] saved line binary: " << (fs::path(opt.outDir) / "projection_lines_bin.png").string() << "\n";
        std::cout << "[projection] saved thin line binary: " << (fs::path(opt.outDir) / "projection_lines_thin.png").string() << "\n";
        std::cout << "[projection] saved refined line binary: " << (fs::path(opt.outDir) / "projection_lines_refined.png").string() << "\n";
        std::cout << "[projection] saved overlay: " << (fs::path(opt.outDir) / "projection_lines_overlay.png").string() << "\n";
    }

    return 0;
}
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: pcd_projection <input.pcd/ply> <out_dir> [gridSize] [resolutionScale] [lowerZ] [upperZ] [voxel]"
                     " [saveLineBinary=1] [occThreshold=1] [medianKSize=3] [binDilate=0] [binClose=1]"
                     " [closeKernelSize=3] [ccEnable=0] [ccMinArea=50] [ccMaxArea=10000] [ccAspectRatio=10] [ccAdaptive=0] [exportLineLikePcd=1]"
                     " [useXimgprocThinning=1] [houghThreshold=20] [houghMinLineLength=10] [houghMaxLineGap=8] [mergeAngleDeg=8] [mergeDistPx=4] [mergeGapPx=20] [sharpRaster=1]"
                     " [numSlices=10] [minVotes=6]\n"
                     "(Compatibility: if you still pass the old extra params, they will be ignored and occupancy output is used.)\n";
        return 1;
    }

    ProjectionOptions opt;
    opt.inputPath = argv[1];
    opt.outDir = argv[2];
    opt.gridSize = (argc >= 4) ? std::stod(argv[3]) : opt.gridSize;
    opt.resolutionScale = (argc >= 5) ? std::stod(argv[4]) : opt.resolutionScale;
    opt.lowerZ = (argc >= 6) ? static_cast<float>(std::stod(argv[5])) : opt.lowerZ;
    opt.upperZ = (argc >= 7) ? static_cast<float>(std::stod(argv[6])) : opt.upperZ;
    opt.voxel = (argc >= 8) ? static_cast<float>(std::stod(argv[7])) : opt.voxel;
    opt.saveLineBinary = (argc >= 9) ? (std::stoi(argv[8]) != 0) : opt.saveLineBinary;

    // Preferred new short form: ... [saveLineBinary] [occThreshold] [medianKSize] [binDilate] [binClose]
    if (argc >= 10)
        opt.occThreshold = std::stoi(argv[9]);
    if (argc >= 11)
        opt.medianKSize = std::stoi(argv[10]);
    if (argc >= 12)
        opt.binDilate = std::stoi(argv[11]);
    if (argc >= 13)
        opt.binClose = std::stoi(argv[12]);

    // Extended post-process params (ported from img_plus), appended after binClose.
    // If you pass more args beyond these, they are ignored.
    if (argc >= 14)
        opt.closeKernelSize = std::stoi(argv[13]);
    if (argc >= 15)
        opt.ccEnable = (std::stoi(argv[14]) != 0);
    if (argc >= 16)
        opt.ccMinArea = std::stod(argv[15]);
    if (argc >= 17)
        opt.ccMaxArea = std::stod(argv[16]);
    if (argc >= 18)
        opt.ccAspectRatio = std::stod(argv[17]);
    if (argc >= 19)
        opt.ccAdaptive = (std::stoi(argv[18]) != 0);
    if (argc >= 20)
        opt.exportLineLikePcd = (std::stoi(argv[19]) != 0);

    if (argc >= 21)
        opt.useXimgprocThinning = (std::stoi(argv[20]) != 0);
    if (argc >= 22)
        opt.houghThreshold = std::stoi(argv[21]);
    if (argc >= 23)
        opt.houghMinLineLength = std::stod(argv[22]);
    if (argc >= 24)
        opt.houghMaxLineGap = std::stod(argv[23]);
    if (argc >= 25)
        opt.mergeAngleDeg = std::stod(argv[24]);
    if (argc >= 26)
        opt.mergeDistPx = std::stod(argv[25]);
    if (argc >= 27)
        opt.mergeGapPx = std::stod(argv[26]);
    if (argc >= 28)
        opt.sharpRaster = (std::stoi(argv[27]) != 0);
    if (argc >= 29)
        opt.numSlices = std::stoi(argv[28]);
    if (argc >= 30)
        opt.minVotes = std::stoi(argv[29]);

    try
    {
        return runPcdProjection(opt);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}