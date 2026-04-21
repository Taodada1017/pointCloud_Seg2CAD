#include "frontend/geometry/point_cloud.h"
#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <limits>

namespace {
bool GetCloudZRange(const pcl::PointCloud<PointT>::Ptr& cloud, float& zMin, float& zMax) {
	if (!cloud || cloud->empty()) {
		zMin = 0.0f;
		zMax = 0.0f;
		return false;
	}
	zMin = std::numeric_limits<float>::max();
	zMax = std::numeric_limits<float>::lowest();
	for (const auto& p : cloud->points) {
		zMin = std::min(zMin, p.z);
		zMax = std::max(zMax, p.z);
	}
	return true;
}
}

PCD2d::PCD2d() : wallsPcd_(new pcl::PointCloud<pcl::PointXYZ>()), groundPcd_(new pcl::PointCloud<pcl::PointXYZ>()),
				upperBound_(1.9f), lowerBound_(0.2f),
				voxelSize_(0.01f),
				T_rp_(Eigen::Matrix4d::Identity()),
				T_rp_inv_(Eigen::Matrix4d::Identity()), ogWalls_(new pcl::PointCloud<PointT>()),
				ogGround_(new pcl::PointCloud<PointT>()){
	ogWallsCropped_.reset(new pcl::PointCloud<PointT>);
	walls2d_.reset(new pcl::PointCloud<PointT>);
};

PCD2d::PCD2d(
		pcl::PointCloud<pcl::PointXYZ>::Ptr groundPcd
) : wallsPcd_(new pcl::PointCloud<pcl::PointXYZ>()), groundPcd_(groundPcd),
	upperBound_(1.9f), lowerBound_(0.2f),
	voxelSize_(0.01f),
	T_rp_(Eigen::Matrix4d::Identity()),
	T_rp_inv_(Eigen::Matrix4d::Identity()), ogWalls_(new pcl::PointCloud<PointT>()),
	ogGround_(groundPcd) {
	ogWallsCropped_.reset(new pcl::PointCloud<PointT>);
	walls2d_.reset(new pcl::PointCloud<PointT>);

//	pcl::VoxelGrid<PointT> voxelFilter;
//	voxelFilter.setInputCloud(groundPcd_);
//	voxelFilter.setLeafSize(1.0f, 1.0f, 1.0f); // 设置体素大小
//	voxelFilter.filter(*groundPcd_);
}
	
PCD2d::PCD2d(
		pcl::PointCloud<pcl::PointXYZ>::Ptr wallsPcd,
		pcl::PointCloud<pcl::PointXYZ>::Ptr groundPcd
) : wallsPcd_(wallsPcd), groundPcd_(groundPcd),
	upperBound_(1.9f), lowerBound_(0.2f),
	voxelSize_(0.01f), T_rp_(Eigen::Matrix4d::Identity()),
	T_rp_inv_(Eigen::Matrix4d::Identity()), ogWalls_(new pcl::PointCloud<PointT>()), ogGround_(new pcl::PointCloud<PointT>()) {
	ogWallsCropped_.reset(new pcl::PointCloud<PointT>);
	walls2d_.reset(new pcl::PointCloud<PointT>);
}

bool PCD2d::GetPcd2d(const std::string &lidarOrigin) {
	if (!ResetRP(lidarOrigin)) return false;
	CropWalls();
	FlattenWalls();
	return true;
}

bool PCD2d::ResetRP(const std::string &lidarOrigin) {
//	auto downGround = groundPcd_->VoxelDownSample(1.0);
//	if (downGround->points_.size() < 3) {
//		std::cout << "Not enough points to fit the plane." << std::endl;
//		return false;
//	}
//	auto plane = downGround->SegmentPlane(0.1, 3, 20).first;
	// timer.tic("Reset RP");
	if (groundPcd_->points.size() < 3) {
		if (!wallsPcd_ || wallsPcd_->points.size() < 3) {
			std::cout << "[PCD2d] Not enough points to fit plane and walls too small." << std::endl;
			return false;
		}
		float zMin = 0.0f, zMax = 0.0f;
		GetCloudZRange(wallsPcd_, zMin, zMax);
		T_rp_ = Eigen::Matrix4d::Identity();
		T_rp_(2, 3) = -static_cast<double>(zMin);
		T_rp_inv_ = T_rp_.inverse();

		*ogWalls_ = *wallsPcd_;
		*ogGround_ = *groundPcd_;
		Eigen::Matrix4f T_rp_float = T_rp_.cast<float>();
		pcl::transformPointCloud(*ogWalls_, *ogWalls_, T_rp_float);
		if (!ogGround_->empty()) {
			pcl::transformPointCloud(*ogGround_, *ogGround_, T_rp_float);
		}

		float ogZMin = 0.0f, ogZMax = 0.0f;
		GetCloudZRange(ogWalls_, ogZMin, ogZMax);
		std::cout << "[PCD2d] Not enough ground points for plane fit. Fallback to Z-shift"
				  << " (z_min=" << zMin << ", src_z_range=[" << zMin << ", " << zMax
				  << "], rp_z_range=[" << ogZMin << ", " << ogZMax << "])." << std::endl;
		return true;
	}
	pcl::ModelCoefficients::Ptr planeCoefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	
	pcl::SACSegmentation<PointT> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(1000);
	seg.setDistanceThreshold(0.3);
	seg.setInputCloud(groundPcd_);
	seg.segment(*inliers, *planeCoefficients);
	if (inliers->indices.size() == 0) {
		if (!wallsPcd_ || wallsPcd_->points.size() < 3) {
			std::cerr << "[PCD2d] Could not estimate plane and walls too small." << std::endl;
			return false;
		}
		float zMin = 0.0f, zMax = 0.0f;
		GetCloudZRange(wallsPcd_, zMin, zMax);
		T_rp_ = Eigen::Matrix4d::Identity();
		T_rp_(2, 3) = -static_cast<double>(zMin);
		T_rp_inv_ = T_rp_.inverse();

		*ogWalls_ = *wallsPcd_;
		*ogGround_ = *groundPcd_;
		Eigen::Matrix4f T_rp_float = T_rp_.cast<float>();
		pcl::transformPointCloud(*ogWalls_, *ogWalls_, T_rp_float);
		if (!ogGround_->empty()) {
			pcl::transformPointCloud(*ogGround_, *ogGround_, T_rp_float);
		}

		float ogZMin = 0.0f, ogZMax = 0.0f;
		GetCloudZRange(ogWalls_, ogZMin, ogZMax);
		std::cout << "[PCD2d] Plane fitting failed. Fallback to Z-shift"
				  << " (z_min=" << zMin << ", src_z_range=[" << zMin << ", " << zMax
				  << "], rp_z_range=[" << ogZMin << ", " << ogZMax << "])." << std::endl;
		return true;
	}
	Eigen::Vector4d plane;
	plane << planeCoefficients->values[0], planeCoefficients->values[1],
			planeCoefficients->values[2], planeCoefficients->values[3];
	Eigen::Vector3d t_l2w(0, 0, 0);
	
	t_l2w = ProjPt2Pl(t_l2w, plane);
	
	
	Eigen::Vector3d v1(0, 0, 1);
	Eigen::Vector3d v2(plane.head<3>());
	Eigen::Matrix3d R = RFrom2Vecs(v1, v2);
	Eigen::Vector3d t = -R.transpose() * t_l2w;
	
	T_rp_.block<3, 3>(0, 0) = R.transpose();
	T_rp_.block<3, 1>(0, 3) = t;
	
	T_rp_inv_ = T_rp_.inverse();
	
	// Deep copy input clouds before transformation (similar to Python's deepcopy)
	if (wallsPcd_ && !wallsPcd_->empty()) {
		*ogWalls_ = *wallsPcd_;
	}
	if (groundPcd_ && !groundPcd_->empty()) {
		*ogGround_ = *groundPcd_;
	}
	
	// Transform walls and ground to reference plane
	Eigen::Matrix4f T_rp_float = T_rp_.cast<float>();
	if (ogWalls_ && !ogWalls_->empty()) {
		pcl::transformPointCloud(*ogWalls_, *ogWalls_, T_rp_float);
	}
	if (ogGround_ && !ogGround_->empty()) {
		pcl::transformPointCloud(*ogGround_, *ogGround_, T_rp_float);
	}

	float wallZMin = 0.0f, wallZMax = 0.0f;
	if (GetCloudZRange(ogWalls_, wallZMin, wallZMax)) {
		std::cout << "[PCD2d] ResetRP done. walls_rp points=" << ogWalls_->points.size()
				  << ", z_range=[" << wallZMin << ", " << wallZMax << "]"
				  << ", height_band=[" << lowerBound_ << ", " << upperBound_ << "]"
				  << std::endl;
	}
	// timer.toc("Reset RP");
	return true;
}

void PCD2d::CropWalls() {
	if (!ogWalls_ || ogWalls_->empty()) {
		ogWallsCropped_->clear();
		std::cout << "[PCD2d] CropWalls skipped: ogWalls is empty." << std::endl;
		return;
	}
	float inZMin = 0.0f, inZMax = 0.0f;
	GetCloudZRange(ogWalls_, inZMin, inZMax);

	pcl::CropBox<PointT> cropBox = GetCropBox(GetBoundPcd(ogWalls_), upperBound_, lowerBound_);
	cropBox.setInputCloud(ogWalls_);
	cropBox.filter(*ogWallsCropped_);

	if (ogWallsCropped_->empty()) {
		*ogWallsCropped_ = *ogWalls_;
		std::cout << "[PCD2d] WARNING: crop produced empty cloud, fallback to uncropped walls. "
				  << "height_band=[" << lowerBound_ << ", " << upperBound_ << "], "
				  << "z_range=[" << inZMin << ", " << inZMax << "], "
				  << "input_points=" << ogWalls_->points.size() << std::endl;
		return;
	}

	float outZMin = 0.0f, outZMax = 0.0f;
	GetCloudZRange(ogWallsCropped_, outZMin, outZMax);
	std::cout << "[PCD2d] CropWalls: input_points=" << ogWalls_->points.size()
			  << ", output_points=" << ogWallsCropped_->points.size()
			  << ", input_z=[" << inZMin << ", " << inZMax << "]"
			  << ", output_z=[" << outZMin << ", " << outZMax << "]"
			  << ", height_band=[" << lowerBound_ << ", " << upperBound_ << "]"
			  << std::endl;
}

void PCD2d::FlattenWalls() {
	pcl::PointCloud<PointT>::Ptr tempCloud(new pcl::PointCloud<PointT>(*ogWallsCropped_));
	for (auto &point: tempCloud->points) {
		point.z = 0;
	}
	pcl::VoxelGrid<PointT> voxelGrid;
	voxelGrid.setInputCloud(tempCloud);
	// Clamp voxel size to reasonable range: [0.005, 0.05] to avoid losing too many points
	double effective_voxel = std::max(0.005, std::min(static_cast<double>(voxelSize_), 0.05));
	voxelGrid.setLeafSize(effective_voxel, effective_voxel, effective_voxel);
	voxelGrid.filter(*walls2d_);
}

Eigen::Matrix3d PCD2d::RFrom2Vecs(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2) {
	Eigen::Vector3d v1Norm = v1.normalized();
	Eigen::Vector3d v2Norm = v2.normalized();
	Eigen::Vector3d axis = v1Norm.cross(v2Norm);
	double axisLength = axis.norm();
	double dot = std::clamp(v1Norm.dot(v2Norm), -1.0, 1.0);
	double angle = acos(dot);
	if (axisLength < 1e-12) {
		if (dot > 0.0) {
			return Eigen::Matrix3d::Identity();
		}
		Eigen::Vector3d ortho = v1Norm.unitOrthogonal();
		return Eigen::AngleAxisd(std::acos(-1.0), ortho).toRotationMatrix();
	}
	return Eigen::AngleAxisd(angle, axis / axisLength).toRotationMatrix();
}


double PCD2d::GetPt2PlDist(const Eigen::Vector3d &pt, const Eigen::Vector4d &plane) {
	double a = plane[0], b = plane[1], c = plane[2], d = plane[3];
	return (a * pt[0] + b * pt[1] + c * pt[2] + d) / std::sqrt(a * a + b * b + c * c);
}

Eigen::Vector3d PCD2d::ProjPt2Pl(const Eigen::Vector3d &pt, const Eigen::Vector4d &plane) {
	double a = plane[0], b = plane[1], c = plane[2], d = plane[3];
	double x = pt[0], y = pt[1], z = pt[2];
	double pt2pl_dist = GetPt2PlDist(pt, plane);
	double norm = std::sqrt(a * a + b * b + c * c);
	return Eigen::Vector3d(
			x - a * pt2pl_dist / norm,
			y - b * pt2pl_dist / norm,
			z - c * pt2pl_dist / norm
	);
}

pcl::PointCloud<PointT>::Ptr PCD2d::GetBoundPcd(const pcl::PointCloud<PointT>::Ptr &pcd) {
	Eigen::Vector4f minPt, maxPt;
	pcl::getMinMax3D(*pcd, minPt, maxPt);
	
	pcl::PointCloud<PointT>::Ptr boundPcd(new pcl::PointCloud<PointT>);
	boundPcd->emplace_back(minPt.x(), minPt.y(), minPt.z());
	boundPcd->emplace_back(maxPt.x(), minPt.y(), minPt.z());
	boundPcd->emplace_back(maxPt.x(), maxPt.y(), minPt.z());
	boundPcd->emplace_back(minPt.x(), maxPt.y(), minPt.z());
	
	return boundPcd;
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> PCD2d::GetMinMax(const pcl::PointCloud<PointT>::Ptr &boundPcd) {
	Eigen::Vector4f minPt, maxPt;
	pcl::getMinMax3D(*boundPcd, minPt, maxPt);
	
	Eigen::Vector3f minBound(minPt.x(), minPt.y(), 0.0f);
	Eigen::Vector3f maxBound(maxPt.x(), maxPt.y(), 0.0f);
	
	return {minBound, maxBound};
}

pcl::CropBox<PointT>
PCD2d::GetCropBox(const pcl::PointCloud<PointT>::Ptr &boundPcd, float upperBound, float lowerBound) {
	auto [minBound, maxBound] = GetMinMax(boundPcd);
	
	minBound.z() = lowerBound;
	maxBound.z() = upperBound;
	
	pcl::CropBox<PointT> cropBox;
	cropBox.setMin(Eigen::Vector4f(minBound.x(), minBound.y(), minBound.z(), 1.0));
	cropBox.setMax(Eigen::Vector4f(maxBound.x(), maxBound.y(), maxBound.z(), 1.0));
	
	return cropBox;
}

void PCD2d::SetHeightBand(float lower, float upper) {
	if (upper < lower) std::swap(upper, lower);
	lowerBound_ = lower;
	upperBound_ = upper;
}

void PCD2d::SetVoxelSize(float voxel) {
	if (voxel <= 0.0f) return;
	voxelSize_ = voxel;
}
