/**
** Created by Zhijian QIAO.
** UAV Group, Hong Kong University of Science and Technology
** email: zqiaoac@connect.ust.hk
**/

#ifndef BIMREG_POINT_CLOUD_H
#define BIMREG_POINT_CLOUD_H
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
typedef pcl::PointXYZ PointT;
class PCD2d {
public:
	PCD2d();
	PCD2d(pcl::PointCloud<pcl::PointXYZ>::Ptr groundPcd);
	PCD2d(pcl::PointCloud<pcl::PointXYZ>::Ptr wallsPcd,
		  pcl::PointCloud<pcl::PointXYZ>::Ptr groundPcd);
	
	bool GetPcd2d(const std::string& lidarOrigin = "default");
	Eigen::Matrix4d GetT_rp() const { return T_rp_; }
	pcl::PointCloud<pcl::PointXYZ>::Ptr getWalls2d() const { return walls2d_; }
	pcl::PointCloud<pcl::PointXYZ>::Ptr getWallsRaw() const { return ogWalls_; }
	pcl::PointCloud<pcl::PointXYZ>::Ptr getWallsCropped() const { return ogWallsCropped_; }
	
	void SetT_rp(const Eigen::Matrix4d& T_rp) { T_rp_ = T_rp; }
	void SetHeightBand(float lower, float upper);
	void SetVoxelSize(float voxel);

private:
	bool ResetRP(const std::string& lidarOrigin);
	void CropWalls();
	void FlattenWalls();
	
	
	double GetPt2PlDist(const Eigen::Vector3d& pt, const Eigen::Vector4d& plane);
	Eigen::Vector3d ProjPt2Pl(const Eigen::Vector3d& pt, const Eigen::Vector4d& plane);
	Eigen::Matrix3d RFrom2Vecs(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2);
	
	pcl::PointCloud<PointT>::Ptr GetBoundPcd(const pcl::PointCloud<PointT>::Ptr& pcd);
	std::pair<Eigen::Vector3f, Eigen::Vector3f> GetMinMax(const pcl::PointCloud<PointT>::Ptr& boundPcd);
	pcl::CropBox<PointT> GetCropBox(const pcl::PointCloud<PointT>::Ptr& boundPcd, float upperBound, float lowerBound);
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr wallsPcd_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr groundPcd_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr ogWalls_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr ogGround_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr ogWallsCropped_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr walls2d_;
	
	Eigen::Matrix4d T_rp_;
	Eigen::Matrix4d T_rp_inv_;
	
	float upperBound_;
	float lowerBound_;
	float voxelSize_;
};

#endif //BIMREG_POINT_CLOUD_H
