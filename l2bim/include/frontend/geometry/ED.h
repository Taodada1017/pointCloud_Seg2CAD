/**************************************************************************************************************
* Edge Drawing (ED) and Edge Drawing Parameter Free (EDPF) source codes.
* Copyright (C) Cihan Topal & Cuneyt Akinlar 
* E-mails of the authors:  cihantopal@gmail.com, cuneytakinlar@gmail.com
*
* Please cite the following papers if you use Edge Drawing library:
*
* [1] C. Topal and C. Akinlar, “Edge Drawing: A Combined Real-Time Edge and Segment Detector,”
*     Journal of Visual Communication and Image Representation, 23(6), 862-872, DOI: 10.1016/j.jvcir.2012.05.004 (2012).
*
* [2] C. Akinlar and C. Topal, “EDPF: A Real-time Parameter-free Edge Segment Detector with a False Detection Control,”
*     International Journal of Pattern Recognition and Artificial Intelligence, 26(1), DOI: 10.1142/S0218001412550026 (2012).
**************************************************************************************************************/

#ifndef _ED_
#define _ED_

#include <opencv2/opencv.hpp>
#include "EDColor.h"

// 添加命名空间以避免与项目中的类冲突
namespace edlines {

/// 特殊宏定义
#define EDGE_VERTICAL   1       // 垂直边缘标识
#define EDGE_HORIZONTAL 2       // 水平边缘标识

#define ANCHOR_PIXEL  254       // 锚点像素值
#define EDGE_PIXEL    255       // 边缘像素值

#define ED_LEFT  1              // 左方向
#define ED_RIGHT 2              // 右方向
#define ED_UP    3              // 上方向
#define ED_DOWN  4              // 下方向

/**
 * @brief 梯度算子枚举
 * @details 定义了可用的梯度计算算子类型
 */
enum GradientOperator {
    PREWITT_OPERATOR = 101,     // Prewitt算子，简单的梯度计算
    SOBEL_OPERATOR = 102,       // Sobel算子，常用的梯度计算
    SCHARR_OPERATOR = 103,      // Scharr算子，高精度的梯度计算
    LSD_OPERATOR = 104          // LSD算子，用于LSD算法的梯度计算
};

/**
 * @brief 栈节点结构体
 * @details 用于边缘检测过程中的栈操作，存储边缘点信息
 */
struct StackNode {
	int r, c;                   // 起始像素点的行列坐标
	int parent;                 // 父链索引，-1表示没有父链
	int dir;                    // 应该前进的方向
};

/**
 * @brief 链结构体
 * @details 用于边缘链接过程中，存储边缘链的信息
 */
struct Chain {

	int dir;                    // 链的方向
	int len;                    // 链中的像素数量
	int parent;                 // 该节点的父节点索引，-1表示没有父节点
	int children[2];            // 该节点的子节点索引数组，-1表示没有子节点
	cv::Point *pixels;          // 指向像素数组开头的指针
};

/**
 * @brief Edge Drawing (ED) 类
 * @details 实现了Edge Drawing算法，用于实时边缘和线段检测
 */
class ED {
							
public:
	/**
	 * @brief 主构造函数
	 * @details 从源图像创建ED对象，执行边缘检测
	 * @param _srcImage 输入源图像
	 * @param _op 梯度算子类型，默认为PREWITT_OPERATOR
	 * @param _gradThresh 梯度阈值，默认为20
	 * @param _anchorThresh 锚点阈值，默认为0
	 * @param _scanInterval 扫描间隔，默认为1
	 * @param _minPathLen 最小路径长度，默认为10
	 * @param _sigma 高斯平滑的sigma值，默认为1.0
	 * @param _sumFlag 求和标志，默认为true
	 */
	ED(cv::Mat _srcImage, GradientOperator _op = PREWITT_OPERATOR, int _gradThresh = 20, int _anchorThresh = 0, int _scanInterval = 1, int _minPathLen = 10, double _sigma = 1.0, bool _sumFlag = true);
	
	/**
	 * @brief 拷贝构造函数
	 * @details 从另一个ED对象创建副本
	 * @param cpyObj 要拷贝的ED对象
	 */
	ED(const ED &cpyObj); 
	
	/**
	 * @brief 从梯度图像和方向图像创建ED对象
	 * @details 直接使用预先计算好的梯度和方向图像进行边缘检测
	 * @param gradImg 梯度图像数据指针
	 * @param dirImg 方向图像数据指针
	 * @param _width 图像宽度
	 * @param _height 图像高度
	 * @param _gradThresh 梯度阈值
	 * @param _anchorThresh 锚点阈值
	 * @param _scanInterval 扫描间隔，默认为1
	 * @param _minPathLen 最小路径长度，默认为10
	 * @param selectStableAnchors 是否选择稳定锚点，默认为true
	 */
	ED(short* gradImg, uchar *dirImg, int _width, int _height, int _gradThresh, int _anchorThresh, int _scanInterval = 1, int _minPathLen = 10, bool selectStableAnchors = true);
	
	/**
	 * @brief 从EDColor对象创建ED对象
	 * @details 从彩色图像的边缘检测结果创建ED对象
	 * @param cpyObj 要拷贝的EDColor对象
	 */
	ED(EDColor &cpyObj);
	
	/**
	 * @brief 默认构造函数
	 * @details 创建一个空的ED对象
	 */
	ED();

	/**
	 * @brief 获取边缘图像
	 * @return 边缘图像
	 */
	cv::Mat getEdgeImage();
	
	/**
	 * @brief 获取锚点图像
	 * @return 锚点图像
	 */
	cv::Mat getAnchorImage();
	
	/**
	 * @brief 获取平滑后的图像
	 * @return 平滑图像
	 */
	cv::Mat getSmoothImage();
	
	/**
	 * @brief 获取梯度图像
	 * @return 梯度图像
	 */
	cv::Mat getGradImage();
	
	/**
	 * @brief 获取线段数量
	 * @return 检测到的线段数量
	 */
	int getSegmentNo();
	
	/**
	 * @brief 获取锚点数量
	 * @return 检测到的锚点数量
	 */
	int getAnchorNo();
	
	/**
	 * @brief 获取锚点坐标
	 * @return 锚点坐标的向量
	 */
	std::vector<cv::Point> getAnchorPoints();
	
	/**
	 * @brief 获取检测到的线段
	 * @return 线段向量，每个线段由一系列点组成
	 */
	std::vector<std::vector<cv::Point>> getSegments();
	
	/**
	 * @brief 获取排序后的线段
	 * @return 按长度排序后的线段向量
	 */
	std::vector<std::vector<cv::Point>> getSortedSegments();
	
	/**
	 * @brief 绘制特定的线段
	 * @param list 要绘制的线段索引列表
	 * @return 绘制了特定线段的图像
	 */
	cv::Mat drawParticularSegments(std::vector<int> list);

protected:
	int width;                 // 源图像宽度
	int height;                // 源图像高度
	uchar *srcImg;             // 源图像数据指针
	std::vector<std::vector< cv::Point> > segmentPoints; // 线段点集合
	double sigma;              // 高斯平滑的sigma值
	cv::Mat smoothImage;       // 平滑后的图像
	uchar *edgeImg;            // 边缘图像数据指针
	uchar *smoothImg;          // 平滑图像数据指针
	int segmentNos;            // 线段数量
	int minPathLen;            // 最小路径长度
	cv::Mat srcImage;          // 源图像

private:
	/**
	 * @brief 计算梯度
	 * @details 对图像进行梯度计算，生成梯度图像和方向图像
	 */
	void ComputeGradient();
	
	/**
	 * @brief 计算锚点
	 * @details 从梯度图像中检测锚点
	 */
	void ComputeAnchorPoints();
	
	/**
	 * @brief 使用排序后的锚点连接锚点
	 * @details 将排序后的锚点连接成边缘链
	 */
	void JoinAnchorPointsUsingSortedAnchors(); 
	
	/**
	 * @brief 按梯度值对锚点进行排序
	 * @details 根据锚点的梯度值进行排序
	 */
	void sortAnchorsByGradValue();
	
	/**
	 * @brief 按梯度值对锚点进行排序（返回索引数组）
	 * @return 排序后的锚点索引数组
	 */
	int* sortAnchorsByGradValue1();

	/**
	 * @brief 查找最长链
	 * @param chains 链数组
	 * @param root 根链索引
	 * @return 最长链的长度
	 */
	static int LongestChain(Chain *chains, int root);
	
	/**
	 * @brief 检索链数量
	 * @param chains 链数组
	 * @param root 根链索引
	 * @param chainNos 存储链数量的数组
	 * @return 链数量
	 */
	static int RetrieveChainNos(Chain *chains, int root, int chainNos[]);

	int anchorNos;             // 锚点数量
	std::vector<cv::Point> anchorPoints; // 锚点坐标集合
	std::vector<cv::Point> edgePoints;   // 边缘点坐标集合

	cv::Mat edgeImage;         // 边缘图像
	cv::Mat gradImage;         // 梯度图像

	uchar *dirImg;             // 方向图像数据指针
	short *gradImg;            // 梯度图像数据指针

	GradientOperator op;       // 用于梯度计算的算子
	int gradThresh;            // 梯度阈值
	int anchorThresh;          // 锚点阈值
	int scanInterval;          // 扫描间隔
	bool sumFlag;              // 求和标志
};


} // namespace edlines

#endif
