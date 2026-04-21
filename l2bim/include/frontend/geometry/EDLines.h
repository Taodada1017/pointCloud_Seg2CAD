/**************************************************************************************************************
* EDLines 源代码
* 版权所有 (C) Cuneyt Akinlar & Cihan Topal
* 作者邮箱: cuneytakinlar@gmail.com, cihantopal@gmail.com
*
* 如果使用 EDLines 库，请引用以下论文:
*
* [1] C. Akinlar 和 C. Topal, "EDLines: A Real-time Line Segment Detector with a False Detection Control,"
*     Pattern Recognition Letters, 32(13), 1633-1642, DOI: 10.1016/j.patrec.2011.06.001 (2011).
*
* [2] C. Akinlar 和 C. Topal, "EDLines: Realtime Line Segment Detection by Edge Drawing (ED),"
*     IEEE Int'l Conf. on Image Processing (ICIP), 2011年9月.
**************************************************************************************************************/

#ifndef _EDLines_
#define _EDLines_

#include "ED.h"          // 包含边缘检测类
#include "EDColor.h"     // 包含彩色图像边缘检测类
#include "NFA.h"         // 包含非随机算法类

// 添加命名空间以避免与项目中的LineSegment类冲突
namespace edlines {

// 线段端点类型定义
#define EDLINES_SS 0  // 线段起始点-起始点连接
#define EDLINES_SE 1  // 线段起始点-结束点连接
#define EDLINES_ES 2  // 线段结束点-起始点连接
#define EDLINES_EE 3  // 线段结束点-结束点连接

/**
 * @brief 轻量级线段结构体，仅包含线段的起点和终点坐标
 */
struct LS {
	cv::Point2d start;  // 线段起点坐标
	cv::Point2d end;    // 线段终点坐标

	/**
	 * @brief 构造函数
	 * @param _start 线段起点
	 * @param _end 线段终点
	 */
	LS(cv::Point2d _start, cv::Point2d _end)
	{
		start = _start;
		end = _end;
	}
};


/**
 * @brief 详细的线段结构体，包含线段的完整参数
 */
struct LineSegment {
	double a, b;          // 直线方程参数：y = a + bx (当invert=0时) || x = a + by (当invert=1时)
	int invert;           // 是否交换x和y坐标（用于避免垂直线斜率计算中的除零问题）

	double sx, sy;        // 线段起点的x和y坐标
	double ex, ey;        // 线段终点的x和y坐标

	int segmentNo;        // 该线段所属的边缘段编号
	int firstPixelIndex;  // 线段第一个像素在边缘段中的索引
	int len;              // 组成该线段的像素数量

	/**
	 * @brief 构造函数
	 * @param _a 直线方程参数a
	 * @param _b 直线方程参数b
	 * @param _invert 是否交换x和y坐标
	 * @param _sx 起点x坐标
	 * @param _sy 起点y坐标
	 * @param _ex 终点x坐标
	 * @param _ey 终点y坐标
	 * @param _segmentNo 所属边缘段编号
	 * @param _firstPixelIndex 第一个像素索引
	 * @param _len 线段长度（像素数）
	 */
	LineSegment(double _a, double _b, int _invert, double _sx, double _sy, double _ex, double _ey, int _segmentNo, int _firstPixelIndex, int _len) {
		a = _a;
		b = _b;
		invert = _invert;
		sx = _sx;
		sy = _sy;
		ex = _ex;
		ey = _ey;
		segmentNo = _segmentNo;
		firstPixelIndex = _firstPixelIndex;
		len = _len;
	}
};


/**
 * @brief EDLines 类，用于从图像中检测线段
 * @details 继承自 ED 类，实现了基于边缘检测的线段检测算法
 */
class EDLines : public ED {
public:
	/**
	 * @brief 从灰度图像创建 EDLines 对象
	 * @param srcImage 输入灰度图像
	 * @param _line_error 直线拟合误差阈值
	 * @param _min_line_len 最小线段长度（像素），-1表示自动计算
	 * @param _max_distance_between_two_lines 两条线之间的最大距离（用于合并共线线段）
	 * @param _max_error 线段验证的最大误差
	 */
	EDLines(cv::Mat srcImage, double _line_error = 1.0, int _min_line_len = -1, double _max_distance_between_two_lines = 6.0, double _max_error = 1.3);
	
	/**
	 * @brief 从 ED 对象创建 EDLines 对象
	 * @param obj 已有的 ED 对象
	 * @param _line_error 直线拟合误差阈值
	 * @param _min_line_len 最小线段长度（像素），-1表示自动计算
	 * @param _max_distance_between_two_lines 两条线之间的最大距离（用于合并共线线段）
	 * @param _max_error 线段验证的最大误差
	 */
	EDLines(ED obj, double _line_error = 1.0, int _min_line_len = -1, double _max_distance_between_two_lines = 6.0, double _max_error = 1.3);
	
	/**
	 * @brief 从 EDColor 对象创建 EDLines 对象
	 * @param obj 已有的 EDColor 对象
	 * @param _line_error 直线拟合误差阈值
	 * @param _min_line_len 最小线段长度（像素），-1表示自动计算
	 * @param _max_distance_between_two_lines 两条线之间的最大距离（用于合并共线线段）
	 * @param _max_error 线段验证的最大误差
	 */
	EDLines(EDColor obj, double _line_error = 1.0, int _min_line_len = -1, double _max_distance_between_two_lines = 6.0, double _max_error = 1.3);
	
	/**
	 * @brief 默认构造函数
	 */
	EDLines();

	/**
	 * @brief 获取检测到的线段列表
	 * @return 包含线段起点和终点的向量
	 */
	std::vector<LS> getLines();
	
	/**
	 * @brief 获取检测到的线段数量
	 * @return 线段数量
	 */
	int getLinesNo();
	
	/**
	 * @brief 获取只包含线段的图像
	 * @return 线段图像
	 */
	cv::Mat getLineImage();
	
	/**
	 * @brief 在原图上绘制检测到的线段
	 * @return 绘制了线段的原图
	 */
	cv::Mat drawOnImage();

	/**
	 * @brief 将边缘段分割为线段（静态方法，供 EDCircle 使用）
	 * @param x 边缘段像素的x坐标数组
	 * @param y 边缘段像素的y坐标数组
	 * @param noPixels 边缘段的像素数量
	 * @param segmentNo 边缘段编号
	 * @param lines 输出的线段向量
	 * @param min_line_len 最小线段长度
	 * @param line_error 直线拟合误差阈值
	 */
	static void SplitSegment2Lines(double *x, double *y, int noPixels, int segmentNo, std::vector<LineSegment> &lines, int min_line_len = 6, double line_error = 1.0);

private:
	std::vector<LineSegment> lines;                    // 检测到的有效线段
	std::vector<LineSegment> invalidLines;             // 无效的线段
	std::vector<LS> linePoints;                        // 转换为LS格式的线段
	int linesNo;                                       // 线段数量
	int min_line_len;                                  // 最小线段长度
	double line_error;                                 // 直线拟合误差阈值
	double max_distance_between_two_lines;             // 两条线之间的最大距离（用于合并）
	double max_error;                                  // 线段验证的最大误差
	double prec;                                       // 计算精度
	NFALUT *nfa;                                       // 非随机算法查找表
	
	/**
	 * @brief 计算最小线段长度
	 * @return 最小线段长度
	 */
	int ComputeMinLineLength();
	
	/**
	 * @brief 将边缘段分割为线段
	 * @param x 边缘段像素的x坐标数组
	 * @param y 边缘段像素的y坐标数组
	 * @param noPixels 边缘段的像素数量
	 * @param segmentNo 边缘段编号
	 */
	void SplitSegment2Lines(double *x, double *y, int noPixels, int segmentNo);
	
	/**
	 * @brief 合并共线的线段
	 */
	void JoinCollinearLines();
	
	/**
	 * @brief 验证线段的有效性
	 */
	void ValidateLineSegments();
	
	/**
	 * @brief 验证线段的矩形区域
	 * @param x 像素x坐标数组
	 * @param y 像素y坐标数组
	 * @param ls 线段指针
	 * @return 线段是否有效
	 */
	bool ValidateLineSegmentRect(int *x, int *y, LineSegment *ls);
	
	/**
	 * @brief 尝试合并两条线段
	 * @param ls1 第一条线段
	 * @param ls2 第二条线段
	 * @param changeIndex 变化索引
	 * @return 是否成功合并
	 */
	bool TryToJoinTwoLineSegments(LineSegment *ls1, LineSegment *ls2, int changeIndex);
	
	/**
	 * @brief 计算点到直线的最小距离
	 * @param x1 点的x坐标
	 * @param y1 点的y坐标
	 * @param a 直线方程参数a
	 * @param b 直线方程参数b
	 * @param invert 是否交换x和y坐标
	 * @return 点到直线的最小距离
	 */
	static double ComputeMinDistance(double x1, double y1, double a, double b, int invert);
	
	/**
	 * @brief 计算点到直线的最近点
	 * @param x1 点的x坐标
	 * @param y1 点的y坐标
	 * @param a 直线方程参数a
	 * @param b 直线方程参数b
	 * @param invert 是否交换x和y坐标
	 * @param xOut 输出的最近点x坐标
	 * @param yOut 输出的最近点y坐标
	 */
	static void ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double &xOut, double &yOut);
	
	/**
	 * @brief 直线拟合（已知invert参数）
	 * @param x 点的x坐标数组
	 * @param y 点的y坐标数组
	 * @param count 点的数量
	 * @param a 输出的直线参数a
	 * @param b 输出的直线参数b
	 * @param invert 是否交换x和y坐标
	 */
	static void LineFit(double *x, double *y, int count, double &a, double &b, int invert);
	
	/**
	 * @brief 直线拟合（自动确定invert参数）
	 * @param x 点的x坐标数组
	 * @param y 点的y坐标数组
	 * @param count 点的数量
	 * @param a 输出的直线参数a
	 * @param b 输出的直线参数b
	 * @param e 输出的拟合误差
	 * @param invert 输出的invert参数
	 */
	static void LineFit(double *x, double *y, int count, double &a, double &b, double &e, int &invert);
	
	/**
	 * @brief 计算两条线段之间的最小距离
	 * @param ls1 第一条线段
	 * @param ls2 第二条线段
	 * @param pwhich 输出的距离类型（SS, SE, ES, EE）
	 * @return 两条线段之间的最小距离
	 */
	static double ComputeMinDistanceBetweenTwoLines(LineSegment *ls1, LineSegment *ls2, int *pwhich);
	
	/**
	 * @brief 更新线段参数
	 * @param ls 线段指针
	 */
	static void UpdateLineParameters(LineSegment *ls);
	
	/**
	 * @brief 枚举矩形区域内的点
	 * @param sx 矩形起点x坐标
	 * @param sy 矩形起点y坐标
	 * @param ex 矩形终点x坐标
	 * @param ey 矩形终点y坐标
	 * @param ptsx 输出的x坐标数组
	 * @param ptsy 输出的y坐标数组
	 * @param pNoPoints 输出的点数量
	 * @param maxPoints 数组的最大容量，防止缓冲区溢出
	 */
	static void EnumerateRectPoints(double sx, double sy, double ex, double ey,int ptsx[], int ptsy[], int *pNoPoints, int maxPoints);

	// 工具数学函数
	
};

} // namespace edlines

#endif 
