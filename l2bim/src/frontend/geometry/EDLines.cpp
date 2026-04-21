#include "EDLines.h"
#include "EDColor.h"
#include "NFA.h"

using namespace cv;
using namespace std;

// 添加命名空间以匹配头文件
namespace edlines {

/**
 * @brief 从灰度图像创建 EDLines 对象
 * @param srcImage 输入灰度图像
 * @param _line_error 直线拟合误差阈值，默认为1.0
 * @param _min_line_len 最小线段长度（像素），-1表示自动计算，默认为-1
 * @param _max_distance_between_two_lines 两条线之间的最大距离（用于合并共线线段），默认为6.0
 * @param _max_error 线段验证的最大误差，默认为1.3
 * @details 从灰度图像创建EDLines对象，执行边缘检测、线段分割、共线合并和线段验证
 */
EDLines::EDLines(Mat srcImage ,  double _line_error, int _min_line_len, double _max_distance_between_two_lines , double _max_error)
	:ED(srcImage, SOBEL_OPERATOR, 36, 8) 
{
	// 初始化线段检测参数
	min_line_len = _min_line_len;           // 最小线段长度
	line_error = _line_error;               // 直线拟合误差阈值
	max_distance_between_two_lines = _max_distance_between_two_lines; // 共线线段合并的最大距离
	max_error = _max_error;                 // 线段验证的最大误差

	// 如果没有提供最小线段长度，自动计算
	if(min_line_len == -1) 
		min_line_len = ComputeMinLineLength();

	// 确保最小线段长度不小于9像素，避免结果中出现过小的线段
	if (min_line_len < 9)
		min_line_len = 9;

	// 计算用于直线拟合的临时缓冲区大小
	// 初始大小为图像宽高之和的8倍
	size_t buffer_size = (width + height) * 8;
	// 遍历所有边缘段，确保缓冲区能容纳最长的边缘段
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		auto segment_size = segmentPoints[segmentNumber].size();
		buffer_size = std::max(buffer_size, segment_size);
	}
	// 分配临时缓冲区，用于存储边缘段的像素坐标
	double* x = new double[buffer_size];
	double* y = new double[buffer_size];

	linesNo = 0; // 初始化检测到的线段数量
	
	// 遍历所有边缘段，将每个边缘段分割为线段
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		std::vector<Point> segment = segmentPoints[segmentNumber];
		// 将边缘段的像素坐标复制到临时缓冲区
		for (int k = 0; k < segment.size(); k++) {
			x[k] = segment[k].x;
			y[k] = segment[k].y;
		}
		// 将边缘段分割为线段
		SplitSegment2Lines(x, y, (int)segment.size(), segmentNumber);
	}

	/*----------- 合并共线线段 ----------------*/
	JoinCollinearLines();

	/*----------- 验证线段 ----------------*/
#define PRECISON_ANGLE 22.5 
	// 将角度精度转换为弧度
	prec = (PRECISON_ANGLE / 180)*M_PI;
	// 设置概率阈值
	double prob = 0.125;
#undef PRECISON_ANGLE

	// 计算logNT值，用于NFA算法
	double logNT = 2.0*(log10((double)width) + log10((double)height));

	// 计算NFA查找表大小
	int lutSize = (width + height) / 8;
	// 创建NFA查找表，用于快速验证线段
	nfa = new NFALUT(lutSize, prob, logNT);
	
	// 使用NFA算法验证所有线段
	ValidateLineSegments();

	// 删除无效线段，只保留有效线段
	int size = (int)lines.size();
	for (int i = 1; i <= size - linesNo; i++)
		lines.pop_back();
	
	// 将有效线段转换为LS格式，方便外部使用
	for (int i = 0; i<linesNo; i++) {
		Point2d start(lines[i].sx, lines[i].sy);
		Point2d end(lines[i].ex, lines[i].ey);
		
		linePoints.push_back(LS(start, end));
	} //end-for

	// 释放临时缓冲区
	delete[] x;
	delete[] y;
	delete nfa;
}


/**
 * @brief 从ED对象创建 EDLines 对象
 * @param obj 已有的ED对象
 * @param _line_error 直线拟合误差阈值，默认为1.0
 * @param _min_line_len 最小线段长度（像素），-1表示自动计算，默认为-1
 * @param _max_distance_between_two_lines 两条线之间的最大距离（用于合并共线线段），默认为6.0
 * @param _max_error 线段验证的最大误差，默认为1.3
 * @details 从已有的ED对象创建EDLines对象，避免重复进行边缘检测
 */
EDLines::EDLines(ED obj, double _line_error, int _min_line_len, double _max_distance_between_two_lines, double _max_error)
	:ED(obj) 
{
	// 初始化线段检测参数
	min_line_len = _min_line_len;           // 最小线段长度
	line_error = _line_error;               // 直线拟合误差阈值
	max_distance_between_two_lines = _max_distance_between_two_lines; // 共线线段合并的最大距离
	max_error = _max_error;                 // 线段验证的最大误差

	// 如果没有提供最小线段长度，自动计算
	if (min_line_len == -1)
		min_line_len = ComputeMinLineLength();

	// 确保最小线段长度不小于9像素，避免结果中出现过小的线段
	if (min_line_len < 9)
		min_line_len = 9;

	// 计算用于直线拟合的临时缓冲区大小
	// 初始大小为图像宽高之和的8倍
	size_t buffer_size = (width + height) * 8;
	// 遍历所有边缘段，确保缓冲区能容纳最长的边缘段
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		auto segment_size = segmentPoints[segmentNumber].size();
		buffer_size = std::max(buffer_size, segment_size);
	}
	// 分配临时缓冲区，用于存储边缘段的像素坐标
	double* x = new double[buffer_size];
	double* y = new double[buffer_size];

	linesNo = 0; // 初始化检测到的线段数量

	// 遍历所有边缘段，将每个边缘段分割为线段
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		std::vector<Point> segment = segmentPoints[segmentNumber];
		// 将边缘段的像素坐标复制到临时缓冲区
		for (int k = 0; k < segment.size(); k++) {
			x[k] = segment[k].x;
			y[k] = segment[k].y;
		}
		// 将边缘段分割为线段
		SplitSegment2Lines(x, y, (int)segment.size(), segmentNumber);
	}

	/*----------- 合并共线线段 ----------------*/
	JoinCollinearLines();

	/*----------- 验证线段 ----------------*/
#define PRECISON_ANGLE 22.5 
	// 将角度精度转换为弧度
	prec = (PRECISON_ANGLE / 180)*M_PI;
	// 设置概率阈值
	double prob = 0.125;
#undef PRECISON_ANGLE

	// 计算logNT值，用于NFA算法
	double logNT = 2.0*(log10((double)width) + log10((double)height));

	// 计算NFA查找表大小
	int lutSize = (width + height) / 8;
	// 创建NFA查找表，用于快速验证线段
	nfa = new NFALUT(lutSize, prob, logNT);

	// 使用NFA算法验证所有线段
	ValidateLineSegments();

	// 删除无效线段，只保留有效线段
	int size = (int)lines.size();
	for (int i = 1; i <= size - linesNo; i++)
		lines.pop_back();

	// 将有效线段转换为LS格式，方便外部使用
	for (int i = 0; i<linesNo; i++) {
		Point2d start(lines[i].sx, lines[i].sy);
		Point2d end(lines[i].ex, lines[i].ey);

		linePoints.push_back(LS(start, end));
	} //end-for

	// 释放临时缓冲区
	delete[] x;
	delete[] y;
	delete nfa;
}

/**
 * @brief 从EDColor对象创建 EDLines 对象
 * @param obj 已有的EDColor对象
 * @param _line_error 直线拟合误差阈值，默认为1.0
 * @param _min_line_len 最小线段长度（像素），-1表示自动计算，默认为-1
 * @param _max_distance_between_two_lines 两条线之间的最大距离（用于合并共线线段），默认为6.0
 * @param _max_error 线段验证的最大误差，默认为1.3
 * @details 从已有的EDColor对象创建EDLines对象，利用彩色图像的边缘检测结果
 */
EDLines::EDLines(EDColor obj, double _line_error, int _min_line_len, double _max_distance_between_two_lines, double _max_error)
	:ED(obj)
{
	// 初始化线段检测参数
	min_line_len = _min_line_len;           // 最小线段长度
	line_error = _line_error;               // 直线拟合误差阈值
	max_distance_between_two_lines = _max_distance_between_two_lines; // 共线线段合并的最大距离
	max_error = _max_error;                 // 线段验证的最大误差

	// 如果没有提供最小线段长度，自动计算
	if (min_line_len == -1)
		min_line_len = ComputeMinLineLength();

	// 确保最小线段长度不小于9像素，避免结果中出现过小的线段
	if (min_line_len < 9)
		min_line_len = 9;

	// 计算用于直线拟合的临时缓冲区大小
	// 初始大小为图像宽高之和的8倍
	size_t buffer_size = (width + height) * 8;
	// 遍历所有边缘段，确保缓冲区能容纳最长的边缘段
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		auto segment_size = segmentPoints[segmentNumber].size();
		buffer_size = std::max(buffer_size, segment_size);
	}
	// 分配临时缓冲区，用于存储边缘段的像素坐标
	double* x = new double[buffer_size];
	double* y = new double[buffer_size];

	linesNo = 0; // 初始化检测到的线段数量

	// 遍历所有边缘段，将每个边缘段分割为线段
	for (int segmentNumber = 0; segmentNumber < segmentPoints.size(); segmentNumber++) {
		std::vector<Point> segment = segmentPoints[segmentNumber];
		// 将边缘段的像素坐标复制到临时缓冲区
		for (int k = 0; k < segment.size(); k++) {
			x[k] = segment[k].x;
			y[k] = segment[k].y;
		}
		// 将边缘段分割为线段
		SplitSegment2Lines(x, y, (int)segment.size(), segmentNumber);
	}

	/*----------- 合并共线线段 ----------------*/
	JoinCollinearLines();

	/*----------- 线段验证准备 ----------------*/
#define PRECISON_ANGLE 22.5 
	// 将角度精度转换为弧度
	prec = (PRECISON_ANGLE / 180)*M_PI;
	// 设置概率阈值
	double prob = 0.125;
#undef PRECISON_ANGLE

	// 计算logNT值，用于NFA算法
	double logNT = 2.0*(log10((double)width) + log10((double)height));

	// 计算NFA查找表大小
	int lutSize = (width + height) / 8;
	// 创建NFA查找表，用于快速验证线段
	nfa = new NFALUT(lutSize, prob, logNT);

	// 注意：由于边缘段在EDColor中已经过验证，因此在线段检测中不再重复验证
	// 这是与其他构造函数的主要区别
	// TODO :: 可以考虑添加进一步的验证
	// ValidateLineSegments(); 

	// 删除无效线段，只保留有效线段
	int size = (int)lines.size();
	for (int i = 1; i <= size - linesNo; i++)
		lines.pop_back();

	// 将有效线段转换为LS格式，方便外部使用
	for (int i = 0; i<linesNo; i++) {
		Point2d start(lines[i].sx, lines[i].sy);
		Point2d end(lines[i].ex, lines[i].ey);

		linePoints.push_back(LS(start, end));
	} //end-for

	// 释放临时缓冲区
	delete[] x;
	delete[] y;
	delete nfa;
}

/**
 * @brief 默认构造函数
 * @details 创建一个空的EDLines对象，需要后续初始化
 */
EDLines::EDLines()
{
	//
}

vector<LS> EDLines::getLines()
{
	return linePoints;
}

int EDLines::getLinesNo()
{
	return linesNo;
}

Mat EDLines::getLineImage()
{
	Mat lineImage = Mat(height, width, CV_8UC1, Scalar(255));
	for (int i = 0; i < linesNo; i++) {
		line(lineImage, linePoints[i].start, linePoints[i].end, Scalar(0), 1, LINE_AA, 0);
	}

	return lineImage;
}

Mat EDLines::drawOnImage()
{
	Mat colorImage = Mat(height, width, CV_8UC1, srcImg);
	cvtColor(colorImage, colorImage, COLOR_GRAY2BGR);
	for (int i = 0; i < linesNo; i++) {
		line(colorImage, linePoints[i].start, linePoints[i].end, Scalar(0, 255, 0), 1, LINE_AA, 0); // draw lines as green on image
	}

	return colorImage;
}

//-----------------------------------------------------------------------------------------
/**
 * @brief 使用NFA公式计算最小线段长度
 * @return 最小线段长度
 * @details 根据图像宽度和高度计算理论最小线段长度，
 *          将理论值除以2是因为现在使用宽度为2的线支持区域矩形测试短线段，
 *          这意味着长度为"l"的线段在其线支持区域矩形内有"2*l"个像素，
 *          因此长度为"l"的线段有机会通过NFA验证。
 */
int EDLines::ComputeMinLineLength() {
	double logNT = 2.0*(log10((double)width) + log10((double)height));
	return (int) round((-logNT / log10(0.125))*0.5);
} //end-ComputeMinLineLength

//-----------------------------------------------------------------
/**
 * @brief 将完整的边缘段分割为线段
 * @param x 边缘段像素的x坐标数组
 * @param y 边缘段像素的y坐标数组
 * @param noPixels 边缘段的像素数量
 * @param segmentNo 边缘段编号
 * @details 该函数将一个完整的边缘段分割为多个线段，
 *          首先尝试拟合最小长度的线段，然后尝试扩展该线段，
 *          直到无法继续扩展或达到边缘段末尾。
 */
void EDLines::SplitSegment2Lines(double * x, double * y, int noPixels, int segmentNo)
{
	// 线段在边缘段内的第一个像素索引
	int firstPixelIndex = 0;

	// 当剩余像素数量大于等于最小线段长度时，继续分割
	while (noPixels >= min_line_len) {
		// 开始时，对最小线段长度的像素进行直线拟合
		bool valid = false;
		double lastA, lastB, error; // 直线参数和拟合误差
		int lastInvert; // 是否交换x和y坐标

		// 尝试找到可以拟合直线的最小线段长度的像素
		while (noPixels >= min_line_len) {
			LineFit(x, y, min_line_len, lastA, lastB, error, lastInvert);
			if (error <= 0.5) { // 拟合误差足够小，有效
				valid = true;
				break;
			}

#if 1
			noPixels -= 1;   // 缓慢前进，每次移动1个像素
			x += 1; y += 1;
			firstPixelIndex += 1;
#else
			noPixels -= 2;   // 快速前进，每次移动2个像素（为了速度）
			x += 2; y += 2;
			firstPixelIndex += 2;
#endif
		} //end-while

		if (valid == false) return; // 无法找到有效的直线拟合，退出

		// 现在尝试扩展这条直线
		int index = min_line_len; // 当前处理的像素索引
		int len = min_line_len;   // 当前线段长度

		// 继续处理剩余的像素
		while (index < noPixels) {
			int startIndex = index;       // 当前扩展起始索引
			int lastGoodIndex = index - 1; // 最后一个符合条件的像素索引
			int goodPixelCount = 0;       // 符合条件的像素计数
			int badPixelCount = 0;        // 不符合条件的像素计数
			
			// 检查当前像素是否符合直线条件
			while (index < noPixels) {
				// 计算当前像素到直线的距离
				double d = ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert);

				if (d <= line_error) { // 距离在误差范围内，符合条件
					lastGoodIndex = index;
					goodPixelCount++;
					badPixelCount = 0;
				}
				else { // 距离超出误差范围，不符合条件
					badPixelCount++;
					if (badPixelCount >= 5) break; // 连续5个不符合条件的像素，停止扩展
				} //end-if

				index++;
			} //end-while

			if (goodPixelCount >= 2) { // 有足够多的符合条件的像素，继续扩展线段
				len += lastGoodIndex - startIndex + 1; // 更新线段长度
				LineFit(x, y, len, lastA, lastB, lastInvert);  // 重新拟合直线（更快的LineFit）
				index = lastGoodIndex + 1; // 继续处理下一个像素
			} // end-if

			// 无法继续扩展或已处理完所有像素，结束当前线段
			if (goodPixelCount < 2 || index >= noPixels) {
				// 线段结束，计算端点
				double sx, sy, ex, ey; // 线段的起点和终点

				// 找到线段的起点
				int index = 0;
				while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index++;
				ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, sx, sy);
				int noSkippedPixels = index; // 跳过的像素数量

				// 找到线段的终点
				index = lastGoodIndex;
				while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index--;
				ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, ex, ey);

				if ((sx == ex) && (sy == ey)) // 起点和终点相同，跳过
					break;

				// 将线段添加到结果中
				lines.push_back(LineSegment(lastA, lastB, lastInvert, sx, sy, ex, ey, segmentNo, firstPixelIndex + noSkippedPixels, index - noSkippedPixels + 1));
				linesNo++; // 线段数量加1
				len = index + 1; // 更新处理的像素长度
				break;
			} //end-else
		} //end-while

		// 更新剩余像素数量和指针位置
		noPixels -= len;
		x += len;
		y += len;
		firstPixelIndex += len;
	} //end-while
}

//------------------------------------------------------------------
/**
 * @brief 遍历原始线段，合并属于同一边缘段的共线线段
 */
void EDLines::JoinCollinearLines()
{
	int lastLineIndex = -1;   // 合并后的线段列表中最后一条线段的索引
	int i = 0;
	// 遍历所有线段
	while (i < linesNo) {
		int segmentNo = lines[i].segmentNo; // 当前处理的线段所属的边缘段编号

		lastLineIndex++;
		// 如果当前索引不是lastLineIndex，复制线段
		if (lastLineIndex != i) 
			lines[lastLineIndex] = lines[i];
		
		int firstLineIndex = lastLineIndex;  // 当前边缘段中第一条线段的索引

		int count = 1; // 当前边缘段中的线段数量
		// 遍历当前边缘段的后续线段
		for (int j = i + 1; j< linesNo; j++) {
			if (lines[j].segmentNo != segmentNo) break; // 不属于同一边缘段，退出循环

			// 尝试将当前线段与前一条线段合并
			if (TryToJoinTwoLineSegments(&lines[lastLineIndex], &lines[j],
				lastLineIndex) == false) {
				// 合并失败，将当前线段添加到结果中
				lastLineIndex++;
				if (lastLineIndex != j) 
					lines[lastLineIndex] = lines[j];
				
			} //end-if

			count++;
		} //end-for

		// 尝试合并当前边缘段的第一条和最后一条线段
		if (firstLineIndex != lastLineIndex) {
			if (TryToJoinTwoLineSegments(&lines[firstLineIndex], &lines[lastLineIndex],
				firstLineIndex)) {
				lastLineIndex--; // 合并成功，减少线段计数
			} //end-if
		} //end-if

		i += count; // 跳过已处理的线段
	} //end-while

	linesNo = lastLineIndex + 1; // 更新线段数量
}

/**
 * @brief 验证线段的有效性，使用NFA算法筛选真实线段
 */
void EDLines::ValidateLineSegments()
{
	// 分配用于存储矩形区域点的临时数组
	int *x = new int[(width + height) * 4];
	int *y = new int[(width + height) * 4];

	int noValidLines = 0; // 有效线段的数量
	// 遍历所有线段
	for (int i = 0; i< linesNo; i++) {
		LineSegment *ls = &lines[i]; // 当前处理的线段

		// 计算线段的角度
		double lineAngle;

		if (ls->invert == 0) {
			// 直线方程为 y = a + bx
			lineAngle = atan(ls->b);
		}
		else {
			// 直线方程为 x = a + by
			lineAngle = atan(1.0 / ls->b);
		} //end-else

		// 确保角度在[0, π)范围内
		if (lineAngle < 0) lineAngle += M_PI;

		// 获取当前线段所属边缘段的像素
		Point *pixels = &(segmentPoints[ls->segmentNo][0]);
		int noPixels = ls->len; // 线段的像素数量

		bool valid = false; // 线段是否有效的标志

		// 非常长的线段直接通过验证，它们几乎从不失效
		if (ls->len >= 80) {
			valid = true;
		}
		// 短线段（长度<=25）使用宽度为2的线支持区域矩形进行验证
		else if (ls->len <= 25) {
			valid = ValidateLineSegmentRect(x, y, ls);
		}
		// 中等长度的线段
		else {
			// 首先使用宽度为1的线支持区域矩形进行验证（为了速度）
			// 如果线段仍然无效，则尝试使用宽度为2的线支持区域矩形
			// 如果线段两次测试都失败，则被丢弃
			int aligned = 0; // 方向一致的像素数量
			int count = 0;   // 有效像素数量
			for (int j = 0; j<noPixels; j++) {
				int r = pixels[j].x; // 当前像素的行坐标
				int c = pixels[j].y; // 当前像素的列坐标

				// 跳过边界像素
				if (r <= 0 || r >= height - 1 || c <= 0 || c >= width - 1) continue;

				count++;

				// 使用简单的[-1 -1 -1; 1 1 1]滤波器计算梯度gx和gy
				// 以下是更快的计算方法：
				// A B C
				// D x E
				// F G H
				// gx = (C-A) + (E-D) + (H-F)
				// gy = (F-A) + (G-B) + (H-C)
				// 优化后：
				// com1 = (H-A)
				// com2 = (C-F)
				// gx = com1 + com2 + (E-D) = (H-A) + (C-F) + (E-D) = (C-A) + (E-D) + (H-F)
				// gy = com2 - com1 + (G-B) = (H-A) - (C-F) + (G-B) = (F-A) + (G-B) + (H-C)
				int com1 = srcImg[(r + 1)*width + c + 1] - srcImg[(r - 1)*width + c - 1];
				int com2 = srcImg[(r - 1)*width + c + 1] - srcImg[(r + 1)*width + c - 1];

				int gx = com1 + com2 + srcImg[r*width + c + 1] - srcImg[r*width + c - 1];
				int gy = com1 - com2 + srcImg[(r + 1)*width + c] - srcImg[(r - 1)*width + c];
				
				// 计算当前像素的梯度角度
				double pixelAngle = nfa->myAtan2((double)gx, (double)-gy);
				// 计算与线段角度的差异
				double diff = fabs(lineAngle - pixelAngle);

				// 如果角度差异在允许范围内，则认为方向一致
				if (diff <= prec || diff >= M_PI - prec) aligned++;
			} //end-for

			// 使用NFA查找表快速检查验证
			valid = nfa->checkValidationByNFA(count, aligned);
			// 如果验证失败，尝试使用矩形区域进行验证
			if (valid == false) valid = ValidateLineSegmentRect(x, y, ls);
		} //end-else

		// 如果线段有效，保留它
		if (valid) {
			if (i != noValidLines) lines[noValidLines] = lines[i];
			noValidLines++;
		}
		// 否则，将其添加到无效线段列表
		else {
			invalidLines.push_back(lines[i]);
		} //end-else
	} //end-for

	// 更新有效线段数量
	linesNo = noValidLines;

	// 释放临时数组
	delete x;
	delete y;
}

/**
 * @brief 使用矩形区域验证线段的有效性
 * @param x 存储矩形区域点x坐标的数组
 * @param y 存储矩形区域点y坐标的数组
 * @param ls 要验证的线段指针
 * @return 线段是否有效的布尔值
 * @details 通过检查线段周围矩形区域内的像素梯度方向，使用NFA算法验证线段的真实性
 */
bool EDLines::ValidateLineSegmentRect(int * x, int * y, LineSegment * ls)
{
	// 计算线段的角度
	double lineAngle;

	if (ls->invert == 0) {
		// 直线方程为 y = a + bx
		lineAngle = atan(ls->b);
	}
	else {
		// 直线方程为 x = a + by
		lineAngle = atan(1.0 / ls->b);
	} //end-else

	// 确保角度在[0, π)范围内
	if (lineAngle < 0) lineAngle += M_PI;

	int noPoints = 0;

	// 枚举线段周围矩形区域内的所有像素点
	int maxPoints = (width + height) * 4; // 与分配的数组大小保持一致
	EnumerateRectPoints(ls->sx, ls->sy, ls->ex, ls->ey, x, y, &noPoints, maxPoints);

	int count = 0;   // 有效像素数量
	int aligned = 0; // 方向一致的像素数量

	// 遍历矩形区域内的所有点
	for (int i = 0; i<noPoints; i++) {
		int r = y[i]; // 当前点的行坐标
		int c = x[i]; // 当前点的列坐标

		// 跳过边界像素
		if (r <= 0 || r >= height - 1 || c <= 0 || c >= width - 1) continue;

		count++;

		// 使用简单的[-1 -1 -1; 1 1 1]滤波器计算梯度gx和gy
		// 以下是更快的计算方法：
		// A B C
		// D x E
		// F G H
		// gx = (C-A) + (E-D) + (H-F)
		// gy = (F-A) + (G-B) + (H-C)
		// 优化后：
		// com1 = (H-A)
		// com2 = (C-F)
		// gx = com1 + com2 + (E-D) = (H-A) + (C-F) + (E-D) = (C-A) + (E-D) + (H-F)
		// gy = com2 - com1 + (G-B) = (H-A) - (C-F) + (G-B) = (F-A) + (G-B) + (H-C)
		int com1 = srcImg[(r + 1)*width + c + 1] - srcImg[(r - 1)*width + c - 1];
		int com2 = srcImg[(r - 1)*width + c + 1] - srcImg[(r + 1)*width + c - 1];

		int gx = com1 + com2 + srcImg[r*width + c + 1] - srcImg[r*width + c - 1];
		int gy = com1 - com2 + srcImg[(r + 1)*width + c] - srcImg[(r - 1)*width + c];
		// 计算当前像素的梯度角度
		double pixelAngle = nfa->myAtan2((double)gx, (double)-gy);
		// 计算与线段角度的差异
		double diff = fabs(lineAngle - pixelAngle);

		// 如果角度差异在允许范围内，则认为方向一致
		if (diff <= prec || diff >= M_PI - prec) aligned++;
	} //end-for

	// 使用NFA算法验证线段
	return nfa->checkValidationByNFA(count, aligned);
}



/**
 * @brief 计算点到直线的最小距离
 * @param x1 点的x坐标
 * @param y1 点的y坐标
 * @param a 直线方程参数a
 * @param b 直线方程参数b
 * @param invert 是否交换x和y坐标
 * @return 点到直线的最小距离
 * @details 根据直线方程的形式（y = a + bx 或 x = a + by），计算点到直线的垂直距离
 */
double EDLines::ComputeMinDistance(double x1, double y1, double a, double b, int invert)
{
	double x2, y2; // 直线上距离点(x1,y1)最近的点的坐标

	if (invert == 0) {
		// 直线方程为 y = a + bx
		if (b == 0) {
			// 水平线，最近点的x坐标与原坐标相同
			x2 = x1;
			y2 = a;
		}
		else {
			// 计算过点(x1,y1)且垂直于原直线的直线方程
			// 原直线斜率为b，垂线斜率为-1/b
			double d = -1.0 / b;
			double c = y1 - d*x1; // 垂线的截距

			// 计算两条直线的交点
			x2 = (a - c) / (d - b);
			y2 = a + b*x2;
		} //end-else

	}
	else {
		// 直线方程为 x = a + by
		if (b == 0) {
			// 垂直线，最近点的y坐标与原坐标相同
			x2 = a;
			y2 = y1;
		}
		else {
			// 计算过点(x1,y1)且垂直于原直线的直线方程
			// 原直线斜率为1/b，垂线斜率为-b
			double d = -1.0 / b;
			double c = x1 - d*y1; // 垂线的截距

			// 计算两条直线的交点
			y2 = (a - c) / (d - b);
			x2 = a + b*y2;
		} //end-else
	} //end-else

	// 计算两点之间的距离
	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

//---------------------------------------------------------------------------------
/**
 * @brief 计算点到直线的最近点坐标
 * @param x1 输入点的x坐标
 * @param y1 输入点的y坐标
 * @param a 直线方程参数a
 * @param b 直线方程参数b
 * @param invert 是否交换x和y坐标
 * @param xOut 输出最近点的x坐标
 * @param yOut 输出最近点的y坐标
 * @details 根据直线方程的形式（y = a + bx 或 x = a + by），计算直线上距离给定点最近的点
 */
void EDLines::ComputeClosestPoint(double x1, double y1, double a, double b, int invert, double &xOut, double &yOut)
{
	double x2, y2; // 直线上距离点(x1,y1)最近的点的坐标

	if (invert == 0) {
		// 直线方程为 y = a + bx
		if (b == 0) {
			// 水平线，最近点的x坐标与原坐标相同
			x2 = x1;
			y2 = a;
		}
		else {
			// 计算过点(x1,y1)且垂直于原直线的直线方程
			// 原直线斜率为b，垂线斜率为-1/b
			double d = -1.0 / b;
			double c = y1 - d*x1; // 垂线的截距

			// 计算两条直线的交点
			x2 = (a - c) / (d - b);
			y2 = a + b*x2;
		} //end-else

	}
	else {
		// 直线方程为 x = a + by
		if (b == 0) {
			// 垂直线，最近点的y坐标与原坐标相同
			x2 = a;
			y2 = y1;
		}
		else {
			// 计算过点(x1,y1)且垂直于原直线的直线方程
			// 原直线斜率为1/b，垂线斜率为-b
			double d = -1.0 / b;
			double c = x1 - d*y1; // 垂线的截距

			// 计算两条直线的交点
			y2 = (a - c) / (d - b);
			x2 = a + b*y2;
		} //end-else
	} //end-else

	// 输出结果
	xOut = x2;
	yOut = y2;
}

//-----------------------------------------------------------------------------------
/**
 * @brief 直线拟合（已知直线方向）
 * @param x 点的x坐标数组
 * @param y 点的y坐标数组
 * @param count 点的数量
 * @param a 输出的直线参数a
 * @param b 输出的直线参数b
 * @param invert 是否交换x和y坐标（0表示y=a+bx，1表示x=a+by）
 * @details 已知直线方向的情况下，使用最小二乘法拟合直线
 */
void EDLines::LineFit(double * x, double * y, int count, double &a, double &b, int invert)
{
	if (count<2) return; // 至少需要两个点才能拟合直线

	// 计算总和：S=count, Sx=Σx, Sy=Σy
	double S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
	for (int i = 0; i<count; i++) {
		Sx += x[i];
		Sy += y[i];
	} //end-for

	if (invert) {
		// 垂直线情况，交换x和y坐标，以及对应的总和
		double *t = x;
		x = y;
		y = t;

		double d = Sx;
		Sx = Sy;
		Sy = d;
	} //end-if

	// 计算Sxx=Σx², Sxy=Σxy
	for (int i = 0; i<count; i++) {
		Sxx += x[i] * x[i];
		Sxy += x[i] * y[i];
	} //end-for

	// 计算行列式D
	double D = S*Sxx - Sx*Sx;
	// 使用最小二乘法计算直线参数
	a = (Sxx*Sy - Sx*Sxy) / D;
	b = (S*Sxy - Sx*Sy) / D;
}


//-----------------------------------------------------------------------------------
/**
 * @brief 直线拟合（自动确定直线方向）
 * @param x 点的x坐标数组
 * @param y 点的y坐标数组
 * @param count 点的数量
 * @param a 输出的直线参数a
 * @param b 输出的直线参数b
 * @param e 输出的拟合误差
 * @param invert 输出是否交换x和y坐标（0表示y=a+bx，1表示x=a+by）
 * @details 自动确定直线方向，使用最小二乘法拟合直线，并计算拟合误差
 */
void EDLines::LineFit(double * x, double * y, int count, double &a, double &b, double &e, int &invert)
{
	if (count<2) return; // 至少需要两个点才能拟合直线

	// 计算总和：S=count, Sx=Σx, Sy=Σy
	double S = count, Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
	for (int i = 0; i<count; i++) {
		Sx += x[i];
		Sy += y[i];
	} //end-for

	// 计算x和y的平均值
	double mx = Sx / count;
	double my = Sy / count;

	// 计算x和y方向的方差（用于确定直线方向）
	double dx = 0.0;
	double dy = 0.0;
	for (int i = 0; i < count; i++) {
		dx += (x[i] - mx)*(x[i] - mx);
		dy += (y[i] - my)*(y[i] - my);
	} //end-for

	// 根据方差确定直线方向
	if (dx < dy) {
		// 垂直线情况，交换x和y坐标
		invert = 1;
		double *t = x;
		x = y;
		y = t;

		// 交换对应的总和
		double d = Sx;
		Sx = Sy;
		Sy = d;
	}
	else {
		// 水平线情况，不交换坐标
		invert = 0;
	} //end-else  

	// 计算Sxx=Σx², Sxy=Σxy
	for (int i = 0; i<count; i++) {
		Sxx += x[i] * x[i];
		Sxy += x[i] * y[i];
	} //end-for

	// 计算行列式D
	double D = S*Sxx - Sx*Sx;
	// 使用最小二乘法计算直线参数
	a = (Sxx*Sy - Sx*Sxy) / D;
	b = (S*Sxy - Sx*Sy) / D;

	// 计算拟合误差
	if (b == 0.0) {
		// 垂直线或水平线，使用简单的距离计算
		double error = 0.0;
		for (int i = 0; i<count; i++) {
			error += fabs((a) - y[i]);
		} //end-for
		e = error / count;
	}
	else {
		// 一般直线情况，计算点到直线的垂直距离
		double error = 0.0;
		for (int i = 0; i<count; i++) {
			// 计算过点(x[i],y[i])且垂直于原直线的直线方程
			double d = -1.0 / b;
			double c = y[i] - d*x[i];
			// 计算两条直线的交点
			double x2 = ((a) - c) / (d - (b));
			double y2 = (a) + (b)*x2;

			// 计算点到直线的距离平方
			double dist = (x[i] - x2)*(x[i] - x2) + (y[i] - y2)*(y[i] - y2);
			error += dist;
		} //end-for

		// 计算平均距离（均方根误差）
		e = sqrt(error / count);
	} //end-else
}

//-----------------------------------------------------------------
/**
 * @brief 检查两条线段是否共线，如果共线则合并它们
 * @param ls1 第一条线段指针，合并后将被更新
 * @param ls2 第二条线段指针，合并后不会被修改
 * @param changeIndex 需要更新的线段在lines数组中的索引
 * @return 合并是否成功的布尔值
 * @details 检查两条线段是否共线且距离足够近，如果满足条件则合并它们，更新ls1的参数
 */
bool EDLines::TryToJoinTwoLineSegments(LineSegment * ls1, LineSegment * ls2, int changeIndex)
{
	int which; // 记录两条线段端点之间的最小距离类型
	// 计算两条线段端点之间的最小距离
	double dist = ComputeMinDistanceBetweenTwoLines(ls1, ls2, &which);
	// 如果距离超过最大允许距离，无法合并
	if (dist > max_distance_between_two_lines) return false;

	// 计算两条线段的长度，使用较长的线段作为基准
	double dx = ls1->sx - ls1->ex;
	double dy = ls1->sy - ls1->ey;
	double prevLen = sqrt(dx*dx + dy*dy);

	dx = ls2->sx - ls2->ex;
	dy = ls2->sy - ls2->ey;
	double nextLen = sqrt(dx*dx + dy*dy);

	// 确定较长和较短的线段
	LineSegment *shorter = ls1;
	LineSegment *longer = ls2;

	if (prevLen > nextLen) { shorter = ls2; longer = ls1; }

	// 使用3个点检查共线性（更快）
	dist = ComputeMinDistance(shorter->sx, shorter->sy, longer->a, longer->b, longer->invert);
	dist += ComputeMinDistance((shorter->sx + shorter->ex) / 2.0, (shorter->sy + shorter->ey) / 2.0, longer->a, longer->b, longer->invert);
	dist += ComputeMinDistance(shorter->ex, shorter->ey, longer->a, longer->b, longer->invert);

	dist /= 3.0; // 计算平均距离

	// 如果平均距离超过最大误差，无法合并
	if (dist > max_error) return false;

	// 确定合并后的线段端点，选择距离最远的两个端点
	/// 4种情况：1:(s1, s2), 2:(s1, e2), 3:(e1, s2), 4:(e1, e2)

	/// case 1: (s1, s2)
	dx = fabs(ls1->sx - ls2->sx);
	dy = fabs(ls1->sy - ls2->sy);
	double d = dx + dy;
	double max = d;
	which = 1;

	/// case 2: (s1, e2)
	dx = fabs(ls1->sx - ls2->ex);
	dy = fabs(ls1->sy - ls2->ey);
	d = dx + dy;
	if (d > max) {
		max = d;
		which = 2;
	} //end-if

	/// case 3: (e1, s2)
	dx = fabs(ls1->ex - ls2->sx);
	dy = fabs(ls1->ey - ls2->sy);
	d = dx + dy;
	if (d > max) {
		max = d;
		which = 3;
	} //end-if

	/// case 4: (e1, e2)
	dx = fabs(ls1->ex - ls2->ex);
	dy = fabs(ls1->ey - ls2->ey);
	d = dx + dy;
	if (d > max) {
		max = d;
		which = 4;
	} //end-if

	// 根据最远点组合更新ls1的端点
	if (which == 1) {
		// (s1, s2) - 合并后的线段为s1到e2
		ls1->ex = ls2->sx;
		ls1->ey = ls2->sy;
	}
	else if (which == 2) {
		// (s1, e2) - 合并后的线段为s1到e2
		ls1->ex = ls2->ex;
		ls1->ey = ls2->ey;
	}
	else if (which == 3) {
		// (e1, s2) - 合并后的线段为s2到e1
		ls1->sx = ls2->sx;
		ls1->sy = ls2->sy;
	}
	else {
		// (e1, e2) - 合并后的线段为e1到e2
		ls1->sx = ls1->ex;
		ls1->sy = ls1->ey;
		ls1->ex = ls2->ex;
		ls1->ey = ls2->ey;
	} //end-else

	// 更新第一条线段的长度
	if (ls1->firstPixelIndex + ls1->len + 5 >= ls2->firstPixelIndex) {
		// 如果两条线段在边缘段中是连续的，合并长度
		ls1->len += ls2->len;
	}
	else if (ls2->len > ls1->len) {
		// 否则，使用较长线段的长度信息
		ls1->firstPixelIndex = ls2->firstPixelIndex;
		ls1->len = ls2->len;
	} //end-if

	// 更新线段参数
	UpdateLineParameters(ls1);
	// 更新lines数组中的线段
	lines[changeIndex] = *ls1;

	return true;
}

//-------------------------------------------------------------------------------
/**
 * @brief 计算两条线段端点之间的最小距离
 * @param ls1 第一条线段指针
 * @param ls2 第二条线段指针
 * @param pwhich 输出最小距离对应的端点组合类型（SS, SE, ES, EE）
 * @return 两条线段端点之间的最小距离
 * @details 计算两条线段的四个端点之间的所有组合距离，返回最小值
 */
double EDLines::ComputeMinDistanceBetweenTwoLines(LineSegment * ls1, LineSegment * ls2, int * pwhich)
{
	// 计算ls1的起点到ls2的起点的距离
	double dx = ls1->sx - ls2->sx;
	double dy = ls1->sy - ls2->sy;
	double d = sqrt(dx*dx + dy*dy);
	double min = d;
	int which = EDLINES_SS; // 初始最小距离为SS类型

	// 计算ls1的起点到ls2的终点的距离
	dx = ls1->sx - ls2->ex;
	dy = ls1->sy - ls2->ey;
	d = sqrt(dx*dx + dy*dy);
	if (d < min) { min = d; which = EDLINES_SE; }

	// 计算ls1的终点到ls2的起点的距离
	dx = ls1->ex - ls2->sx;
	dy = ls1->ey - ls2->sy;
	d = sqrt(dx*dx + dy*dy);
	if (d < min) { min = d; which = EDLINES_ES; }

	// 计算ls1的终点到ls2的终点的距离
	dx = ls1->ex - ls2->ex;
	dy = ls1->ey - ls2->ey;
	d = sqrt(dx*dx + dy*dy);
	if (d < min) { min = d; which = EDLINES_EE; }

	// 如果pwhich不为nullptr，输出最小距离类型
	if (pwhich) *pwhich = which;
	// 返回最小距离
	return min;
}

//-----------------------------------------------------------------------------------
// Uses the two end points (sx, sy)----(ex, ey) of the line segment
// and computes the line that passes through these points (a, b, invert)
//
void EDLines::UpdateLineParameters(LineSegment * ls)
{
	double dx = ls->ex - ls->sx;
	double dy = ls->ey - ls->sy;

	if (fabs(dx) >= fabs(dy)) {
		/// Line will be of the form y = a + bx
		ls->invert = 0;
		if (fabs(dy) < 1e-3) { ls->b = 0; ls->a = (ls->sy + ls->ey) / 2; }
		else {
			ls->b = dy / dx;
			ls->a = ls->sy - (ls->b)*ls->sx;
		} //end-else

	}
	else {
		/// Line will be of the form x = a + by
		ls->invert = 1;
		if (fabs(dx) < 1e-3) { ls->b = 0; ls->a = (ls->sx + ls->ex) / 2; }
		else {
			ls->b = dx / dy;
			ls->a = ls->sx - (ls->b)*ls->sy;
		} //end-else
	} //end-else
}

void EDLines::EnumerateRectPoints(double sx, double sy, double ex, double ey, int ptsx[], int ptsy[], int * pNoPoints, int maxPoints)
{
	double vxTmp[4], vyTmp[4];
	double vx[4], vy[4];
	int n, offset;

	double x1 = sx;
	double y1 = sy;
	double x2 = ex;
	double y2 = ey;
	double width = 2;

	double dx = x2 - x1;
	double dy = y2 - y1;
	double vLen = sqrt(dx*dx + dy*dy);

	// make unit vector
	dx = dx / vLen;
	dy = dy / vLen;

	/* build list of rectangle corners ordered
	in a circular way around the rectangle */
	vxTmp[0] = x1 - dy * width / 2.0;
	vyTmp[0] = y1 + dx * width / 2.0;
	vxTmp[1] = x2 - dy * width / 2.0;
	vyTmp[1] = y2 + dx * width / 2.0;
	vxTmp[2] = x2 + dy * width / 2.0;
	vyTmp[2] = y2 - dx * width / 2.0;
	vxTmp[3] = x1 + dy * width / 2.0;
	vyTmp[3] = y1 - dx * width / 2.0;

	/* compute rotation of index of corners needed so that the first
	point has the smaller x.

	if one side is vertical, thus two corners have the same smaller x
	value, the one with the largest y value is selected as the first.
	*/
	if (x1 < x2 && y1 <= y2) offset = 0;
	else if (x1 >= x2 && y1 < y2) offset = 1;
	else if (x1 > x2 && y1 >= y2) offset = 2;
	else                          offset = 3;

	/* apply rotation of index. */
	for (n = 0; n<4; n++) {
		vx[n] = vxTmp[(offset + n) % 4];
		vy[n] = vyTmp[(offset + n) % 4];
	} //end-for

	  /* Set a initial condition.

	  The values are set to values that will cause 'ri_inc' (that will
	  be called immediately) to initialize correctly the first 'column'
	  and compute the limits 'ys' and 'ye'.

	  'y' is set to the integer value of vy[0], the starting corner.

	  'ys' and 'ye' are set to very small values, so 'ri_inc' will
	  notice that it needs to start a new 'column'.

	  The smaller integer coordinate inside of the rectangle is
	  'ceil(vx[0])'. The current 'x' value is set to that value minus
	  one, so 'ri_inc' (that will increase x by one) will advance to
	  the first 'column'.
	  */
	int x = (int)ceil(vx[0]) - 1;
	int y = (int)ceil(vy[0]);
	double ys = -DBL_MAX, ye = -DBL_MAX;

	int noPoints = 0;
	while (1) {
		/* if not at end of exploration,
		increase y value for next pixel in the 'column' */
		y++;

		/* if the end of the current 'column' is reached,
		and it is not the end of exploration,
		advance to the next 'column' */
		while (y > ye && x <= vx[2]) {
			/* increase x, next 'column' */
			x++;

			/* if end of exploration, return */
			if (x > vx[2]) break;

			/* update lower y limit (start) for the new 'column'.

			We need to interpolate the y value that corresponds to the
			lower side of the rectangle. The first thing is to decide if
			the corresponding side is

			vx[0],vy[0] to vx[3],vy[3] or
			vx[3],vy[3] to vx[2],vy[2]

			Then, the side is interpolated for the x value of the
			'column'. But, if the side is vertical (as it could happen if
			the rectangle is vertical and we are dealing with the first
			or last 'columns') then we pick the lower value of the side
			by using 'inter_low'.
			*/
			if ((double)x < vx[3]) {
				/* interpolation */
				if (fabs(vx[0] - vx[3]) <= 0.01) {
					if (vy[0]<vy[3]) ys = vy[0];
					else if (vy[0]>vy[3]) ys = vy[3];
					else     ys = vy[0] + (x - vx[0]) * (vy[3] - vy[0]) / (vx[3] - vx[0]);
				}
				else
					ys = vy[0] + (x - vx[0]) * (vy[3] - vy[0]) / (vx[3] - vx[0]);

			}
			else {
				/* interpolation */
				if (fabs(vx[3] - vx[2]) <= 0.01) {
					if (vy[3]<vy[2]) ys = vy[3];
					else if (vy[3]>vy[2]) ys = vy[2];
					else     ys = vy[3] + (x - vx[3]) * (y2 - vy[3]) / (vx[2] - vx[3]);
				}
				else
					ys = vy[3] + (x - vx[3]) * (vy[2] - vy[3]) / (vx[2] - vx[3]);
			} //end-else

			  /* update upper y limit (end) for the new 'column'.

			  We need to interpolate the y value that corresponds to the
			  upper side of the rectangle. The first thing is to decide if
			  the corresponding side is

			  vx[0],vy[0] to vx[1],vy[1] or
			  vx[1],vy[1] to vx[2],vy[2]

			  Then, the side is interpolated for the x value of the
			  'column'. But, if the side is vertical (as it could happen if
			  the rectangle is vertical and we are dealing with the first
			  or last 'columns') then we pick the lower value of the side
			  by using 'inter_low'.
			  */
			if ((double)x < vx[1]) {
				/* interpolation */
				if (fabs(vx[0] - vx[1]) <= 0.01) {
					if (vy[0]<vy[1]) ye = vy[1];
					else if (vy[0]>vy[1]) ye = vy[0];
					else     ye = vy[0] + (x - vx[0]) * (vy[1] - vy[0]) / (vx[1] - vx[0]);
				}
				else
					ye = vy[0] + (x - vx[0]) * (vy[1] - vy[0]) / (vx[1] - vx[0]);

			}
			else {
				/* interpolation */
				if (fabs(vx[1] - vx[2]) <= 0.01) {
					if (vy[1]<vy[2]) ye = vy[2];
					else if (vy[1]>vy[2]) ye = vy[1];
					else     ye = vy[1] + (x - vx[1]) * (vy[2] - vy[1]) / (vx[2] - vx[1]);
				}
				else
					ye = vy[1] + (x - vx[1]) * (vy[2] - vy[1]) / (vx[2] - vx[1]);
			} //end-else

			  /* new y */
			y = (int)ceil(ys);
		} //end-while

		  // Are we done?
		if (x > vx[2]) break;

		ptsx[noPoints] = x;
		ptsy[noPoints] = y;
		noPoints++;
	} //end-while

	*pNoPoints = noPoints;
}

void EDLines::SplitSegment2Lines(double * x, double * y, int noPixels, int segmentNo, vector<LineSegment> &lines, int min_line_len, double line_error)
{
	// First pixel of the line segment within the segment of points
	int firstPixelIndex = 0;

	while (noPixels >= min_line_len) {
		// Start by fitting a line to MIN_LINE_LEN pixels
		bool valid = false;
		double lastA, lastB, error;
		int lastInvert;

		while (noPixels >= min_line_len) {
			LineFit(x, y, min_line_len, lastA, lastB, error, lastInvert);
			if (error <= 0.5) { valid = true; break; }

#if 1
			noPixels -= 1;   // Go slowly
			x += 1; y += 1;
			firstPixelIndex += 1;
#else
			noPixels -= 2;   // Go faster (for speed)
			x += 2; y += 2;
			firstPixelIndex += 2;
#endif
		} //end-while

		if (valid == false) return;

		// Now try to extend this line
		int index = min_line_len;
		int len = min_line_len;

		while (index < noPixels) {
			int startIndex = index;
			int lastGoodIndex = index - 1;
			int goodPixelCount = 0;
			int badPixelCount = 0;
			while (index < noPixels) {
				double d = ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert);

				if (d <= line_error) {
					lastGoodIndex = index;
					goodPixelCount++;
					badPixelCount = 0;

				}
				else {
					badPixelCount++;
					if (badPixelCount >= 5) break;
				} //end-if

				index++;
			} //end-while

			if (goodPixelCount >= 2) {
				len += lastGoodIndex - startIndex + 1;
				LineFit(x, y, len, lastA, lastB, lastInvert);  // faster LineFit
				index = lastGoodIndex + 1;
			} // end-if

			if (goodPixelCount < 2 || index >= noPixels) {
				// End of a line segment. Compute the end points
				double sx, sy, ex, ey;

				int index = 0;
				while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index++;
				ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, sx, sy);
				int noSkippedPixels = index;

				index = lastGoodIndex;
				while (ComputeMinDistance(x[index], y[index], lastA, lastB, lastInvert) > line_error) index--;
				ComputeClosestPoint(x[index], y[index], lastA, lastB, lastInvert, ex, ey);

				// Add the line segment to lines
				lines.push_back(LineSegment(lastA, lastB, lastInvert, sx, sy, ex, ey, segmentNo, firstPixelIndex + noSkippedPixels, index - noSkippedPixels + 1));
				//linesNo++;
				len = index + 1;
				break;
			} //end-else
		} //end-while

		noPixels -= len;
		x += len;
		y += len;
		firstPixelIndex += len;
	} //end-while
}
} //namespce endline