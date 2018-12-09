
#ifndef TRACKER_CUH_
#define TRACKER_CUH_

#include <complex>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>

#include <chrono>
#include <iostream>
#include <ostream>
#include <vector>
#include <cmath>


using namespace cv;

class CUTracker {

public:
	CUTracker();
	bool init( const Mat& image, const Rect2d& boundingBox );
	bool update( const Mat& image, Rect2d& boundingBox );

protected:

	void get_features();
	void get_featuresNoHanning();
	void initRes();

	void fourier_transform_features();

	bool uploadToCuda( const Mat& image, const Rect2d &boundingBox );

	cv::Mat gaussian_shaped_labels(double sigma, int dim1, int dim2);
	void CircShift(cv::Mat &x, cv::Size k);
	cv::Mat CreateGaussian1D(int n, float sigma);
	cv::Mat CreateGaussian2D(cv::Size sz, float sigma);
	cv::Mat GaussianShapedLabels(float sigma, cv::Size sz);
	bool Learn(float rate);
	cv::Mat GaussianCorrelation(std::vector<cv::Mat> xf, std::vector<cv::Mat> yf);
	cv::Mat LinearCorrelation(std::vector<cv::Mat> xf, std::vector<cv::Mat> yf);
	cv::Mat ComplexDiv(const cv::Mat &x1, const cv::Mat &x2);
	cv::Mat ComplexMul(const cv::Mat &x1, const cv::Mat &x2);

private:
	////////////////////////////////////////////
	float padding_ = 2;
	float output_sigma_factor_ = 0.1;
	float kernel_sigma_ = 0.2;
	int cell_size_ = 4;
	float lambda_ = 1e-4;
	float interp_factor_ = 0.075;

	bool features_hog_ = true;
	int features_hog_orientations_ = 9;
	////////////////////////////////////////////

	Point2f pos_;
	Size size_;
	Size origSize_;


	Mat model_alphaf_;

	Mat patchRGB;
	Mat patchGRAY;
	Mat patchHSV;

	std::vector<Mat> labels;
	std::vector<Mat> hanWindows;
	std::vector<Mat> features;

	std::vector<Mat> filters;

	std::vector<Mat> model_xf_;
	std::vector<Mat> labelsX;
	std::vector<Mat> featuresX;

	/////////////////////////////////////////////

	cuda::GpuMat dImageRGB;
	cuda::GpuMat dImageGRAY;
	cuda::GpuMat dImageHSV;

	std::vector<cuda::GpuMat> dLabels;
	std::vector<cuda::GpuMat> dLabelsX;
	std::vector<cuda::GpuMat> dHanning;

	std::vector<cuda::GpuMat> model_xfGpu;
	std::vector<cuda::GpuMat> featuresGpu;
	std::vector<cuda::GpuMat> featuresXGpu;
	std::vector<cuda::GpuMat> filtersGpu;

	std::chrono::time_point<std::chrono::system_clock> start, end;
};


#endif /* TRACKER_CUH_ */

