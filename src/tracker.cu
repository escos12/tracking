#include "tracker.cuh".cuh"

#include <iostream>

CUTracker::CUTracker()
{

}

Mat get_hogdescriptor_visu(const Mat& color_origImg, std::vector<float>& descriptorValues, const Size & size )
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size( (int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac) ) );

	int cellSize        = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI/(float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter   = new int*[cells_in_y_dir];
	for (int y=0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x=0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin=0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
			// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr=0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr==1) celly++;
				if (cellNr==2) cellx++;
				if (cellNr==3)
				{
					cellx++;
					celly++;
				}

				for (int bin=0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[ descriptorDataIdx ];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


					// note: overlapping blocks lead to multiple updates of this sum!
					// we therefore keep track how often a cell was updated,
					// to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	// compute average gradient strengths
	for (celly=0; celly<cells_in_y_dir; celly++)
	{
		for (cellx=0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin=0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly=0; celly<cells_in_y_dir; celly++)
	{
		for (cellx=0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize/2;
			int my = drawY + cellSize/2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX+cellSize)*zoomFac), (int)((drawY+cellSize)*zoomFac)), Scalar(100,100,100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin=0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength==0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;

				float dirVecX = cos( currRad );
				float dirVecY = sin( currRad );
				float maxVecLen = (float)(cellSize/2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

				// compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac),(int)(y1*zoomFac)), Point((int)(x2*zoomFac),(int)(y2*zoomFac)), Scalar(0,255,0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	// don't forget to free memory allocated by helper data structures!
	for (int y=0; y<cells_in_y_dir; y++)
	{
		for (int x=0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

}
cv::Mat CUTracker::CreateGaussian1D(int n, float sigma) {
	cv::Mat kernel(n, 1, CV_32F);
	float* cf = kernel.ptr<float>();

	double sigmaX = sigma > 0 ? sigma : ((n - 1)*0.5 - 1)*0.3 + 0.8;
	double scale2X = -0.5 / (sigmaX*sigmaX);

	for (int i = 0; i < n; ++i) {
		double x = i - floor(n / 2) + 1;
		double t = std::exp(scale2X * x * x);
		cf[i] = (float)t;
	}

	return kernel;
}

cv::Mat CUTracker::CreateGaussian2D(cv::Size sz, float sigma) {
	cv::Mat a = CreateGaussian1D(sz.height, sigma);
	cv::Mat b = CreateGaussian1D(sz.width, sigma);
	return a * b.t();
}

cv::Mat CUTracker::gaussian_shaped_labels(double sigma, int dim1, int dim2)
{
	cv::Mat labels = CreateGaussian2D(size_, sigma);
	cv::Size shift_temp = cv::Size(-cvFloor(size_.width * (1./2)), -cvFloor(size_.height * (1./2)));
	shift_temp.width += 1;
	shift_temp.height += 1;
	CircShift(labels, shift_temp);
	return labels;
}

void CUTracker::CircShift(cv::Mat &x, cv::Size k)
{
	int cx, cy;
	if (k.width < 0)
		cx = -k.width;
	else
		cx = x.cols - k.width;

	if (k.height < 0)
		cy = -k.height;
	else
		cy = x.rows - k.height;

	cv::Mat q0(x, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(x, cv::Rect(cx, 0, x.cols - cx, cy));  // Top-Right
	cv::Mat q2(x, cv::Rect(0, cy, cx, x.rows -cy));  // Bottom-Left
	cv::Mat q3(x, cv::Rect(cx, cy, x.cols -cx, x.rows-cy)); // Bottom-Right

	cv::Mat tmp1, tmp2;                           // swap quadrants (Top-Left with Bottom-Right)
	cv::hconcat(q3, q2, tmp1);
	cv::hconcat(q1, q0, tmp2);
	cv::vconcat(tmp1, tmp2, x);
}

bool CUTracker::uploadToCuda( const Mat& image, const Rect2d &boundingBox )
{
	//dImageRGB.upload(window);
}

cv::Mat CUTracker::GaussianShapedLabels(float sigma, cv::Size sz) {
	cv::Mat labels = CreateGaussian2D(sz, sigma);
	cv::Size shift_temp = cv::Size(-cvFloor(sz.width * (1./2)), -cvFloor(sz.height * (1./2)));
	shift_temp.width += 1;
	shift_temp.height += 1;
	CircShift(labels, shift_temp);
	return labels;
}

bool CUTracker::init(const Mat &im, const Rect2d &boundingBox)
{
	if (im.channels() != 3)
		return false;

	int w = getOptimalDFTSize(int(boundingBox.width));
	int h = getOptimalDFTSize(int(boundingBox.height));

	//Get the center position
	pos_.x = boundingBox.x + boundingBox.width/2;
	pos_.y = boundingBox.y + boundingBox.height/2;


	size_.width = w * padding_;
	size_.height = h * padding_;
	origSize_.width = boundingBox.width;
	origSize_.height = boundingBox.height;


	// scale and save patch
	getRectSubPix(im, size_, pos_, patchRGB);
	cvtColor(patchRGB, patchGRAY, CV_BGR2GRAY);

	imshow("patchGRAY", patchGRAY);


	initRes();

	featuresX.resize(labels.size());
	labelsX.resize(labels.size());
	model_xf_.resize(labels.size());

	for(unsigned int i = 0; i < labels.size(); i++)
	{
		cv::dft(labels[i], labelsX[i], DFT_COMPLEX_OUTPUT);
	}

	Learn(1.0);

}

bool CUTracker::Learn(float rate)
{
	get_features();

	for (unsigned int i = 0; i < features.size(); i++)
	{
		cv::dft(features[i], featuresX[i], DFT_COMPLEX_OUTPUT);
	}

	cv::Mat kf;
	kf = GaussianCorrelation(featuresX, featuresX);
	//kf = LinearCorrelation(featuresX, featuresX);

	for (unsigned int i = 0; i < features.size(); i++)
	{
		cv::Mat alphaf = ComplexDiv(labelsX[i], kf + cv::Scalar(lambda_, 0));
		if (rate > 0.99)
		{
			model_alphaf_ = alphaf;
			model_xf_.clear();
			for (unsigned int i = 0; i < featuresX.size(); ++i)
				model_xf_.push_back(featuresX[i].clone());
		} else {
			model_alphaf_ = (1.0 - rate) * model_alphaf_ + rate * alphaf;
			for (unsigned int i = 0; i < featuresX.size(); ++i)
				model_xf_[i] = (1.0 - rate) * model_xf_[i] + rate * featuresX[i].clone();
		}
	}

}

cv::Mat CUTracker::GaussianCorrelation(std::vector<cv::Mat> xf, std::vector<cv::Mat> yf)
{
	//showSpectrum(xf[0], "CU_");
	int N = xf[0].size().area();
	double xx = 0, yy = 0;

	std::vector<cv::Mat> xyf_vector(xf.size());
	cv::Mat xy(xf[0].size(), CV_32FC1, Scalar(0.0)), xyf, xy_temp;
	for (unsigned int i = 0; i < xf.size(); ++i)
	{
		xx += cv::norm(xf[i]) * cv::norm(xf[i]) / N;
		yy += cv::norm(yf[i]) * cv::norm(yf[i]) / N;
		cv::mulSpectrums(xf[i], yf[i], xyf, 0, true);
		cv::idft(xyf, xy_temp, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
		xy += xy_temp;
	}
	float numel_xf = N * xf.size();
	cv::Mat k, kf;

	exp((-1 / (kernel_sigma_ * kernel_sigma_)) * max(0.0, (xx + yy - 2 * xy) / numel_xf), k);

	k.convertTo(k, CV_32FC1);
	cv::dft(k, kf, DFT_COMPLEX_OUTPUT);
	return kf;
}

cv::Mat CUTracker::LinearCorrelation(std::vector<cv::Mat> xf, std::vector<cv::Mat> yf)
{
	cv::Mat kf(xf[0].size(), CV_32FC2, cv::Scalar(0)), xyf;
	for (unsigned int i = 0; i < xf.size(); ++i) {
		cv::mulSpectrums(xf[i], yf[i], xyf, 0, true);
		kf += xyf;
	}
	float numel_xf = xf[0].size().area() * xf.size();
	return kf/numel_xf;
}



bool CUTracker::update( const Mat& image, Rect2d& boundingBox )
{

	getRectSubPix(image, size_, pos_, patchRGB);
	cvtColor(patchRGB, patchGRAY, CV_BGR2GRAY);

	imshow("patchGRAY_", patchGRAY);

	//get_featuresNoHanning();
	get_features();


	for (unsigned int i = 0; i < features.size(); i++)
	{
		cv::dft(features[i], featuresX[i], DFT_COMPLEX_OUTPUT);
	}
	//imshow("CU features", features[0]);
	//showSpectrum(featuresX[0], "CU");

	cv::Mat kzf;
	kzf = GaussianCorrelation(featuresX, model_xf_);
	//kzf = LinearCorrelation(featuresX, model_xf_);


	cv::Mat response;
	cv::idft(ComplexMul(model_alphaf_, kzf), response, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT

	imshow("response", response);

	cv::Point maxLoc;
	cv::minMaxLoc(response, NULL, NULL, NULL, &maxLoc);

	if ((maxLoc.x + 1) > (response.cols / 2))
		maxLoc.x = maxLoc.x - response.cols;
	if ((maxLoc.y + 1) > (response.rows / 2))
		maxLoc.y = maxLoc.y - response.rows;

	pos_.x +=  maxLoc.x;
	pos_.y +=  maxLoc.y;

	boundingBox.x = pos_.x - origSize_.width/2;
	boundingBox.y = pos_.y - origSize_.height/2;

	getRectSubPix(image, size_, pos_, patchRGB);
	cvtColor(patchRGB, patchGRAY, CV_BGR2GRAY);

	Learn(interp_factor_);

}

void CUTracker::get_featuresNoHanning()
{
	int cnt = 0;

	Mat f;
	patchGRAY.convertTo(f, CV_32FC1, 1.0 / 255);
	f = f - cv::mean(f).val[0];
	features[cnt] = f;
	cnt++;

	imshow("features[cnt]", features[cnt-1]);
}

void CUTracker::get_features()
{
	int cnt = 0;

	Mat f;
	patchGRAY.convertTo(f, CV_32FC1, 1.0 / 255);
	f = f - cv::mean(f).val[0];
	Mat gx, gy;
	Sobel(f, gx, CV_32F, 1, 0, 1);
	Sobel(f, gy, CV_32F, 0, 1, 1);
	Mat mag, angle;
	cartToPolar(gx, gy, mag, angle, 1);


	angle.convertTo(angle, CV_32FC1, 1.0 / 255);

	angle = angle - cv::mean(angle).val[0];
	//angle = angle.mul(hanWindows[cnt]);
	features[cnt] = angle.clone();

	imshow("angle", angle);
	imshow("mag", mag);
	cnt++;

/*
	int img_width  = patchGRAY.cols;
	int img_height = patchGRAY.rows;
	int block_size = 16;
	int bin_number = 9;

	cv::Ptr<cv::cuda::HOG> cuda_hog = cuda::HOG::create(Size(img_width, img_height),
	                                                    Size(block_size, block_size),
	                                                    Size(block_size/2, block_size/2),
	                                                    Size(block_size/2, block_size/2),
	                                                    bin_number);

	cuda_hog->setDescriptorFormat(cuda::HOG::DESCR_FORMAT_COL_BY_COL);
	cuda_hog->setGammaCorrection(true);
	cuda_hog->setWinStride(Size(patchGRAY.cols, patchGRAY.rows));

	cv::cuda::GpuMat image;
	cv::cuda::GpuMat descriptor;

	image.upload(patchRGB);

	cv::cuda::GpuMat image_alpha;
	cuda::cvtColor(image, image_alpha, COLOR_BGR2BGRA, 4);

	cuda_hog->compute(image_alpha, descriptor);

	cv::Mat dst;
	image_alpha.download(dst);
*/

	/*
	HOGDescriptor d;
	std::vector<float> descriptorsValues;
	std::vector<Point> locations;
	d.compute( patchGRAY, descriptorsValues, Size(0,0), Size(0,0), locations);


	Mat ttt = get_hogdescriptor_visu(patchRGB, descriptorsValues, Size(4,4));

	imshow("ttt", ttt);
	 */
	//Mat hg;
	//std::vector<float> featureVector;
	//hog.compute(patchGRAY, featureVector, Size(cell_size_, cell_size_), Size(0, 0));


	//std::vector<cuda::GpuMat> features;

	///GRAY features
	//resize(gray_m, gray_m, feature_size, 0, 0, INTER_CUBIC);
	//dImageGRAY.convertTo(dImageGRAYF, CV_32FC1, 1.0/255.0, -0.5);
	//features.push_back(dImageGRAYF);

	/*
    	///HSV features
    	//resize(gray_m, gray_m, feature_size, 0, 0, INTER_CUBIC);
    	dImageGRAY.convertTo(dImageGRAY, CV_32FC1, 1.0/255.0, -0.5);
    	features.push_back(dImageGRAY);

    	///HOG features
    	//resize(gray_m, gray_m, feature_size, 0, 0, INTER_CUBIC);
    	dImageGRAY.convertTo(dImageGRAY, CV_32FC1, 1.0/255.0, -0.5);
    	features.push_back(dImageGRAY);
    /*
    if (params.use_hog) {
        std::vector<Mat> hog = get_features_hog(patch, cell_size);
        features.insert(features.end(), hog.begin(),
                hog.begin()+params.num_hog_channels_used);
    }
    if (params.use_color_names) {
        std::vector<Mat> cn;
        cn = get_features_cn(patch, feature_size);
        features.insert(features.end(), cn.begin(), cn.end());
    }
    if(params.use_gray) {
        Mat gray_m;
        cvtColor(patch, gray_m, COLOR_BGR2GRAY);
        resize(gray_m, gray_m, feature_size, 0, 0, INTER_CUBIC);
        gray_m.convertTo(gray_m, CV_32FC1, 1.0/255.0, -0.5);
        features.push_back(gray_m);
    }
    if(params.use_rgb) {
        std::vector<Mat> rgb_features = get_features_rgb(patch, feature_size);
        features.insert(features.end(), rgb_features.begin(), rgb_features.end());
    }

    for (size_t i = 0; i < features.size(); ++i) {
        features.at(i) = features.at(i).mul(window);
    }
	 */
}
void CUTracker::initRes()
{
	labels.clear();
	hanWindows.clear();
	features.clear();

	Mat L;
	L = GaussianShapedLabels(std::sqrt(float(patchGRAY.size().area())) * output_sigma_factor_, patchGRAY.size());
	labels.push_back(L.clone());

	Mat H;
	createHanningWindow(H, patchGRAY.size(), CV_32F);
	hanWindows.push_back(H.clone());
	imshow("HanningWindow", H);

	/*
	Mat HH;
	Mat mH;
	std::cout << "origSize_" << origSize_;
	createHanningWindow(mH, origSize_, CV_32F);
	int d = (patchGRAY.cols - origSize_.width)/2;
	int s = (patchGRAY.rows - origSize_.height)/2;
	copyMakeBorder( mH, HH, s, s, d, d, BORDER_CONSTANT, Scalar(0,0,0) );
	hanWindows.push_back(HH.clone());
	imshow("HH", HH);
	 */
	Mat F;
	patchGRAY.convertTo(F, CV_32FC1, 1.0 / 255);
	F = F - cv::mean(F).val[0];
	F = F.mul(H);
	features.push_back(F.clone());

}

cv::Mat CUTracker::ComplexMul(const cv::Mat &x1, const cv::Mat &x2)
{
	std::vector<cv::Mat> planes1;
	cv::split(x1, planes1);
	std::vector<cv::Mat> planes2;
	cv::split(x2, planes2);
	std::vector<cv::Mat>complex(2);
	complex[0] = planes1[0].mul(planes2[0]) - planes1[1].mul(planes2[1]);
	complex[1] = planes1[0].mul(planes2[1]) + planes1[1].mul(planes2[0]);
	Mat result;
	cv::merge(complex, result);
	return result;
}

cv::Mat CUTracker::ComplexDiv(const cv::Mat &x1, const cv::Mat &x2)
{
	std::vector<cv::Mat> planes1;
	cv::split(x1, planes1);
	std::vector<cv::Mat> planes2;
	cv::split(x2, planes2);
	std::vector<cv::Mat>complex(2);

	cv::Mat cc = planes2[0].mul(planes2[0]);
	cv::Mat dd = planes2[1].mul(planes2[1]);

	complex[0] = (planes1[0].mul(planes2[0]) + planes1[1].mul(planes2[1])) / (cc + dd);
	complex[1] = (-planes1[0].mul(planes2[1]) + planes1[1].mul(planes2[0])) / (cc + dd);
	cv::Mat result;
	cv::merge(complex, result);
	return result;
}
