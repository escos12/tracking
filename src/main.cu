
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include "tracker.cuh"

#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cassert>
#include <sstream>
#include <experimental/filesystem>

using namespace cv;
using namespace std;
namespace fs = experimental::filesystem;

int posX = 0;
int posY = 0;
cv::Size aim(60, 90);

bool startTracking = false;
std::chrono::time_point<std::chrono::system_clock> start, stop;

Mat translateImg(Mat &img, int offsetx, int offsety){
	Mat out;
	Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
	warpAffine(img,out,trans_mat,img.size());
	return out;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	posX = x;
	posY = y;
	if  ( event == EVENT_LBUTTONDOWN )
	{
		startTracking = !startTracking;
		cout << startTracking << endl;
	}
	else if  ( event == EVENT_RBUTTONDOWN )
	{
	}
	else if  ( event == EVENT_MBUTTONDOWN )
	{
	}
	else if ( event == EVENT_MOUSEMOVE )
	{
	}
}

vector<float> stringToFloat(string s)
																								{
	std::size_t pos = 0;
	std::vector< float > vd;
	double d = 0.0;
	// convert ',' to ' '
	while (pos < s.size ())
		if ((pos = s.find_first_of (',',pos)) != std::string::npos)
			s[pos] = ' ';
	std::stringstream ss(s);
	while (ss >> d)
		vd.push_back (d);
	return vd;
																								}

int main(int argc, char **argv)
{
	vector<string> paths;
	std::string path = "/home/timyr/tracking";
	for (const auto & p : fs::directory_iterator(path))
	{
		paths.push_back(p.path() + "/");
		std::cout << p << std::endl; // "p" is the directory entry. Get the path with "p.path()"
	}

	int cn = 1;


	for (int i = 0; i < paths.size(); i++)
	{
		ifstream groundtruth;
		groundtruth.open(paths[i] + "groundtruth.txt");
		while(true)
		{
			std::string gt;
			std::getline(groundtruth, gt);
			vector<float> ps = stringToFloat(gt);
			std::stringstream buffer;
			buffer << std::setw(8) << std::setfill('0') << cn;
			std::string s = buffer.str();
			cv::Mat img = imread(paths[i] + s + ".jpg", 1);
			if(! img.data )                              // Check for invalid input
			{
				cn = 1;
				cout <<  "Could not open or find the image" << std::endl;
				break;
			}
			rectangle(img, Point(ps[0], ps[1]), Point(ps[4], ps[5]), Scalar(0, 255, 0), 1, 8, 0);

			imshow("img", img);
			char k = waitKey(0);
			if(k == 27)
				break;
			cn++;
		}
	}
	/*
	string line;
	ifstream myfile ("example.txt");
	if (myfile.is_open())
	{
		while ( getline (myfile,line) )
		{
			cout << line << '\n';
		}
		myfile.close();
	}
	 */
	//stringToFloat(Numbers);

	// If possible, always prefer std::vector to naked array
	std::vector<float> v;

	// Build an istream that holds the input string
	/*std::istringstream iss(Numbers);

	// Iterate over the istream, using >> to grab floats
	// and push_back to store them in the vector
	std::copy(std::istream_iterator<float>(iss),
			std::istream_iterator<float>(),
			std::back_inserter(v));

	// Put the result on standard out
	std::copy(v.begin(), v.end(),
			std::ostream_iterator<float>(std::cout, ", "));
	std::cout << "\n";
	 */
	CUTracker track;

	/*
	const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)120/1 ! \
					nvvidconv flip-method=4 ! video/x-raw, format=(string)BGRx ! \
					videoconvert ! video/x-raw, format=(string)BGR ! \
					appsink";*/

	cv::VideoCapture cap(0);

	if(!cap.isOpened())
	{
		std::cout<<"Failed to open camera."<<std::endl;
		return -1;
	}

	unsigned int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	unsigned int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	unsigned int pixels = width*height;

	std::cout <<"Frame size : "<<width<<" x "<<height<<", "<<pixels<<" Pixels "<<std::endl;

	cv::namedWindow("MyCameraPreview", CV_WINDOW_AUTOSIZE);
	cv::setMouseCallback("MyCameraPreview", CallBackFunc, NULL);

	cv::Mat frame_in(width, height, CV_8UC3);


	start = std::chrono::system_clock::now();
	int counter = 0;
	bool firstFrame = 0;

	Rect2d r;
	while(true)
	{
		cap.read(frame_in);
		if (!frame_in.data) {
			std::cout<<"Capture read error"<<std::endl;
			break;
		}

		counter++;

		int elapsed_seconds = 0;

		if(elapsed_seconds > 1000000)
		{
			counter = 0;
		}

		if(startTracking)
		{
			if(!firstFrame)
			{
				r =  Rect(posX - aim.height/2, posY - aim.width/2, aim.height, aim.width);
				track.init(frame_in, Rect(posX - aim.height/2, posY - aim.width/2, aim.height, aim.width));
				firstFrame = 1;
			}
			else
			{

				start = std::chrono::system_clock::now();
				track.update(frame_in, r);
				stop = std::chrono::system_clock::now();
				elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
			}
		}
		else
		{
			firstFrame = 0;
		}

		cv::putText(frame_in, to_string(1000000/elapsed_seconds) + " FPS", Point(10, 40), 1, 1, Scalar(0, 0, 255), 1, 1, 0);
		if(startTracking)
		{
			cv::rectangle(frame_in, r, cv::Scalar(255, 255, 0), 1, 8, 0);
		}
		else
		{
			cv::rectangle(frame_in, Rect(posX - aim.height/2, posY - aim.width/2, aim.height, aim.width), cv::Scalar(0, 255, 0), 1, 8, 0);
		}

		if(counter%4 == 0)
		{
			cv::imshow("MyCameraPreview",frame_in);
			char k = cv::waitKey(2); // let imshow draw and wait for next frame 8 ms for 120 fps
			if (k == 27)
			{
				break;
			}
		}

	}

	double pi = 3.14159265359/60;
	double dpi = 2.0;

	Mat lena = imread("/home/nvidia/lena.jpg", CV_LOAD_IMAGE_COLOR);

	cuda::GpuMat image1;
	image1.upload(lena);
	//Mat lena1 = imread("/home/nvidia/lena1.png", CV_LOAD_IMAGE_COLOR);

	cv::Mat out = cv::Mat::zeros(lena.size(), lena.type());

	Rect roi = Rect(240, 240, 60, 60);

	Rect roiKCF = roi;
	Rect2d roiCU = roi;
	Rect2d roiGPU = roi;


	CUTracker cuTracker;
	//CUTrackerGPU trackerGPU;

	cuTracker.init(lena, roiCU);
	//trackerGPU.init(lena, roiGPU);


	int dx, dy;

	//while(true)
	for(int j = 0; j < 200; j++)
	{
		dx = 60 * cos(dpi);
		dy = 20 * sin(dpi);
		out = translateImg(lena, dx, dy);

		start = std::chrono::system_clock::now();
		cuTracker.update(out, roiCU);
		stop = std::chrono::system_clock::now();
		int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();

		/*
		start = std::chrono::system_clock::now();
		roiKCF = tr.Update(out);
		end = std::chrono::system_clock::now();

		int elapsed_seconds1 = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
		 */

		//start = std::chrono::system_clock::now();
		//trackerGPU.update(out, roiGPU);
		//end = std::chrono::system_clock::now();

		/*
		int elapsed_seconds2 = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

		rectangle(out, roiGPU, Scalar(255, 0, 255), 4, 8, 0);
		rectangle(out, roiKCF, Scalar(0, 255, 0), 2, 8, 0);
		//rectangle(out, roiCU, Scalar(0, 0, 255), 1, 8, 0);

		putText(out, to_string(1000000/elapsed_seconds) + " FPS", Point(10, 40), 1, 1, Scalar(0, 0, 255), 1, 1, 0);
		putText(out, to_string(1000000/elapsed_seconds1) + " FPS", Point(10, 80), 1, 1, Scalar(0, 255, 0), 1, 1, 0);

		 */
		std::cout << 1000000/elapsed_seconds << std::endl;

		//imshow("lena", out);

		dpi += pi;
		/*
		char k = waitKey(1);
		if(k == 27)
			break;
		 */
	}


	std::cout << "end!-!";
	return 0;
}
