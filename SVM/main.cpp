#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

int g_C = 100.0;
int g_gamma = 100;
int g_degree = 1;

void onSlider_C_Gamma(int, void*)
{
	const int size = 10;
	int label[size] = { 1, 1, 1 , 2 , 2 ,2 ,3, 3 ,3, 3 };
	float traindata[size][2] = { {110, 233 },{ 202, 353 },{ 97, 198 }, {11,79}, { 16, 108},  { 22, 106 },  { 64, 196 },  { 55, 204 },  { 57, 208 }, {100, 170} };
	//转为Mat以调用
	Mat trainMat(10, 2, CV_32FC1, traindata);
	Mat	labelMat(10, 1, CV_32SC1, label);
	//训练的初始化
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::POLY);
	g_C = (g_C == 0 ? 1 : g_C);
	svm->setC(g_C*0.01);
	g_gamma = (g_gamma == 0 ? 1 : g_gamma);
	svm->setGamma(g_gamma*0.01);	
	g_degree = (g_degree == 0 ? 1 : g_degree);
	svm->setDegree(g_degree);
	cout << "C=" << g_C * 0.01 << " " << "gamma=" << g_gamma * 0.01 <<" degree="<<g_degree<<endl;
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	//开始训练
	svm->train(trainMat, ROW_SAMPLE, labelMat);
	//-----------无关紧要的美工的部分-----------------------	
	//----其实对每个像素点的坐标也进行了分类----------------
	int width = 512, height = 512;
	Mat dispImage = Mat::zeros(width, height, CV_8UC3);
	Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255), black(0, 0, 0);
	for (int i = 0; i < dispImage.rows; ++i)
		for (int j = 0; j < dispImage.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = svm->predict(sampleMat);
			if (response == 1)
				dispImage.at<Vec3b>(i, j) = red;
			else if (response == 2)
				dispImage.at<Vec3b>(i, j) = green;
			else if (response == 3)
				dispImage.at<Vec3b>(i, j) = blue;
			else if (response == 4)
				dispImage.at<Vec3b>(i, j) = black;
		}
	//--------把初始化训练的点画进图片------------
	int thickness = -1;
	int lineType = 8;
	for (int idx = 0; idx < sizeof(label) / sizeof(int); idx++) {
		circle(dispImage, Point(traindata[idx][0], traindata[idx][1]), 10, Scalar(255, 255, 255), thickness, -1);
	}
	// 把 support vectors  cout粗来看看……
	Mat sv = svm->getSupportVectors();
	//cout << "Support Vectors为：" << endl;
	for (int i = 0; i < sv.rows; ++i)
	{
		const float* v = sv.ptr<float>(i);
		//cout << v[0] << " " << v[1] << endl;
		circle(dispImage, Point(v[0], v[1]), 10, Scalar(25, 25, 25), 2);
	}
	imshow("win", dispImage);
}

void testSVM()
{
	const int size = 2;
	int label[size] = { 1 , 2 };
	float traindata[size][2] = { {110, 233 }, {11,79}};
	//转为Mat以调用
	Mat trainMat(size, 2, CV_32FC1, traindata);
	Mat	labelMat(size, 1, CV_32SC1, label);
	//训练的初始化
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setC(100);
	//svm->setGamma(1);
	//svm->setDegree(2);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	//开始训练
	svm->train(trainMat, ROW_SAMPLE, labelMat);
	//-----------无关紧要的美工的部分-----------------------	
	//----其实对每个像素点的坐标也进行了分类----------------
	int width = 512, height = 512;
	Mat dispImage = Mat::zeros(width, height, CV_8UC3);
	Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255), black(0, 0, 0);
	for (int i = 0; i < dispImage.rows; ++i)
		for (int j = 0; j < dispImage.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = svm->predict(sampleMat);
			if (response == 1)
				dispImage.at<Vec3b>(i, j) = red;
			else if (response == 2)
				dispImage.at<Vec3b>(i, j) = green;
			else if (response == 3)
				dispImage.at<Vec3b>(i, j) = blue;
			else if (response == 4)
				dispImage.at<Vec3b>(i, j) = black;
		}
	//--------把初始化训练的点画进图片------------
	int thickness = -1;
	int lineType = 8;
	for (int idx = 0; idx < sizeof(label) / sizeof(int); idx++) {
		//circle(dispImage, Point(traindata[idx][0], traindata[idx][1]), 10, Scalar(255, 255, 255), thickness, -1);
	}
	// 把 support vectors  cout粗来看看……
	Mat sv = svm->getSupportVectors();
	//cout << "Support Vectors为：" << endl;
	for (int i = 0; i < sv.rows; ++i)
	{
		const float* v = sv.ptr<float>(i);
		//cout << v[0] << " " << v[1] << endl;
		circle(dispImage, Point(v[0], v[1]), 10, Scalar(255, 128, 255), 2);
	}
	imshow("win2", dispImage);
}


int main()
{
	//训练需要用到的数据
	namedWindow("win", 0);
	cv::resizeWindow("win", 500, 500);
	cv::createTrackbar("Cx0.01", "win", &g_C, 10000, onSlider_C_Gamma);
	onSlider_C_Gamma(0, 0);
	cv::createTrackbar("gammax0.01", "win", &g_gamma, 10000, onSlider_C_Gamma);
	onSlider_C_Gamma(0, 0);
	cv::createTrackbar("degree", "win", &g_degree, 50, onSlider_C_Gamma);
	onSlider_C_Gamma(0, 0);
	testSVM();
	cout << "done!" << endl;
	cv::waitKey(0);
}




