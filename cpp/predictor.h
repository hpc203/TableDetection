#ifndef PREDICTOR_H
#define PREDICTOR_H
#include "utils.h"
#include <opencv2/dnn.hpp>


class YoloDet
{
public:
	YoloDet(const std::string model_path);
	std::vector<Bbox> infer(const cv::Mat& srcimg, const float score=0.4f);
private:
	const int resize_shape[2] = {928, 928};
	std::vector<std::string> outlayer_names;
    cv::dnn::Net model;
};

class YoloSeg
{
public:
	YoloSeg(const std::string model_path);
	std::tuple<cv::Mat, cv::Point, cv::Point, cv::Point, cv::Point> infer(const cv::Mat& srcimg);
private:
	const int resize_shape[2] = {800, 800};
	cv::Mat img_postprocess(const std::vector<cv::Mat>& predict_maps);
	void adjust_coordinates(cv::Mat& box, const int left, const int top, const int resize_w, const int resize_h, const int destWidth, const int destHeight);
	void sort_and_clip_coordinates(const cv::Mat& box, cv::Point& lt, cv::Point& lb, cv::Point& rt, cv::Point& rb);
	std::vector<std::string> outlayer_names;
    cv::dnn::Net model;
};

class PPLCNet
{
public:
	PPLCNet(const std::string model_path);
	int infer(const cv::Mat& srcimg);
private:
	const int resize_shape[2] = {624, 624};
	std::vector<std::string> outlayer_names;
	cv::dnn::Net model;
};

#endif