#ifndef INFERENCE_H
#define INFERENCE_H
#include "predictor.h"

typedef struct
{
    Bbox box;
    cv::Point lb;
    cv::Point lt;
    cv::Point rb;
    cv::Point rt;
} Bbox_Points;

class TableDetector
{
public:
    TableDetector(const std::string obj_model_path, const std::string edge_model_path, const std::string cls_model_path);
    std::vector<Bbox_Points> detect(const cv::Mat& srcimg, const float det_accuracy=0.7);
private:
    std::shared_ptr<YoloDet> obj_detector{nullptr};
    std::shared_ptr<YoloSeg> segnet{nullptr};
    std::shared_ptr<PPLCNet> pplcnet{nullptr};

    void get_box_points(const Bbox& box, cv::Point& lt, cv::Point& rt, cv::Point& rb, cv::Point& lb);
    Bbox pad_box_points(const int h, const int w, const int xmax, const int xmin, const int ymax, const int ymin, const int pad);
    void adjust_edge_points_axis(cv::Mat& edge_box, cv::Point& lb, cv::Point& lt, cv::Point& rb, cv::Point& rt, const int xmin_edge, const int ymin_edge);
    void add_pre_info_for_cls(cv::Mat& cls_img, const cv::Mat& edge_box, const int xmin_cls, const int ymin_cls);
    void get_real_rotated_points(const cv::Point& lb, const cv::Point& lt, const int pred_label, const cv::Point& rb, const cv::Point& rt, cv::Point& lb1, cv::Point& lt1, cv::Point& rb1, cv::Point& rt1);
};

#endif