#include "inference.h"


using namespace std;
using namespace cv;


TableDetector::TableDetector(const string obj_model_path, const string edge_model_path, const string cls_model_path)
{
    this->obj_detector = std::make_shared<YoloDet>(obj_model_path);
    this->segnet = std::make_shared<YoloSeg>(edge_model_path);
    this->pplcnet = std::make_shared<PPLCNet>(cls_model_path);
}

vector<Bbox_Points> TableDetector::detect(const Mat& srcimg, const float det_accuracy)
{
    Mat img;
    cvtColor(srcimg, img, COLOR_BGR2RGB);
    const int h = img.rows;
    const int w = img.cols;
    vector<Bbox_Points> result;
    
    vector<Bbox> obj_det_res = this->obj_detector->infer(img, det_accuracy);

    for(int i=0;i<obj_det_res.size();i++)
    {

        Point lb, lt, rb, rt;
        this->get_box_points(obj_det_res[i], lt, rt, rb, lb);

        Bbox edge_ = this->pad_box_points(h, w, obj_det_res[i].xmax, obj_det_res[i].xmin, obj_det_res[i].ymax, obj_det_res[i].ymin, 10);
        Rect roi = Rect(edge_.xmin, edge_.ymin, edge_.xmax-edge_.xmin, edge_.ymax-edge_.ymin);
        Mat crop_img;
        img(roi).copyTo(crop_img);
        std::tuple<Mat, Point, Point, Point, Point> seg_res = this->segnet->infer(crop_img);
        Mat edge_box = get<0>(seg_res);   //// 4x2的矩阵
        if(edge_box.empty())
        {
            continue;
        }
        
        lt = get<1>(seg_res);
        lb = get<2>(seg_res);
        rt = get<3>(seg_res);
        rb = get<4>(seg_res);
        this->adjust_edge_points_axis(edge_box, lb, lt, rb, rt, edge_.xmin, edge_.ymin);

        Bbox cls_ = this->pad_box_points(h, w, obj_det_res[i].xmax, obj_det_res[i].xmin, obj_det_res[i].ymax, obj_det_res[i].ymin, 5);
        roi = Rect(cls_.xmin, cls_.ymin, cls_.xmax-cls_.xmin, cls_.ymax-cls_.ymin);
        Mat cls_img;
        img(roi).copyTo(cls_img);

        this->add_pre_info_for_cls(cls_img, edge_box, cls_.xmin, cls_.ymin);
        const int pred_label = this->pplcnet->infer(cls_img);

        Bbox_Points box_points;
        this->get_real_rotated_points(lb, lt, pred_label, rb, rt, box_points.lb, box_points.lt, box_points.rb, box_points.rt);
        box_points.box = obj_det_res[i];
        result.emplace_back(box_points);
    }
    return result;
}

void TableDetector::get_box_points(const Bbox& box, Point& lt, Point& rt, Point& rb, Point& lb)
{
    lt = Point(box.xmin, box.ymin);
    rt = Point(box.xmax, box.ymin);
    rb = Point(box.xmax, box.ymax);
    lb = Point(box.xmin, box.ymax);
}

Bbox TableDetector::pad_box_points(const int h, const int w, const int xmax, const int xmin, const int ymax, const int ymin, const int pad)
{
    Bbox edge;
    edge.xmin = max(xmin-pad, 0);
    edge.ymin = max(ymin-pad, 0);
    edge.xmax = min(xmax+pad, w);
    edge.ymax = min(ymax+pad, h);
    edge.score=1.f;  ////忽律，没用的
    return edge;
}

void TableDetector::adjust_edge_points_axis(Mat& edge_box, Point& lb, Point& lt, Point& rb, Point& rt, const int xmin_edge, const int ymin_edge)
{
    edge_box.col(0) += xmin_edge;
    edge_box.col(1) += ymin_edge;
    lt.x += xmin_edge;
    lt.y += ymin_edge;
    lb.x += xmin_edge;
    lb.y += ymin_edge;
    rt.x += xmin_edge;
    rt.y += ymin_edge;
    rb.x += xmin_edge;
    rb.y += ymin_edge;
}

void TableDetector::add_pre_info_for_cls(cv::Mat& cls_img, const cv::Mat& edge_box, const int xmin_cls, const int ymin_cls)
{
    vector<Point> cls_box(edge_box.rows);
    for(int i=0;i<edge_box.rows;i++)
    {
        cls_box[i] = Point(edge_box.ptr<float>(i)[0] - xmin_cls, edge_box.ptr<float>(i)[1] - ymin_cls);
    }
    cv::polylines(cls_img, cls_box, true, Scalar(255, 0, 255), 5);
}

void TableDetector::get_real_rotated_points(const Point& lb, const Point& lt, const int pred_label, const Point& rb, const Point& rt, Point& lb1, Point& lt1, Point& rb1, Point& rt1)
{
    if(pred_label == 0)
    {
        lt1 = lt;
        rt1 = rt;
        rb1 = rb;
        lb1 = lb;
    }
    else if(pred_label == 1)
    {
        lt1 = rt;
        rt1 = rb;
        rb1 = lb;
        lb1 = lt;
    }
    else if(pred_label == 2)
    {
        lt1 = rb;
        rt1 = lb;
        rb1 = lt;
        lb1 = rt;
    }
    else if(pred_label == 3)
    {
        lt1 = lb;
        rt1 = lt;
        rb1 = rt;
        lb1 = rb;
    }
    else
    {
        lt1 = lt;
        rt1 = rt;
        rb1 = rb;
        lb1 = lb;
    }
}