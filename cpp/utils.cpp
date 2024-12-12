#include "utils.h"
#include <vector>
#include <cmath>
#include <algorithm>


using namespace std;
using namespace cv;

Mat sortMat(const Mat &stats, int colId)
{
    //根据指定列以行为单位排序
    
    Mat sorted_index;
    cv::sortIdx(stats, sorted_index, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
    // 降序是DESCENDING 升序是ASCENDING
    
    sorted_index = sorted_index.col(colId);
    Mat sorted_stats = stats.clone();
    int row_num = sorted_index.rows;
    for(int i = 0; i < row_num; i++){
        int _idx = sorted_index.at<int>(i, 0);
        sorted_stats.row(i) = stats.row(_idx) + 0;//必须加0否则会出很难debug的错误
    }
    return sorted_stats;
}


std::tuple<vector<Point2f>, float> get_mini_boxes(const vector<Point>& contour)
{
    RotatedRect bounding_box = cv::minAreaRect(contour);
    cv::Mat rect;
    cv::boxPoints(bounding_box, rect);
    Mat points = sortMat(rect, 0);

    int index_1 = 0;
    int index_2 = 1;
    int index_3 = 2;
    int index_4 = 3;
    if(points.ptr<float>(1)[1] > points.ptr<float>(0)[1])
    {
        index_1 = 0;
        index_4 = 1;
    }
    else
    {
        index_1 = 1;
        index_4 = 0;
    }
    if(points.ptr<float>(3)[1] > points.ptr<float>(2)[1])
    {
        index_2 = 2;
        index_3 = 3;
    } 
    else
    {
        index_2 = 3;
        index_3 = 2;
    }
        
    vector<Point2f> box = {Point2f(points.ptr<float>(index_1)[0], points.ptr<float>(index_1)[1]),
                           Point2f(points.ptr<float>(index_2)[0], points.ptr<float>(index_2)[1]),
                           Point2f(points.ptr<float>(index_3)[0], points.ptr<float>(index_3)[1]),
                           Point2f(points.ptr<float>(index_4)[0], points.ptr<float>(index_4)[1])};
    std::tuple<vector<Point2f>, float> result = std::make_tuple(box, std::min(bounding_box.size.width, bounding_box.size.height));
    return result;
}

Mat get_inv(const Mat& concat) {
    double a = concat.at<double>(0, 0);
    double b = concat.at<double>(0, 1);
    double c = concat.at<double>(1, 0);
    double d = concat.at<double>(1, 1);
    double det_concat = a * d - b * c;
    Mat inv_result = (Mat_<double>(2, 2) << d / det_concat, -b / det_concat, -c / det_concat, a / det_concat);
    return inv_result;
}

vector<vector<int>> nchoosek(int startnum, int endnum, int step = 1, int n = 1) {
    vector<vector<int>> c;
    vector<int> range;
    for (int i = startnum; i <= endnum; i += step) {
        range.push_back(i);
    }
    vector<int> combination;
    function<void(int, int)> combine = [&](int offset, int k) {
        if (k == 0) {
            c.push_back(combination);
            return;
        }
        for (int i = offset; i <= range.size() - k; ++i) {
            combination.push_back(range[i]);
            combine(i + 1, k - 1);
            combination.pop_back();
        }
    };
    combine(0, n);
    return c;
}

vector<Point> minboundquad(const vector<Point>& hull) 
{
    int len_hull = hull.size();
    vector<Point2f> xy(hull.begin(), hull.end());
    vector<int> idx(len_hull);
    iota(idx.begin(), idx.end(), 0);
    vector<int> idx_roll(len_hull);
    rotate_copy(idx.begin(), idx.begin() + 1, idx.end(), idx_roll.begin());
    vector<vector<int>> edges(len_hull, vector<int>(2));
    for (int i = 0; i < len_hull; ++i) {
        edges[i][0] = idx[i];
        edges[i][1] = idx_roll[i];
    }
    vector<pair<double, int>> edgeangles1;
    for (int i = 0; i < len_hull; ++i) {
        double y = xy[edges[i][1]].y - xy[edges[i][0]].y;
        double x = xy[edges[i][1]].x - xy[edges[i][0]].x;
        double angle = atan2(y, x);
        if (angle < 0) {
            angle += 2 * M_PI;
        }
        edgeangles1.emplace_back(angle, i);
    }
    sort(edgeangles1.begin(), edgeangles1.end());
    vector<vector<int>> edges1;
    vector<double> edgeangle1;
    for (const auto& item : edgeangles1) {
        edges1.push_back(edges[item.second]);
        edgeangle1.push_back(item.first);
    }
    vector<double> edgeangles(edgeangle1.begin(), edgeangle1.end());
    edges = edges1;
    double eps = 2.2204e-16;
    double angletol = eps * 100;
    vector<bool> k(edgeangles.size() - 1);
    adjacent_difference(edgeangles.begin(), edgeangles.end(), k.begin(), [&](double a, double b) { return (b - a) < angletol; });
    vector<int> idx_to_delete;
    for (int i = 0; i < k.size(); ++i) {
        if (k[i]) {
            idx_to_delete.push_back(i);
        }
    }
    for (int i = idx_to_delete.size() - 1; i >= 0; --i) {
        edges.erase(edges.begin() + idx_to_delete[i]);
        edgeangles.erase(edgeangles.begin() + idx_to_delete[i]);
    }
    int nedges = edges.size();
    vector<vector<int>> edgelist = nchoosek(0, nedges - 1, 1, 4);
    vector<int> k_idx;
    for (int i = 0; i < edgelist.size(); ++i) {
        if (edgeangles[edgelist[i][3]] - edgeangles[edgelist[i][0]] <= M_PI) {
            k_idx.push_back(i);
        }
    }
    for (int i = k_idx.size() - 1; i >= 0; --i) {
        edgelist.erase(edgelist.begin() + k_idx[i]);
    }
    int nquads = edgelist.size();
    double quadareas = numeric_limits<double>::infinity();
    vector<Point> cnt(4);
    for (int i = 0; i < nquads; ++i) {
        vector<int> edgeind = edgelist[i];
        edgeind.push_back(edgelist[i][0]);
        vector<vector<int>> edgesi;
        vector<double> edgeang;
        for (int idx : edgeind) {
            edgesi.push_back(edges[idx]);
            edgeang.push_back(edgeangles[idx]);
        }
        bool is_continue = false;
        for (int j = 0; j < edgeang.size() - 1; ++j) {
            if (edgeang[j + 1] - edgeang[j] > M_PI) {
                is_continue = true;
                break;
            }
        }
        if (is_continue) {
            continue;
        }
        vector<double> qxi(4), qyi(4);
        for (int j = 0; j < 4; ++j) {
            int jplus1 = j + 1;
            vector<int> shared;
            set_intersection(edgesi[j].begin(), edgesi[j].end(), edgesi[jplus1].begin(), edgesi[jplus1].end(), back_inserter(shared));
            if (!shared.empty()) {
                qxi[j] = xy[shared[0]].x;
                qyi[j] = xy[shared[0]].y;
            } else {
                Point2f A = xy[edgesi[j][0]];
                Point2f B = xy[edgesi[j][1]];
                Point2f C = xy[edgesi[jplus1][0]];
                Point2f D = xy[edgesi[jplus1][1]];
                Mat concat = (Mat_<double>(2, 2) << A.x - B.x, D.x - C.x, A.y - B.y, D.y - C.y);
                Mat div = (Mat_<double>(2, 1) << A.x - C.x, A.y - C.y);
                Mat inv_result = get_inv(concat);
                double a = inv_result.at<double>(0, 0);
                double b = inv_result.at<double>(0, 1);
                double c = inv_result.at<double>(1, 0);
                double d = inv_result.at<double>(1, 1);
                double e = div.at<double>(0, 0);
                double f = div.at<double>(1, 0);
                vector<double> ts1 = {a * e + b * f, c * e + d * f};
                Point2f Q = A + (B - A) * ts1[0];
                qxi[j] = Q.x;
                qyi[j] = Q.y;
            }
        }
        vector<Point> contour;
        for (int j = 0; j < 4; ++j) {
            contour.emplace_back(qxi[j], qyi[j]);
        }
        double A_i = contourArea(contour);
        if (A_i < quadareas) {
            quadareas = A_i;
            cnt = contour;
        }
    }
    return cnt;
}


Mat ResizePad(const Mat& img, const int target_size, int& new_w, int& new_h, int& left, int& top)
{
    const int h = img.rows;
    const int w = img.cols;
    const int m = max(h, w);
    const float ratio = (float)target_size / (float)m;
    new_w = int(ratio * w);
    new_h = int(ratio * h);
    Mat dstimg;
    resize(img, dstimg, Size(new_w, new_h), 0, 0, INTER_LINEAR);
    top = (target_size - new_h) / 2;
    int bottom = (target_size - new_h) - top;
    left = (target_size - new_w) / 2;
    int right = (target_size - new_w) - left;
    copyMakeBorder(dstimg, dstimg, top, bottom, left, right, BORDER_CONSTANT, Scalar(114, 114, 114));
    return dstimg;
}

Mat get_max_adjacent_bbox(const Mat& mask)
{
    vector<vector<Point>> contours;
    cv::findContours(mask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    float max_size;
    vector<Point> cnt_save;
    for(int i=0;i<contours.size();i++)
    {
        std::tuple<vector<Point2f>, float> result = get_mini_boxes(contours[i]);
        //vector<Point2f> points = std::get<0>(result); ////没有用
        float sside = std::get<1>(result);
        if(sside > max_size)
        {
            max_size = sside;
            cnt_save = contours[i];
        }
    }
    if(cnt_save.size() > 0)
    {
        float epsilon = 0.01 * cv::arcLength(cnt_save, true);
        vector<Point> box;
        cv::approxPolyDP(cnt_save, box, epsilon, true);
        vector<Point> hull;
        cv::convexHull(box, hull);
        std::tuple<vector<Point2f>, float> result = get_mini_boxes(cnt_save);
        vector<Point2f> points = std::get<0>(result);
        const int len_hull = hull.size();

        if(len_hull==4)
        {
            Mat tar_box = Mat(hull.size(), 2, CV_32FC1);
            for(int i=0;i<hull.size();i++)
            {
                tar_box.ptr<float>(i)[0] = hull[i].x;
                tar_box.ptr<float>(i)[1] = hull[i].y;
            }
            return tar_box;   ////也可以返回vector<Point>这种格式的
        }
        else if(len_hull > 4)
        {
            vector<Point> target_box = minboundquad(hull);
            Mat tar_box = Mat(target_box.size(), 2, CV_32FC1);
            for(int i=0;i<target_box.size();i++)
            {
                tar_box.ptr<float>(i)[0] = target_box[i].x;
                tar_box.ptr<float>(i)[1] = target_box[i].y;
            }
            return tar_box;
        }
        else
        {
            Mat tar_box = Mat(points.size(), 2, CV_32FC1);
            for(int i=0;i<points.size();i++)
            {
                tar_box.ptr<float>(i)[0] = points[i].x;
                tar_box.ptr<float>(i)[1] = points[i].y;
            }
            return tar_box;
        }
    }
    else
    {
        return cv::Mat();
    }
}


void visuallize(cv::Mat& img, const Bbox& box, const Point& lt, const Point& rt, const Point& rb, const Point& lb)
{
    vector<Point> draw_box = {lt, rt, rb, lb};
    circle(img, lt, 50, Scalar(255, 0, 0), 10);
    rectangle(img, Point(box.xmin, box.ymin), Point(box.xmax, box.ymax), Scalar(255, 0, 0), 10);
    cv::polylines(img, draw_box, true, Scalar(255, 0, 255), 6);
}

Mat extract_table_img(const Mat&img, const Point& lt, const Point& rt, const Point& rb, const Point& lb)
{
    Point2f src_points[4] = {lt, rt, lb, rb};
    const float width_a = sqrt(pow(rb.x - lb.x, 2) + pow(rb.y - lb.y, 2));
    const float width_b = sqrt(pow(rt.x - lt.x, 2) + pow(rt.y - lt.y, 2));
    const float max_width = max(width_a, width_b);

    const float height_a = sqrt(pow(rt.x - rb.x, 2) + pow(rt.y - rb.y, 2));
    const float height_b = sqrt(pow(lt.x - lb.x, 2) + pow(lt.y - lb.y, 2));
    const float max_height = max(height_a, height_b);

    Point2f dst_points[4] = {Point2f(0, 0), Point2f(max_width - 1, 0), Point2f(0, max_height - 1), Point(max_width - 1, max_height - 1)};
    Mat M;
    M = cv::getPerspectiveTransform(src_points, dst_points);
    Mat warped ;
    cv::warpPerspective(img, warped, M, Size(max_width, max_height));
    return warped;
}