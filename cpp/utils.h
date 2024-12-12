#ifndef UTIL_H
#define UTIL_H
#include <iostream>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

typedef struct
{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    float score;
} Bbox;

cv::Mat ResizePad(const cv::Mat& img, const int target_size, int& new_w, int& new_h, int& left, int& top);
cv::Mat get_max_adjacent_bbox(const cv::Mat& mask);
void visuallize(cv::Mat& img, const Bbox& box, const cv::Point& lt, const cv::Point& rt, const cv::Point& rb, const cv::Point& lb);
cv::Mat extract_table_img(const cv::Mat&img, const cv::Point& lt, const cv::Point& rt, const cv::Point& rb, const cv::Point& lb);

template<typename T> std::vector<int> argsort_descend(const std::vector<T>& array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    std::iota(array_index.begin(), array_index.end(), 0);

    std::sort(array_index.begin(), array_index.end(),
        [&array](int pos1, int pos2) {return (array[pos1] > array[pos2]); });

    return array_index;
}

template<typename T> std::vector<int> argsort_ascend(const std::vector<T>& array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    std::iota(array_index.begin(), array_index.end(), 0);

    std::sort(array_index.begin(), array_index.end(),
        [&array](int pos1, int pos2) {return (array[pos1] < array[pos2]); });

    return array_index;
}


#endif