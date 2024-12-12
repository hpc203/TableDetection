#include "predictor.h"


using namespace std;
using namespace cv;
using namespace dnn;


YoloDet::YoloDet(const string model_path)
{
    this->model = readNet(model_path);
	this->outlayer_names = this->model.getUnconnectedOutLayersNames();
}

vector<Bbox> YoloDet::infer(const Mat& srcimg, const float score)
{
    const int ori_h = srcimg.rows;
    const int ori_w = srcimg.cols;
    ////img_preprocess////
    Mat img;
    int new_w, new_h, left, top;
    img = ResizePad(srcimg, this->resize_shape[0], new_w, new_h, left, top);
    img.convertTo(img, CV_32FC3, 1.0/255.0);
    Mat blob = blobFromImage(img);

    this->model.setInput(blob);
    std::vector<Mat> outs;
    this->model.forward(outs, this->outlayer_names);

    ////img_postprocess////
    const float x_factor = (float)ori_w / new_w;
    const float y_factor = (float)ori_h / new_h;
    vector<Rect> boxes;
    vector<float> scores;
    const int rows = outs[0].size[2];
    for(int i=0;i<rows;i++)
    {
        float max_score = outs[0].ptr<float>(0, 4)[i];
        if(max_score >= score)
        {
            float x = outs[0].ptr<float>(0, 0)[i];
            float y = outs[0].ptr<float>(0, 1)[i];
            float w = outs[0].ptr<float>(0, 2)[i];
            float h = outs[0].ptr<float>(0, 3)[i];
            int xmin = max(int((x - w / 2 - left) * x_factor), 0);
            int ymin = max(int((y - h / 2 - top) * y_factor), 0);
            boxes.emplace_back(Rect(xmin, ymin, int(w * x_factor), int(h * y_factor)));
            scores.emplace_back(max_score);
        }
    }

    vector<int> indices;
    NMSBoxes(boxes, scores, score, 0.4, indices);
    const int num_keep = indices.size();
    vector<Bbox> bboxes(num_keep);
    for(int i=0;i<num_keep;i++)
    {
        const int ind = indices[i];
        bboxes[i] = {boxes[ind].x, boxes[ind].y, min(boxes[ind].x + boxes[ind].width, ori_w-1), min(boxes[ind].y + boxes[ind].height, ori_h-1), scores[ind]};
    }
    return bboxes;
}

YoloSeg::YoloSeg(const string model_path)
{
    this->model = readNet(model_path);
	this->outlayer_names = this->model.getUnconnectedOutLayersNames();
}

std::tuple<Mat, Point, Point, Point, Point> YoloSeg::infer(const Mat& srcimg)
{
    const int destHeight = srcimg.rows;
    const int destWidth = srcimg.cols;
    ////img_preprocess////
    Mat img;
    int resize_h, resize_w, left, top;
    img = ResizePad(srcimg, this->resize_shape[0], resize_w, resize_h, left, top);
    // img.convertTo(img, CV_32FC3, 1.0/255.0);  ///也可以
    Mat blob = blobFromImage(img, 1.0/255.0);

    this->model.setInput(blob);
    std::vector<Mat> predict_maps;
    this->model.forward(predict_maps, this->outlayer_names);

    Mat pred = this->img_postprocess(predict_maps);
    if(pred.empty())
    {
        return std::make_tuple(Mat(), Point(), Point(), Point(), Point());
    }
    Mat mask = pred > 0.8;
    mask.convertTo(mask, CV_8UC1);
    
    Mat box = get_max_adjacent_bbox(mask);
    if(!box.empty())
    {
        this->adjust_coordinates(box, left, top, resize_w, resize_h, destWidth, destHeight);
        Point lt, lb, rt, rb;
        this->sort_and_clip_coordinates(box, lt, lb, rt, rb);
        return std::make_tuple(box, lt, lb, rt, rb);
    }
    else
    {
        return std::make_tuple(Mat(), Point(), Point(), Point(), Point());
    }
}

void YoloSeg::adjust_coordinates(Mat& box, const int left, const int top, const int resize_w, const int resize_h, const int destWidth, const int destHeight)
{
    for(int i=0;i<box.rows;i++)
    {
        float x = (box.ptr<float>(i)[0] - left) / resize_w * destWidth;
        float y = (box.ptr<float>(i)[1] - top) / resize_h * destHeight;
        box.ptr<float>(i)[0] = (int)std::min(std::max(x, 0.0f), (float)destWidth-1);
        box.ptr<float>(i)[1] = (int)std::min(std::max(y, 0.0f), (float)destHeight-1);
    }
}

void YoloSeg::sort_and_clip_coordinates(const Mat& box, Point& lt, Point& lb, Point& rt, Point& rb)
{
    vector<float> x = box.col(0).reshape(1);
    vector<int> l_idx = argsort_ascend(x);
    int l_box[2][2] = {{(int)box.ptr<float>(l_idx[0])[0], (int)box.ptr<float>(l_idx[0])[1]}, {(int)box.ptr<float>(l_idx[1])[0], (int)box.ptr<float>(l_idx[1])[1]}};
    int r_box[2][2] = {{(int)box.ptr<float>(l_idx[2])[0], (int)box.ptr<float>(l_idx[2])[1]}, {(int)box.ptr<float>(l_idx[3])[0], (int)box.ptr<float>(l_idx[3])[1]}};

    int l_idx_1[2] = {0, 1};
    if(l_box[0][1] > l_box[1][1])
    {
        l_idx_1[0] = 1;
        l_idx_1[1] = 0;
    }
    lt = Point(std::max(l_box[l_idx_1[0]][0], 0), std::max(l_box[l_idx_1[0]][1], 0));
    lb = Point(std::max(l_box[l_idx_1[1]][0], 0), std::max(l_box[l_idx_1[1]][1], 0));

    int r_idx_1[2] = {0, 1};
    if(r_box[0][1] > r_box[1][1])
    {
        r_idx_1[0] = 1;
        r_idx_1[1] = 0;
    }
    rt = Point(std::max(r_box[r_idx_1[0]][0], 0), std::max(r_box[r_idx_1[0]][1], 0));
    rb = Point(std::max(r_box[r_idx_1[1]][0], 0), std::max(r_box[r_idx_1[1]][1], 0));
}

Mat YoloSeg::img_postprocess(const vector<Mat>& predict_maps)
{
    Mat mask_output = predict_maps[1];
    const int len = predict_maps[0].size[1];
    const int num_proposals = predict_maps[0].size[2];
    Mat predictions = predict_maps[0].reshape(0, len).t();
    Mat scores = predictions.col(4);
    double max_class_socre;;
    Point classIdPoint;
    minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
    int highest_score_index = classIdPoint.y;

    Mat highest_score_prediction = predictions.row(highest_score_index);
    float x = highest_score_prediction.ptr<float>(0)[0];
    float y = highest_score_prediction.ptr<float>(0)[1];
    float w = highest_score_prediction.ptr<float>(0)[2];
    float h = highest_score_prediction.ptr<float>(0)[3];
    float highest_score = highest_score_prediction.ptr<float>(0)[4];
    if(highest_score < 0.7)
    {
        return Mat();
    }
    Mat mask_predictions = highest_score_prediction.colRange(5, len);
    const int num_mask = mask_output.size[1];
    const int mask_height = mask_output.size[2];
    const int mask_width = mask_output.size[3];
    const std::vector<int> newshape = {num_mask, mask_height*mask_width};
    Mat mask_output_reshaped = mask_output.reshape(0, newshape);   ////不考虑batchsize
    Mat masks = mask_predictions * mask_output_reshaped;
    cv::exp(-masks, masks);
    masks = 1.f / (1 + masks);
    Mat mask = masks.reshape(0, mask_height);     ////不考虑batchsize

    const int small_w = 200;
    const int small_h = 200;
    int small_x_min = max(0, int((x - w / 2) * small_w / 800.0));
    int small_x_max = min(small_w, int((x + w / 2) * small_w / 800.0));
    int small_y_min = max(0, int((y - h / 2) * small_h / 800.0));
    int small_y_max = min(small_h, int((y + h / 2) * small_h / 800.0));
    
    Mat filtered_mask = Mat::zeros(small_h, small_w, CV_32FC1);
    Rect crop_rect(small_x_min, small_y_min, small_x_max-small_x_min, small_y_max-small_y_min);
    mask(crop_rect).copyTo(filtered_mask(crop_rect));
    Mat resized_mask;
    resize(filtered_mask, resized_mask, Size(800, 800), 0, 0, INTER_CUBIC);
    return resized_mask;
}


PPLCNet::PPLCNet(const string model_path)
{
    this->model = readNet(model_path);
    this->outlayer_names = this->model.getUnconnectedOutLayersNames();
}

int PPLCNet::infer(const Mat& srcimg)
{
    ////img_preprocess////
    Mat img;
    int new_w, new_h, left, top;
    img = ResizePad(srcimg, this->resize_shape[0], new_w, new_h, left, top);
    img.convertTo(img, CV_32FC3, 1.0/255.0);
    Mat blob = blobFromImage(img);

    this->model.setInput(blob);
    std::vector<Mat> outs;
    this->model.forward(outs, this->outlayer_names);

    ////img_postprocess////
    const int cols = outs[0].size[1];
    float* pdata = (float*)outs[0].data;
    int maxPosition = std::max_element(pdata, pdata+cols) - pdata;
    return maxPosition;
}