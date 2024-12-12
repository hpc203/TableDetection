#include"inference.h"


using namespace std;
using namespace cv;


int main()
{
    const string imgpath = "/home/wangbo/project/my_table_det/images/doc5.jpg";
    const string obj_model_path = "/home/wangbo/project/my_table_det/weights/yolo_obj_det.onnx";
    const string edge_model_path = "/home/wangbo/project/my_table_det/weights/yolo_edge_det.onnx";
    const string cls_model_path = "/home/wangbo/project/my_table_det/weights/paddle_cls.onnx";

    TableDetector table_det(obj_model_path, edge_model_path, cls_model_path);
    Mat srcimg = imread(imgpath);
    std::vector<Bbox_Points> result = table_det.detect(srcimg);

    ////输出可视化
    Mat draw_img = srcimg.clone();
    for(int i=0;i<result.size();i++)
    {
        Bbox box = result[i].box;
        Point lt = result[i].lt;
        Point rt = result[i].rt;
        Point rb = result[i].rb;
        Point lb = result[i].lb;
        visuallize(draw_img, box, lt, rt, rb, lb);
        Mat wrapped_img = extract_table_img(srcimg, lt, rt, rb, lb);
        string savepath = "extract-"+to_string(i)+".jpg";
        imwrite(savepath, wrapped_img);
    }
    imwrite("visualize.jpg", draw_img);
}