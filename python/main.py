import os
import cv2
from inference import TableDetector
from utils import visuallize, extract_table_img


if __name__=='__main__':
    img_path = "images/chip2.jpg"
    table_det = TableDetector("weights/yolo_obj_det.onnx", "weights/yolo_edge_det.onnx", "weights/paddle_cls.onnx")

    srcimg = cv2.imread(img_path)
    result = table_det.detect(srcimg.copy())
    
    # 输出可视化
    file_name_with_ext = os.path.basename(img_path)
    file_name, file_ext = os.path.splitext(file_name_with_ext)
    out_dir = "outputs"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    draw_img = srcimg.copy()
    for i, res in enumerate(result):
        box = res["box"]
        lt, rt, rb, lb = res["lt"], res["rt"], res["rb"], res["lb"]
        # 带识别框和左上角方向位置
        draw_img = visuallize(draw_img, box, lt, rt, rb, lb)
        # 透视变换提取表格图片
        wrapped_img = extract_table_img(srcimg, lt, rt, rb, lb)
        cv2.imwrite(f"{out_dir}/{file_name}-extract-{i}.jpg", wrapped_img)
    cv2.imwrite(f"{out_dir}/{file_name}-visualize.jpg", draw_img)