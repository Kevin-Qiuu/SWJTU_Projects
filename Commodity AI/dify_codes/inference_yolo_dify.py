import os

import zxing

from ultralytics import YOLO
import cv2
import time
import dify_api
from correct_barcode import correct_barcode_image


def make_bboxes(model, results, results_outputdir='yolo_inference'):
    img = results[0].orig_img  # 提取原始BGR图像
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 转换为numpy数组
    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    bboxes = []

    for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        label = f"bbox{i} : {model.names[int(cls)]}  {conf:.2f}"
        print(label)

        if int(cls) == 0:
            # 条形码照片四周均扩大 40 个像素
            y1 = max(0, y1 - 20)
            y2 = min(img.shape[0], y2 + 20)
            x1 = max(0, x1 - 20)
            x2 = min(img.shape[1], x2 + 20)

        bbox_img = img[y1:y2, x1:x2]
        bbox_path = results_outputdir + "/" + str(int(time.time())) + f'_{i}_{int(cls)}' + '.png'
        bbox_cls = int(cls)
        bbox = {"cls": bbox_cls, "path": bbox_path}
        print(f"Process: {bbox_path}, cls: {bbox_cls}")
        cv2.imwrite(bbox_path, bbox_img)
        bboxes.append(bbox)

    return bboxes

def inference_barcode(barcode_bboxes):
    reader = zxing.BarCodeReader()
    results = []
    for bbox_path in barcode_bboxes:
        # print(f"Process: {bbox_path}")
        result = reader.decode(bbox_path)

        # 如果识别为空，尝试纠正图像，进行两次旋转验证
        if result.raw is None:
            corrected_image = correct_barcode_image(bbox_path)
            cv2.imwrite(bbox_path, corrected_image)
            result = reader.decode(bbox_path)
            if result.raw is None:
                # 如果识别为空，尝试顺时针旋转 90 度，可能是因为条形码没有水平
                rotated_image = cv2.rotate(corrected_image, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(bbox_path, rotated_image)
                result = reader.decode(bbox_path)

        results.append(result.raw)

    return results


def inference_dify(datecode_bboxes_path, api_key, base_url, user="kevinqiu"):
    results = []

    for bbox_path in datecode_bboxes_path:
        results.append(dify_api.execute_on_dify(bbox_path, api_key, base_url, user))

    return results


def inference_work(weight, input_img, api_key, base_url, user="kevinqiu"):
    model = YOLO(weight)
    results = model(input_img)
    bboxes = make_bboxes(model, results)

    barcode_bboxes_paths = [bbox['path'] for bbox in bboxes if bbox['cls'] == 0]
    datecode_bboxes_paths = [bbox['path'] for bbox in bboxes if bbox['cls'] != 0]

    barcode_results = inference_barcode(barcode_bboxes_paths)  # 识别条形码
    datecode_results = inference_dify(datecode_bboxes_paths, api_key, base_url, user)  # 识别日期码

    results = [barcode_results, datecode_results]
    return results




if __name__ == '__main__':

    rets = inference_work(weight="model_weights/best-yolov8-400.pt",  # yolo 模型权重
                          input_img="test_imgs/english_example.png",  # 商品 外包装图像
                          api_key="app-ScBngBz8Or67tKg3h9QwyI7i",  # dify 的 API 密钥
                          base_url="http://127.0.0.1/v1",  # 后端的 API 服务器 IP
                          user="kevinqiu")  # 向 dify 服务器发送请求的用户名称


    print("============ Barcode ===============")
    for i, ret in enumerate(rets[0]):
        print(ret)


    print("============ Datecode =============")
    for i, ret in enumerate(rets[1]):
        print(ret)
