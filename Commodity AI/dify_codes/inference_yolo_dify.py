from ultralytics import YOLO
import cv2
import time

import dify_api


def make_bboxes(model, results):
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
            continue  # 跳过条形码，交给其他方法处理
        bbox = img[y1:y2, x1:x2]
        bbox_path = 'yolo_inference/' + str(int(time.time())) + f'_{i}_{int(cls)}' + '.png'
        cv2.imwrite(bbox_path, bbox)
        bboxes.append(bbox_path)

    return bboxes



def inference_work(weight, input_img, api_key, base_url):
    model = YOLO(weight)
    results = model(input_img)
    bboxes = make_bboxes(model, results)
    results = []
    for bbox in bboxes:
        results.append(dify_api.execute_on_dify(bbox, api_key, base_url))
    return results


if __name__ == '__main__':
    rets = inference_work(weight="yolov11n_all_commodity.pt",  # yolo 模型权重
                          input_img="test_imgs/test_3_qi_ma_bar.jpg",  # 商品外包装图像
                          api_key="app-ScBngBz8Or67tKg3h9QwyI7i",  # dify 的 API 密钥
                          base_url="http://127.0.0.1/v1")  # 后端的 API 服务器 IP
    for i, ret in enumerate(rets):
        print(f"======== 第 {i} 个 bbox 信息 ========")
        print(ret)
        print("====================================")

