from pyzbar import pyzbar
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
        # if int(cls) == 0:
        #     continue  # 跳过条形码，交给其他方法处理
        bbox_img = img[y1:y2, x1:x2]
        bbox_path = 'yolo_inference/' + str(int(time.time())) + f'_{i}_{int(cls)}' + '.png'
        bbox_cls = int(cls)
        bbox = {"cls": bbox_cls, "path": bbox_path}
        cv2.imwrite(bbox_path, bbox_img)
        bboxes.append(bbox)

    return bboxes


def inference_barcode(barcode_bboxes):
    results = []

    for bbox_path in barcode_bboxes:
        bbox_img = cv2.imread(bbox_path)
        barcodes = pyzbar.decode(bbox_img)
        barcodes_rets = [barcode.data.decode('utf-8') for barcode in barcodes]
        if barcodes_rets:
            results.append(*barcodes_rets)

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

    rets = inference_work(weight="yolov11n_all_commodity.pt",  # yolo 模型权重
                          input_img="test_imgs/so_many_codes.jpg",  # 商品外包装图像
                          api_key="app-ScBngBz8Or67tKg3h9QwyI7i",  # dify 的 API 密钥
                          base_url="http://127.0.0.1/v1",
                          user="kevinqiu")  # 后端的 API 服务器 IP

    print("============ Barcode ===============")

    for i, ret in enumerate(rets[0]):
        # print(f"======== 第 {i} 个 bbox 信息 ========")
        print(ret)
        # print("====================================")

    print("============ Datecode =============")

    for i, ret in enumerate(rets[1]):
        # print(f"======== 第 {i} 个 bbox 信息 ========")
        print(ret)
        # print("====================================")
