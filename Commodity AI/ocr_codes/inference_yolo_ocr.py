import os
import paddleocr
import zxing
import cv2
import time
from ultralytics import YOLO
from correct_barcode import correct_barcode_image
from identify_datecode import inference_datecode, inference_datecode_with_cls
from pyzbar.pyzbar import decode

# 定义类别名称
CLASS_NAMES = ['条形码', '一期码', '二期码', '三期码']


def make_bboxes(model, results, results_output=True, results_outputdir='yolo_inference'):
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

        bbox_img = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        bbox_cls = int(cls)
        bbox = {"bbox_cls": bbox_cls, "bbox_img": bbox_img}
        bboxes.append(bbox)

        if results_output:
            if not os.path.exists(results_outputdir):
                os.makedirs(results_outputdir)
            bbox_path = results_outputdir + "/" + str(int(time.time())) + f'_{i}_{int(cls)}' + '.png'
            print(f"Process: {bbox_path}, bbox_cls: {bbox_cls}")
            cv2.imwrite(bbox_path, bbox_img)

    return bboxes


def inference_barcode(barcode_bboxes):
    # reader = zxing.BarCodeReader()
    results = []
    for bbox in barcode_bboxes:
        # print(f"Process: {bbox_path}")
        # cv2.imshow("bbox", bbox)
        # cv2.waitKey(0)
        decode_data = decode(bbox)
        result = []
        for barcode in decode_data:
            result.append(barcode.data.decode("utf-8"))

        # 如果识别为空，尝试纠正图像，进行两次旋转验证
        if result.__len__() == 0:
            corrected_image = correct_barcode_image(bbox)
            decode_data = decode(corrected_image)
            result = []
            for barcode in decode_data:
                result.append(barcode.data.decode("utf-8"))
            if result.__len__() == 0:
                # 如果识别为空，尝试顺时针旋转 90 度，可能是因为条形码没有水平
                rotated_image = cv2.rotate(corrected_image, cv2.ROTATE_90_CLOCKWISE)
                decode_data = decode(rotated_image)
                for barcode in decode_data:
                    result.append(barcode.data.decode("utf-8"))

        if result.__len__() > 0:
            results.append(*result)

    return results


def inference_work(yolo_model, ocr_model, input_img, result_output=True, results_outputdir='yolo_inference'):
    results = yolo_model(input_img)
    bboxes = make_bboxes(yolo_model, results, result_output, results_outputdir)
    barcode_bboxes = [bbox['bbox_img'] for bbox in bboxes if bbox['bbox_cls'] == 0]
    # datecode_bboxes_paths = [bbox['path'] for bbox in bboxes if bbox['cls'] != 0]
    datecode_bboxes = [bbox for bbox in bboxes if bbox['bbox_cls'] != 0]

    barcode_results = inference_barcode(barcode_bboxes)  # 识别条形码
    # datecode_results = inference_datecode(ocr_model, datecode_bboxes)  # 识别日期码
    datecode_results = inference_datecode_with_cls(ocr_model, datecode_bboxes)  # 识别日期码

    results = [barcode_results, datecode_results]
    return results


def show_inference_work(yolo_model, ocr_model, input_img):
    strat_time = time.time()
    [barcode_rets, datecode_rets] = inference_work(yolo_model, ocr_model, input_img)
    print("\n********* 识别结果如下：********")

    print("======= 条形码的识别结果 ========")
    if barcode_rets.__len__() == 0:
        print("没有识别到条形码")
    else:
        for barcode_result in barcode_rets:
            print(barcode_result)

    print("======== 日期码的识别结果 =======")
    if datecode_rets.__len__() == 0:
        print("YOLO 未识别到日期码")
    else:
        for datecode_result in datecode_rets:
            print("当前日期码为：" + CLASS_NAMES[datecode_result[0]])
            if datecode_result.__len__() == 1:
                print("OCR 识别日期码失败")
            for datecode in datecode_result[1:]:
                print(datecode)

    used_time = time.time() - strat_time
    print("==============================")
    print(f"本次识别花费时间：{used_time} 秒")


if __name__ == '__main__':
    Yolo_model = YOLO("model_weights/yolov11s_best.pt")
    Ocr_model = paddleocr.PaddleOCR(use_angle_cls=True, lang='ch')  # 反向检测，语言
    print("**********************************************")
    print("*   Load yolo and ocr models successfully!   *")
    print("**********************************************")

    show_inference_work(Yolo_model, Ocr_model,
                        "dataset/qiu_add_again/IMG_008.jpg")


