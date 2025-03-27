import os
import paddleocr
import zxing
import cv2
import time
from ultralytics import YOLO
from correct_barcode import correct_barcode_image
from identify_datecode import inference_datecode


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


def inference_work(yolo_model, ocr_model, input_img, results_outputdir='yolo_inference'):
    results = yolo_model(input_img)
    bboxes = make_bboxes(yolo_model, results, results_outputdir)
    barcode_bboxes_paths = [bbox['path'] for bbox in bboxes if bbox['cls'] == 0]
    datecode_bboxes_paths = [bbox['path'] for bbox in bboxes if bbox['cls'] != 0]

    barcode_results = inference_barcode(barcode_bboxes_paths)  # 识别条形码
    datecode_results = inference_datecode(ocr_model, datecode_bboxes_paths)  # 识别日期码

    results = [barcode_results, datecode_results]
    return results


def show_inference_work(yolo_model, ocr_model, input_img):
    strat_time = time.time()
    [barcode_rets, datecode_rets] = inference_work(yolo_model, ocr_model, input_img)

    print("\n==============================")
    print("Identified barcode results:")
    for barcode_result in barcode_rets:
        print(barcode_result)

    print("\n==============================")
    print("Identified date text results:")
    for datecode_result in datecode_rets:
        print(datecode_result)

    used_time = time.time() - strat_time
    print(f"\nTotal time: {used_time}s")


# 在一个文件夹中找到条形码识别失败的图片
def find_none_bar(weight, input_dir):
    model = YOLO(weight)
    print("Load yolo successfully!")
    files = os.listdir(input_dir)
    img_in_dir = []
    none_bar_img = []
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        img_in_dir.append(file_path)

    print(f"总共有 {len(img_in_dir)} 张图片")
    print("====== Inference Begin ======")
    time_start = time.time()
    for i, img_path in enumerate(img_in_dir):
        results = model(img_path)
        bboxes = make_bboxes(model, results, "yolo_inference")
        barcode_bboxes_paths = [bbox['path'] for bbox in bboxes if bbox['cls'] == 0]
        if len(barcode_bboxes_paths) == 0:
            continue
        # print(barcode_bboxes_paths)

        rets = inference_barcode(barcode_bboxes_paths)
        for (ret, img_path_bbox) in zip(rets, barcode_bboxes_paths):
            if ret is None:
                none_bar_img.append(img_path)
                cv2.imwrite("no_barcode_img/"+f"{i}.jpg", cv2.imread(img_path_bbox))
                print(img_path)
                file_path = 'none_bar_img.txt'

                # 使用 'a' 模式追加数据
                with open(file_path, 'a') as file:
                    file.write(img_path_bbox + "--->" + img_path + '\n')

    used_time = time.time() - time_start
    print(f"====== End, time: {used_time} ========")
    return none_bar_img



if __name__ == '__main__':
    Yolo_model = YOLO("model_weights/yolov11s_best.pt")
    Ocr_model = paddleocr.PaddleOCR(use_angle_cls=True, lang='ch')  # 反向检测，语言
    print("**********************************************")
    print("*   Load yolo and ocr models successfully!   *")
    print("**********************************************")

    show_inference_work(Yolo_model, Ocr_model, "test_imgs/so_many_codes.jpg")





    # input_dir = "test_imgs"
    # file_name = os.listdir(input_dir)
    # s_time = time.time()
    # for file in file_name:
    #     if file == ".DS_Store":
    #         continue
    #     img_path = os.path.join(input_dir, file)
    #     show_inference_work(yolo_model, ocr_model, img_path)
    #
    # u_time = time.time() - s_time
    # print("\n==============================")
    # print(f"Total time: {u_time}s")
    # print(f"Average time: {u_time/len(file_name)}s")
