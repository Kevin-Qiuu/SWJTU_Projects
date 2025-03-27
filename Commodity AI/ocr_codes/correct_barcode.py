import cv2
import numpy as np


def correct_barcode_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 转为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # 如果无法找到轮廓，表示图像可能太不清晰了，直接返回原图
    if len(contours) == 0:
        return image

    # 找到最大轮廓，假设它是条形码的矩形
    largest_contour = max(contours, key=cv2.contourArea)

    # 计算最小外接矩形
    rect = cv2.minAreaRect(largest_contour)

    # 获取矩形的角度和尺寸
    angle = rect[2]


    # 获取矩形的尺寸
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 计算旋转矩阵
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后的图像大小（防止图像被裁剪）
    (h, w) = image.shape[:2]
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # 调整旋转矩阵以考虑新的图像大小
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    # 进行图像旋转
    rotated_image = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

    return rotated_image


if __name__ == "__main__":
    # 测试函数
    image_path = 'yolo_inference/img_1.png'
    corrected_image = correct_barcode_image(image_path)
    # 顺时针旋转 90 度
    rotated_image = cv2.rotate(corrected_image, cv2.ROTATE_90_CLOCKWISE)

    # 显示矫正后的图像
    cv2.imshow('Corrected Barcode', corrected_image)
    cv2.waitKey(0)
    cv2.imshow('Rotated Barcode', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

