import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

img_1 = cv2.imread('./SomeFakeImages/fake_1.png')

# 与以左上角元素作为参考所得结果大致相同
def interpolate(img, factor):
    return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
                         scale_factor=1 / factor, mode='nearest', recompute_scale_factor=True)

# 通过最小尺度的裁剪以满足当前图像可以被分成 n 个 block_size * block_size 的块
def trim_image_to_blocks(image, block_size):
    # 获取原始图像尺寸
    height, width, channels = image.shape

    # 计算新的图像尺寸
    new_height = height - (height % block_size)
    new_width = width - (width % block_size)

    # 裁剪图像
    trimmed_image = image[:new_height, :new_width, :]

    return trimmed_image

# 三种 NPR 的操作，top 为性能最佳
def calculate_npr(image, l=2, method='top'):
    # 获取图像尺寸
    height, width, channels = image.shape

    # 初始化NPR特征数组
    npr_features = np.zeros(channels * height * width, dtype=np.double).reshape(height, width, channels)
    print(npr_features.shape)
    # 对每个通道进行处理
    for i in range(channels):
        channel = image[:, :, i]
        # 遍历调整后的通道中的每个l×l块
        for j in range(0, height - l + 1, l):
            for k in range(0, width - l + 1, l):
                block = channel[j:j + l, k:k + l]  # 提取l×l块
                if method == 'top': reference_pixel = block[0, 0]  # 使用左上角像素作为参考
                elif method == 'avg': reference_pixel = np.average(block)  # 使用像素平均值作为参考
                else: reference_pixel = np.max(block)  # 使用像素最大值作为参考
                # 计算与参考像素的差异并进行赋值操作
                differences = block - reference_pixel
                npr_features[j:j + l, k:k + l, i] = differences

    return npr_features.astype(np.uint8)

if __name__ == '__main__':
    NPR = calculate_npr(img_1, l=12, method='top')
    # img_1 = torch.from_numpy(img_1).unsqueeze(0).permute(0, 3, 1, 2)
    # NPR = (img_1 - interpolate(img_1, 0.5)).squeeze(0).permute(1, 2, 0)
    plt.imshow(cv2.cvtColor(NPR, cv2.COLOR_RGB2GRAY), cmap='YlGnBu')
    # plt.imshow(NPR, cmap='cool')
    plt.imsave('nprResults/fake_npr_int_l12.png', cv2.cvtColor(NPR, cv2.COLOR_RGB2GRAY), cmap='YlGnBu')
    # plt.show()
