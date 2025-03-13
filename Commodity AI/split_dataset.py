import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm


def split_dataset(img_dir, label_dir, output_dir, ratios=(0.8, 0.2)):
    """
    划分数据集到COCO8.yaml 格式目录结构
    :param img_dir: 原始图片文件夹路径（如：/dataset/images）
    :param label_dir: 原始标签文件夹路径（如：/dataset/labels）
    :param output_dir: 输出根目录（自动创建images/train, images/val等子目录）
    :param ratios: 训练集与验证集比例（默认8:2）
    """
    # 创建目录结构 [8]()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dirs = {
        'train_img': os.path.join(output_dir, 'images/train'),
        'val_img': os.path.join(output_dir, 'images/val'),
        'train_label': os.path.join(output_dir, 'labels/train'),
        'val_label': os.path.join(output_dir, 'labels/val')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 获取并打乱文件列表 [14]()
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(img_files)

    # 计算分割点 [3]()
    split_idx = int(len(img_files) * ratios[0])
    train_files = img_files[:split_idx]
    val_files = img_files[split_idx:]

    # 复制文件到目标目录 [2]()
    for phase, files in [('train', train_files), ('val', val_files)]:
        print("===============================")
        print(f"开始划分 {phase} 的图片和标签...")
        for img_file in tqdm(files):
            # 复制图片
            src_img = os.path.join(img_dir, img_file)
            dst_img = os.path.join(dirs[f'{phase}_img'], img_file)
            shutil.copy2(src_img, dst_img)

            # 复制对应标签
            label_file = Path(img_file).stem + '.txt'
            src_label = os.path.join(label_dir, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(dirs[f'{phase}_label'], label_file)
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告：缺失标签文件 {label_file}")
        print(f"{phase} 的图片和标签划分完成！")
        print("===============================")
        

    print(f"数据集划分完成：\n"
          f"训练集: {len(train_files)} images\n"
          f"验证集: {len(val_files)} images\n"
          f"输出目录结构：\n{os.linesep.join(dirs.values())}")


# 使用示例（修改路径参数）
split_dataset(
    img_dir="datasets/All_Images/image",
    label_dir="datasets/All_Images/txt",
    output_dir="datasets/all_images_coco8_dataset",
    ratios=(0.9, 0.1)  # 可调整为其他比例如(0.9, 0.1)
)