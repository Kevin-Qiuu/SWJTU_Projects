import yaml
from pathlib import Path
import cv2

def validate_yolo_dataset(dataset_path, yaml_path):
    """ 验证YOLO格式数据集的结构和内容 """
    dataset_path = "datasets/" + dataset_path
    errors = []
    # 验证目录结构
    required_dirs = ['images/train', 'labels/train', 
                    'images/val', 'labels/val']
    for dir_path in required_dirs:
        if not (Path(dataset_path) / dir_path).exists():
            errors.append(f" 缺失目录: {dataset_path}/{dir_path}")

    # 验证data.yaml 配置文件
    # yaml_path = Path(dataset_path) / 'data.yaml' 
    yaml_path = Path(yaml_path)
    if not yaml_path.exists(): 
        errors.append(" 缺少coco8.yaml 配置文件")
    else:
        with open(yaml_path) as f:
            data_cfg = yaml.safe_load(f) 
            
            # print("***********")
            # print(data_cfg)
            # print("***********")
            
            # 验证关键字段
            for key in ['names', 'train', 'val']:
                if key not in data_cfg:
                    errors.append(f"data.yaml 缺少必要字段: {key}")
            
            # 验证路径有效性
            for phase in ['train', 'val']:
                if not Path("datasets/" + data_cfg['path']+"/"+data_cfg[phase]).exists():
                    not_exit_path = data_cfg['path'] + "/" + data_cfg[phase]
                    errors.append(f"{phase} 路径不存在: {not_exit_path}")

    # 随机抽样验证标注文件
    sample_files = list((Path(dataset_path)/'labels/train').glob('*.txt'))[:5]
    for label_file in sample_files:
        # 验证图像文件对应关系
        img_path = label_file.parent.parent.parent  / 'images' / 'train' / label_file.with_suffix('.jpg').name 
        if not img_path.exists(): 
            errors.append(f" 图像文件缺失: {img_path}")
        
        # 验证标注内容格式
        with open(label_file) as f:
            for line in f.readlines(): 
                parts = line.strip().split() 
                if len(parts) != 5:
                    errors.append(f" 标注格式错误: {label_file} -> {line}")
                    continue
                
                # 验证数值范围
                try:
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    if not (0 <= coords[0] <=1 and 0 <= coords[1] <=1 
                           and 0 <= coords[2] <=1 and 0 <= coords[3] <=1):
                        errors.append(f" 坐标值超出范围: {label_file} -> {coords}")
                    
                    # 验证类别ID有效性
                    if 'names' in data_cfg and cls_id >= len(data_cfg['names']):
                        errors.append(f" 无效类别ID: {cls_id} (最大应为{len(data_cfg['names'])-1})")
                except ValueError:
                    errors.append(f" 数值解析失败: {label_file} -> {line}")

    # 输出验证结果
    if not errors:
        print("✅ 数据集格式验证通过")
    else:
        print("❌ 发现以下问题：")
        for i, error in enumerate(errors[:10], 1):
            print(f"{i}. {error}")
        if len(errors) > 10:
            print(f"...（共发现{len(errors)}个问题，仅显示前10项）")

if __name__ == "__main__":
    dataset_path = "all_images_coco8_dataset"  # 修改为实际路径
    yaml_path = "coco8.yaml"  # yaml 文件路径
    validate_yolo_dataset(dataset_path, yaml_path)