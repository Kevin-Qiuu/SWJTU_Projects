from ultralytics import YOLO

model = YOLO("model/yolo11n.pt")  


results = model.train(
        data='coco8.yaml',  # 数据集配置文件路径
        epochs=16,  # 训练轮数
        imgsz=640,  # 输入图像大小
        # batch=4,  # 批量大小
        # workers=2,  # 数据加载线程数
        # project='medicine/runs',  # 指定输出目录为 yolov11 下的 runs 文件夹
        name='train_all_images'  
    )

results = model.train(data="coco8.yaml", epochs=400, imgsz=640)
