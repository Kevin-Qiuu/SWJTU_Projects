# -*- coding: utf-8 -*-
import json
import os
import argparse
from tqdm import tqdm
import chardet


def convert_label_json(json_dir, save_dir, classes):
    json_paths = os.listdir(json_dir)
    classes = classes.split(',')

    for json_path in tqdm(json_paths):
        # for json_path in json_paths:
        path = os.path.join(json_dir, json_path)
        file_name, file_extension = os.path.splitext(path)
        if (file_extension != '.json'):
            continue
        # print(path)

        with open(path, 'rb') as f:
            raw_data = f.read(1024)
            encoding = chardet.detect(raw_data)['encoding']

        # 使用检测到的编码重新打开文件
        with open(path, 'r', encoding=encoding) as load_f:
            json_dict = json.load(load_f)
            # print(json_dict)
        # with open(path, 'r', encoding='utf-8') as load_f:
        #     json_dict = json.load(load_f,)
        #     print(json_dict)
        image_h, image_w = json_dict['imageHeight'], json_dict['imageWidth']

        # save txt path
        txt_path = os.path.join(save_dir, json_path.replace('json', 'txt'))
        txt_file = open(txt_path, 'w')

        for shape_dict in json_dict['shapes']:
            label = shape_dict['label']
            label_index = classes.index(label)
            points = shape_dict['points']

            x1 = points[0][0]
            y1 = points[0][1]
            x2 = points[1][0]
            y2 = points[1][1]

            w = abs(x1 - x2)
            h = abs(y1 - y2)
            xc = min(x1, x2) + w / 2
            yc = min(y1, y2) + h / 2

            # 归一化
            xc = xc / image_w
            yc = yc / image_h
            w = w / image_w
            h = h / image_h

            points_nor_list = [xc, yc, w, h]

            # for point in points:
            #     points_nor_list.append(point[0] / w)
            #     points_nor_list.append(point[1] / h)

            points_nor_list = list(map(lambda x: str(x), points_nor_list))
            points_nor_str = ' '.join(points_nor_list)

            label_str = str(label_index) + ' ' + points_nor_str + '\n'
            txt_file.writelines(label_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='json convert to txt params')
    parser.add_argument('--json-dir', type=str, default=r'datasets/All_Images/Medicine/label',
                        help='json path dir')  # json目录
    parser.add_argument('--save-dir', type=str, default=r'datasets/All_Images/Medicine/txt',
                        help='txt save dir')  # 存放结果的目录
    parser.add_argument('--classes', type=str, default='0,1,2,3', help='classes')  # 设置类别，多类别格式‘A,B,C,D’
    args = parser.parse_args()
    json_dir = args.json_dir
    save_dir = args.save_dir
    classes = args.classes
    convert_label_json(json_dir, save_dir, classes)

