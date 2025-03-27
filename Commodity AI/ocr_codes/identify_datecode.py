import re
from datetime import datetime

import paddleocr


def extract_dates(text):
    """支持全格式灵活解析的日期提取函数"""
    patterns = [
        # 带分隔符格式（支持单数字月日）
        r'(?<!\d)(19|20)\d{2}[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])(?!\d)',
        # 中文格式（支持单数字月日）
        r'(?<!\d)(19|20)\d{2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12]\d|3[01])日(?!\d)',
        # 紧凑格式（严格8位数字）
        r'(?<!\d)((19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)'
    ]

    date_candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            date_str = match.group()
            # 格式统一预处理
            if len(date_str) == 8 and date_str.isdigit():  # 紧凑格式处理
                processed = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:  # 其他格式处理
                processed = re.sub(r'[ 年月日/\.]', '-', date_str)
                # 仅对年和月替换为-，日直接移除
                # processed = re.sub(r' 年|月', '-', date_str).replace('日', '')
                parts = processed.split('-')[0:3]
                if len(parts) == 3:
                    # 智能补零逻辑（兼容单数字）
                    processed = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
            date_candidates.append(processed)

            # 有效性验证与格式化
    valid_dates = []
    for date_str in date_candidates:
        try:
            if datetime.strptime(date_str, "%Y-%m-%d"):
                valid_dates.append(date_str)
        except ValueError:
            continue

            # 去重且保序
    seen = set()
    return [x for x in valid_dates if not (x in seen or seen.add(x))]


def ocr_identify_date(ocr_model, input_path):
    ocr_results = ocr_model.ocr(input_path, cls=True)
    date_texts = []
    for result in ocr_results:
        if not result:
            continue
        for line in result:
            linetext = line[1][0]
            date_text = extract_dates(linetext)
            if date_text:
                date_texts.extend(date_text)
                # print(f"识别到日期：{date_text}")

    return date_texts


def inference_datecode(ocr_model, datecode_bboxes_paths):
    datecode_results = []
    for datecode_bbox_path in datecode_bboxes_paths:
        date_texts = ocr_identify_date(ocr_model, datecode_bbox_path)
        for date_text in date_texts:
            datecode_results.append(date_text)
    return datecode_results


if __name__ == "__main__":
    ocr_model = paddleocr.PaddleOCR(use_angle_cls=True, lang='ch')  # 反向检测，语言
    input_path = ["some_demo_image/3_qi_ma/Snipaste_2025-03-16_00-06-08.png"]
    ret = inference_datecode(ocr_model, input_path)
    print(ret)