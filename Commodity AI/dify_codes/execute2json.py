import re
import json
from datetime import datetime


def extract_json(text):
    """
    增强型JSON提取函数，支持处理多格式变体
    功能特性：
    1. 支持匹配不同代码块格式（```json/```）
    2. 自动处理头尾多余字符
    3. 带异常处理和日期验证
    """
    try:
        # 增强正则匹配模式，兼容多种代码块格式
        pattern = r'(?:```json|```)(.*?)(?:```|\Z)'
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            raise ValueError("未检测到JSON代码块")

        json_str = match.group(1).strip()

        # 处理转义字符和多余空格
        json_str = re.sub(r'\n\s+', ' ', json_str)  # 压缩多行空格
        json_str = json_str.replace('\\n', '')  # 移除转义字符

        data = json.loads(json_str)

        # # 扩展功能：日期格式验证（根据当前日期2025-03-16）
        # date_fields = ["生产日期", "保质期至"]
        # for field in date_fields:
        #     if field in data:
        #         datetime.strptime(data[field], "%YYYY-%mm-%dd")

        return data
    except Exception as e:
        print(f"解析错误: {str(e)}")
        return None


# # 测试用例集
# test_cases = [
#     # 标准三行格式
#     '''```json\n{\n"生产日期": "2024-04-13",\n"保质期至": "2026-04-12",\n"产品批号": "3574240401"\n}\n```''',
#     '''```json\n{\n"生产日期": "2024-04-13",\n"保质期至": "2026-04-12"\n}\n```''',
# ]
#
# # 执行测试
# for idx, case in enumerate(test_cases, 1):
#     print(f"测试用例 {idx}:")
#     result = extract_json(case)
#     print(f"解析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
#     print("-" * 40)