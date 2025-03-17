import requests
import json

import execute2json


def upload_file(file_path, api_key, base_url, user="kevinqiu"):
    upload_url = f"{base_url}/files/upload"
    headers = {
        # "Authorization": f"Bearer {api_key}",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        # print("上传文件中...")
        with open(file_path, 'rb') as file:
            files = {
                'file': (file_path, file, 'text/plain')  # 确保文件以适当的MIME类型上传
            }
            data = {
                "user": user,
                "type": "PNG"  # 设置文件类型为TXT
            }

            response = requests.post(upload_url, headers=headers, files=files, data=data)
            if response.status_code == 201:  # 201 表示创建成功
                # print("文件上传成功")
                return response.json().get("id")  # 获取上传的文件 ID
            else:
                print(f"文件上传失败，状态码: {response.status_code}")
                return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None


def run_workflow(file_id, api_key, base_url, user="kevinqiu", response_mode="blocking"):
    workflow_url = f"{base_url}/workflows/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": {
            "input_img": {
                "transfer_method": "local_file",
                "upload_file_id": file_id,
                "type": "image"
            }
        },
        "response_mode": response_mode,
        "user": user
    }

    try:
        # print("运行工作流...")
        response = requests.post(workflow_url, headers=headers, json=data)
        if response.status_code == 200:
            # print("工作流执行成功")
            return response.json()
        else:
            print(f"工作流执行失败，状态码: {response.status_code}")
            return {"status": "error", "message": f"Failed to execute workflow, status code: {response.status_code}"}
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return {"status": "error", "message": str(e)}


def execute_on_dify(file_path, api_key, base_url, user="KevinQiu"):
    # 上传文件
    file_id = upload_file(file_path, api_key, base_url, user)
    if file_id:
        # 文件上传成功，继续运行工作流
        result = run_workflow(file_id, api_key, base_url, user)
        # print(result['data']['outputs']['output'])
        result = execute2json.extract_json(result.get('data').get('outputs').get('output'))
        return result
    else:
        print("dify : 文件上传失败，无法执行工作流")

if __name__ == '__main__':
    ret = execute_on_dify("yolo_inference/1742145129_0.png",
                          api_key="app-ScBngBz8Or67tKg3h9QwyI7i",
                          base_url="http://127.0.0.1/v1")
    print(ret)
