from openai import OpenAI
import os

# 初始化OpenAI客户端
client = OpenAI(
    # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
    api_key="sk-d77a9ebd93c943be8bae5158585871f0",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def main():
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复

    # 创建聊天完成请求
    stream = client.chat.completions.create(
        model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称
        messages=[
            {"role": "user", "content": "你是谁"},
        ],
        stream=True,
        # 解除以下注释会在最后一个chunk返回Token使用量
        stream_options={
            "include_usage": True
        }
    )

    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

    for chunk in stream:
        # 处理usage信息
        if not getattr(chunk, 'choices', None):
            print("\n" + "=" * 20 + "Token 使用情况" + "=" * 20 + "\n")
            print(chunk.usage)
            continue

        delta = chunk.choices[0].delta

        # 检查是否有reasoning_content属性
        if not hasattr(delta, 'reasoning_content'):
            continue

        # 处理空内容情况
        if not getattr(delta, 'reasoning_content', None) and not getattr(delta, 'content', None):
            continue

        # 处理开始回答的情况
        if not getattr(delta, 'reasoning_content', None) and not is_answering:
            print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
            is_answering = True

        # 处理思考过程
        if getattr(delta, 'reasoning_content', None):
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        # 处理回复内容
        elif getattr(delta, 'content', None):
            print(delta.content, end='', flush=True)
            answer_content += delta.content

    # 如果需要打印完整内容，解除以下的注释
    """
    print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
    print(reasoning_content)
    print("=" * 20 + "完整回复" + "=" * 20 + "\n")
    print(answer_content)
    """

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误：{e}")