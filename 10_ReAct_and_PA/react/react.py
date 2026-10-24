import json
from main import get_order, get_shipment

import os

from dotenv import load_dotenv
from openai import OpenAI

# 将 .env 中的配置加载到环境变量
load_dotenv()

client = OpenAI(
    api_key=os.environ["CHATANYWHERE_API_KEY"],
    base_url=os.environ["CHATANYWHERE_BASE_URL"],
    timeout=30.0,
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "根据订单号查询订单信息，返回商品和运单号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，例如 A123",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shipment",
            "description": "根据运单号查询物流状态和当前位置。",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_number": {
                        "type": "string",
                        "description": "运单号，必须来自用户或订单查询结果。",
                    }
                },
                "required": ["tracking_number"],
                "additionalProperties": False,
            },
        },
    },
]
# 工具名称 → 真正的 Python 函数
tool_functions = {
    "get_order": get_order,
    "get_shipment": get_shipment,
}

messages = [
    {
        "role": "system",
        "content": (
            "你是一个客服助手。"
            "查询订单和物流时必须使用工具，不能编造查询结果。"
            "缺少运单号时，先查询订单获取运单号。"
            "工具返回错误时，根据错误处理；无法继续时如实说明。"
            "获得足够信息后，用简短的中文回答用户。"
        ),
    },
    {
        "role": "user",
        "content": "帮我查一下订单 A123 的物流状态。",
    },
]

MAX_ROUNDS = 8

for round_index in range(MAX_ROUNDS):
    print(f"\n--- 第 {round_index + 1} 轮 ---")

    # 1. 把当前上下文交给模型，让它决定下一步
    response = client.chat.completions.create(
        model=os.environ["MODEL_NAME"],
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )

    message = response.choices[0].message

    # 2. 保存模型回复，包括它提出的工具调用请求
    messages.append(message.model_dump(exclude_none=True))

    # 3. 没有工具调用请求，则输出回复并结束循环
    if not message.tool_calls:
        print("最终回答：", message.content)
        break

    # 4. 执行模型请求的工具
    for call in message.tool_calls:
        tool_name = call.function.name
        print("调用工具：", tool_name)
        print("调用参数：", call.function.arguments)

        try:
            arguments = json.loads(call.function.arguments)

            if tool_name not in tool_functions:
                raise ValueError(f"未知工具：{tool_name}")

            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")

            if not all(isinstance(v, str) for v in arguments.values()):
                raise ValueError("本例的工具参数必须是字符串")

            function = tool_functions[tool_name]
            result = function(**arguments)

        except (json.JSONDecodeError, TypeError, ValueError) as error:
            result = {"error": str(error)}

        print("工具结果：", result)

        # 5. 将真实执行结果回传到上下文
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

else:
    print("达到最大执行轮数，任务尚未完成。")
