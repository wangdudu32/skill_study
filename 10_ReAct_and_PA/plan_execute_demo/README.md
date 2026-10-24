# 手写 Plan-and-Execute

沿用订单查询示例，适用于 WSL Ubuntu 22.04、Python 3.10+、ChatAnywhere、gpt-4o-mini、python-dotenv。不依赖 LangChain / LangGraph。

## 运行

已激活原来的 `.react` 虚拟环境时可以直接安装依赖，无需创建新环境。

```bash
cd plan_execute_demo
python -m pip install -r requirements.txt
cp .env.example .env
nano .env
python plan_execute.py
```

在 `.env` 中填写自己的 ChatAnywhere 密钥。默认 `load_dotenv` 不覆盖已有环境变量；如果以前 export 过同名配置，先 unset 对应变量或确保两者一致。不要将 `.env` 提交到 Git。

如果没有现成虚拟环境，可先运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

其他输入：

```bash
python plan_execute.py '帮我查一下订单 A124 的物流状态。'
python plan_execute.py '帮我查一下订单 UNKNOWN 的物流状态。'
```

示例数据完全是本地模拟：A123 对应 SF001，运输中，位于上海转运中心；A124 未发货；UNKNOWN 不存在。实际服务需要替换两个工具的内部实现。

## 结构及阅读顺序

1. `get_order` / `get_shipment`：实际工具函数。
2. `TOOLS`：模型看到的工具描述和参数 Schema。
3. `execute_tool`：工具白名单、参数检查、函数执行与错误回传。
4. `planner`：通过 JSON mode 生成有序计划；本地检查结构。
5. `executor`：内层 ReAct 式工具循环，执行当前步骤。工具调用及其原始结果会保留。
6. `replanner`：依据目标、原计划和真实执行记录，返回剩余计划或最终回答。
7. `run`：外层调度循环，将这些组件串起来。

三个角色复用同一个模型实例，使用不同提示词。外层保存显式计划；内层根据工具结果决定下一次行动。它与此前纯 ReAct 的区别在于增加了独立规划阶段和重新规划阶段。

## 输出状态

外层状态：`completed`（模型判断信息足够并给出答复）、`blocked`（无法继续或缺少信息）、`limit_reached`（达到上限）。

执行器的 `finished` 只表示当前执行器完成一轮工作，不能直接当作用户目标完成。外层由 Replanner 再评估。

正常情况下会看到：

```text
[Planner] 初始计划：查询订单 -> 查询物流
[Executor] 第1步：查询订单
  [Tool] get_order -> SF001
[Replanner] continue，剩余步骤为查询物流
[Executor] 第2步：查询物流
  [Tool] get_shipment -> 运输中、上海转运中心
[Replanner] completed
[最终回答] 订单 A123 运输中，目前位于上海转运中心。
```

这是流程示意，实际计划措辞、步骤数量及回答可能不同。未发货或订单不存在时，Replanner 应删除不再可执行的物流查询步骤或返回阻塞说明，而不是猜测运单号。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试使用模拟模型响应，无需密钥、无需网络，也不产生 API 费用。覆盖正常执行、重新规划、业务失败、工具参数异常、JSON 修正和循环上限。它验证的是程序控制流，不能证明真实模型一定按指令行动。

## 实现边界

- JSON mode 只约束 JSON 输出形式，本地 validator 进一步检查字段和类型；两者都不保证业务语义正确。
- Replanner 依据真实工具记录做判断，但仍是模型判断，生产系统需要为关键完成条件增加程序校验。
- 状态仅在内存中；没有跨进程恢复、数据库检查点或并行任务调度。
- 工具当前都是只读操作，不包含订单修改或发送消息。
- 每步最多调用模型 5 轮，外层最多执行 6 步，结构不合法的 JSON 最多修正一次，网络请求超时 30 秒并最多重试一次。
- 实际调用需要 ChatAnywhere 账户能使用该模型，且接口支持 tools 和 JSON mode；不支持时会显示异常，不会伪装成成功。
- 此版本未使用你的密钥进行真实 API 联调。

参考：
- LangChain 的 Plan-and-Execute 架构介绍：https://www.langchain.com/blog/planning-agents
- ChatAnywhere 接口说明：https://github.com/chatanywhere/GPT_API_free
