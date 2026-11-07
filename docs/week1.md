# DataInsight · 第 1 周

一个基于 LLM 的表格数据分析 Agent。本周只干一件事：**让 Agent 循环转起来**。

> 不做界面、不做沙箱、不装框架。先把"模型要工具 → 本地执行 → 结果回传"这条链路走通。
> 这条链路没走通之前，任何架构都是纸上谈兵。

---

## 为什么零依赖

`step1_loop.py` 只用 Python 标准库：

| 需求 | 用了什么 | 为什么不用常见方案 |
|---|---|---|
| 调模型 | `urllib.request` 手写 HTTP | 直接看到发出去的 JSON，不隔着 SDK 猜 |
| 查数据 | `sqlite3`（内存库） | 不用 `pip install duckdb/pandas`，下载即跑 |
| 读文件 | `csv` | 同上 |

Python 3.10+ 即可，**不需要 pip install 任何东西**。

---

## 三步跑起来

```bash
cd datainsight

# 第 1 步：先看循环长什么样（不用 Key，不发请求）
python step1_loop.py

# 第 2 步：配好 Key
cp .env.example .env       # 然后编辑 .env，填 API_KEY

# 第 3 步：接真实模型
python step1_loop.py --real
```

其他用法：

```bash
python step1_loop.py --real -q "哪个类目的客单价最高？"   # 换问题
python step1_loop.py --real --dump                       # 打印完整 messages，看清协议结构
python step1_loop.py --real --steps 3                     # 收紧步数预算，看它被强制中止
```

---

## 输出怎么读

```
[第 1 步]  模型思考完毕  (1.8s)
[动作] run_sql  参数={"sql": "SELECT city, SUM(amount) ..."}
[观察] city | total
上海 | 241589.06
...

[第 2 步]  模型思考完毕  (1.2s)
[最终回答]
销售额最高的是上海，241,589.06 元……
```

- **动作** = 模型返回了 `tool_calls`，不是文本
- **观察** = 你本地函数返回的字符串，被塞进了对话
- **最终回答** = 模型这次没要工具，循环结束

---

## 这段代码里，真正重要的只有 5 行

```python
for step in range(1, max_steps + 1):                       # ① 循环 + 预算
    message = llm(messages)                                # ② 问模型
    if not message.get("tool_calls"):                      # ③ 出口条件
        return message["content"]
    messages.append(normalize(message))                    # ④ 把模型的"意图"存进对话
    ...
    messages.append({"role": "tool",                        # ⑤ 把执行结果存回对话
                     "tool_call_id": call["id"],
                     "content": observation})
```

Agent 不是框架，不是 prompt 技巧，就是**这五步转圈**。
框架（LangGraph、AutoGen）做的事情，是把这五步包装得更好维护，但内核一模一样。

理解这一点，你在面试里就不会被"你用的什么框架"这类问题带偏。

---

## 三个动手练习（按顺序做）

**练习 1 · 读懂协议（15 分钟）**
跑 `python step1_loop.py --real --dump`，把打印出来的 messages 数组抄到纸上，标出：
哪条是 `system`、哪条带 `tool_calls`、哪条是 `tool`、`tool_call_id` 和 `id` 是怎么对上的。

**练习 2 · 加第二个工具（30 分钟）**
仿照 `run_sql`，加一个 `list_columns()` 工具，返回表的所有字段名和类型。
然后问 `"这张表有哪些字段？"`，看模型会不会主动去调它。

这一步会逼你面对第一个真实问题：**工具多了，模型怎么选？描述写得好不好，直接决定选得对不对。**

**练习 3 · 观察自修复（20 分钟）**
在 `run_sql` 里临时把 `sales` 表改名成 `sale`，让模型生成的 SQL 必然报错。
观察它收到 `no such table: sales` 之后，会不会自己改。会改几次？改不对会怎样？

---

## 本周验收标准

能不看代码，口头答出这三题就算过关：

1. 循环靠什么结束？如果模型一直要工具会怎样？
2. `role=tool` 的消息为什么必须带 `tool_call_id`？去掉会怎样？
3. 说出三种会让循环"卡死"或"空转"的情况。

（参考答案都在代码注释里，先自己想，再对照。）

---

## 常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| `没有找到 API_KEY` | 没建 `.env` 或没填 | `cp .env.example .env` 后填 Key |
| `HTTP 401` | Key 错、或复制时带了空格 | 检查 Key，注意别把引号抄进去 |
| `HTTP 402` | 账户余额不足 | 充值，或换一家 |
| `HTTP 400 ... tools` | 该模型不支持 function calling | 换 `deepseek-chat` / `qwen-plus` 这类 |
| 中文打印成乱码 | Windows 终端编码 | `set PYTHONUTF8=1` 或 `chcp 65001` |
| `no such table: sales` | 没跑 `make_data.py` | `python data/make_data.py` |

---

## 目录

```
datainsight/
├── data/
│   ├── make_data.py     # 生成假数据（固定种子，可复现）
│   └── sales.csv        # 320 行电商订单
├── step1_loop.py        # 本周全部代码
├── .env.example
└── README.md
```

---

## 下周（第 2 周）预告

把 `step1_loop.py` 拆成模块：工具注册表自动生成 schema、4 个工具、会话状态。
届时引入 `duckdb`（比 SQLite 更擅长聚合）和 `pydantic`（参数校验）。
