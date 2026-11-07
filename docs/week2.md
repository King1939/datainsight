# 第 2 周 · 从"一个脚本"到"一个系统"

## 本周只解决一个痛点

第 1 周你手写了这段：

```python
TOOLS = [{"type": "function", "function": {"name": "run_sql", ...}}]   # 手写
def run_sql(sql): ...                                                   # 手写
TOOL_IMPL = {"run_sql": run_sql}                                        # 手写
```

三处要手动保持一致。想加第二个工具，你得同时改三个地方，漏一个就出诡异 bug。

**本周把它变成：写一个函数，就等于加一个工具。**（加限定：函数里**写好描述**才算真的加上了）

```python
@registry.tool
def run_sql(sql: str, max_rows: int = 30) -> str:
    """执行一条只读 SQL……"""
```

签名、类型注解、docstring 全部被翻译成 JSON Schema，这就是 `app/schema.py` 干的事。

---

## 先说清楚：什么自动了，什么没有

这里很容易被夸大成"文档自动生成"，其实不是。它只自动化了一半：

| 部分 | 谁负责 | 为什么 |
|---|---|---|
| 工具名、参数名 | 签名 → 自动 | 机器可读 |
| 参数类型（`str`→`string`） | 类型注解 → 自动 | 机器可读 |
| `required` 列表 | 有没有默认值 → 自动 | 机器可读 |
| `enum` 候选值 | `Literal[...]` → 自动 | 机器可读 |
| **工具什么时候该用** | **人写 docstring** | 机器推不出来 |
| **每个参数是什么意思** | **人写 `Args:` 段** | 机器推不出来 |
| **什么时候不该用它** | **人写，最容易被漏** | 机器推不出来 |

所以真正的收益不是"少写文档"，而是**结构只有一份真相**。

第 1 周你手写的时候，同一件事有 5 个可能互相对不上的地方：

```
① TOOLS 里的 name ── ② 函数名 ── ③ TOOL_IMPL 的 key
④ TOOLS 里的 required ── ⑤ 函数的实际参数
```

改一次参数名，得同步改三处；漏了，报错只会在模型调用时才出现，而且报错信息完全指不到根因。

现在只剩 2 处，且分工明确：结构归签名（改一处就全对），语义归 docstring（本来就得你自己想清楚）。

这个说法是可以被测试验证的，见 `tests/test_week2.py::test_schema_always_matches_signature`——
它断言"注册表里存的 schema 必须能由签名原样重建"，一旦有人绕过装饰器手改 schema，测试立刻红。

---

## 跑起来

```bash
pip install -r requirements.txt        # duckdb / pandas / matplotlib

python step2_agent.py --tools          # ← 先看这个，不用 Key
python step2_agent.py --selftest       # ← 再跑这个，不用 Key
python step2_agent.py --chat           # 需要 .env 里的 Key
```

`--tools` 会把按签名生成的 schema 打印出来，注意这两处细节：

```json
"kind": {"type": "string", "enum": ["bar", "line", "pie"], "default": "bar"}
"required": ["sql"]
```

`Literal` 变成了 `enum`（模型只能在三个值里选，出错率大幅下降），
有默认值的参数没有进 `required`（它知道那是不用填的）。

---

## 代码地图

| 文件 | 一句话职责 | 这周要理解的 |
|---|---|---|
| `app/schema.py` | 类型注解 → JSON Schema | 为什么"手写 schema"迟早会不同步 |
| `app/tools/registry.py` | 注册 + 统一调用入口 | **所有错误都返回字符串** |
| `app/tools/data_tools.py` | 看表结构、跑 SQL | 为什么先 `inspect_table` 能减少一半错误 |
| `app/tools/code_tools.py` | 执行 Python | 这版不安全，第 3 周才加固 |
| `app/tools/chart_tools.py` | 出图 | 中文字体不做处理就是一排方块 |
| `app/agent.py` | 循环 + 事件回调 | 同一个循环，为什么这次能接 UI |
| `app/session.py` | 多轮对话状态 | 为什么状态要独立于 Agent |
| `tests/test_week2.py` | 11 项回归测试 | 用假模型测循环，不花钱 |

---

## 三个关键设计

### 1. 工具不返回数据，返回"文本"

`run_sql` 明明查出了 DataFrame，为什么要 `str` 化再返回？

因为**模型只读得懂 token**。返回 `str(df)` 会带一堆 dtype 和索引噪音，返回几万行更是灾难。
所以 `data_tools` 里做了三件事：只取前 N 行、拼成 `a | b | c` 的表格、末尾注明总行数。

> 面试点：工具返回值的**信息密度**直接决定 Agent 的成功率，这件事比换模型影响更大。

### 2. 统一错误出口

看 `registry.call()`：未知工具、参数名写错、缺必填项、函数内部炸了——**全部返回字符串**。

好处是模型能看见错误并自我修正：

```
[动作] run_sql  {"sql": "SELECT revenue FROM sales"}
[观察] SQL 执行失败：Binder Error: Referenced column "revenue" not found...
       Candidate bindings: "order_date", "order_id"
[动作] run_sql  {"sql": "SELECT amount FROM sales ..."}     ← 它自己改对了
```

如果这里抛异常，程序直接退出，模型永远没有第二次机会。

> 面试点：这是"Agent 可用"和"Agent 能用"的分界线。

### 3. 三个刹车

| 刹车 | 位置 | 防的是 |
|---|---|---|
| `max_steps` | `agent.py` | 反复试错停不下来 |
| `token_budget` | `agent.py` | 长对话把成本烧穿 |
| 重复调用检测 | `agent.py` | 同一条 SQL 连发三次 |

Agent 出错的时候，**很少是报错，多数是安静地烧钱**。这三个是保命的。

---

## 三个练习

**练习 1 · 加一个第五工具（30 分钟）**
在 `data_tools.py` 里加 `top_n(column: str, n: int = 5)`，按某列聚合后取前 N。
不加任何 schema——跑 `--tools` 看它有没有自动出现。这是本周的核心体感。

**练习 2 · 制造一次参数错误（20 分钟）**
把 `run_sql` 的参数从 `sql` 改名成 `statement`（只改函数签名，别改 docstring 里的 `Args`）。
观察 `--tools` 输出变成什么样，再想：如果模型还在传 `sql` 会报什么错？

**练习 3 · 读一次测试（20 分钟）**
跑 `python -m unittest discover -s tests -v`，然后打开 `test_repeat_abort`。
把 `repeat >= 3` 改成 `2`，测试还过吗？为什么这个测试能发现你改坏了？

---

## 验收标准

1. 能说清 `@registry.tool` 从函数上都"读"走了什么，变成了 schema 的哪部分。
2. 能解释为什么工具报错要返回字符串，而不是 `raise`。
3. 能指出 `run_python` 这一版的安全漏洞在哪（提示：黑名单永远挡不全）。

---

## 常见坑

| 现象 | 原因 |
|---|---|
| `ModuleNotFoundError: No module named 'duckdb'` | **装到了另一个 Python**。见下方「环境第一坑」 |
| `工具 xxx 缺少 docstring 描述` | 注册表强制要求描述，模型没有描述就不知道何时调用 |
| 模型死活不调 `make_chart` | 工具描述里没写"什么时候该用它"，只写了"它能干什么" |
| 图里中文全是方块 | 没设 `font.sans-serif`，见 `chart_tools.py` |
| `Binder Error` | DuckDB 和 MySQL 语法有差异，别让模型照 MySQL 写 |
| 观察结果被截断 | `MAX_OBS_CHARS`，正常，说明你该在 SQL 里聚合而不是拉全表 |

### 环境第一坑：电脑上有好几个 Python

`pip install` 和 `python xxx.py` 必须指向**同一个解释器**，否则就是"明明装了却说没有"。

```bash
python -c "import sys; print(sys.executable)"   # 先确认你用的是哪一个
python -m pip install -r requirements.txt       # 用同一个 python 装
```

用 `-m pip` 而不是裸 `pip`，就能保证装到当前这个解释器里。
如果系统里有多个 Python（Anaconda、官网版、Microsoft Store 版），建议给项目建独立环境：

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows cmd
python -m pip install -r requirements.txt
```

之后每次开新终端先 `activate`，就永远不会有"装哪去了"的问题。

---

## 下周（第 3 周）预告

`run_python` 现在是进程内 `exec`，用户（或模型幻觉）写 `open("x","w")` 就能删你的文件。
第 3 周把它换成**子进程 + 超时 + 内存上限 + AST 白名单**。

这才是这个项目里真正的工程难点，也是面试时最能拉开差距的一段。
