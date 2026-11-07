# 第 4 周：把 Agent 变成一个能演示的网页

> 一句话：前三天做完的东西都藏在一个黑框里，这周把它打开给人看。

## 一、本周只加了一层壳，但动了三处契约

界面本身不复杂：`st.chat_input` 收问题，`on_event` 收过程，`st.image` 显示图。
真正要动脑的是三个"看着能凑合、但迟早会咬人"的地方。

### 决策 1：上传什么文件，表名都是同一个常量

上传功能最直觉的做法是"文件叫什么表就叫什么"，然后你会发现：

- `inspect_table()` 的默认参数要改
- 两个工具 docstring 里的"表 dataset"要改
- `make_chart`、`run_python` 里的示例要改
- system prompt 里"表 dataset"要改

四处都要跟着变，而且**模型看到的表名每换一个文件就变一次**，它会时不时写错。

所以反过来做：**不管上传什么，都建一个固定名字的视图**（`config.TABLE_NAME`，现在是 `dataset`）。

```python
con.register(TABLE_NAME, frame)          # 上传的 DataFrame
# 或
CREATE VIEW dataset AS SELECT * FROM read_csv_auto('...')
```

工具层一个字都没改。这是一个通用的判断法则：**当外部输入要进入一个已经很稳定的系统时，优先在边界上做适配，而不是让内部跟着外部变。**

> 补充（同周修正）：这个常量最初叫 `sales`，因为第 1 周的示例数据是电商订单表。名字也叫 `sales`
> 会带来两个隐性成本：prompt 里"面对一张电商订单表 sales"在别人传医院数据时变成错误暗示；
> 兜底 prompt（没有数据时用）描述一张根本不存在的表，模型会照着去猜列名。
> 现在常量叫 `dataset`，并且 **prompt 里的表名从 `config.TABLE_NAME` 取，不再手写**。

### 决策 2：system prompt 必须跟着数据集走

第 2 周那句话现在成了 bug：

```
你是一个数据分析助手，面对一张电商订单表 sales（字段见工具说明）。
```

别人传一张医院门诊表进来，这句话就是**错误信息**——模型会按订单表的思路去猜"city""amount"。所以 `agent.py` 加了 `make_system_prompt(columns, rows, label)`：

```
你是一个数据分析助手，面对一张表 dataset，它是从 `门诊.csv` 载入的。
表 dataset，共 1,204 行，字段（列名）：
科室, 就诊日期, 费用, 医保类型
```

没有数据时用的兜底 prompt 必须写成"当前没有数据"，不能描述任何具体表——否则模型会照着
一张不存在的表去猜列名，然后连续几步都在报错。

这是本周唯一改到 Agent 的地方（3 行）。上一周我说过"Agent 一行都不用改"，那个说法在**功能**上成立，在**语义**上不成立：界面一旦支持任意数据，"我是谁、我在看什么"这个前提就必须变成参数。

### 决策 3：图表路径在界面侧解析，不改工具返回

`make_chart` 返回的是给人看、也给模型看的一句话：

```
图表已生成：C:\...\output\chart_1793673912336.png
类型 bar，3 个数据点，横轴 产品，纵轴 销量
```

界面要把图显示出来，就得知道路径。有两个选择：

| 做法 | 代价 |
|---|---|
| 改工具返回结构化对象 | 侵入 `registry.call` 的返回契约，四个工具都要跟着改 |
| 界面侧正则提取 `图表已生成：(.+?\.png)` | 5 行，工具层零改动 |

选后者。理由不是"省事"，而是：**工具的返回值是给模型看的契约**。一旦为了界面去改它，工具描述、实际返回、模型的理解就会开始漂移，而你不会立刻发现。

代价是界面依赖了 `make_chart` 的文案格式。所以我在 `ui_render.py` 里把这个正则单独放了一行，并写了注释——**知道自己在依赖什么，比假装没有依赖更安全**。

## 二、Streamlit 的两个坑

### 坑 1：每次交互，整个脚本从头跑一遍

这不是 bug，是 Streamlit 的设计。它带来一个硬性约束：

```python
agent = Agent()        # ✗ 放在模块级：一刷新页面就没了，提问两次互相不认识

if "agent" not in st.session_state:      # ✓
    st.session_state.agent = Agent()
```

同理 `Session`、对话历史、上传的 DataFrame，全部必须进 `st.session_state`。

理解这一点，你就能解释一个现象：**为什么点个按钮，页面上所有东西都重新算了一遍。**

### 坑 2：多个标签页共用一个进程

`st.session_state` 是按浏览器会话隔离的，但 Python 进程只有一个。而 `data_tools` 里的 duckdb 连接原本是**模块级单例**：

```python
_conn = None             # 第 2 周：整个进程一个连接
```

两个人同时打开页面、各传一份 CSV，第二个人一上传就把第一个人的视图覆盖了。第 2 周那样写没问题（只有一份数据），接上界面的那一刻就变成 bug。

现在连接按 key 分开存，key 是会话 id：

```python
_conns: dict[str, duckdb.DuckDBPyConnection] = {}
_ACTIVE: contextvars.ContextVar[str]   # 当前用哪个 key，按执行上下文隔离
```

每次页面重跑，第一件事就是 `dt.use_dataset(st.session_state.sid)`，把自己那份接回来。

**这里踩过一个更深的坑（2026-11-07 修）**：一开始"当前用哪份数据"写成了模块级变量
`_active = "default"`。连接虽然按 key 分开了，但"现在是哪一份"仍然是全进程共享的。
A 标签页问一句话，Agent 要跑十几秒；这期间 B 一刷新，`use_dataset(B)` 就把这个变量改了，
A 后面几步的 `run_sql` 全部查到 B 的表——**不报错、不提示，只是数字换了**。
`run_python` 更糟：它按当前 key 找 CSV，会拿别人的文件算完一份"像模像样"的结果。
用两个线程复现过一次：A 绑北京的 999，200ms 后查出来的是 B 的表。

修法是 `contextvars.ContextVar`：每个标签页在自己的线程/上下文里 set，互不可见。
工具里一律现取（`current_key()`），不缓存到局部变量跨步使用。注入点有两处——
界面每次 rerun 的 `_init_state()`，和 `Agent.ask()` 的入口。

**只写，不还原。** 一开始还写了个 `reset_key()` 用 `try/finally` 还原，后来删了：
一次 rerun 就是一根新线程，下次刷新必然重设，"留在上下文里的值"活不过一次刷新。
实测把还原改成空函数，全部测试里只有两条专门测还原的会红，功能一条没坏。
留着它反而要多解释一个 Token 概念。

为什么不用"给工具加个 sid 参数"：工具签名会被 `build_tool_spec` 直接变成 JSON
Schema 交给模型，多一个参数模型就会自己编一个值塞进来。上下文注入是唯一不改进
工具签名、也不出现在 schema 里的办法。

**这仍然不是多用户方案**：进程只有一个、没有鉴权、所有连接都在同一个字典里。
但"同一个人的两个标签页互相踩"这件事已经修掉了。正经上线要按请求建连接，
这一点面试被问"能多用户吗"时要主动说。

## 三、一个必须处理的隐藏不一致

上传的 DataFrame 注册进了 duckdb，SQL 能看到新数据。但 `run_python` 跑在**子进程**里，它靠 `pd.read_csv(data_file)` 拿数据——如果 `data_file` 还是 `data/sales.csv`，就会出现：

```
run_sql     → 3 行
run_python  → 320 行
```

这种"两个工具说的不是一张表"的问题最难查，因为两个工具单独看都正常。

修法是让上传的数据**同时落一份 CSV 到 `data/uploads/<会话id>.csv`**，然后：

```python
sandbox.run(code, data_file=active_data_file())   # 子进程读同一份文件
```

代价是磁盘上会多出几个文件，`.gitignore` 已经把 `data/uploads/` 排除了。

为此专门写了一条测试（`test_sql_and_python_see_same_rows`），断言两边行数一致。**这种跨组件的隐式约定，只靠人记一定会忘。**

## 四、改动清单

**新增 3 个**

| 文件 | 行数 | 作用 |
|---|---|---|
| `app/ui_streamlit.py` | ~250 | 页面布局、会话状态、上传处理 |
| `app/ui_render.py` | ~150 | 事件 → 展示内容，纯函数，可单测 |
| `step4_app.py` | ~20 | streamlit 入口壳（保证 `from app...` 能导入） |

**修改 4 个**

| 文件 | 改了什么 | 为什么 |
|---|---|---|
| `app/tools/data_tools.py` | 单例 → 按 key 存的字典；新增 `bind_dataset` / `use_dataset` / `active_data_file` / `column_names` | 支持上传且多会话不串 |
| `app/agent.py` | 加 `system_prompt` 参数 + `make_system_prompt()` | prompt 里的表描述必须跟着数据走 |
| `app/sandbox.py` | `run()` 加 `data_file` 参数 | 子进程要读上传的那份数据 |
| `app/config.py` | 加 `UPLOAD_DIR` | 上传副本的落盘位置 |

**一行没动**：`llm.py`、`session.py`、`registry.py`、`schema.py`、`chart_tools.py`、三个 `step*.py`。

**新增测试 39 项**（`tests/test_week4.py` 34 项 + `tests/test_week4_ui.py` 5 项），总计 67 项。
界面那 5 项用 Streamlit 官方的 `AppTest` 真的执行一遍脚本，断言"不抛异常、标题在、示例问题按列名生成"——
因为 Streamlit 的报错显示在页面上，HTTP 仍然是 200，健康检查查不出来。

## 五、怎么跑

```bash
cd C:\Users\13107\WorkBuddy\2026-11-02-15-46-26\datainsight
python -m streamlit run step4_app.py
```

浏览器会自动打开 `http://localhost:8501`。第一次建议按这个顺序试：

1. **先上传一份 CSV**（网页端没有默认数据，不传就只有一个上传框）
2. **问一句**：`各城市销售额排名前五` → 看左侧"思考过程"逐步出现
3. **展开思考过程**：能看到 `第 1 步 → 要调用：查看表结构 → 返回结果 → 第 2 步 → 执行 SQL`
4. **要求画图**：`画个柱状图看各渠道销售额` → 图直接出现在回答上方
5. **追问**：`那北京呢` → 它能接上，因为 `Session.messages` 里还有上文
6. **换数据**：侧边栏传另一张 CSV → 对话被清空（这是刻意的），示例问题跟着列名变

回归测试（不需要 Key）：

```bash
python -m unittest discover -s tests
```

> 第 1~3 周的 `step1_loop.py` / `step2_agent.py` / `step3_sandbox.py` 已移除，
> 项目现在只有 `step4_app.py` 一个入口；对应任务书仍在 `docs/` 里，作为当时的记录。

## 六、三个练习

1. **把"示例问题"改成点击后直接发送**（现在是填进输入框再点发送）。体会一下 Streamlit 里"按钮触发一次完整的 Agent 调用"为什么要绕 `st.session_state`。
2. **给侧边栏加一个"导出本次对话"按钮**，把 `session.to_dict()` 存成 JSON 下载。`Session.save()` 已经有了，你只需要接上 `st.download_button`。
3. **故意传一个只有一列数字的 CSV**，看会发生什么。现在的守卫是"少于两列就拒绝"，想一想：如果两列都是数值、没有维度列，`make_chart` 会画出什么来？这个坑第 5 周做评测时会遇到。

## 七、验收标准

- [ ] `python -m streamlit run step4_app.py` 能打开页面，无报错
- [ ] **不传文件时是空状态**：侧边栏写"还没有数据"，没有提问框，也不会自动冒出 320 行数据
- [ ] 上传 CSV 后问"各城市销售额"，能看到 3 步以上的思考过程
- [ ] 要求画图，PNG 显示在页面上，中文不是方块
- [ ] 上传一张自己的 CSV（至少两列），示例问题跟着变
- [ ] 刷新页面（F5），对话还在
- [ ] `python -m unittest discover -s tests` 73 项全绿

## 七点五、追加改动：网页端不再默认读取 `data/sales.csv`

最初的做法是"没上传就用示例数据"，图省事。它有两个问题：

1. **数据来源不可见**。打开页面就能问"各城市销售额"，看的人不知道这些数字来自哪里。
2. **底层藏着隐式回退**。`get_conn()` 里有一句"没有连接就读 DATA_FILE"，任何忘了绑定的路径都会
   悄悄拿到 320 行示例数据——比如 `run_python` 没拿到 `data_file` 时会退回示例文件，于是出现
   "SQL 查的是你上传的表、Python 算的是示例表"。

改法是**把回退删掉，让缺失变成显式错误**：

| 位置 | 改法 |
|---|---|
| `data_tools.get_conn()` | 没有绑定时抛 `NoDataset`，不再读 `DATA_FILE` |
| `data_tools.active_data_file()` | 返回 `None`（原来是回退到示例文件） |
| `data_tools.bind_sample()` | 新增；只有 CLI 和测试显式调用 |
| `code_tools.run_python` | 拿不到路径就直接返回"还没有选择数据集" |
| `ui_streamlit._apply_dataset()` | 没有上传就返回 `False`，并指向本会话专属的空 key |
| `ui_streamlit._render_sidebar()` | 空状态下隐藏"清空对话"和示例问题 |

这里有个顺手修掉的真实缺陷：侧边栏显示的表名读的是"当前 key 的元信息"，而这个 key
以前是进程级全局，A 标签页没传数据时它可能还停在 B 的数据集上，于是 A 的侧边栏会
显示别人的文件名。现在 key 跟着会话走（`sid`），没绑定就查不到 label，
`dataset_badge()` 显示"未选择数据"，不会编一个文件名出来。

测试也跟着改了两处：`test_back_to_default` → `test_back_to_sample`（必须显式加载），
新增 `test_no_silent_fallback`（断言"没数据时 run_sql / run_python 都报错，且不含 320"）。
界面测试拆成 `TestEmptyState` 与 `TestAfterUpload` 两组，因为**默认状态本身就是一条要验证的需求**。

## 八、还缺什么（第 5 周要补）

- **没有评测**：现在只能说"能跑"，说不出"准确率多少"。50 条用例 + 标注答案是这个项目最值钱的部分。
- **没有失败归因**：模型答错时，你看不出是选错工具、写错 SQL、还是理解错问题。
- **上下文会膨胀**：`Session.messages` 只增不减，聊十轮以上 prompt 会变得很贵。
- **单进程**：标签页之间的数据已经用 `ContextVar` 隔开了，但连接、资料全在一个进程的
  字典里，没有鉴权也没有超时回收。这是"单机演示"和"能给别人用"之间的差距。
