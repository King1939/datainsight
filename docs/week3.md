# 第 3 周任务书：把 `exec` 关进子进程

> 本周只有一句话：**先假设模型会写出危险代码，再设计执行它的人。**

这是整个项目最难的一周，也是面试最能展开的一周。难点不在写代码，在**想清楚边界**。

---

## 一、先说第 2 周留下了什么坑

第 2 周的 `run_python` 是在**你自己进程里**执行模型写的代码：

```python
exec(head, namespace)      # 和你的 Agent 共用同一个解释器、同一个文件系统、同一个 Key
```

三个后果，每一个都会真发生：

| 场景 | 后果 |
|---|---|
| 模型写 `while True:` | 你的 CLI 卡死，Ctrl+C 都可能没反应 |
| 模型写 `open("x","w")` 或 `os.remove` | 真的会写、真的会删 |
| 模型写了 `import 没装的包` | 报错在你自己进程里，可能留下半初始化状态 |

黑名单（正则匹配 `import os`）挡住了 90% 的手滑，但**本质上是不可靠的**：
`getattr(x, "__cl" + "ass__")` 就能绕过。黑名单永远列不全，这是安全领域的常识。

---

## 二、本周的两道关

```
模型给的代码
    │
    ├─ 第一道：AST 白名单（不执行，只看语法树）        便宜，~0 ms
    │     · import 白名单外的模块        -> 拒
    │     · 出现 eval/exec/open/getattr  -> 拒
    │     · 访问 __class__ 这类内部属性  -> 拒
    │     · 语法错                        -> 拒（带行号）
    │
    └─ 第二道：独立子进程执行                        贵，~0.9 s
          · 超时 8 秒 -> 强制终止
          · 输出上限 4000 字符
          · POSIX 上再限制 512 MB 内存
          · 状态天然隔离：每跑一次都是新的进程
```

**顺序是刻意的**：先做便宜的静态检查，96% 的越权代码在第一道就被拒了，
根本不用付出起进程的代价。`--attack` 的耗时对比就是证据：AST 拦截 0 ms，超时要等满 2 秒。

### 为什么是白名单，不是黑名单

黑名单要枚举"所有危险的东西"，你永远枚举不完。
白名单只回答"我要什么"，剩下的全拒——**未知的东西默认不安全**。

代价是模型偶尔会被无辜拦住（比如它想 `import os` 只是想拿个路径）。
这个代价可以接受：错误信息会告诉它该怎么改，它下一轮就绕过去了。

---

## 三、新增文件

| 文件 | 行数 | 职责 |
|---|---|---|
| `app/sandbox.py` | ~250 | AST 策略 + 子进程调度 + 结果封装 |
| `app/_worker.py` | ~150 | 子进程里的执行体（stdin 收 JSON，stdout 回 JSON） |
| `app/tools/code_tools.py` | **35** | 从 100 行瘦到 35 行，只做"结果→文本"的翻译 |
| `step3_sandbox.py` | ~250 | 演示入口：`--demo` `--attack` `--limits` `--repair` |
| `tests/test_week3.py` | ~200 | 三层测试：策略 / 执行 / 集成 |

注意 `code_tools.py` 从 100 行变成 35 行——**这不是删功能，是把职责搬走**。
工具层不该关心"怎么执行"，只该关心"执行结果怎么表达"。
将来把子进程换成 Docker 容器，这个文件一行都不用改。

---

## 四、本周踩到的真坑（值得记）

### 坑 1：行号定位错位

第 2 周的写法是"把最后一行切出来 `eval`，剩下的 `exec`"：

```python
head, tail = "\n".join(lines[:-1]), lines[-1]
exec(head, ns)          # 从第 1 行开始编号
eval(tail, ns)          # 又从第 1 行开始编号 ← 报错位置永远指错
```

模型写：

```python
x = 1
y = 0
print('算到一半')
z = x / y        ← 真正出错的是第 4 行
```

报错会说"第 1 行：`x = 1`"。模型拿着这个信息去改第 1 行，越改越错。

更糟的是 `while True:` 这种跨行语句被切成 `while True:` 和 `    pass` 两段，
直接变成语法错误——模型写完全正确的代码，却收到"语法错误"。

**修法**：先 `ast.parse` 得到语法树，用 `ast.Module` / `ast.Expression` 分别编译。
AST 节点自带原始 `lineno`，traceback 就能准确指回用户写的那一行。

```python
tree = ast.parse(code)
last_expr = tree.body[-1] if isinstance(tree.body[-1], ast.Expr) else None
head = tree.body[:-1] if last_expr else tree.body
exec(compile(ast.Module(body=head, type_ignores=[]), "<string>", "exec"), ns)
value = eval(compile(ast.Expression(body=last_expr.value), "<string>", "eval"), ns)
```

现在报错是：`出错位置：第 4 行：z = x / y`。模型看到这个，一次就能改对。

### 坑 2：Windows 上没有 `resource`

`resource.setrlimit(RLIMIT_AS)` 只在 Unix 存在。Windows 上限制内存要靠
Job Object（`pywin32`）或 `psutil` 轮询，都不是标准库能干净做到的。

所以这版**内存限制只在 POSIX 生效**，Windows 靠超时兜底。
文档里必须写清楚，不能假装它有。

### 坑 3：stdout 是协议通道，不能有杂音

用户代码里的 `print` 如果直接进 stdout，就和父进程的 JSON 协议混在一起，
解析直接崩。所以 worker 里要用 `contextlib.redirect_stdout(buffer)` 把 print 截到内存。

### 坑 4：同一份代码，两个 pandas 版本跑出两种结果

演示脚本里原本写的是：

```python
((df['amount'] - df['amount'].mean()) / df['amount'].std()).abs().max().round(3)
```

在 pandas 3.1 下正常返回 `6.862`，换成 Anaconda 的 pandas 2.2.3 直接报：

```
代码执行失败（AttributeError）：'float' object has no attribute 'round'
```

原因是 `Series.max()` 的返回类型在两个版本间变了：

| 版本 | `Series.max()` 返回 | 有 `.round()` |
|---|---|---|
| pandas 2.2.3 | Python `float` | ✗ |
| pandas 3.1.0 | `numpy.float64` | ✓ |

**修法**：标量取整一律用内置 `round(x, 3)`，它是内置函数，两种返回类型都吃。
`Series.round()` 不受影响，两者别混：`.sum().round(2)` 在 `sum()` 上拿到的是
`numpy.float64`（有方法），而 `.max()` 拿到 `float`（没有），**同样是"聚合到一个标量"，
两个函数的行为却不一致**——这种不一致正是最容易漏测的地方。

**教训比 bug 本身重要**：沙箱把 pandas 预置进了执行环境，意味着**你的工具行为
依赖宿主机的第三方库版本**。所以测试必须在用户实际使用的解释器里跑一遍，
而不是"我这台能跑就行"。本项目验证过的两个环境：

| 解释器 | numpy / pandas | 28 项测试 |
|---|---|---|
| Anaconda 3.13 | 2.1.3 / 2.2.3 | 全绿 |
| WorkBuddy 3.13 | 2.5.4 / 3.1.0 | 全绿 |

---

## 五、性能代价，要心里有数

```
一次 run_python 约 0.9 秒，其中 0.8 秒是启动子进程 + import pandas/duckdb
```

对比第 2 周（进程内 exec）约 0.01 秒。**慢了 90 倍，换来的是进程隔离。**

这个代价现在可以接受：一个 Agent 通常只调 1~3 次 `run_python`。
但如果以后要跑评测集（50 条 × 每条 3 次调用），就是 2 分钟纯启动开销。

优化方向（**本周不做**，留给以后的你）：

1. **常驻 worker**：起一个进程池，用 stdin/stdout 长连接复用，省掉重复 import
2. **预热**：程序启动时先起一个空 worker，把 import 的开销藏到用户提问之前
3. 换更轻的运行时（但要小心别把 pandas 弄没了）

记住：**先能隔离，再谈性能**。反过来做的项目，最后通常两个都没做好。

---

## 六、自修复：错误信息必须"可行动"

沙箱只会拒绝是不够的。如果错误只说"执行失败"，模型会原地重试同一段代码，
直到撞上重复调用检测——这等于把工具废掉了。

一条合格的错误信息包含三件事：

```
代码执行失败（KeyError）：'revenue'          ← ① 什么错
出错位置：第 1 行：`df['revenue'].sum()`     ← ② 哪一行
提示：环境里已有 df、pd、con，直接用；       ← ③ 怎么改
```

跑 `python step3_sandbox.py --repair` 能看到完整过程：模型猜错列名 → 收到 KeyError
→ 自己改成 `amount` → 成功。**这两次调用之间的信息传递，就是"自修复"的全部含义。**

它也解释了为什么 `run_python` 的 docstring 要写得那么啰嗦：**描述得越清楚，第一次就写对的概率越高**，
而每一次重试都是一次 API 费用。

---

## 七、必须说清楚的边界

**这不是安全沙箱，是护栏（guardrail）。** 两者的区别：

| 能挡住 | 挡不住 |
|---|---|
| 模型手滑写 `open(..., "w")` | `pandas.read_csv("C:/任意路径")` —— pandas 的方法没被禁 |
| 模型写 `import os` | 蓄意构造的 Python 逃逸技巧（理论上存在） |
| 死循环占满 CPU | 子进程联网（`pd.read_csv("http://...")` 就能发请求） |
| 输出爆炸烧上下文 | 内存耗尽（Windows 上没有内存限制） |

真正的隔离要靠：容器 / 低权限用户 / 网络命名空间 / seccomp。

**简历上写"受限执行环境"是诚实的，写"安全沙箱"是吹牛。**
面试被追问时，主动说出上面这张表，比被问出来强一百倍。

---

## 八、练习（约 1.5 小时）

1. **加一条策略，并写测试。**
   让 `df.to_csv("x.csv")` 也被拒绝（提示这是写文件）。
   提示：`ast.Attribute` 的 `attr` 是 `"to_csv"`，但它和"读"共用同一个名字空间，
   你要想清楚拦的是方法名还是参数。

2. **把超时改成可配置，并观察代价。**
   给 `sandbox.run` 加 `timeout` 参数（已经有了），试着用 1 秒跑一个真实的大计算，
   看它是不是"该超时就超时"，而不是把正确结果也杀掉。

3. **让模型自己撞一次墙。**
   真实跑一句：`python step2_agent.py -q "用 Python 把结果写到 result.csv"`
   观察它被拒绝后怎么反应——它是换思路，还是原地重试三次然后被强制中止？
   这个观察结果，就是你第 5 周写"失败模式分析"的第一手材料。

---

## 九、验收标准

```bash
python step3_sandbox.py --policy    # 打印策略
python step3_sandbox.py --demo      # 5/5 通过
python step3_sandbox.py --attack    # 拦截 12/12，退出码 0
python step3_sandbox.py --limits    # 超时/截断/隔离三项正常
python step3_sandbox.py --repair    # 演示"报错 -> 改对"
python -m unittest discover -s tests -v    # 28 项全绿
```

全部通过，第 3 周就完成了。**下一周是界面（Streamlit），会比这周轻松很多。**
