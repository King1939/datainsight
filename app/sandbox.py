"""受控执行环境：AST 白名单 + 独立子进程 + 超时 + 输出上限。

先说清楚它**不是**什么，这一点比它能干什么更重要：

    它不是安全沙箱。它能挡住"模型手滑"和"明显越权"，挡不住蓄意攻击。
    pandas.read_csv 照样能读任意文件；子进程也照样能联网。
    真正的隔离要靠容器、低权限用户、网络命名空间。

    它的定位是 guardrail（护栏），不是 security boundary（安全边界）。
    简历上写"受限执行环境"是诚实的，写"安全沙箱"是吹牛。

它解决的是四个真实问题（第 2 周全都会发生在你自己的进程里）：

    1. 模型写 `while True:` -> 你的进程直接卡死。现在超时即杀，主进程无感。
    2. 模型写 `open("x","w")` / `os.remove` -> 现在 AST 直接拒，并把
       "第几行、为什么" 原样写回给它，它下一次就能改对。
    3. 模型 import 一个没装的包 -> 报错留在子进程，不污染主进程状态。
    4. 模型 print 十万行 -> 输出截断，不会一次把上下文预算烧光。

两道关的顺序是刻意的：**先静态检查（便宜、快），再启动进程（贵、慢）**。
大多数越权代码在第一步就被拒了，根本不需要付出起进程的 300 毫秒。
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import MAX_OBS_CHARS, ROOT, TABLE_NAME

# 白名单：只允许纯计算相关的库。装不装得上另说，先在这里划清边界。
ALLOWED_MODULES = {
    "pandas",
    "numpy",
    "math",
    "statistics",
    "json",
    "re",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "decimal",
    "fractions",
    "random",
    "string",
    "textwrap",
    "typing",
}

# 完全工作区模式下额外放开的模块。刻意**没有** subprocess / socket：
# "可以改我的文件"和"可以起进程、可以联网"是两件事，后者不该被一个模式开关顺带送出去。
EXTRA_MODULES = {
    "os",
    "sys",
    "io",
    "csv",
    "shutil",
    "pathlib",
    "glob",
    "tempfile",
    "time",
    "hashlib",
    "base64",
}

# 这些方法名一出现就落盘。只读模式必须拦它们——早先只拦了 open()，
# 结果 df.to_csv("C:/Users/x/Desktop/test.txt") 能大摇大摆写出去，
# "环境只能做纯计算"那句话就成了假话。拦属性名比拦函数名更接近真实意图。
WRITE_METHODS = {
    "to_csv",
    "to_excel",
    "to_pickle",
    "to_parquet",
    "to_feather",
    "to_hdf",
    "to_stata",
    "to_html",
    "to_json",
    "to_xml",
    "savefig",
    "savetxt",
    "savez",
    "savez_compressed",
}

# 绘图库。它们**永远**不在白名单里，两种模式都一样。
#
# 不是因为危险，而是因为没用：run_python 跑在一次性子进程里，Agg 后端
# 没有窗口可弹；savefig 写出来的文件也不会被界面当成图表渲染——界面只认
# make_chart 返回的「图表已生成：<路径>」那一行。放行只会让模型以为画好了。
#
# 单独列出来是为了在报错里直接指路。只说"不允许 import matplotlib"，
# 模型会去试 seaborn、plotly、PIL，一步步烧掉预算。
CHART_MODULES = {
    "matplotlib",
    "pylab",
    "seaborn",
    "plotly",
    "bokeh",
    "altair",
    "pyecharts",
    "PIL",
}

_CHART_HINT = (
    "要出图请直接调用 make_chart 工具（bar / line / pie），"
    "在这里画出来的图回不到界面。"
)

# 文件解析库。同样不在白名单里，但单列出来是为了**指路**，不是拦得更狠。
#
# 真实教训：有人让模型"读一下桌面那份 PPT，再往里加一页"，它先 import pptx 被拒，
# 再试 zipfile、zlib、__import__，连着六次无效调用才绕到 sys.modules。
# 白名单第一次报错就写全了，缺的是一句"那你该用哪个工具"。
# 解析要按格式分派、要处理分页和编码，这本来就是工具层的活，不该让模型现写。
FILE_MODULES = {
    "zipfile",
    "zlib",
    "gzip",
    "tarfile",
    "pptx",
    "docx",
    "openpyxl",
    "xlrd",
    "pypdf",
    "PyPDF2",
    "fitz",
}

_FILE_HINT = (
    "读 PDF / Word / PPT / Excel 用 read_file，改已有的 PPT 用 edit_pptx。"
    "这些格式要按格式分派解析、还要分页，不适合在这里手写。"
)

# 这些名字一旦出现就拒绝。注意 getattr 也在里面——
# getattr(x, "__class__") 能绕过属性检查，所以不能只拦属性。
DENY_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "exit",
    "quit",
    "help",
    "memoryview",
}

DEFAULT_TIMEOUT = 8.0
MEM_LIMIT_MB = 512


class SandboxError(Exception):
    """代码在策略层面被拒绝（还没开始执行）。

    kind 用来区分"语法写错了"和"越权了"——对模型来说这是两种完全不同的信号：
    一个该改语法，一个该换思路。混成一种它会一直重试同一行。
    """

    def __init__(self, message: str, kind: str = "PermissionError") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class ExecResult:
    ok: bool
    output: str
    error_type: str = ""
    lineno: int | None = None
    source_line: str = ""
    printed: str = ""
    write_mode: bool = False
    has_data: bool = False

    def as_observation(self) -> str:
        """给模型看的文本。必须包含"哪里错、为什么错、怎么改"三件事。"""
        if self.ok:
            return self.output or (
                "执行成功，但没有输出。最后一行需要是一个表达式，或者用 print 打印。"
            )

        parts = [f"代码执行失败（{self.error_type}）：{self.output}"]
        if self.lineno:
            location = f"出错位置：第 {self.lineno} 行"
            if self.source_line:
                location += f"：`{self.source_line.strip()[:120]}`"
            parts.append(location)
        if self.printed.strip():
            parts.append("崩溃前的 print 输出：\n" + self.printed.strip()[:800])

        # 有没有 df 必须分开说。固定写"环境里已有 df"，在完全工作区模式下
        # 会导致模型对着 None 调 .head()，然后收到一份莫名其妙的 AttributeError。
        if self.has_data:
            parts.append("提示：环境里已有 df（整张表的 DataFrame）、pd、con，可以直接用。")
        else:
            parts.append(
                "提示：当前没有数据表，df 是 None，不要对它调方法；pd 和 con 仍然可用，"
                "但 con 里没有 dataset 视图。"
            )

        if self.write_mode:
            parts.append(
                "当前是「完全工作区」模式，可以用 open() 或 pd 的读写方法改本机文件。"
            )
        else:
            parts.append(
                "当前是「只读」模式，不能读写文件、不能 import os/sys/subprocess。"
                "确实需要写文件时，请让用户把模式切到「完全工作区」。"
            )
        return "\n".join(parts)[:MAX_OBS_CHARS]


# ---------------------------------------------------------------- 第一道关：AST


def check_code(code: str, allow_write: bool = False) -> None:
    """静态检查。不执行任何代码，只看语法树。

    allow_write=True 是「完全工作区」模式：放开 open 和 os/shutil/pathlib 等模块，
    但仍然拦 eval / exec / compile / __import__ / 双下划线属性——
    "允许改文件"不等于"允许构造任意代码"。

    拒绝的情况一律抛 SandboxError，消息里必须带行号——
    没有行号，模型就不知道该改哪一行。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SandboxError(f"语法错误：第 {exc.lineno} 行，{exc.msg}", "SyntaxError") from exc

    allowed = ALLOWED_MODULES | EXTRA_MODULES if allow_write else ALLOWED_MODULES
    denied = DENY_NAMES - {"open"} if allow_write else DENY_NAMES

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in CHART_MODULES:
                    raise SandboxError(
                        f"第 {node.lineno} 行：不允许 import {alias.name}。{_CHART_HINT}"
                    )
                if root in FILE_MODULES:
                    raise SandboxError(
                        f"第 {node.lineno} 行：不允许 import {alias.name}。{_FILE_HINT}"
                    )
                if root not in allowed:
                    raise SandboxError(
                        f"第 {node.lineno} 行：不允许 import {alias.name}。"
                        f"允许的模块：{', '.join(sorted(allowed))}"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in CHART_MODULES:
                raise SandboxError(
                    f"第 {node.lineno} 行：不允许 from {node.module} import ...。{_CHART_HINT}"
                )
            if root in FILE_MODULES:
                raise SandboxError(
                    f"第 {node.lineno} 行：不允许 from {node.module} import ...。{_FILE_HINT}"
                )
            if root not in allowed:
                raise SandboxError(
                    f"第 {node.lineno} 行：不允许 from {node.module} import ...。"
                    f"允许的模块：{', '.join(sorted(allowed))}"
                )
        elif isinstance(node, ast.Name) and node.id in denied:
            raise SandboxError(
                f"第 {node.lineno} 行：不允许调用 {node.id}()。"
                "这个环境只能做纯计算。"
            )
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxError(
                f"第 {node.lineno} 行：不允许访问内部属性 {node.attr}。"
                "（__class__ / __globals__ 这类属性是逃逸的常用入口）"
            )
        elif (
            not allow_write
            and isinstance(node, ast.Attribute)
            and node.attr in WRITE_METHODS
        ):
            raise SandboxError(
                f"第 {node.lineno} 行：只读模式不允许 {node.attr}() 写文件。"
                "要写本机文件，请让用户把输入框上方的模式切到「完全工作区」。"
            )


# ---------------------------------------------------------------- 第二道关：子进程


def _limit_memory() -> None:  # pragma: no cover - 只在 POSIX 生效
    """Linux/macOS 上限制地址空间。Windows 没有 resource 模块，只能靠超时兜底。"""
    try:
        import resource

        limit = MEM_LIMIT_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:  # noqa: BLE001 - 限制失败不该导致执行失败
        pass


def run(
    code: str,
    data_file: str | Path | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    allow_write: bool = False,
) -> ExecResult:
    """在独立子进程里执行代码。永远返回 ExecResult，永远不抛异常。

    data_file 是子进程要读的那份 CSV。**没有默认值**——早先这里默认读
    data/sales.csv，结果是"没上传也能算出 320 行"，一个看不见的假数据来源。
    确实没有数据集时（比如完全工作区里只想改个文本文件）传 None，
    子进程会把 df 置成 None，代码里判断一下就行。

    allow_write 由 files 的模式决定。它同时影响 AST 白名单和允许的模块，
    所以"只读"和"完全工作区"的差别在沙箱这一层就能验出来，不用跑一遍才知道。
    """
    has_data = bool(data_file)
    try:
        check_code(code, allow_write=allow_write)
    except SandboxError as exc:
        return ExecResult(
            False, str(exc), exc.kind, write_mode=allow_write, has_data=has_data
        )

    task = json.dumps(
        {
            "code": code,
            "data_file": Path(data_file).as_posix() if data_file else "",
            "table": TABLE_NAME,
        },
        ensure_ascii=False,
    )
    worker = Path(__file__).with_name("_worker.py")

    # -I：隔离模式，忽略 PYTHONPATH 和用户 site，也不把 cwd 加进 sys.path
    # -X utf8：Windows 控制台默认 GBK，不强制 UTF-8 的话中文会乱码/解码失败
    cmd = [sys.executable, "-I", "-X", "utf8", str(worker)]
    kwargs: dict = {
        "input": task,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
        "cwd": str(ROOT),
    }
    if os.name == "posix":
        kwargs["preexec_fn"] = _limit_memory

    try:
        proc = subprocess.run(cmd, **kwargs)
    except subprocess.TimeoutExpired:
        return ExecResult(
            False,
            f"执行超过 {timeout:g} 秒，子进程已被强制终止。"
            "请减少计算量，或先用 SQL 聚合再交给 Python。",
            "TimeoutError",
            write_mode=allow_write,
            has_data=has_data,
        )
    except OSError as exc:
        return ExecResult(
            False, f"无法启动执行进程：{exc}", "OSError", write_mode=allow_write,
            has_data=has_data,
        )

    # 子进程可能因为内存被杀、或自身崩溃，此时 stdout 不是 JSON
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        detail = (proc.stderr or proc.stdout or "").strip()[-600:]
        return ExecResult(
            False,
            f"执行进程异常退出（returncode={proc.returncode}）：{detail}",
            "WorkerCrash",
            write_mode=allow_write,
            has_data=has_data,
        )

    ok = bool(data.get("ok"))
    body = data.get("output") if ok else data.get("error")
    result = ExecResult(
        ok=ok,
        output=str(body or ""),
        error_type=str(data.get("error_type") or ""),
        lineno=data.get("lineno"),
        source_line=str(data.get("line") or ""),
        printed="" if ok else str(data.get("output") or ""),
        write_mode=allow_write,
        has_data=has_data,
    )
    result.output = result.output[:MAX_OBS_CHARS]
    return result
