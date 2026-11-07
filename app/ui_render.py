"""把 Agent 事件翻译成界面能显示的东西。
输入是 Agent._emit 发出的那六种事件：
    step / tool_call / observation / final / warning / error
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .readers import read_table as read_any_csv

# make_chart 的返回值固定是「图表已生成：<绝对路径>」开头
_CHART_RE = re.compile(r"图表已生成：(.+?\.png)")
# make_pptx / edit_pptx 用的是另一句开头。图片能直接嵌进页面，pptx 只能给下载按钮，
# 所以两者要分开认——混在一起会把 pptx 塞给 st.image，报一堆打不开。
_FILE_RE = re.compile(r"(?:演示文稿已生成|已更新)：(?P<path>.+?\.pptx)")

TOOL_LABELS = {
    "inspect_table": "查看表结构",
    "run_sql": "执行 SQL",
    "run_python": "运行 Python",
    "make_chart": "画图",
    "make_mindmap": "画思维导图",
    "make_pptx": "做演示文稿",
    "edit_pptx": "改演示文稿",
    "list_documents": "列出资料",
    "read_document": "读资料",
    "search_document": "搜资料",
    "list_files": "列目录",
    "read_file": "读文件",
    "write_text_file": "写文件",
    "find_program": "找程序",
    "open_file": "打开",
}

# 观察结果太长时截断。界面上一条 4000 字符的表格会把整页挤满，
# 而且模型看到的是全文，人只需要看个大概。
MAX_BODY_CHARS = 1500

# 「思考过程」里不铺工具返回的正文，只留一行摘要。
# read_file 读一份 PPT 就是两三千字大纲，全铺出来面板直接顶穿；而且这些字
# **模型已经看过了**，人不需要再读一遍——人要看的是"它调了什么、拿到了多大的
# 返回"，不是文件本身。
#
# 门槛不能只看长度：edit_pptx / make_pptx 的返回是**状态**（改了哪几页、备份在
# 哪、有没有压到页脚），三五百字很正常，一刀切折叠等于把执行结果藏起来。
# 所以分两档：
#   1. 读内容类工具（返回就是文件本身）按 160 字折叠，哪怕只是一份 4 页的 PPT；
#   2. 其余工具超过 600 字才折叠，留出状态信息的空间。
CONTENT_TOOLS = {"read_file", "read_document", "search_document"}
CONTENT_CHARS = 160
INLINE_CHARS = 600
# 首行通常是工具自己写的摘要（`<路径>（12 页 · 16:9 · 商务蓝 · 边距 0.625 英寸）`），
# 这条比"共 1735 字"有用得多，所以留着。上限放到 200 是因为绝对路径本身就有
# 一百多字——卡 120 会把整条摘要一起丢掉，只剩一个字数。
# 超过这个长度的首行不当作摘要（可能是正文第一段），那就只报字数。
SUMMARY_CHARS = 200
# 报错必须原样显示：工具坏了却因为"太长"被折叠，界面上就什么都看不见了。
_KEEP_PREFIXES = ("错误", "失败", "Traceback")
# 折叠正文时这几类行单独捞出来保留。它们的共同点是**说问题**：
# 「注意：第 3 页正文压到页脚区」「注意：内容过长已截断」。
# 字数用户不关心，出没出问题才关心。
_NOTICE_PREFIXES = ("注意", "警告", "提示")


def size_label(num_bytes: int) -> str:
    """把字节数说成人话。侧边栏一格里放不下 209715200 这种数字。"""
    size = float(num_bytes or 0)
    if size < 1024:
        return f"{size:.0f} B"
    for unit in ("KB", "MB", "GB"):
        size /= 1024
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
    return f"{size:.1f} GB"


def extract_chart(text: str) -> str | None:
    """从工具返回文本里取出 PNG 路径。取不到返回 None。

    之所以在界面侧解析而不是改工具返回结构：工具的返回值是**给模型看的契约**，
    为了界面好看去动它，会让工具描述和实际返回越来越难对齐。
    """
    match = _CHART_RE.search(text or "")
    if not match:
        return None
    path = Path(match.group(1).strip())
    return str(path) if path.exists() else None


def extract_file(text: str) -> str | None:
    """从工具返回文本里取出 pptx 路径。取不到返回 None。"""
    match = _FILE_RE.search(text or "")
    if not match:
        return None
    path = Path(match.group(1).strip())
    return str(path) if path.exists() else None


def prettify_args(raw: str) -> str:
    """工具参数是 JSON 字符串，直接显示会挤成一坨。"""
    try:
        return json.dumps(json.loads(raw or "{}"), ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        return raw or ""


def tool_label(name: str) -> str:
    return TOOL_LABELS.get(name, name)


def observation_body(text: str, name: str = "") -> str:
    """工具返回的正文 -> 界面上显示的那一行。

    短返回原样留着；长的只留首行摘要、带「注意 / 警告」的那几行，再加一句原文字数。
    首行通常是工具自己写的摘要（`<路径>（12 页 · 16:9 · 商务蓝）`、
    `已更新：…（12 页 → 13 页）`），留着有用；字数则让人知道它到底读进去多少。
    """
    text = text or ""
    limit = CONTENT_CHARS if name in CONTENT_TOOLS else INLINE_CHARS
    if len(text) <= limit or text.startswith(_KEEP_PREFIXES):
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # 只有一行、又不长：这已经是它的全部内容了。折成「共 190 字，正文已省略」
    # 会变成先把内容显示一遍、再说正文省略了，自相矛盾。
    if len(lines) <= 1 and len(text) <= SUMMARY_CHARS:
        return text

    kept: list[str] = []
    if lines and len(lines[0]) <= SUMMARY_CHARS:
        kept.append(lines[0])
    kept += [line for line in lines if line.startswith(_NOTICE_PREFIXES)]
    # 首行本身就是「注意」开头的，上面两段会捞到同一条
    kept = list(dict.fromkeys(kept))
    kept.append(f"…（共 {len(text):,} 字，正文已省略）")
    return "\n".join(kept)


@dataclass
class TraceItem:
    kind: str
    title: str
    body: str = ""
    code: bool = False
    chart: str | None = None
    file: str | None = None

    def to_markdown(self) -> str:
        lines = [f"**{self.title}**"]
        body = self.body
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS] + f"\n…（截断，原长 {len(self.body)} 字符）"
        if body:
            lines.append(f"```\n{body}\n```" if self.code else body)
        return "\n\n".join(lines)


@dataclass
class TraceLog:
    """一轮提问里发生的全部事件。"""

    items: list[TraceItem] = field(default_factory=list)

    def push(self, kind: str, payload: dict) -> None:
        item = build_item(kind, payload)
        if item is not None:
            self.items.append(item)

    @property
    def charts(self) -> list[str]:
        return [i.chart for i in self.items if i.chart]

    @property
    def files(self) -> list[str]:
        return [i.file for i in self.items if i.file]

    def to_markdown(self) -> str:
        if not self.items:
            return "_等待模型响应…_"
        return "\n\n---\n\n".join(i.to_markdown() for i in self.items)


def build_item(kind: str, payload: dict) -> TraceItem | None:
    """一种事件 -> 一条展示记录。返回 None 表示这类事件不用显示。"""
    if kind == "step":
        latency = payload.get("latency") or 0
        return TraceItem(
            kind="step",
            title=f"第 {payload.get('step', '?')} 步",
            body=f"输出 {payload.get('tokens', 0)} tokens，耗时 {latency:.1f}s",
        )

    if kind == "tool_call":
        name = payload.get("name", "")
        return TraceItem(
            kind="tool_call",
            title=f"要调用：{tool_label(name)}（{name}）",
            body=prettify_args(payload.get("args", "")),
            code=True,
        )

    if kind == "observation":
        # chart / file 仍然从**原始返回**里取：正文被折叠了，嵌图和下载按钮不能跟着丢。
        name = payload.get("name", "")
        text = payload.get("text", "")
        return TraceItem(
            kind="observation",
            title=f"返回结果（{tool_label(name)}）",
            body=observation_body(text, name),
            code=True,
            chart=extract_chart(text),
            file=extract_file(text),
        )

    if kind == "final":
        return TraceItem(kind="final", title="最终回答", body=payload.get("text", ""))

    if kind == "warning":
        return TraceItem(kind="warning", title="提前中止", body=payload.get("message", ""))

    if kind == "error":
        return TraceItem(kind="error", title="调用失败", body=payload.get("message", ""))

    return None


def messages_to_chat(messages: list[dict]) -> list[dict]:
    """把协议层的 messages 摊平成「你问了什么 / 它答了什么」。

    中间的 assistant(tool_calls) 和 tool 结果不进聊天气泡——它们属于思考过程，
    已经由 TraceLog 展示过了。这里只留能当对话看的部分，避免界面被协议细节淹没。
    """
    turns: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            turns.append({"role": "user", "text": message.get("content", "")})
        elif role == "assistant" and not message.get("tool_calls"):
            turns.append({"role": "assistant", "text": message.get("content", "")})
    return turns


def dataset_badge(meta: dict) -> str:
    """侧边栏那行数据集描述。拿不到文件名就只说规模，不编文件名也不写引导语。"""
    label = meta.get("label") or "已加载的表"
    rows = meta.get("rows")
    columns = meta.get("columns")
    if rows is None:
        return f"`{label}`"
    return f"`{label}` · {rows:,} 行 × {columns} 列"


def documents_badge(docs: list[dict]) -> str:
    """资料文件的汇总行。没有资料时返回空串，让调用方决定要不要显示这一段。"""
    if not docs:
        return ""
    chars = sum(int(d.get("chars") or 0) for d in docs)
    return f"{len(docs)} 份资料 · 共 {chars:,} 字"


def suggest_questions(columns: list[str]) -> list[str]:
    """按列名给几个示例问题，降低"不知道该问什么"的门槛。"""
    lowered = [c.lower() for c in columns]
    picks: list[str] = []
    if any("city" in c or "城市" in c for c in lowered):
        picks.append("各城市销售额排名前五")
    if any("date" in c or "日期" in c or "time" in c for c in lowered):
        picks.append("按月看销售额趋势，画成折线图")
    if any("category" in c or "类目" in c or "品类" in c for c in lowered):
        picks.append("各类目的销售额占比，画饼图")
    if not picks:
        picks = ["这张表有多少行？", "随便做个汇总统计"]
    return picks
