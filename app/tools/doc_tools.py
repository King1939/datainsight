"""资料类工具：读 PDF / Word / PPT / 文本 / 日志。

为什么这些事必须由工具做、不能让模型写 Python 自己读：
  1. run_python 跑在沙箱子进程里，AST 白名单禁掉了 open，模型根本读不到磁盘；
  2. 就算放开了，每份 PDF 都要模型自己调 pypdf、算分页、处理加密，
     一次问答可能烧掉十几步——这属于"环境该提供的确定性能力"，不该让模型现学；
  3. 抽取的边界（扫描件没文字、PPT 图表没数值）需要**固定话术**告诉模型，
     它才不会对着一份空文本编内容。

三个工具的分工，故意拆细：
  list_documents  -> 不知道有哪些文件时先看一眼（省得瞎猜文件名）
  read_document   -> 顺序读，一次一段（长文档必须分页，否则上下文直接爆）
  search_document -> 知道要找什么时直接定位（比顺序读完 7 段便宜得多）

为什么不做模糊匹配文件名：模型猜错一个字，拿到一份无关文档，还会认真总结它。
把"可选名单"回给它，它下一轮就改对了。
"""

from __future__ import annotations

from ..config import MAX_OBS_CHARS
from ..readers import Doc
from . import registry
from .data_tools import documents, find_document

# 一次搜索最多报几处命中。多了就是变相把全文倒进上下文。
MAX_HITS = 8
# 每处命中带几行上下文
CONTEXT_LINES = 1


def _missing(name: str) -> str:
    """找不到文件时的统一话术：把真实名单给它。"""
    docs = documents()
    if not docs:
        return "错误：当前还没有上传任何资料文件。"
    names = "、".join(f"`{d.name}`" for d in docs)
    return (
        f"错误：没有名为 `{name}` 的资料。已上传的有：{names}。"
        "请用完全一致的文件名重新调用。"
    )


@registry.tool
def list_documents() -> str:
    """列出已上传的资料文件（PDF / Word / PPT / 文本 / 日志），含类型、字数、段数。

    不知道有哪些文件、或者 read_document 报"没有该文件"时先调用它。
    它只给目录，不给正文——正文要用 read_document 分页读。
    """
    docs = documents()
    if not docs:
        return (
            "当前还没有上传任何资料文件。"
            "用 run_sql / run_python 处理已上传的数据表；"
            "文档里的数字要出图时，直接把两列数传给 make_chart 的 labels/values。"
        )

    lines = [f"已上传 {len(docs)} 份资料："]
    for index, doc in enumerate(docs, 1):
        detail = f"{doc.kind} · {doc.chars:,} 字 · {doc.parts} 段"
        if doc.note:
            detail += f" · {doc.note}"
        lines.append(f"{index}. `{doc.name}`（{detail}）")
    lines.append(
        '读正文：read_document(name="文件名", part=1)；'
        "找关键词：search_document(name=\"文件名\", keyword=\"...\")。"
    )
    return "\n".join(lines)


@registry.tool
def read_document(name: str, part: int = 1) -> str:
    """读取一份资料文件的第 part 段正文（每段约 2000 字）。

    返回结果末尾会告诉你「还有几段没读」，需要后文就换 part 继续。
    **不要一次读完整份**：先看 list_documents 里的段数，只读和问题相关的部分。

    适合：总结文档、回答"文件里怎么说的"、把文档内容和表数据对照。

    Args:
        name: 文件名，必须和 list_documents 显示的完全一致（含后缀）
        part: 第几段，从 1 开始
    """
    doc: Doc | None = find_document(name)
    if doc is None:
        return _missing(name)

    if not doc.text.strip():
        reason = doc.note or "没有抽到任何文字"
        return f"`{doc.name}` 里没有可读的文字：{reason}"

    try:
        part = int(part)
    except (TypeError, ValueError):
        return f"错误：part 必须是数字，收到的是 `{part}`。"

    if part < 1 or part > doc.parts:
        return (
            f"错误：`{doc.name}` 一共 {doc.parts} 段，part 必须在 1 到 {doc.parts} 之间，"
            f"收到的是 {part}。"
        )

    tail = f"\n\n（第 {part}/{doc.parts} 段，全文 {doc.chars:,} 字）"
    if part < doc.parts:
        tail += f"。后面还有 {doc.parts - part} 段，用 part={part + 1} 继续。"
    if doc.note and part == 1:
        tail += f"\n注意：{doc.note}"

    body = f"【{doc.name}】\n{doc.chunks[part - 1]}{tail}"
    return body[:MAX_OBS_CHARS]


@registry.tool
def search_document(name: str, keyword: str) -> str:
    """在一份资料文件里搜关键词，返回命中的原文片段和所在段号。

    比顺序读完省钱得多：先搜"违约金"，命中在第 4 段，再用 read_document(part=4) 读全段。

    适合：在长报告/日志里定位某个指标、某个日期、某条错误信息。
    不适合：问"整体讲了什么"——那种情况直接用 read_document 从头读。

    Args:
        name: 文件名，必须和 list_documents 显示的完全一致（含后缀）
        keyword: 要找的词，区分中英文大小写不敏感；越具体命中越准
    """
    doc: Doc | None = find_document(name)
    if doc is None:
        return _missing(name)

    keyword = (keyword or "").strip()
    if not keyword:
        return "错误：keyword 不能为空。给一个具体的词，比如「营收」或「ERROR」。"
    if not doc.text.strip():
        return f"`{doc.name}` 里没有可读的文字：{doc.note or '没有抽到任何文字'}"

    needle = keyword.lower()
    hits: list[str] = []
    total = 0

    for part_index, chunk in enumerate(doc.chunks, 1):
        lines = chunk.splitlines()
        for line_index, line in enumerate(lines):
            if needle not in line.lower():
                continue
            total += 1
            if len(hits) >= MAX_HITS:
                continue
            start = max(0, line_index - CONTEXT_LINES)
            end = min(len(lines), line_index + CONTEXT_LINES + 1)
            snippet = []
            for offset in range(start, end):
                content = lines[offset].strip()
                if not content:  # 段与段之间的空行不进结果，否则命中块看起来松松垮垮
                    continue
                mark = ">" if offset == line_index else " "
                snippet.append(f"{mark} {content[:160]}")
            hits.append(f"[第 {part_index} 段 · 第 {line_index + 1} 行]\n" + "\n".join(snippet))

    if not total:
        return (
            f"在 `{doc.name}` 里没找到「{keyword}」。\n"
            "建议：换一个更短的词、或换同义词再试；也可以 read_document 从第 1 段顺序读。"
        )

    header = f"在 `{doc.name}` 里命中 {total} 处「{keyword}」，显示前 {len(hits)} 处："
    tail = ""
    if total > len(hits):
        tail = f"\n\n（还有 {total - len(hits)} 处未显示。换个更具体的词能缩小范围。）"

    text = header + "\n\n" + "\n\n".join(hits) + tail
    text += "\n\n要看某处的前后完整内容，用 read_document(name, part=段号)。"
    return text[:MAX_OBS_CHARS]


__all__ = ["list_documents", "read_document", "search_document"]
