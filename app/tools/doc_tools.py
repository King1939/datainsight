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

search_document 的设计约束，都是"省 token"和"别骗模型"的具体做法：

  1. **默认跨全部资料，不必先指名文件。** 之前要求 name 必填，模型得先
     list_documents 猜哪份文件里可能有——猜错就白跑一轮。跨文件扫一遍很便宜，
     比多一轮对话划算得多。

  2. **只给候选卡片，不给正文。** 每段只回「文件名 + 段号 + 命中那一小截」，
     三段加起来几百字。旧版直接把命中的整行（连同上下文）塞回去，一份长文档
     命中八处就能撑满 4000 字符上限。现在让模型看完卡片再决定读哪一段，
     read_document 只读真正需要的那 2000 字。

  3. **两路召回再融合，而不是单靠字面。** 精确关键词（错误码、专有名词、多个
     指标）走 BM25；"这件事是怎么说的"这种问法走向量语义。两路排名用 RRF 合并，
     所以"词对得上"和"意思对得上"的段落都能进候选，不会因为用户换了个说法就零结果。
     向量没配置、超时、断网时自动退回纯 BM25，检索本身不会失败。
"""

from __future__ import annotations

from .. import images
from ..config import MAX_OBS_CHARS
from ..hybrid_search import hybrid_search
from ..readers import Doc
from . import registry
from .data_tools import documents, find_document

# 一次给几段候选。三段的量刚好够模型做一次选择，再多就是变相把全文倒进上下文。
MAX_CANDIDATES = 3


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
        img_part = f" · {doc.image_count} 图" if doc.image_count else ""
        detail = f"{doc.kind} · {doc.chars:,} 字 · {doc.parts} 段{img_part}"
        if doc.note:
            detail += f" · {doc.note}"
        lines.append(f"{index}. `{doc.name}`（{detail}）")
    lines.append(
        '读正文：read_document(name="文件名", part=1)；'
        '找关键词：search_document(keyword="...")（默认搜全部资料，'
        '加 name 可限定在某一份里）。'
    )
    return "\n".join(lines)


@registry.tool
def read_document(name: str, part: int = 1) -> str:
    """读取一份**已上传资料**（临时上传或储藏区的文档与图片文件）的第 part 段内容。

    适用对象：上面「资料文件」清单或 list_documents 列出的文件（包括 PDF / Word / PPT / Excel / PNG / JPG 等图片）。
    若该段含图片或本身是图片文件，会自动一并提交给模型查看。
    返回结果末尾会告诉你「还有几段没读」，需要后文就换 part 继续。
    **不要一次读完整份**：先看 list_documents 里的段数，只读和问题相关的部分。

    适合：总结文档、查看已上传图片内容、回答"文件里怎么说的"、把文档内容和表数据对照。

    Args:
        name: 文件名，必须和 list_documents 显示的完全一致（含后缀）
        part: 第几段，从 1 开始
    """
    doc: Doc | None = find_document(name)
    if doc is None:
        return _missing(name)

    if not doc.text.strip() and not doc.images:
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

    part_images = doc.images_for_part(part)
    attached = images.attach_step_images(part_images) if part_images else []

    tail = f"\n\n（第 {part}/{doc.parts} 段，全文 {doc.chars:,} 字）"
    if part < doc.parts:
        tail += f"。后面还有 {doc.parts - part} 段，用 part={part + 1} 继续。"
    if attached:
        names = "、".join(img.name for img in attached)
        tail += f"\n（本段附带 {len(attached)} 张图片：{names}，已提交给模型查看）"
    elif doc.images and part == 1:
        tail += f"\n（本文档共含 {len(doc.images)} 张图片，可在对应分段读取）"
    if doc.note and part == 1:
        tail += f"\n注意：{doc.note}"

    body = f"【{doc.name}】\n{doc.chunks[part - 1]}{tail}"
    return body[:MAX_OBS_CHARS]


def _card_line(rank: int, card) -> str:
    """一张候选卡片。三种来源的标记不同，模型据此判断"该不该信这一条"。"""
    head = (
        f"\n[{rank}] `{card.doc.name}` · 第 {card.part}/{card.total_parts} 段"
    )

    if card.bm25_rank is not None and card.vector_rank is not None:
        matched = card.matched_terms
        words = "、".join(matched[:4]) + (f" 等{len(matched)}个" if len(matched) > 4 else "")
        return (
            f"{head} · 向量+BM25双命中 · 命中 {len(matched)} 个词（{words}）"
            f" · 语义 {card.vector_score:.2f}\n    {card.snippet}"
        )

    if card.bm25_rank is not None:
        matched = card.matched_terms
        # 只列前四个词：再长就变成"解释我为什么排这个序"，模型不需要
        words = "、".join(matched[:4]) + (f" 等{len(matched)}个" if len(matched) > 4 else "")
        return f"{head} · 命中 {len(matched)} 个词（{words}）\n    {card.snippet}"

    return f"{head} · 语义匹配（相似度 {card.vector_score:.2f}）\n    {card.snippet}"


@registry.tool
def search_document(keyword: str, name: str = "") -> str:
    """在资料里搜关键词，返回最相关的几段"候选卡片"（文件名 + 段号 + 命中片段）。

    默认在**全部**资料里搜，不用先知道是哪份文件；要限定在某一份里就传 name。
    既支持精确关键词，也支持用一句话描述你要找的内容：

    - 精确：错误码、专有名词、条款名、多个指标；多个词用空格或逗号分开（OR）。
    - 语义：一句话描述（"未履行协议怎么算赔偿"），同义改写也能召回。
    - 两者会一起用，同时命中的段落排最前。

    比顺序读完省钱得多：拿到段号后用 read_document(name, part=段号) 只读那一段。

    适合：在长报告/日志里定位某个指标、某个日期、某条错误信息、某个条款、
    或者某个用自己话说不清但意思明确的概念。
    不适合：问"整体讲了什么"——那种情况直接用 read_document 从头读。

    Args:
        keyword: 要找的词或一句话；多个词用空格或逗号分隔；英文不区分大小写
        name: 可选，限定在某一份文件里搜；必须和 list_documents 显示的完全一致（含后缀）
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return "错误：keyword 不能为空。给一个具体的词，比如「营收」或「ERROR」。"

    if name:
        doc = find_document(name)
        if doc is None:
            return _missing(name)
        docs = [doc]
        where = f"在 `{doc.name}` 里"
    else:
        docs = documents()
        if not docs:
            return "错误：当前还没有上传任何资料文件。先用 list_documents 看看有没有东西可查。"
        where = f"在 {len(docs)} 份资料里"

    result = hybrid_search(keyword, docs, top_k=MAX_CANDIDATES)

    if not result.candidates:
        prefix = (
            "（向量检索暂时不可用，已尝试 BM25 词匹配。）\n"
            if result.vector_degraded
            else ""
        )
        return prefix + (
            f"{where}没找到「{keyword}」。\n"
            "建议：换一个更短的词、或换同义词再试；"
            "也可以 read_document 从第 1 段顺序读。"
        )

    lines: list[str] = []
    if result.vector_degraded:
        lines.append("（向量检索暂时不可用，已自动降级为 BM25 词匹配。）")
    elif not result.used_vector and result.fuzzy:
        lines.append("（整词没命中，下面是按字面模糊匹配的结果，可能是同义也可能只是字面相近，请自行判断。）")

    shown = result.candidates
    if result.used_vector and result.used_bm25:
        lines.append(
            f"{where}混合检索（向量+BM25+RRF）命中「{keyword}」，"
            f"共 {result.total_candidates} 段候选，最相关的 {len(shown)} 段："
        )
    elif result.used_vector:
        lines.append(
            f"{where}语义检索命中「{keyword}」，"
            f"共 {result.total_candidates} 段候选，最相关的 {len(shown)} 段："
        )
    else:
        lines.append(
            f"{where}命中「{keyword}」，"
            f"共 {result.total_candidates} 段候选，最相关的 {len(shown)} 段："
        )

    for rank, card in enumerate(shown, 1):
        lines.append(_card_line(rank, card))

    # 直接把下一步该写的调用拼好：模型照抄就行，省得它自己拼错文件名
    top = shown[0]
    lines.append(
        f"\n读其中一段的完整内容："
        f'read_document(name="{top.doc.name}", part={top.part})'
    )
    if result.total_candidates > len(shown):
        lines.append(
            f"还有 {result.total_candidates - len(shown)} 段没列出来。想更准就多加一个词缩小范围。"
        )
    return "\n".join(lines)[:MAX_OBS_CHARS]


__all__ = ["list_documents", "read_document", "search_document"]
